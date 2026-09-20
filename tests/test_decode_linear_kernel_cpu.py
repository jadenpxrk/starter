"""Execute the kernel Python bodies with CPU pointer/arithmetic semantics.

This is NOT Triton interpretation, compilation, or GPU validation. It detects
indexing, masking, coverage, and cast-placement defects in the authored bodies.
Run: python -m unittest discover -s tests -p test_decode_linear_kernel_cpu.py -v
"""

import ast
from pathlib import Path
import unittest

import torch


class Pointer:
    def __init__(self, data, offset=0, writes=None):
        self.data = data.view(-1)
        self.offset = offset
        self.writes = writes

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset, self.writes)


class Language:
    float32 = torch.float32
    bfloat16 = torch.bfloat16
    constexpr = int
    pid = (0, 0, 0)

    def program_id(self, axis):
        return self.pid[axis]

    @staticmethod
    def cdiv(a, b):
        return (a + b - 1) // b

    arange = staticmethod(torch.arange)
    @staticmethod
    def zeros(shape, dtype):
        return torch.zeros(shape, dtype=dtype)

    trans = staticmethod(torch.t)

    @staticmethod
    def sum(x, axis):
        return x.sum(dim=axis)

    @staticmethod
    def dot(a, b, acc, input_precision=None):
        assert a.dtype == b.dtype == torch.bfloat16
        assert acc.dtype == torch.float32 and input_precision == "ieee"
        return acc + a.float() @ b.float()

    @staticmethod
    def load(pointer, mask=None, other=0):
        index = torch.as_tensor(pointer.offset).long()
        keep = torch.ones_like(index, dtype=torch.bool) if mask is None else mask.expand(index.shape)
        valid = index[keep]
        assert bool(((valid >= 0) & (valid < pointer.data.numel())).all())
        index = torch.where(keep, index, 0)
        return torch.where(keep, pointer.data[index], torch.as_tensor(other, dtype=pointer.data.dtype))

    @staticmethod
    def store(pointer, value, mask=None):
        index = torch.as_tensor(pointer.offset).long()
        keep = torch.ones_like(index, dtype=torch.bool) if mask is None else mask.expand(index.shape)
        index, value = index[keep], value.expand(keep.shape)[keep]
        assert bool(((index >= 0) & (index < pointer.data.numel())).all())
        pointer.data[index] = value.to(pointer.data.dtype)
        if pointer.writes is not None:
            pointer.writes.view(-1).scatter_add_(0, index, torch.ones_like(index, dtype=torch.int32))


def load_bodies():
    file = Path(__file__).resolve().parents[1] / "engine/kernels/decode_linear.py"
    definitions = []
    for node in ast.parse(file.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_gemv", "_small_mm", "_finish"):
            node.decorator_list = []
            definitions.append(node)
    tl = Language()
    namespace = {"tl": tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), str(file), "exec"), namespace)
    return tl, namespace


def simulate_mm(x, weight, split):
    tl, functions = load_bodies()
    m, k = x.shape
    n = weight.shape[0]
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16)
    partial = torch.full((split, m, n), float("nan"), dtype=torch.float32)
    ow, pw = torch.zeros_like(out, dtype=torch.int32), torch.zeros_like(partial, dtype=torch.int32)
    for sn in range((n + 63) // 64):
        for sm in range((m + 15) // 16):
            for sk in range(split):
                tl.pid = (sn, sm, sk)
                functions["_small_mm"](Pointer(x), Pointer(weight), Pointer(out, writes=ow),
                                       Pointer(partial, writes=pw), m, n, k, split, 16, 64, 64)
    if split > 1:
        assert bool((pw == 1).all()), "Every FP32 partial must be overwritten exactly once"
        assert bool(torch.isfinite(partial).all())
        for block in range((m * n + 255) // 256):
            tl.pid = (block, 0, 0)
            functions["_finish"](Pointer(partial), Pointer(out, writes=ow), m * n, split, 256)
    assert bool((ow == 1).all()), "Every output must be written exactly once"
    return out, partial


class KernelCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_simt_body_masks_and_exact_products(self):
        torch.manual_seed(117)
        tl, functions = load_bodies()
        for n in (1, 9, 65):
            for k in (3, 513, 2560, 9728):
                with self.subTest(n=n, k=k):
                    x = (torch.randint(-8, 9, (1, k)).float() / 8).bfloat16()
                    w = (torch.randint(-8, 9, (n, k)).float() / 8).bfloat16()
                    saved = [t.clone() for t in (x, w)]
                    out = torch.full((1, n), float("nan"), dtype=torch.bfloat16)
                    writes = torch.zeros_like(out, dtype=torch.int32)
                    for block in range((n + 7) // 8):
                        tl.pid = (block, 0, 0)
                        functions["_gemv"](Pointer(x), Pointer(w), Pointer(out, writes=writes), n, k, 8, 512)
                    torch.testing.assert_close(out, (x.float() @ w.float().T).bfloat16(), rtol=0, atol=0)
                    self.assertTrue(bool((writes == 1).all()))
                    for t, old in zip((x, w), saved):
                        torch.testing.assert_close(t, old, rtol=0, atol=0)

    def test_tensorcore_body_coverage_empty_splits_and_casts(self):
        torch.manual_seed(23)
        cases = ((1, 7, 3), (3, 65, 257), (16, 129, 2560), (17, 71, 513), (32, 9, 9728))
        for m, n, k in cases:
            # These exact dyadic products remove FP32 associativity uncertainty
            # and make a strict bitwise indexing/cast check possible on CPU.
            x = (torch.randint(-8, 9, (m, k)).float() / 8).bfloat16()
            w = (torch.randint(-8, 9, (n, k)).float() / 8).bfloat16()
            saved = [t.clone() for t in (x, w)]
            for split in (1, 2, 4, 8, 16):
                with self.subTest(m=m, n=n, k=k, split=split):
                    actual, _ = simulate_mm(x, w, split)
                    torch.testing.assert_close(actual, (x.float() @ w.float().T).bfloat16(), rtol=0, atol=0)
                    for t, old in zip((x, w), saved):
                        torch.testing.assert_close(t, old, rtol=0, atol=0)

    def test_bf16_partial_sum_negative_control(self):
        torch.manual_seed(123)
        x = (torch.randint(-8, 9, (3, 2560)).float() / 8).bfloat16()
        w = (torch.randint(-8, 9, (65, 2560)).float() / 8).bfloat16()
        actual, partial = simulate_mm(x, w, 4)
        bad = partial.bfloat16().float().sum(0).bfloat16()
        self.assertGreater(int((bad != actual).sum()), 0)
        torch.testing.assert_close(actual, (x.float() @ w.float().T).bfloat16(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
