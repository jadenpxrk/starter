"""Decode projection checks. CPU kernel-body execution through a Torch shim is
NOT Triton or CUDA validation; it checks indexing, masking, split coverage and
cast placement of the actual kernel source. Performance is not measured here."""

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
KERNELS = ENGINE / "kernels"
HAS_TRITON = importlib.util.find_spec("triton") is not None


def source_functions(path, names, env):
    """Execute selected function bodies from a kernel file without Triton."""
    tree = ast.parse(path.read_text())
    picked = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            for arg in node.args.args:
                arg.annotation = None
            picked.append(node)
    module = ast.Module(body=picked, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env


class Pointer:
    def __init__(self, data, offsets=0, writes=None):
        self.data, self.offsets = data.reshape(-1), offsets
        self.writes = torch.zeros(self.data.numel(), dtype=torch.int64) if writes is None else writes
        self.dtype = SimpleNamespace(element_ty=self.data.dtype)

    def __add__(self, value):
        return Pointer(self.data, self.offsets + value, self.writes)


class Ops:
    """Checked CPU pointer/math shim: bounds, masks, dtypes; no warps or codegen."""
    int64, float32, bfloat16 = torch.int64, torch.float32, torch.bfloat16
    constexpr = int
    arange, rsqrt, exp = staticmethod(torch.arange), staticmethod(torch.rsqrt), staticmethod(torch.exp)
    static_range = staticmethod(range)
    math = SimpleNamespace(rsqrt=torch.rsqrt)
    program = (0, 0)

    def program_id(self, axis):
        return self.program[axis]

    @staticmethod
    def zeros(shape, dtype):
        return torch.zeros(shape, dtype=dtype)

    @staticmethod
    def sum(x, axis):
        return x.sum(dim=axis)

    @staticmethod
    def where(test, a, b):
        return torch.where(torch.as_tensor(test), torch.as_tensor(a), torch.as_tensor(b))

    @staticmethod
    def trans(x):
        return x.T

    @staticmethod
    def dot(a, b, acc=None):
        # BF16 operands, FP32 accumulate: the tensor-core arithmetic class.
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
        assert a.shape[0] >= 16 and b.shape[1] >= 16 and a.shape[1] >= 16
        product = a.float() @ b.float()
        return product if acc is None else acc + product

    @staticmethod
    def _indices(p, mask):
        i = torch.as_tensor(p.offsets, dtype=torch.int64)
        if mask is not None:
            mask = torch.as_tensor(mask)
            i, mask = torch.broadcast_tensors(i, mask)
            i = torch.where(mask, i, torch.zeros_like(i))
        if bool(((i < 0) | (i >= p.data.numel())).any()):
            raise AssertionError("out-of-bounds pointer access")
        return i, mask

    def load(self, p, mask=None, other=0.0):
        i, mask = self._indices(p, mask)
        values = p.data[i]
        if mask is not None:
            values = torch.where(mask, values, torch.full_like(values, other))
        return values

    def store(self, p, value, mask=None):
        i, mask = self._indices(p, mask)
        value = torch.as_tensor(value)
        if mask is not None:
            value, _ = torch.broadcast_tensors(value, mask)
            i, value = i[mask], value[mask]
        p.data[i] = value.to(p.data.dtype)
        p.writes.scatter_add_(0, i.flatten(), torch.ones_like(i.flatten()))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if HAS_TRITON:
    skinny = load_module("skinny_gemm_under_test", KERNELS / "skinny_gemm.py")
else:
    with mock.patch.dict(sys.modules, {"triton": mock.Mock(), "triton.language": mock.Mock()}):
        skinny = load_module("skinny_gemm_under_test", KERNELS / "skinny_gemm.py")

BN, BK = skinny.BN, skinny.BK


def fake(shape, dtype=torch.bfloat16, cuda=True, contiguous=True):
    return SimpleNamespace(ndim=len(shape), shape=tuple(shape), dtype=dtype, is_cuda=cuda,
                           device=torch.device("cuda" if cuda else "cpu"),
                           is_contiguous=lambda: contiguous)


class CpuKernelBodies:
    def __init__(self):
        self.ops = Ops()
        env = source_functions(KERNELS / "skinny_gemm.py",
                               {"_partials_kernel", "_silu_mul_kernel", "_qkv_head_kernel"},
                               {"tl": self.ops})
        self.partials, self.silu = env["_partials_kernel"], env["_silu_mul_kernel"]
        self.qkv_head = env["_qkv_head_kernel"]

    def run_partials(self, x, w, splits, bm):
        m, k = x.shape
        n = w.shape[0]
        out = torch.full((splits, m, n), float("nan"))
        xp, wp, pp = Pointer(x), Pointer(w), Pointer(out)
        for column in range(n // BN):
            for split in range(splits):
                self.ops.program = (column, split)
                self.partials(xp, wp, pp, m, n, k, k // splits, bm, BN, BK)
        assert bool((pp.writes == 1).all()), "every partial written exactly once"
        assert bool((xp.writes == 0).all()) and bool((wp.writes == 0).all())
        return out

    def run_qkv_head(self, x, w, qw, kw, cos, sin, position, keys, values, bm, eps=(1e-6, 1e-6)):
        m, k = x.shape
        batch, nk, capacity, d = keys.shape
        nq = w.shape[0] // d - 2 * nk
        q = torch.full((m, nq, 1, d), float("nan")).bfloat16()
        pointers = [Pointer(t) for t in (x, w, qw, kw, cos, sin, position, q, keys, values)]
        for head in range(nq + 2 * nk):
            self.ops.program = (head, 0)
            self.qkv_head(*pointers, m, k, capacity, nq, nk, d, eps[0], eps[1], bm, BK)
        assert bool((pointers[7].writes == 1).all()), "every Q value written exactly once"
        slot = torch.zeros(batch, nk, capacity, d, dtype=torch.int64)
        slot[:, :, int(position)] = 1
        for pointer in pointers[8:]:
            assert torch.equal(pointer.writes.view_as(slot), slot), "only the current slot is written"
        assert all(bool((pointer.writes == 0).all()) for pointer in pointers[:7]), "inputs are read-only"
        return q

    def run_silu(self, x, w, bm):
        m, k = x.shape
        n = w.shape[0] // 2
        out = torch.full((m, n), float("nan")).bfloat16()
        xp, wp, yp = Pointer(x), Pointer(w), Pointer(out)
        for column in range(n // BN):
            self.ops.program = (column, 0)
            self.silu(xp, wp, yp, m, n, k, bm, BN, BK)
        assert bool((yp.writes == 1).all())
        return out


def reference_silu_mul(x, w):
    gate, up = F.linear(x.float(), w.float()).to(torch.bfloat16).chunk(2, dim=-1)
    return (F.silu(gate.float()).to(torch.bfloat16).float() * up.float()).to(torch.bfloat16)


def restate_qkv_head(projection, qw, kw, cos, sin, position, nq, nk, d, eps=(1e-6, 1e-6)):
    """The fused epilogue's cast chain in Torch: per-head halves, same order.

    Returns (q [B,Nq,1,D], k [B,Nkv,D], v [B,Nkv,D]) in BF16 from a BF16 projection.
    """
    batch = projection.shape[0]
    rows = projection.float().view(batch, nq + 2 * nk, d)
    left, right = rows[..., :d // 2], rows[..., d // 2:]
    outs = []
    for kind, (gain, e) in enumerate(((qw, eps[0]), (kw, eps[1]))):
        sel = slice(0, nq) if kind == 0 else slice(nq, nq + nk)
        l, r = left[:, sel], right[:, sel]
        inv = torch.rsqrt(((l * l).sum(-1) + (r * r).sum(-1)) / d + e)
        nl = (l * inv[..., None]).bfloat16().float()
        nr = (r * inv[..., None]).bfloat16().float()
        nl = (nl * gain[:d // 2].float()).bfloat16().float()
        nr = (nr * gain[d // 2:].float()).bfloat16().float()
        c, si = cos[position].float(), sin[position].float()
        ol = (nl * c[:d // 2]).bfloat16().float() + (-nr * si[:d // 2]).bfloat16().float()
        orr = (nr * c[d // 2:]).bfloat16().float() + (nl * si[d // 2:]).bfloat16().float()
        outs.append(torch.cat((ol, orr), -1).bfloat16())
    v = rows[:, nq + nk:].bfloat16()
    return outs[0].unsqueeze(2), outs[1], v


def sum_partials_bf16(partials):
    # Match the consumer's sequential FP32 order before its single BF16 cast.
    total = torch.zeros_like(partials[0])
    for partial in partials:
        total = total + partial
    return total.to(torch.bfloat16)


class SkinnyCPU(unittest.TestCase):
    def test_split_plan_for_qwen3_4b_and_alignment(self):
        # Packed Q/K/V [6144, 2560], o_proj [2560, 4096], down_proj [2560, 9728].
        for n, k, expected in ((6144, 2560, 4), (2560, 4096, 8), (2560, 9728, 8)):
            with self.subTest(n=n, k=k):
                splits = skinny.split_count(n, k)
                self.assertEqual(splits, expected)
                self.assertEqual(k % (splits * BK), 0)
                self.assertGreaterEqual(n // BN * splits, min(skinny.TARGET_PROGRAMS, 320))
        self.assertEqual(skinny.split_count(1024, 64), 1)  # K = one tile: no split possible
        self.assertEqual(skinny.split_count(64, 128), 2)   # K = 2 tiles: exactly two splits
        # gate/up [19456, 2560] pairs never split (SiLU needs full K); 304 programs.
        self.assertEqual(19456 // 2 // BN, 304)

    def test_supports_is_a_static_layout_rule(self):
        x, w = fake((4, 2560)), fake((6144, 2560))
        self.assertTrue(skinny.supports(x, w))
        self.assertTrue(skinny.supports(fake((skinny.MAX_ROWS, 2560)), w))
        self.assertTrue(skinny.supports(x, fake((19456, 2560)), pairs=True))
        for bad_x, bad_w, pairs in (
            (fake((skinny.MAX_ROWS + 1, 2560)), w, False),
            (fake((0, 2560)), w, False),
            (fake((4, 2560), cuda=False), fake((6144, 2560), cuda=False), False),
            (fake((4, 2560), dtype=torch.float16), w, False),
            (fake((4, 2560), contiguous=False), w, False),
            (x, fake((6144, 2560), contiguous=False), False),
            (fake((4, 2570)), fake((6144, 2570)), False),        # K not a BK multiple
            (x, fake((6144 + 8, 2560)), False),                  # N not a BN multiple
            (x, fake((2 * BN + BN, 2560)), True),                # pairs need 2*BN
            (fake((4, 2560)), fake((6144, 2561)), False),        # K mismatch
            (fake((2, 4, 2560)), w, False),
        ):
            self.assertFalse(skinny.supports(bad_x, bad_w, pairs=pairs))
        with self.assertRaises(ValueError):
            skinny._dims(fake((4, 2560), cuda=False), fake((6144, 2560), cuda=False))

    def test_kernel_bodies_partials_cover_k_exactly_once(self):
        torch.manual_seed(7)
        bodies = CpuKernelBodies()
        for m, n, k, splits, bm in ((1, 64, 128, 2, 16), (3, 32, 256, 4, 16), (16, 96, 64, 1, 16),
                                    (17, 64, 128, 1, 32), (5, 64, 192, 1, 16)):
            with self.subTest(m=m, n=n, k=k, splits=splits):
                x = torch.randn(m, k).bfloat16()
                w = torch.randn(n, k).bfloat16()
                partials = bodies.run_partials(x, w, splits, bm)
                self.assertTrue(bool(torch.isfinite(partials).all()))
                per_split = k // splits
                for s in range(splits):
                    chunk = slice(s * per_split, (s + 1) * per_split)
                    expected = x[:, chunk].float() @ w[:, chunk].float().T
                    torch.testing.assert_close(partials[s], expected, rtol=1e-5, atol=1e-5)
                # The consumer's sum equals the full-K product up to summation order.
                torch.testing.assert_close(partials.sum(0), x.float() @ w.float().T, rtol=1e-4, atol=1e-4)

    def test_kernel_body_silu_epilogue_cast_order(self):
        torch.manual_seed(8)
        bodies = CpuKernelBodies()
        for m, n, k, bm in ((1, 64, 128, 16), (4, 32, 64, 16), (16, 64, 192, 16), (33, 32, 64, 64)):
            with self.subTest(m=m, n=n, k=k):
                x = torch.randn(m, k).bfloat16() * 2
                w = torch.randn(2 * n, k).bfloat16()
                out = bodies.run_silu(x, w, bm)
                self.assertEqual(out.dtype, torch.bfloat16)
                self.assertTrue(bool(torch.isfinite(out).all()))
                torch.testing.assert_close(out, reference_silu_mul(x, w), rtol=1e-2, atol=1e-3)
        # Negative control: skipping the BF16 rounding of the projections is a different function.
        x = torch.randn(64, 256).bfloat16() * 3
        w = torch.randn(128, 256).bfloat16()
        gate, up = F.linear(x.float(), w.float()).chunk(2, dim=-1)
        unrounded = (F.silu(gate) * up).to(torch.bfloat16)
        self.assertFalse(torch.equal(unrounded, reference_silu_mul(x, w)))

    def test_kernel_body_fused_qkv_head_epilogue(self):
        """Fused head body: rounded projection -> norm -> gain -> RoPE -> Q or cache slot."""
        torch.manual_seed(10)
        bodies = CpuKernelBodies()
        ops = Ops()
        consumer = source_functions(KERNELS / "decode_fused.py", {"_qkv_norm_rope_cache_kernel"},
                                    {"tl": ops})["_qkv_norm_rope_cache_kernel"]
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        for batch, nq, nk, d, k, bm in ((3, 4, 2, 32, 128, 16), (1, 2, 1, 128, 64, 16),
                                        (17, 3, 1, 32, 64, 32)):
            with self.subTest(batch=batch, nq=nq, nk=nk, d=d, k=k):
                capacity, position = 5, torch.tensor([3])
                x = torch.randn(batch, k).bfloat16()
                w = torch.randn((nq + 2 * nk) * d, k).bfloat16() / k ** 0.5
                qw, kw = torch.randn(d).bfloat16(), torch.randn(d).bfloat16()
                cos, sin = torch.randn(capacity, d).bfloat16(), torch.randn(capacity, d).bfloat16()
                keys = torch.full((batch, nk, capacity, d), 33.).bfloat16()
                values = keys.clone()
                q = bodies.run_qkv_head(x, w, qw, kw, cos, sin, position, keys, values, bm)
                self.assertEqual(q.shape, (batch, nq, 1, d))
                self.assertTrue(bool(torch.isfinite(q).all()))
                projection = F.linear(x.float(), w.float()).bfloat16()
                # Exact against the restated chain from the same BF16 projection.
                eq, ek, ev = restate_qkv_head(projection, qw, kw, cos, sin, 3, nq, nk, d)
                torch.testing.assert_close(q, eq, rtol=0, atol=0)
                torch.testing.assert_close(keys[:, :, 3], ek, rtol=0, atol=0)
                torch.testing.assert_close(values[:, :, 3], ev, rtol=0, atol=0)
                # Same function as the split-K consumer body, up to the FP32 norm reduction order.
                ck = torch.full_like(keys, 33.)
                cv = torch.full_like(values, 33.)
                cq = torch.empty_like(q)
                width = (nq + 2 * nk) * d
                pointers = [Pointer(t) for t in (projection, qw, kw, cos, sin, position, cq, ck, cv)]
                for row in range(batch * (nq + nk)):
                    ops.program = (row, 0)
                    consumer(*pointers, width, capacity, batch * width, nq, nk, d, 1e-6, 1e-6, 0)
                torch.testing.assert_close(q, cq, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(keys, ck, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(values, cv, rtol=0, atol=0)
                # Native formula: per-head RMSNorm with gain, then HF rotary at the slot.
                pq, pk, pv = projection.split([nq * d, nk * d, nk * d], dim=-1)
                norm = lambda t, g: (t.float() * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True)
                                                              + 1e-6)).bfloat16() * g
                nq_, nk_ = apply_rotary_pos_emb(norm(pq.view(batch, nq, 1, d), qw),
                                                norm(pk.view(batch, nk, 1, d), kw),
                                                cos[3].view(1, 1, d), sin[3].view(1, 1, d))
                torch.testing.assert_close(q, nq_, rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(keys[:, :, 3], nk_[:, :, 0], rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(values[:, :, 3], pv.view(batch, nk, d), rtol=0, atol=0)
                self.assertTrue(bool((keys[:, :, :3] == 33).all()) and bool((keys[:, :, 4:] == 33).all()))
        # Negative control: an unrounded projection is a different function of the same inputs.
        unrounded = F.linear(x.float(), w.float())
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(
                q, restate_qkv_head(unrounded, qw, kw, cos, sin, 3, nq, nk, d)[0], rtol=0, atol=0)

    def test_supports_qkv_is_static_and_wrapper_validates(self):
        x, w = fake((4, 2560)), fake((6144, 2560))
        keys = fake((4, 8, 600, 128))
        self.assertTrue(skinny.supports_qkv(x, w, keys))
        self.assertTrue(skinny.supports_qkv(fake((skinny.MAX_ROWS, 2560)), w, fake((64, 8, 9, 128))))
        for bad_x, bad_w, bad_keys in (
                (fake((skinny.MAX_ROWS + 1, 2560)), w, fake((65, 8, 600, 128))),
                (x, w, fake((5, 8, 600, 128))),           # cache batch differs from rows
                (x, w, fake((4, 8, 600, 64))),            # head width is not 128
                (x, fake((6144 + 32, 2560)), keys),       # rows are not whole heads
                (x, fake((2048, 2560)), keys),            # no query heads left
                (x, w, fake((4, 600, 128))),              # not [B, Nkv, C, D]
                (fake((4, 2560), cuda=False), fake((6144, 2560), cuda=False), fake((4, 8, 600, 128), cuda=False)),
                (x, fake((6144, 2560), contiguous=False), keys)):
            self.assertFalse(skinny.supports_qkv(bad_x, bad_w, bad_keys))
        # The wrapper refuses CPU tensors before any launch.
        cpu = torch.zeros(2, 64, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            skinny.linear_qkv_norm_rope_cache(cpu, torch.zeros(6 * 128, 64, dtype=torch.bfloat16),
                                              None, None, 0, 0, None, None, None,
                                              torch.zeros(2, 2, 3, 128, dtype=torch.bfloat16), None)

    def test_consumer_kernels_add_partials_then_round_once(self):
        """Run the modified consumer bodies with SPLITS=0 and SPLITS=S on CPU."""
        torch.manual_seed(9)
        ops = Ops()
        env = source_functions(KERNELS / "decode_fused.py",
                               {"_add_rms_norm_kernel", "_qkv_norm_rope_cache_kernel"}, {"tl": ops})
        add_norm, qkv_kernel = env["_add_rms_norm_kernel"], env["_qkv_norm_rope_cache_kernel"]

        rows, cols, splits = 3, 64, 4
        x = torch.randn(rows, cols).bfloat16()
        w = torch.randn(cols).bfloat16()
        partials = torch.randn(splits, rows, cols) * 0.5
        branch = sum_partials_bf16(partials)

        def run_add(y, s):
            total, normed = torch.empty_like(x), torch.empty_like(x)
            pointers = [Pointer(t) for t in (x, y, w, total, normed)]
            for row in range(rows):
                ops.program = (row, 0)
                add_norm(*pointers, cols, 1e-6, rows * cols, 64, s)
            self.assertTrue(bool((pointers[3].writes == 1).all()) and bool((pointers[4].writes == 1).all()))
            return total, normed

        direct = run_add(branch, 0)
        summed = run_add(partials, splits)
        for a, b in zip(direct, summed):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        expected_total = (x.float() + branch.float()).to(torch.bfloat16)
        torch.testing.assert_close(summed[0], expected_total, rtol=0, atol=0)
        f = expected_total.float()
        expected_normed = (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * w
        torch.testing.assert_close(summed[1], expected_normed, rtol=0, atol=0)
        # Partials whose FP32 sum differs from the BF16 branch by less than half an ulp round identically.
        nudged = partials.clone()
        nudged[0] += 1e-6
        torch.testing.assert_close(run_add(nudged, splits)[0], summed[0], rtol=0, atol=0)

        batch, nq, nk, d, capacity = 2, 4, 2, 32, 5
        width = (nq + 2 * nk) * d
        qkv_partials = torch.randn(splits, batch, width) * 0.5
        qkv = sum_partials_bf16(qkv_partials)
        qw, kw = torch.randn(d).bfloat16(), torch.randn(d).bfloat16()
        cos, sin = torch.randn(capacity, d).bfloat16(), torch.randn(capacity, d).bfloat16()
        position = torch.tensor([3])

        def run_qkv(source, s):
            keys = torch.full((batch, nk, capacity, d), 33.).bfloat16()
            values = torch.full_like(keys, 33.)
            q = torch.empty(batch, nq, 1, d).bfloat16()
            pointers = [Pointer(t) for t in (source, qw, kw, cos, sin, position, q, keys, values)]
            for row in range(batch * (nq + nk)):
                ops.program = (row, 0)
                qkv_kernel(*pointers, width, capacity, batch * width, nq, nk, d, 1e-6, 1e-6, s)
            self.assertTrue(bool((pointers[6].writes == 1).all()))
            slot = torch.zeros(batch, nk, capacity, d, dtype=torch.int64)
            slot[:, :, 3] = 1
            for p in pointers[7:]:
                self.assertTrue(torch.equal(p.writes.view_as(slot), slot))
            return q, keys, values

        direct = run_qkv(qkv, 0)
        summed = run_qkv(qkv_partials, splits)
        for a, b in zip(direct, summed):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        v_expected = qkv[:, (nq + nk) * d:].view(batch, nk, d)
        torch.testing.assert_close(summed[2][:, :, 3], v_expected, rtol=0, atol=0)
        self.assertTrue(bool((summed[1][:, :, :3] == 33).all()) and bool((summed[1][:, :, 4:] == 33).all()))

    def test_consumer_wrappers_validate_partials(self):
        with mock.patch.dict(sys.modules, {"triton": mock.Mock(), "triton.language": mock.Mock()}):
            fused = load_module("decode_fused_under_test", KERNELS / "decode_fused.py")
        device = torch.device("cuda")
        self.assertEqual(fused._splits(fake((3, 5, 64), torch.float32), (5, 64), device), 3)
        for bad in (fake((3, 5, 65), torch.float32), fake((0, 5, 64), torch.float32),
                    fake((3, 5, 64), torch.float32, contiguous=False),
                    fake((3, 5, 64), torch.float32, cuda=False), fake((6, 64)), fake((5, 64), cuda=False)):
            with self.assertRaises(ValueError):
                fused._splits(bad, (5, 64), device)
        # A CPU residual is rejected before any launch, with or without partials.
        residual = torch.zeros(2, 64, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            fused.add_rms_norm(residual, torch.zeros(2, 2, 64), torch.ones(64, dtype=torch.bfloat16), 1e-6)
        with self.assertRaises(ValueError):
            fused.qkv_norm_rope_cache(torch.zeros(2, 2, 256), None, None, 0, 0, None, None, None, None, None)

    @torch.inference_mode()
    def test_fused_step_with_partials_matches_native(self):
        """Orchestration: decode_step routes every projection through the split path."""
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("Transformers required")
        from transformers import Qwen3Config, Qwen3ForCausalLM
        sys.path.insert(0, str(ENGINE))
        try:
            with mock.patch.dict(sys.modules, {
                    "kernels.qk_norm_rope": mock.Mock(), "kernels.decode_fused": mock.Mock(),
                    "kernels.skinny_gemm": mock.Mock(supports=mock.Mock(return_value=False),
                                             supports_qkv=mock.Mock(return_value=False))}):
                import decode_step
                import qk_norm_rope as adapter
            from decode import DecodeState
            from decode_attention import install_decode_attention
            from flash_decode import FlashDecodeContext
            from mlp import PackedMLP
        finally:
            sys.path.pop(0)

        config = Qwen3Config(vocab_size=127, hidden_size=64, intermediate_size=96,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=16, max_position_embeddings=32768, rope_theta=5_000_000,
                             tie_word_embeddings=True, sliding_window=None)
        config._attn_implementation = "sdpa"
        torch.manual_seed(1)
        model = Qwen3ForCausalLM(config).eval().to(torch.bfloat16)
        candidate = deepcopy(model)
        install_decode_attention(candidate)
        for layer in candidate.model.layers:
            layer.mlp = PackedMLP(layer.mlp)
            layer.self_attn = adapter.DecodeQKNormRoPE(layer.self_attn)

        class CPUContext(FlashDecodeContext):
            def attention(self, query, key, value, mask, scale):
                batch, heads, _, width = query.shape
                kv_heads = key.shape[1]
                visible = torch.arange(key.shape[2]).view(1, 1, 1, -1) < self.used.view(batch, kv_heads, 1, 1)
                out = F.scaled_dot_product_attention(query.view(batch, kv_heads, heads // kv_heads, width),
                                                     key, value, attn_mask=visible, scale=scale)
                return out.reshape(batch, 1, heads, width)

        def norm(x, weight, eps):
            f = x.float()
            return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight

        def partials(x, weight):
            splits = 4 if x.shape[1] % (4 * BK) == 0 else 1
            per = x.shape[1] // splits
            return torch.stack([x[:, s*per:(s+1)*per].float() @ weight[:, s*per:(s+1)*per].float().T
                                for s in range(splits)])

        def fold(operand):
            return operand.sum(0).to(torch.bfloat16) if operand.dtype == torch.float32 else operand

        def add_rms_norm(residual, branch, weight, eps):
            total = residual + fold(branch)
            return total, norm(total, weight, eps)

        def qkv_norm_rope_cache(qkv, qw, kw, qe, ke, cos, sin, position, keys, values):
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            qkv = fold(qkv)
            batch, kv_heads, _, width = keys.shape
            heads = qkv.shape[1] // width - 2 * kv_heads
            q, k, v = qkv.split([heads * width, kv_heads * width, kv_heads * width], dim=-1)
            q = norm(q.reshape(batch, heads, 1, width), qw, qe)
            k = norm(k.reshape(batch, kv_heads, 1, width), kw, ke)
            c, s = cos.index_select(0, position).unsqueeze(0), sin.index_select(0, position).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, c, s)
            keys.index_copy_(2, position, k)
            values.index_copy_(2, position, v.reshape(batch, kv_heads, 1, width))
            return q.contiguous()

        calls = {"partials": 0, "silu": 0, "supports": 0, "qkv": 0}

        def supports(x, weight, pairs=False):
            calls["supports"] += 1
            return x.ndim == 2 and weight.ndim == 2 and x.shape[0] <= 64

        def supports_qkv(x, weight, keys):
            return supports(x, weight) and keys.ndim == 4 and keys.shape[0] == x.shape[0]

        def linear_qkv_norm_rope_cache(x, weight, *args):
            calls["qkv"] += 1
            return qkv_norm_rope_cache(F.linear(x, weight), *args)

        def linear_silu_mul(x, weight):
            calls["silu"] += 1
            return reference_silu_mul(x, weight)

        def linear_partials(x, weight):
            calls["partials"] += 1
            return partials(x, weight)

        patches = (
            mock.patch.object(decode_step, "supports", side_effect=supports),
            mock.patch.object(decode_step, "linear_partials", side_effect=linear_partials),
            mock.patch.object(decode_step, "linear_silu_mul", side_effect=linear_silu_mul),
            mock.patch.object(decode_step, "add_rms_norm", side_effect=add_rms_norm),
            mock.patch.object(decode_step, "qkv_norm_rope_cache", side_effect=AssertionError("split Q/K/V path")),
            mock.patch.object(decode_step, "silu_mul", side_effect=AssertionError("unfused MLP path")),
            mock.patch.object(decode_step, "supports_qkv", side_effect=supports_qkv),
            mock.patch.object(decode_step, "linear_qkv_norm_rope_cache", side_effect=linear_qkv_norm_rope_cache),
        )
        for batch, length, count in ((1, 3, 3), (3, 7, 4)):
            with self.subTest(batch=batch, length=length):
                state = DecodeState(candidate, batch, length, count)
                state.flash_context = CPUContext(batch, 4, 2, state.key_positions)
                state.fused_forward = decode_step.fused_decode_forward
                prompt = torch.randint(0, 127, (batch, length))
                logits = state.prefill(prompt)
                current, cache = prompt, None
                for step in range(count):
                    expected = model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    torch.testing.assert_close(logits, expected.logits, rtol=2e-2, atol=2e-3)
                    torch.testing.assert_close(state.tokens, expected.logits[:, -1].argmax(-1, keepdim=True))
                    current, cache = state.tokens.clone(), expected.past_key_values
                    if step + 1 < count:
                        before = dict(calls)
                        with (patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],
                              patches[6], patches[7]):
                            logits = state.step()
                        layers = config.num_hidden_layers
                        self.assertEqual(calls["partials"] - before["partials"], 2 * layers)
                        self.assertEqual(calls["qkv"] - before["qkv"], layers)
                        self.assertEqual(calls["silu"] - before["silu"], layers)


@unittest.skipUnless(torch.cuda.is_available() and HAS_TRITON,
                     "CUDA and Triton are required; CPU checks do not validate the kernels")
class SkinnyCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global fused
        sys.path.insert(0, str(ENGINE))
        try:
            from kernels import decode_fused as fused
        finally:
            sys.path.pop(0)

    @torch.inference_mode()
    def test_partials_sum_matches_linear_for_qwen_shapes(self):
        torch.manual_seed(21)
        for m in (1, 4, 16, 33, 64):
            for n, k in ((6144, 2560), (2560, 4096), (2560, 9728), (1024, 256)):
                with self.subTest(m=m, n=n, k=k):
                    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5
                    before = (x.clone(), w.clone())
                    partials = skinny.linear_partials(x, w)
                    self.assertEqual(partials.shape, (skinny.split_count(n, k), m, n))
                    self.assertEqual(partials.dtype, torch.float32)
                    self.assertTrue(bool(torch.isfinite(partials).all()))
                    expected = F.linear(x.float(), w.float())
                    torch.testing.assert_close(partials.sum(0), expected, rtol=2e-3, atol=2e-3)
                    torch.testing.assert_close(partials.sum(0).to(torch.bfloat16), F.linear(x, w),
                                               rtol=1.6e-2, atol=1e-2)
                    for t, old in zip((x, w), before):
                        torch.testing.assert_close(t, old, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            skinny.linear_partials(torch.randn(65, 256, device="cuda", dtype=torch.bfloat16),
                                   torch.randn(64, 256, device="cuda", dtype=torch.bfloat16))

    @torch.inference_mode()
    def test_silu_mul_matches_unfused_path(self):
        torch.manual_seed(22)
        for m, n, k in ((1, 9728, 2560), (4, 9728, 2560), (16, 9728, 2560), (64, 512, 256), (3, 64, 64)):
            with self.subTest(m=m, n=n, k=k):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5 * 4
                out = skinny.linear_silu_mul(x, w)
                self.assertEqual(out.shape, (m, n))
                self.assertTrue(bool(torch.isfinite(out).all()))
                torch.testing.assert_close(out, fused.silu_mul(F.linear(x, w)), rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(out, reference_silu_mul(x, w), rtol=2e-2, atol=2e-2)

    @torch.inference_mode()
    def test_consumers_on_partials_equal_consumers_on_bf16(self):
        torch.manual_seed(23)
        rows, cols = 4, 2560
        x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(cols, device="cuda", dtype=torch.bfloat16)
        partials = torch.randn(8, rows, cols, device="cuda")
        branch = sum_partials_bf16(partials)
        for a, b in zip(fused.add_rms_norm(x, partials, w, 1e-6), fused.add_rms_norm(x, branch, w, 1e-6)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        batch, nq, nk, d, capacity = 2, 32, 8, 128, 9
        width = (nq + 2 * nk) * d
        qkv_partials = torch.randn(4, batch, width, device="cuda")
        qkv = sum_partials_bf16(qkv_partials)
        qw, kw = (torch.randn(d, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        cos, sin = (torch.randn(capacity, d, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        position = torch.tensor([5], device="cuda")
        results = []
        for source in (qkv, qkv_partials):
            keys = torch.full((batch, nk, capacity, d), 33., device="cuda", dtype=torch.bfloat16)
            values = keys.clone()
            q = fused.qkv_norm_rope_cache(source, qw, kw, 1e-6, 1e-6, cos, sin, position, keys, values)
            results.append((q, keys, values))
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    @torch.inference_mode()
    def test_fused_qkv_head_matches_split_consumer_and_slots(self):
        torch.manual_seed(25)
        nq, nk, d, k, capacity = 32, 8, 128, 2560, 9
        w = torch.randn((nq + 2 * nk) * d, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5
        qw, kw = (torch.randn(d, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        cos, sin = (torch.randn(capacity, d, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        for m in (1, 4, 16, 33, 64):
            for slot in (0, 5, capacity - 1):
                with self.subTest(m=m, slot=slot):
                    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                    position = torch.tensor([slot], device="cuda")
                    before = (x.clone(), w.clone())
                    keys = torch.full((m, nk, capacity, d), 33., device="cuda", dtype=torch.bfloat16)
                    values = keys.clone()
                    self.assertTrue(skinny.supports_qkv(x, w, keys))
                    q = skinny.linear_qkv_norm_rope_cache(x, w, qw, kw, 1e-6, 1e-6, cos, sin,
                                                          position, keys, values)
                    self.assertEqual(q.shape, (m, nq, 1, d))
                    self.assertTrue(bool(torch.isfinite(q).all()))
                    for t, old in zip((x, w), before):
                        torch.testing.assert_close(t, old, rtol=0, atol=0)
                    untouched = torch.ones(capacity, dtype=torch.bool, device="cuda")
                    untouched[slot] = False
                    self.assertTrue(bool((keys[:, :, untouched] == 33).all()))
                    self.assertTrue(bool((values[:, :, untouched] == 33).all()))
                    # Split-K partials and full-K accumulation differ only in FP32 order.
                    for source in (skinny.linear_partials(x, w), F.linear(x, w)):
                        ek = torch.full_like(keys, 33.)
                        ev = torch.full_like(values, 33.)
                        eq = fused.qkv_norm_rope_cache(source, qw, kw, 1e-6, 1e-6, cos, sin,
                                                       position, ek, ev)
                        torch.testing.assert_close(q, eq, rtol=2e-2, atol=2e-2)
                        torch.testing.assert_close(keys, ek, rtol=2e-2, atol=2e-2)
                        torch.testing.assert_close(values, ev, rtol=2e-2, atol=2e-2)
                    torch.testing.assert_close(values[:, :, slot].reshape(m, -1),
                                               F.linear(x, w)[:, (nq + nk) * d:], rtol=1.6e-2, atol=1e-2)

    @torch.inference_mode()
    def test_graph_replay_follows_fresh_inputs(self):
        torch.manual_seed(24)
        x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2560, 4096, device="cuda", dtype=torch.bfloat16) / 64
        gate_up = torch.randn(2 * 2560, 4096, device="cuda", dtype=torch.bfloat16) / 64
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                skinny.linear_partials(x, w); skinny.linear_silu_mul(x, gate_up)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            partials = skinny.linear_partials(x, w)
            hidden = skinny.linear_silu_mul(x, gate_up)
        for _ in range(3):
            x.normal_()
            graph.replay()
            torch.testing.assert_close(partials.sum(0), F.linear(x.float(), w.float()), rtol=2e-3, atol=2e-3)
            torch.testing.assert_close(hidden, fused.silu_mul(F.linear(x, gate_up)), rtol=2e-2, atol=2e-2)

    @torch.inference_mode()
    def test_engine_own_prefix_tokens_with_split_projections(self):
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("Transformers required")
        import tempfile
        sys.path.insert(0, str(ENGINE))
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
            from engine import Engine
            import decode_step
        finally:
            sys.path.pop(0)
        config = Qwen3Config(vocab_size=127, hidden_size=256, intermediate_size=512,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=128, max_position_embeddings=32768, rope_theta=5_000_000,
                             tie_word_embeddings=True, sliding_window=None, eos_token_id=0)
        config._attn_implementation = "sdpa"
        torch.manual_seed(25)
        native = Qwen3ForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as path:
            native.save_pretrained(path)
            engine = Engine(path)
            native = native.cuda().bfloat16()
            for batch, length, count in ((3, 7, 5), (1, 1, 3), (2, 129, 9), (65, 5, 3)):
                prompt = torch.randint(0, 127, (batch, length), device="cuda")
                with mock.patch.object(decode_step, "linear_partials", wraps=decode_step.linear_partials) as split:
                    emitted = list(engine.generate(prompt.tolist(), count))
                self.assertEqual(len(emitted), count)
                if count > 1 and engine.decode_state.graph is None:
                    self.fail("decode graph expected")
                tokens = torch.tensor(emitted, device="cuda").T
                prefix = torch.cat((prompt, tokens[:, :-1]), dim=1)
                logits = native(input_ids=prefix, use_cache=False).logits[:, length - 1:].float()
                gaps = logits.amax(-1) - logits.gather(-1, tokens[..., None]).squeeze(-1)
                self.assertLessEqual(float(gaps.max()), 2.0)
                # Prove dispatch in eager mode: 3 split projections per layer per step, none above MAX_ROWS.
                engine.decode_state.prefill(prompt)
                with mock.patch.object(decode_step, "linear_partials", wraps=decode_step.linear_partials) as eager:
                    engine.decode_state.step()
                self.assertEqual(eager.call_count, 0 if batch > skinny.MAX_ROWS else 3 * config.num_hidden_layers)


if __name__ == "__main__":
    unittest.main()
