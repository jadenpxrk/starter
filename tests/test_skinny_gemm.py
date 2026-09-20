"""Decode projection checks. CPU kernel-body execution through a Torch shim is
NOT Triton or CUDA validation; it checks indexing, masking, split coverage and
cast placement of the actual kernel source. Performance is not measured here."""

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import random
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

    def load(self, p, mask=None, other=0.0, cache_modifier=""):
        assert cache_modifier in ("", ".cg")
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

    def atomic_add(self, p, value, sem=None):
        # One arrival per program, as Triton lowers a scalar atomic; returns the old value.
        assert sem == "acq_rel"
        i, _ = self._indices(p, None)
        assert i.ndim == 0, "the arrival counter is one scalar per column tile"
        old = p.data[i].clone()
        p.data[i] += value
        p.writes[i] += 1
        return old

    @staticmethod
    def debug_barrier():
        pass


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


def block_rows(m):
    return max(16, 1 << (m - 1).bit_length())


def native_norm(x, weight, eps):
    f = x.float()
    return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def tile_stats(total, bm):
    """FP32 [N // BN, BM] sums of squares of a BF16 [M, N] sum, padded rows zero."""
    m, n = total.shape
    out = torch.zeros(n // BN, bm)
    out[:, :m] = total.float().square().view(m, n // BN, BN).sum(-1).T
    return out


def norm_from_stats(x, stats, gain, eps):
    m, k = x.shape
    inv = torch.rsqrt(stats.sum(0)[:m] / k + eps)
    return (x.float() * inv[:, None]).to(torch.bfloat16) * gain


def fake(shape, dtype=torch.bfloat16, cuda=True, contiguous=True):
    return SimpleNamespace(ndim=len(shape), shape=tuple(shape), dtype=dtype, is_cuda=cuda,
                           device=torch.device("cuda" if cuda else "cpu"),
                           is_contiguous=lambda: contiguous)


class CpuKernelBodies:
    def __init__(self):
        self.ops = Ops()
        env = source_functions(KERNELS / "skinny_gemm.py",
                               {"_row_rstd", "_normalize", "_partials_kernel",
                                "_partials_stats_kernel", "_silu_mul_kernel"}, {"tl": self.ops})
        self.partials, self.silu = env["_partials_kernel"], env["_silu_mul_kernel"]
        self.stats = env["_partials_stats_kernel"]

    @staticmethod
    def _norm(xp, norm):
        """Kernel arguments (SQ, G, eps, NORM, TILES, TP); inert when norm is None."""
        if norm is None:
            return xp, xp, 0.0, False, 1, 1
        stats, gain, eps = norm
        tiles = stats.shape[0]
        assert stats.dtype == torch.float32 and gain.dtype == torch.bfloat16
        return Pointer(stats), Pointer(gain), eps, True, tiles, 1 << (tiles - 1).bit_length()

    def run_partials(self, x, w, splits, bm, norm=None):
        m, k = x.shape
        n = w.shape[0]
        out = torch.full((splits, m, n), float("nan"))
        xp, wp, pp = Pointer(x), Pointer(w), Pointer(out)
        sq, g, eps, flag, tiles, tp = self._norm(xp, norm)
        for column in range(n // BN):
            for split in range(splits):
                self.ops.program = (column, split)
                self.partials(xp, wp, pp, sq, g, m, n, k, k // splits, eps, bm, BN, BK, flag, tiles, tp)
        assert bool((pp.writes == 1).all()), "every partial written exactly once"
        assert bool((xp.writes == 0).all()) and bool((wp.writes == 0).all())
        assert bool((sq.writes == 0).all()) and bool((g.writes == 0).all())
        return out

    def run_silu(self, x, w, bm, norm=None):
        m, k = x.shape
        n = w.shape[0] // 2
        out = torch.full((m, n), float("nan")).bfloat16()
        xp, wp, yp = Pointer(x), Pointer(w), Pointer(out)
        sq, g, eps, flag, tiles, tp = self._norm(xp, norm)
        for column in range(n // BN):
            self.ops.program = (column, 0)
            self.silu(xp, wp, yp, sq, g, m, n, k, eps, bm, BN, BK, flag, tiles, tp)
        assert bool((yp.writes == 1).all())
        assert bool((sq.writes == 0).all()) and bool((g.writes == 0).all())
        return out

    def run_stats(self, x, w, residual, splits, bm, seed):
        """Programs run in a seeded random order: any split may arrive last at a tile."""
        m, k = x.shape
        n = w.shape[0]
        tiles = n // BN
        partials = torch.full((splits, m, n), float("nan"))
        total = torch.full((m, n), float("nan")).bfloat16()
        stats = torch.full((tiles, bm), float("nan"))
        counters = torch.zeros(tiles, dtype=torch.int32)
        pointers = [Pointer(t) for t in (x, w, partials, residual, total, stats, counters)]
        programs = [(tile, split) for tile in range(tiles) for split in range(splits)]
        random.Random(seed).shuffle(programs)
        for program in programs:
            self.ops.program = program
            self.stats(*pointers, m, n, k, k // splits, bm, BN, BK, splits)
        xp, wp, pp, rp, tp, sp, cp = pointers
        assert bool((pp.writes == 1).all()), "every partial written exactly once"
        assert bool((tp.writes == 1).all()) and bool((sp.writes == 1).all()), "sum/stats once"
        assert bool((xp.writes == 0).all()) and bool((wp.writes == 0).all()) and bool((rp.writes == 0).all())
        assert bool((counters == 0).all()), "every counter is left zeroed for the next launch"
        assert bool((cp.writes == splits + 1).all()), "S arrivals and one reset per tile"
        return partials, total, stats


def reference_silu_mul(x, w):
    gate, up = F.linear(x.float(), w.float()).to(torch.bfloat16).chunk(2, dim=-1)
    return (F.silu(gate.float()).to(torch.bfloat16).float() * up.float()).to(torch.bfloat16)


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

    def test_kernel_body_stats_epilogue_any_arrival_order(self):
        """The last split at a tile finishes the residual add exactly as add_rms_norm."""
        torch.manual_seed(10)
        bodies = CpuKernelBodies()
        ops = Ops()
        add_norm = source_functions(KERNELS / "decode_fused.py", {"_add_rms_norm_kernel"},
                                    {"tl": ops})["_add_rms_norm_kernel"]
        for m, n, k, splits, bm in ((1, 64, 128, 2, 16), (3, 96, 256, 4, 16), (17, 64, 512, 8, 32),
                                    (16, 32, 64, 1, 16), (5, 64, 192, 1, 16)):
            for seed in (0, 1, 2):
                with self.subTest(m=m, n=n, k=k, splits=splits, seed=seed):
                    x = torch.randn(m, k).bfloat16()
                    w = torch.randn(n, k).bfloat16() / k ** 0.5
                    residual = torch.randn(m, n).bfloat16()
                    partials, total, stats = bodies.run_stats(x, w, residual, splits, bm, seed)
                    per_split = k // splits
                    for s in range(splits):
                        chunk = slice(s * per_split, (s + 1) * per_split)
                        torch.testing.assert_close(partials[s], x[:, chunk].float() @ w[:, chunk].float().T,
                                                   rtol=1e-5, atol=1e-5)
                    # Bit-for-bit the existing consumer kernel body on the same partials.
                    gain = torch.randn(n).bfloat16()
                    ref_total, ref_normed = torch.empty_like(total), torch.empty_like(total)
                    pointers = [Pointer(t) for t in (residual, partials, gain, ref_total, ref_normed)]
                    for row in range(m):
                        ops.program = (row, 0)
                        add_norm(*pointers, n, 1e-6, m * n, 1 << (n - 1).bit_length(), splits)
                    torch.testing.assert_close(total, ref_total, rtol=0, atol=0)
                    expected = (residual.float() + sum_partials_bf16(partials).float()).to(torch.bfloat16)
                    torch.testing.assert_close(total, expected, rtol=0, atol=0)
                    self.assertEqual(tuple(stats.shape), (n // BN, bm))
                    self.assertTrue(bool((stats[:, m:] == 0).all()))
                    torch.testing.assert_close(stats, tile_stats(total, bm), rtol=1e-6, atol=0)
                    # The consumers' fixed-order regrouping reproduces the norm.
                    torch.testing.assert_close(norm_from_stats(total, stats, gain, 1e-6), ref_normed,
                                               rtol=1e-2, atol=1e-2)

    def test_kernel_bodies_normalize_on_the_way_in(self):
        """NORM tiles equal the projection of the materialized add_rms_norm output."""
        torch.manual_seed(11)
        bodies = CpuKernelBodies()
        ops = Ops()
        add_norm = source_functions(KERNELS / "decode_fused.py", {"_add_rms_norm_kernel"},
                                    {"tl": ops})["_add_rms_norm_kernel"]
        for m, k, n, splits, bm, exact in ((1, 128, 64, 2, 16, True), (3, 256, 32, 4, 16, True),
                                           (17, 64, 64, 1, 32, True), (4, 192, 64, 1, 16, False),
                                           (33, 128, 32, 2, 64, False)):
            with self.subTest(m=m, k=k, n=n, splits=splits, exact=exact):
                if exact:
                    # Small integers: sums of squares are exact in FP32 in any order,
                    # so the two norms agree bit for bit and so must every product.
                    total = torch.randint(-6, 7, (m, k)).bfloat16()
                else:
                    total = torch.randn(m, k).bfloat16() * 3
                gain = torch.randn(k).bfloat16()
                stats = tile_stats(total, bm)
                ref_total, normed = torch.empty_like(total), torch.empty_like(total)
                pointers = [Pointer(t) for t in (total, torch.zeros_like(total), gain, ref_total, normed)]
                for row in range(m):
                    ops.program = (row, 0)
                    add_norm(*pointers, k, 1e-6, m * k, 1 << (k - 1).bit_length(), 0)
                torch.testing.assert_close(ref_total, total, rtol=0, atol=0)
                w = torch.randn(n, k).bfloat16()
                gate_up = torch.randn(2 * n, k).bfloat16()
                norm = (stats, gain, 1e-6)
                fused = bodies.run_partials(total, w, splits, bm, norm)
                plain = bodies.run_partials(normed, w, splits, bm)
                fused_act = bodies.run_silu(total, gate_up, bm, norm)
                plain_act = bodies.run_silu(normed, gate_up, bm)
                if exact:
                    torch.testing.assert_close(fused, plain, rtol=0, atol=0)
                    torch.testing.assert_close(fused_act, plain_act, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(fused, plain, rtol=2e-3, atol=2e-3)
                    torch.testing.assert_close(fused_act, plain_act, rtol=2e-2, atol=2e-2)
                # Independent of both kernels: the pinned Qwen3RMSNorm module, then FP32 products.
                native = native_norm(total, gain, 1e-6)
                if importlib.util.find_spec("transformers") is not None:
                    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
                    module = Qwen3RMSNorm(k, eps=1e-6)
                    module.weight.data = gain.clone()
                    torch.testing.assert_close(module(total), native, rtol=0, atol=0)
                torch.testing.assert_close(fused.sum(0), native.float() @ w.float().T, rtol=2e-3, atol=2e-3)
                torch.testing.assert_close(fused_act, reference_silu_mul(native, gate_up), rtol=2e-2, atol=2e-2)
        # Negative control: multiplying the gain before the BF16 rounding is a different function.
        total = torch.randn(8, 256).bfloat16() * 3
        gain = torch.randn(256).bfloat16()
        f = total.float()
        unrounded = (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6) * gain.float()).to(torch.bfloat16)
        self.assertFalse(torch.equal(unrounded, native_norm(total, gain, 1e-6)))

    def test_stats_wrappers_validate_before_launch(self):
        with mock.patch.dict(sys.modules, {"triton": mock.Mock(), "triton.language": mock.Mock()}):
            module = load_module("skinny_gemm_wrappers_under_test", KERNELS / "skinny_gemm.py")
        x = fake((4, 2560))
        stats, gain = fake((80, 16), torch.float32), fake((2560,))
        with mock.patch.object(module.triton, "next_power_of_2", side_effect=lambda v: 1 << (v - 1).bit_length()):
            self.assertEqual(module._norm_args(x, None)[2:], (0.0, 1, 1))
            self.assertEqual(module._norm_args(x, (stats, gain, 1e-6))[2:], (1e-6, 80, 128))
            for bad in ((fake((80, 32), torch.float32), gain), (fake((79, 16), torch.float32), gain),
                        (fake((80, 16)), gain), (fake((80, 16), torch.float32, contiguous=False), gain),
                        (fake((80, 16), torch.float32, cuda=False), gain), (stats, fake((2559,))),
                        (stats, fake((2560,), torch.float32)), (stats, fake((2560,), contiguous=False))):
                with self.assertRaises(ValueError):
                    module._norm_args(x, (bad[0], bad[1], 1e-6))
        # CPU operands or a mismatched residual are rejected before any launch.
        cpu_x = torch.zeros(2, 64, dtype=torch.bfloat16)
        cpu_w = torch.zeros(64, 64, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            module.linear_add_stats(cpu_x, cpu_w, torch.zeros(2, 64, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            module.linear_partials(cpu_x, cpu_w, (torch.zeros(2, 16), torch.zeros(64, dtype=torch.bfloat16), 1e-6))
        with mock.patch.object(module, "supports", return_value=True), \
                mock.patch.object(module, "_partials_stats_kernel"):
            for residual in (fake((3, 64)), fake((2, 64), torch.float32), fake((2, 64), contiguous=False),
                             fake((2, 64), cuda=False)):
                with self.assertRaises(ValueError):
                    module.linear_add_stats(fake((2, 64)), fake((64, 64)), residual)

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
                    "kernels.skinny_gemm": mock.Mock(supports=mock.Mock(return_value=False))}):
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

        calls = {"partials": 0, "silu": 0, "supports": 0, "stats": 0, "add": 0, "normed": 0}

        def supports(x, weight, pairs=False):
            calls["supports"] += 1
            return x.ndim == 2 and weight.ndim == 2 and x.shape[0] <= 64

        def take(x, norm):
            if norm is None:
                return x
            calls["normed"] += 1
            return norm_from_stats(x, *norm)

        def linear_silu_mul(x, weight, norm=None):
            calls["silu"] += 1
            return reference_silu_mul(take(x, norm), weight)

        def linear_partials(x, weight, norm=None):
            calls["partials"] += 1
            return partials(take(x, norm), weight)

        def linear_add_stats(x, weight, residual):
            calls["stats"] += 1
            total = residual + fold(partials(x, weight))
            return total, tile_stats(total, block_rows(total.shape[0]))

        def counted_add_rms_norm(residual, branch, weight, eps):
            calls["add"] += 1
            return add_rms_norm(residual, branch, weight, eps)

        patches = (
            mock.patch.object(decode_step, "supports", side_effect=supports),
            mock.patch.object(decode_step, "linear_partials", side_effect=linear_partials),
            mock.patch.object(decode_step, "linear_silu_mul", side_effect=linear_silu_mul),
            mock.patch.object(decode_step, "add_rms_norm", side_effect=counted_add_rms_norm),
            mock.patch.object(decode_step, "qkv_norm_rope_cache", side_effect=qkv_norm_rope_cache),
            mock.patch.object(decode_step, "silu_mul", side_effect=AssertionError("unfused MLP path")),
            mock.patch.object(decode_step, "linear_add_stats", side_effect=linear_add_stats),
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
                        with patches[0], patches[1], patches[2], patches[3], patches[4], \
                                patches[5], patches[6]:
                            logits = state.step()
                        layers = config.num_hidden_layers
                        # Q/K/V per layer plus the last down projection stay partials; every
                        # o_proj and every other down projection finishes its residual add.
                        self.assertEqual(calls["partials"] - before["partials"], layers + 1)
                        self.assertEqual(calls["stats"] - before["stats"], 2 * layers - 1)
                        self.assertEqual(calls["silu"] - before["silu"], layers)
                        self.assertEqual(calls["normed"] - before["normed"], 2 * layers - 1)
                        self.assertEqual(calls["add"] - before["add"], 1)


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
    def test_add_stats_chain_equals_add_rms_norm_chain(self):
        torch.manual_seed(26)
        for m in (1, 4, 16, 33, 64):
            for n, k in ((2560, 4096), (2560, 9728), (64, 128)):
                with self.subTest(m=m, n=n, k=k):
                    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5
                    residual = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
                    gain = torch.randn(n, device="cuda", dtype=torch.bfloat16)
                    before = [t.clone() for t in (x, w, residual, gain)]
                    total, stats = skinny.linear_add_stats(x, w, residual)
                    ref_total, ref_normed = fused.add_rms_norm(residual, skinny.linear_partials(x, w), gain, 1e-6)
                    torch.testing.assert_close(total, ref_total, rtol=0, atol=0)
                    bm = block_rows(m)
                    self.assertEqual(tuple(stats.shape), (n // BN, bm))
                    self.assertTrue(bool((stats[:, m:] == 0).all()))
                    torch.testing.assert_close(stats.sum(0)[:m], ref_total.float().square().sum(-1),
                                               rtol=1e-5, atol=1e-3)
                    self.assertTrue(bool((skinny._arrival_counters(x.device, n // BN) == 0).all()))
                    norm = (stats, gain, 1e-6)
                    w2 = torch.randn(4 * BN, n, device="cuda", dtype=torch.bfloat16) / n ** 0.5
                    gate_up = torch.randn(4 * BN, n, device="cuda", dtype=torch.bfloat16) / n ** 0.5 * 4
                    fused_partials = skinny.linear_partials(total, w2, norm)
                    plain_partials = skinny.linear_partials(ref_normed, w2)
                    torch.testing.assert_close(fused_partials, plain_partials, rtol=2e-3, atol=2e-3)
                    torch.testing.assert_close(skinny.linear_silu_mul(total, gate_up, norm),
                                               skinny.linear_silu_mul(ref_normed, gate_up), rtol=2e-2, atol=2e-2)
                    native = native_norm(ref_total, gain, 1e-6)
                    torch.testing.assert_close(fused_partials.sum(0), F.linear(native.float(), w2.float()),
                                               rtol=2e-3, atol=2e-3)
                    for t, old in zip((x, w, residual, gain), before):
                        torch.testing.assert_close(t, old, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            skinny.linear_add_stats(x, w, residual[:, :32].contiguous())
        with self.assertRaises(ValueError):
            skinny.linear_partials(total, w2, (stats[:1], gain, 1e-6))

    @torch.inference_mode()
    def test_add_stats_graph_replay_leaves_counters_zero(self):
        torch.manual_seed(27)
        x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2560, 4096, device="cuda", dtype=torch.bfloat16) / 64
        residual = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
        gain = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
        gate_up = torch.randn(2 * 2560, 2560, device="cuda", dtype=torch.bfloat16) / 64
        down = torch.randn(2560, 2560, device="cuda", dtype=torch.bfloat16) / 64

        def chain():
            total, stats = skinny.linear_add_stats(x, w, residual)
            hidden = skinny.linear_silu_mul(total, gate_up, (stats, gain, 1e-6))
            total2, stats2 = skinny.linear_add_stats(hidden, down, total)
            return total, hidden, total2, stats2

        def reference():
            total, normed = fused.add_rms_norm(residual, skinny.linear_partials(x, w), gain, 1e-6)
            hidden = skinny.linear_silu_mul(normed, gate_up)
            total2 = fused.add_rms_norm(total, skinny.linear_partials(hidden, down), gain, 1e-6)[0]
            return total, hidden, total2

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                chain()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = chain()
        counters = skinny._arrival_counters(x.device, 2560 // BN)
        for _ in range(3):
            x.normal_(); residual.normal_()
            graph.replay()
            total, hidden, total2 = reference()
            torch.testing.assert_close(outputs[0], total, rtol=0, atol=0)
            torch.testing.assert_close(outputs[1], hidden, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(outputs[2], total2, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(outputs[3].sum(0)[:4], total2.float().square().sum(-1), rtol=1e-4, atol=1e-2)
            self.assertTrue(bool((counters == 0).all()))

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
                emitted = list(engine.generate(prompt.tolist(), count))
                self.assertEqual(len(emitted), count)
                if count > 1 and engine.decode_state.graph is None:
                    self.fail("decode graph expected")
                tokens = torch.tensor(emitted, device="cuda").T
                prefix = torch.cat((prompt, tokens[:, :-1]), dim=1)
                logits = native(input_ids=prefix, use_cache=False).logits[:, length - 1:].float()
                gaps = logits.amax(-1) - logits.gather(-1, tokens[..., None]).squeeze(-1)
                self.assertLessEqual(float(gaps.max()), 2.0)
                # Prove dispatch in eager mode: Q/K/V partials per layer plus the last down
                # projection, and a residual-finishing stream for every other projection.
                engine.decode_state.prefill(prompt)
                with mock.patch.object(decode_step, "linear_partials", wraps=decode_step.linear_partials) as eager, \
                        mock.patch.object(decode_step, "linear_add_stats", wraps=decode_step.linear_add_stats) as stats:
                    engine.decode_state.step()
                layers = config.num_hidden_layers
                self.assertEqual(eager.call_count, 0 if batch > skinny.MAX_ROWS else layers + 1)
                self.assertEqual(stats.call_count, 0 if batch > skinny.MAX_ROWS else 2 * layers - 1)


if __name__ == "__main__":
    unittest.main()
