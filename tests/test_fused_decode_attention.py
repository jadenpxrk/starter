"""Checks for the one-launch decode attention (Q/K/V completion, segments, merge).

CPU tests execute the actual kernel source bodies through a checked Torch
pointer shim: indexing, masks, slot ownership, arrival handling, write
coverage, cast placement and orchestration. They are NOT Triton interpreter,
MMA, warp, GPU memory-model or timing validation; the CUDA class covers the
operator and is skipped without CUDA and Triton.
"""

import ast
import importlib.util
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

ENGINE = Path(__file__).resolve().parents[1] / "engine"
KERNEL = ENGINE / "kernels/small_query_attention.py"
CONSUMER = ENGINE / "kernels/decode_fused.py"
DIAGNOSTICS = {"exact_cases": 0, "random_cases": 0, "pipeline_steps": 0}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Shared shim, legacy kernel bodies and independent references; its test
# classes stay in that module's namespace and are not collected twice here.
base = load("fused_attention_helpers", Path(__file__).resolve().parent / "test_small_query_attention.py")
flash = base.flash
Pointer = base.Pointer


class FusedOps(base.Ops):
    """Adds the scalar atomic, barrier and cache-modifier surface the fused body uses."""
    int32 = torch.int32
    rsqrt = staticmethod(torch.rsqrt)
    static_range = staticmethod(range)

    def __init__(self):
        self.arrivals = []

    @staticmethod
    def debug_barrier():
        pass

    def atomic_add(self, p, value, sem=None):
        assert sem == "acq_rel", "the arrival counter must publish and acquire partials"
        index, _ = self.indices(p, True)
        assert index.ndim == 0, "one counter per program"
        old = p.data[index].clone()
        p.data[index] += value
        p.access[1][index] += 1
        self.arrivals.append((int(index), int(old)))
        return old

    def load(self, p, mask=True, other=0, cache_modifier=None):
        assert cache_modifier in (None, ".cg")
        return super().load(p, mask, other)

    def store(self, p, value, mask=True):
        return super().store(p, torch.as_tensor(value), mask)


class LegacyConsumer:
    """The passing separate Q/K/V consumer body, for bit-level comparison."""
    def __init__(self):
        self.ops = FusedOps()
        self.kernel = base.bodies(CONSUMER, {"_qkv_norm_rope_cache_kernel"},
                                  {"tl": self.ops})["_qkv_norm_rope_cache_kernel"]

    @torch.inference_mode()
    def run(self, qkv, qw, kw, qe, ke, cos, sin, position, keys, values):
        b, hk, cap, d = keys.shape
        width = qkv.shape[-1]
        hq = width // d - 2 * hk
        splits = qkv.shape[0] if qkv.ndim == 3 else 0
        q = torch.full((b, hq, 1, d), float("nan"), dtype=torch.bfloat16)
        ptrs = [Pointer(t) for t in (qkv, qw, kw, cos, sin, position, q, keys, values)]
        for row in range(b * (hq + hk)):
            self.ops.program = (row, 0)
            self.kernel(*ptrs, width, cap, b * width, NQ=hq, NK=hk, D=d,
                        QEPS=qe, KEPS=ke, SPLITS=splits)
        return q


class FusedBody:
    def __init__(self):
        self.ops = FusedOps()
        self.kernel = base.bodies(KERNEL, {"_fused_attention", "_packed_rows", "_norm_rope"}, {"tl": self.ops})["_fused_attention"]

    @torch.inference_mode()
    def run(self, qkv, qw, kw, qe, ke, cos, sin, position, keys, values, scale,
            splits=3, order=None, check=True):
        b, hk, cap, d = keys.shape
        width = qkv.shape[-1]
        hq = width // d - 2 * hk
        g, vh = hq // hk, b * hk
        ksplits = qkv.shape[0] if qkv.ndim == 3 else 0
        tiles = (cap + 63) // 64
        span = ((tiles + splits - 1) // splits) * 64
        partial = torch.full((vh, splits, g, d), float("nan"))
        stats = torch.full((vh, splits, 2, g), float("nan"))
        counters = torch.zeros(vh, dtype=torch.int32)
        out = torch.full((b, 1, hq, d), float("nan"), dtype=torch.bfloat16)
        tensors = (qkv, qw, kw, cos, sin, position, keys, values, partial, stats, counters, out)
        ptrs = [Pointer(t) for t in tensors]
        programs = [(head, segment) for head in range(vh) for segment in range(splits)]
        if order is not None:
            order.shuffle(programs)
        self.ops.arrivals = []
        for program in programs:
            self.ops.program = program
            self.kernel(*ptrs, width, b * width, C=cap, NQ=hq, NK=hk, G=g, S=splits,
                        SPAN=span, SCALE=scale, D=d, QM=16, KN=64,
                        SS=1 << (splits - 1).bit_length(), SPLITS=ksplits, QEPS=qe, KEPS=ke)
        if check:
            pos = int(position)
            for p in ptrs[:6]:
                assert not bool(p.access[1].any()), "input was mutated"
            assert bool((ptrs[-1].access[1] == 1).all()), "output write coverage"
            if splits > 1:
                assert bool((ptrs[8].access[1] == 1).all()), "partial write coverage"
                assert bool((ptrs[9].access[1] == 1).all()), "statistics write coverage"
                assert bool((counters == 0).all()), "counters must end at zero"
                assert bool((ptrs[10].access[1] == splits + 1).all()), "S arrivals and one reset"
                for head in range(vh):
                    seen = sorted(old for index, old in self.ops.arrivals if index == head)
                    assert seen == list(range(splits)), "every segment arrived exactly once"
            else:
                assert not self.ops.arrivals, "a single segment needs no arrival protocol"
            for p in ptrs[6:8]:
                written = p.access[1].view(vh, cap, d)
                assert bool((written[:, pos] == 1).all()), "slot pos written exactly once per head"
                assert int(written.sum()) == vh * d, "no other cache slot written"
                read = p.access[0].view(vh, cap, d)
                assert not bool(read[:, pos + 1:].any()), "invalid capacity was loaded"
                assert bool((read[:, :pos + 1] == 1).all()), "valid K/V not read exactly once per head"
        return out


def integer_inputs(seed, b, hq, hk, cap, ksplits):
    """Inputs whose only reduction-order-sensitive step (sum of squares) is exact."""
    gen = torch.Generator().manual_seed(seed)
    width = (hq + 2 * hk) * 128
    if ksplits:
        qkv = torch.randint(-2, 3, (ksplits, b, width), generator=gen).float()
    else:
        qkv = torch.randint(-2, 3, (b, width), generator=gen).bfloat16()
    gains = torch.tensor([0.5, 1.0, 2.0, -1.0, -0.5])
    qw = gains[torch.randint(0, 5, (128,), generator=gen)].bfloat16()
    kw = gains[torch.randint(0, 5, (128,), generator=gen)].bfloat16()
    return qkv, qw, kw


def random_inputs(seed, b, hq, hk, cap, ksplits):
    gen = torch.Generator().manual_seed(seed)
    width = (hq + 2 * hk) * 128
    if ksplits:
        qkv = torch.randn((ksplits, b, width), generator=gen) * 2
    else:
        qkv = (torch.randn((b, width), generator=gen) * 2).bfloat16()
    qw = (torch.randn(128, generator=gen)).bfloat16()
    kw = (torch.randn(128, generator=gen)).bfloat16()
    return qkv, qw, kw


def tables(cap, seed=0):
    """Angle tables like DecodeState's: BF16 [capacity, 128] cos/sin pairs."""
    gen = torch.Generator().manual_seed(seed)
    angles = torch.rand((cap, 64), generator=gen) * 6.3
    angles = torch.cat((angles, angles), -1)
    return angles.cos().bfloat16().contiguous(), angles.sin().bfloat16().contiguous()


def poisoned_caches(b, hk, cap, pos, seed):
    gen = torch.Generator().manual_seed(seed)
    keys = torch.full((b, hk, cap, 128), float("nan"), dtype=torch.bfloat16)
    values = keys.clone()
    keys[:, :, :pos] = torch.randn((b, hk, pos, 128), generator=gen).bfloat16()
    values[:, :, :pos] = torch.randn((b, hk, pos, 128), generator=gen).bfloat16()
    return keys, values


CASES = (  # batch, query heads, KV heads, capacity, segments, projection splits
    (1, 32, 8, 544, 9, 4), (1, 32, 8, 544, 9, 0), (16, 32, 8, 640, 3, 2),
    (4, 32, 8, 2080, 9, 4), (2, 6, 2, 257, 3, 0), (3, 8, 2, 129, 8, 8),
    (1, 16, 1, 193, 2, 1), (1, 4, 1, 65, 1, 4), (1, 4, 1, 1, 1, 0), (1, 1, 1, 65, 2, 2),
)


class FusedAttentionCPU(unittest.TestCase):
    @torch.inference_mode()
    def test_fused_body_equals_legacy_consumer_plus_attention_bit_for_bit(self):
        fused, consumer, legacy = FusedBody(), LegacyConsumer(), base.KernelBody()
        for case, (b, hq, hk, cap, splits, ksplits) in enumerate(CASES):
            cos, sin = tables(cap, case)
            for pos in sorted({0, cap // 2, cap - 1}):
                with self.subTest(case=(b, hq, hk, cap, splits, ksplits), pos=pos):
                    qkv, qw, kw = integer_inputs(100 + case, b, hq, hk, cap, ksplits)
                    position = torch.tensor([pos])
                    keys, values = poisoned_caches(b, hk, cap, pos, 7 + case)
                    ref_keys, ref_values = keys.clone(), values.clone()
                    before = [t.clone() for t in (qkv, qw, kw, cos, sin, position)]
                    out = fused.run(qkv, qw, kw, 1e-6, 3e-6, cos, sin, position, keys, values,
                                    128 ** -0.5, splits, order=random.Random(case))
                    q = consumer.run(qkv, qw, kw, 1e-6, 3e-6, cos, sin, position, ref_keys, ref_values)
                    used = torch.full((b * hk,), pos + 1, dtype=torch.int32)
                    expected = legacy.run(q, ref_keys, ref_values, used, 128 ** -0.5, splits)
                    self.assertTrue(bool(torch.isfinite(out).all()))
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    torch.testing.assert_close(keys, ref_keys, rtol=0, atol=0, equal_nan=True)
                    torch.testing.assert_close(values, ref_values, rtol=0, atol=0, equal_nan=True)
                    self.assertTrue(bool(keys[:, :, pos + 1:].isnan().all()))
                    for new, old in zip((qkv, qw, kw, cos, sin, position), before):
                        torch.testing.assert_close(new, old, rtol=0, atol=0)
                    # Arrival order must not change the merged result.
                    again = fused.run(qkv, qw, kw, 1e-6, 3e-6, cos, sin, position, keys.clone(),
                                      values.clone(), 128 ** -0.5, splits, order=random.Random(case + 999))
                    torch.testing.assert_close(again, out, rtol=0, atol=0)
                    DIAGNOSTICS["exact_cases"] += 1

    @torch.inference_mode()
    def test_fused_body_matches_independent_reference_on_random_inputs(self):
        fused = FusedBody()
        for case, (b, hq, hk, cap, splits, ksplits) in enumerate(CASES):
            cos, sin = tables(cap, case)
            for pos in sorted({0, cap // 2, cap - 1}):
                with self.subTest(case=(b, hq, hk, cap, splits, ksplits), pos=pos):
                    qkv, qw, kw = random_inputs(200 + case, b, hq, hk, cap, ksplits)
                    position = torch.tensor([pos])
                    keys, values = poisoned_caches(b, hk, cap, pos, 9 + case)
                    ref_keys, ref_values = keys.clone(), values.clone()
                    out = fused.run(qkv, qw, kw, 1e-6, 3e-6, cos, sin, position, keys, values,
                                    128 ** -0.5, splits)
                    q = base.ref_qkv(qkv, qw, kw, 1e-6, 3e-6, cos, sin, position, ref_keys, ref_values)
                    used = torch.full((b * hk,), pos + 1, dtype=torch.int32)
                    expected = base.reference(q, ref_keys, ref_values, used, 128 ** -0.5)
                    self.assertTrue(bool(torch.isfinite(out).all()))
                    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)
                    torch.testing.assert_close(keys[:, :, pos], ref_keys[:, :, pos], rtol=0.02, atol=0.02)
                    torch.testing.assert_close(values[:, :, pos], ref_values[:, :, pos], rtol=0, atol=0)
                    DIAGNOSTICS["random_cases"] += 1

    @torch.inference_mode()
    def test_gain_before_rounding_is_a_different_function(self):
        # Negative control for the cast placement the fused body must keep.
        b, hq, hk, cap, splits, ksplits = (2, 8, 2, 129, 3, 2)
        qkv, qw, kw = random_inputs(5, b, hq, hk, cap, ksplits)
        cos, sin = tables(cap, 3)
        position = torch.tensor([100])
        keys, values = poisoned_caches(b, hk, cap, 100, 4)
        out = FusedBody().run(qkv, qw, kw, 1e-6, 1e-6, cos, sin, position, keys, values, 128 ** -0.5, splits)
        total = qkv[0] + qkv[1]
        q, k, v = total.bfloat16().split((hq * 128, hk * 128, hk * 128), -1)
        def wrong_norm(x, w):
            f = x.float()
            return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6) * w).to(x.dtype)
        wrong_q = wrong_norm(q.reshape(b, hq, 1, 128), qw)
        wrong_k = wrong_norm(k.reshape(b, hk, 1, 128), kw)
        right_q = base.norm(q.reshape(b, hq, 1, 128), qw, 1e-6)
        self.assertGreater(int(torch.count_nonzero(wrong_q != right_q)), 0)
        self.assertGreater(int(torch.count_nonzero(wrong_k != base.norm(k.reshape(b, hk, 1, 128), kw, 1e-6))), 0)
        self.assertTrue(bool(torch.isfinite(out).all()))

    @torch.inference_mode()
    def test_actual_wrapper_single_launch_arguments_and_validation(self):
        calls = []

        class Launcher:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    calls.append((self.name, grid, args, kwargs))
                    args[11].zero_()
                return launch

        proxy = SimpleNamespace(
            device=torch.device, is_grad_enabled=torch.is_grad_enabled,
            float32=torch.float32, bfloat16=torch.bfloat16, int32=torch.int32, int64=torch.int64,
            cuda=SimpleNamespace(get_device_properties=lambda d: SimpleNamespace(multi_processor_count=132)),
            empty=lambda shape, dtype, device: torch.empty(shape, dtype=dtype),
            zeros=lambda shape, dtype, device: torch.zeros(shape, dtype=dtype),
        )
        tree = ast.parse(KERNEL.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SmallQueryAttention")
        env = {"torch": proxy, "math": math, "partition_keys": base.partition,
               "QUERY_TILE": 16, "KEY_TILE": 64,
               "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
               "_segments": Launcher("segments"), "_merge": Launcher("merge"),
               "_fused_attention": Launcher("fused")}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(KERNEL), "exec"), env)
        for batch, capacity, ksplits in ((2, 257, 0), (2, 257, 4), (200, 1, 2)):
            calls.clear()
            plan = env["SmallQueryAttention"](batch, 4, 1, capacity, "cuda:0")
            self.assertEqual(plan.counters.dtype, torch.int32)
            self.assertEqual(tuple(plan.counters.shape), (batch,))
            width = 6 * 128
            qkv = (torch.randn(ksplits, batch, width) if ksplits
                   else torch.randn(batch, width).bfloat16())
            gains = torch.randn(128).bfloat16()
            cos, sin = tables(capacity)
            keys = torch.randn(batch, 1, capacity, 128).bfloat16()
            values = torch.randn_like(keys)
            position = torch.tensor([0])
            out = plan.run_fused(qkv, gains, gains, 1e-6, 1e-6, cos, sin, position, keys, values, 0.25)
            self.assertEqual(out.shape, (batch, 1, 4, 128))
            self.assertEqual(len(calls), 1, "exactly one launch replaces consumer, segments and merge")
            name, grid, args, kwargs = calls[0]
            self.assertEqual(name, "fused")
            self.assertEqual(grid, (batch, plan.splits))
            self.assertIs(args[0], qkv)
            self.assertIs(args[5], position)
            self.assertIs(args[6], keys)
            self.assertIs(args[7], values)
            self.assertIs(args[8], plan.partials)
            self.assertIs(args[9], plan.stats)
            self.assertIs(args[10], plan.counters)
            self.assertEqual(args[12:], (width, batch * width))
            self.assertEqual((kwargs["S"], kwargs["SPAN"], kwargs["SPLITS"]), (plan.splits, plan.span, ksplits))
            self.assertEqual(kwargs["SS"], 1 << (plan.splits - 1).bit_length())
            self.assertEqual((kwargs["G"], kwargs["NQ"], kwargs["NK"], kwargs["C"]), (4, 4, 1, capacity))
            self.assertFalse(kwargs["enable_fp_fusion"])
            for bad in (
                lambda: plan.run_fused(qkv.float() if ksplits == 0 else qkv.bfloat16(), gains, gains, 1e-6, 1e-6, cos, sin, position, keys, values, 0.25),
                lambda: plan.run_fused(qkv, gains, gains, 1e-6, 1e-6, cos, sin, position.int(), keys, values, 0.25),
                lambda: plan.run_fused(qkv, gains, gains, 1e-6, 1e-6, cos[:-1] if capacity > 1 else cos.float(), sin, position, keys, values, 0.25),
                lambda: plan.run_fused(qkv, gains[:64], gains, 1e-6, 1e-6, cos, sin, position, keys, values, 0.25),
                lambda: plan.run_fused(qkv, gains, gains, 1e-6, 1e-6, cos, sin, position, keys.float(), values, 0.25),
                lambda: plan.run_fused(qkv, gains, gains, 1e-6, 1e-6, cos, sin, position, keys, values, float("nan")),
            ):
                with self.assertRaises(ValueError):
                    bad()
            self.assertEqual(len(calls), 1)

    @torch.inference_mode()
    def test_context_dispatch_forwards_or_refuses(self):
        marker = object()
        plan = SimpleNamespace(run_fused=mock.Mock(return_value=marker))
        fake = SimpleNamespace(SmallQueryAttention=mock.Mock(return_value=plan))
        with mock.patch.object(torch.Tensor, "is_cuda", new=property(lambda self: True)), \
                mock.patch.dict(sys.modules, {"kernels.small_query_attention": fake}):
            context = flash.FlashDecodeContext(2, 8, 2, torch.arange(13))
            args = tuple(object() for _ in range(11))
            self.assertIs(context.attention_from_qkv(*args), marker)
            plan.run_fused.assert_called_once_with(*args)
            plan.run_fused.side_effect = RuntimeError("candidate fault")
            with self.assertRaisesRegex(RuntimeError, "candidate fault"):
                context.attention_from_qkv(*args)
            no_plan = flash.FlashDecodeContext(1, 34, 2, torch.arange(13))
            self.assertIsNone(no_plan.small_query)
            with self.assertRaises(ValueError):
                no_plan.attention_from_qkv(*args)

    @torch.inference_mode()
    def test_decode_step_uses_one_launch_only_with_a_plan(self):
        ref = base.load("fused_equations_fixture", ENGINE.parent / "tests/test_prefill.py")
        forward = base.cpu_forward()
        torch.manual_seed(8)
        for batch, length, steps in ((1, 7, 4), (3, 13, 3)):
            model, cache, (cos, sin) = ref.fixture(batch, length, extra=steps + 1)
            for t in cache.key_cache + cache.value_cache:
                t.fill_(float("nan"))
            prompt = torch.randint(0, 127, (batch, length))
            initial = ref.materialized(model, cache, prompt, cos, sin)
            oracle_cache = SimpleNamespace(key_cache=[t.clone() for t in cache.key_cache],
                                           value_cache=[t.clone() for t in cache.value_cache])
            context = flash.FlashDecodeContext(batch, 4, 2, torch.arange(length + steps + 1))
            oracle = flash.FlashDecodeContext(batch, 4, 2, torch.arange(length + steps + 1))
            oracle.attention = lambda q, k, v, m, s: base.reference(q, k, v, oracle.used, s)
            fused_calls, legacy_calls = [], []

            def attention_from_qkv(qkv, qw, kw, qe, ke, c, s, position, keys, values, scale):
                fused_calls.append(qkv.shape)
                q = base.ref_qkv(qkv, qw, kw, qe, ke, c, s, position, keys, values)
                return base.reference(q, keys, values, torch.full((batch * 2,), int(position) + 1, dtype=torch.int32), scale)

            context.small_query = object()
            context.attention_from_qkv = attention_from_qkv
            context.attention = lambda *a: legacy_calls.append(a)
            token = initial[:, -1].argmax(-1, keepdim=True)
            for step in range(steps):
                pos = torch.tensor([length + step])
                mask = context.prepare(pos)
                oracle.prepare(pos)
                actual = forward(model, cache, token, pos, mask, context, cos, sin)
                expected = forward(model, oracle_cache, token, pos, mask, oracle, cos, sin)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.125)
                token = actual[:, -1].argmax(-1, keepdim=True)
                DIAGNOSTICS["pipeline_steps"] += 1
            self.assertEqual(len(fused_calls), steps * len(model.model.layers))
            self.assertEqual(legacy_calls, [])
            for t in cache.key_cache + cache.value_cache:
                self.assertTrue(bool(torch.isfinite(t[:, :, :length + steps]).all()))
                self.assertTrue(bool(torch.isnan(t[:, :, length + steps:]).all()))

    @torch.inference_mode()
    def test_real_transformers_own_prefix_through_fused_body_with_split_partials(self):
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("real Transformers CPU integration requires the pinned package")
        from copy import deepcopy
        from transformers import Qwen3Config, Qwen3ForCausalLM
        sys.path.insert(0, str(ENGINE))
        try:
            from decode import DecodeState
            from mlp import PackedMLP
        finally:
            sys.path.pop(0)
        cfg = Qwen3Config(vocab_size=127, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=128, max_position_embeddings=1024, rope_theta=5_000_000,
                          tie_word_embeddings=True, sliding_window=None)
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(9)
        native = Qwen3ForCausalLM(cfg).eval().bfloat16()
        model = deepcopy(native)
        for layer in model.model.layers:
            a = layer.self_attn
            a.qkv_weight = torch.cat([p.weight for p in (a.q_proj, a.k_proj, a.v_proj)])
            layer.mlp = PackedMLP(layer.mlp)

        def split_partials(x, weight):
            # Two FP32 K-range partial products, the decode projection's shape.
            half = x.shape[1] // 2
            return torch.stack((F.linear(x[:, :half].float(), weight[:, :half].float()),
                                F.linear(x[:, half:].float(), weight[:, half:].float())))

        def silu_mul(p):
            g, u = p.chunk(2, -1)
            return F.silu(g) * u

        forward = base.bodies(ENGINE / "decode_step.py", {"_project", "_mlp_hidden", "fused_decode_forward"},
                              {"torch": torch, "F": F, "supports": lambda x, w, pairs=False: not pairs,
                               "linear_partials": split_partials, "linear_silu_mul": None,
                               "add_rms_norm": base.add_norm, "silu_mul": silu_mul,
                               "qkv_norm_rope_cache": None})["fused_decode_forward"]
        body = FusedBody()
        for batch, length, count in ((1, 7, 5), (3, 13, 4)):
            state = DecodeState(model, batch, length, count)
            state.fused_forward = forward
            context = flash.FlashDecodeContext(batch, 4, 2, state.key_positions)
            context.small_query = body
            shapes = []

            def attention_from_qkv(qkv, *rest, splits=3):
                shapes.append(tuple(qkv.shape))
                return body.run(qkv, *rest, splits=splits)

            context.attention_from_qkv = attention_from_qkv
            state.flash_context = context
            pointers = [t.data_ptr() for t in state.cache.key_cache + state.cache.value_cache]
            for _ in range(2):
                for t in state.cache.key_cache + state.cache.value_cache:
                    t.fill_(float("nan"))
                prompt = torch.randint(0, 127, (batch, length))
                state.prefill(prompt)
                emitted = list(state.emit(count))
                tokens = torch.tensor(emitted).T
                logits = native(input_ids=torch.cat((prompt, tokens[:, :-1]), 1), use_cache=False).logits[:, length - 1:]
                gap = logits.float().amax(-1) - logits.float().gather(-1, tokens[..., None]).squeeze(-1)
                self.assertLessEqual(float(gap.max()), 2.0)
                self.assertEqual(int(state.position), length + count - 1)
                self.assertEqual(pointers, [t.data_ptr() for t in state.cache.key_cache + state.cache.value_cache])
                self.assertTrue(all(type(t) is int for row in emitted for t in row))
                self.assertTrue(shapes and all(s == (2, batch, 8 * 128) for s in shapes))
                DIAGNOSTICS["pipeline_steps"] += count - 1


    @torch.inference_mode()
    def test_spy_on_production_context_class_observes_engine_steps(self):
        """The CUDA engine test patches the class DecodeState really constructs.

        This file's ``flash`` helper is a separate module object loaded by path,
        so a spy on it can never see engine calls. Bind the spy to the module the
        engine imports and prove, on CPU, that DecodeState.step reaches it.
        """
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("real Transformers CPU integration requires the pinned package")
        from copy import deepcopy
        from transformers import Qwen3Config, Qwen3ForCausalLM
        sys.path.insert(0, str(ENGINE))
        try:
            import decode
            import flash_decode
            from decode import DecodeState
            from mlp import PackedMLP
        finally:
            sys.path.pop(0)
        production = flash_decode.FlashDecodeContext
        self.assertIsNot(production, flash.FlashDecodeContext)
        self.assertIs(decode.__dict__.get("make_flash_context", flash_decode.make_flash_context),
                      flash_decode.make_flash_context)
        self.assertIs(flash_decode.make_flash_context.__globals__["FlashDecodeContext"], production)
        cfg = Qwen3Config(vocab_size=127, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=128, max_position_embeddings=1024, rope_theta=5_000_000,
                          tie_word_embeddings=True, sliding_window=None)
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(9)
        native = Qwen3ForCausalLM(cfg).eval().bfloat16()
        model = deepcopy(native)
        for layer in model.model.layers:
            a = layer.self_attn
            a.qkv_weight = torch.cat([p.weight for p in (a.q_proj, a.k_proj, a.v_proj)])
            layer.mlp = PackedMLP(layer.mlp)
        forward = base.cpu_forward()

        def fused_reference(context, qkv, qw, kw, qe, ke, c, s, position, keys, values, scale):
            q = base.ref_qkv(qkv, qw, kw, qe, ke, c, s, position, keys, values)
            used = torch.full((context.virtual_batch,), int(position) + 1, dtype=torch.int32)
            return base.reference(q, keys, values, used, scale)

        for batch, length, count in ((1, 7, 5), (3, 13, 4)):
            state = DecodeState(model, batch, length, count)
            self.assertIsNone(state.flash_context)  # CPU: the engine attaches none itself.
            context = production(batch, 4, 2, state.key_positions)
            context.small_query = object()  # static dispatch marker; the spy replaces the call
            state.flash_context, state.fused_forward = context, forward
            prompt = torch.randint(0, 127, (batch, length))
            with mock.patch.object(production, "attention_from_qkv", autospec=True,
                                   side_effect=fused_reference) as fused, \
                    mock.patch.object(production, "attention", autospec=True,
                                      side_effect=AssertionError("legacy attention must not run")) as legacy:
                state.prefill(prompt)
                emitted = list(state.emit(count))
            self.assertEqual(fused.call_count, (count - 1) * len(model.model.layers))
            self.assertEqual(legacy.call_count, 0)
            self.assertTrue(all(call.args[0] is context for call in fused.call_args_list))
            tokens = torch.tensor(emitted).T
            logits = native(input_ids=torch.cat((prompt, tokens[:, :-1]), 1), use_cache=False).logits[:, length - 1:]
            gap = logits.float().amax(-1) - logits.float().gather(-1, tokens[..., None]).squeeze(-1)
            self.assertLessEqual(float(gap.max()), 2.0)
            # Negative control: the helper copy's class is blind to the same run.
            with mock.patch.object(flash.FlashDecodeContext, "attention_from_qkv", autospec=True) as blind, \
                    mock.patch.object(production, "attention_from_qkv", autospec=True,
                                      side_effect=fused_reference):
                state.prefill(prompt)
                list(state.emit(count))
            self.assertEqual(blind.call_count, 0)


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
                     "CUDA/Triton required; CPU shims are not GPU validation")
class FusedAttentionCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ENGINE))
        try:
            from kernels.decode_fused import qkv_norm_rope_cache
            from kernels.skinny_gemm import linear_partials, supports
            from kernels.small_query_attention import SmallQueryAttention
        finally:
            sys.path.pop(0)
        cls.Plan = SmallQueryAttention
        cls.consumer = staticmethod(qkv_norm_rope_cache)
        cls.partials = staticmethod(linear_partials)
        cls.supports = staticmethod(supports)

    def rotary(self, capacity):
        sys.path.insert(0, str(ENGINE))
        try:
            from decode import DecodeState
        finally:
            sys.path.pop(0)
        helper = load("fused_cuda_step_fixture", ENGINE.parent / "tests/test_decode_step.py")
        model = helper.tiny_model(torch.bfloat16, head_dim=128).cuda()
        table = DecodeState(model, 1, 1, capacity - 1)
        return table.cos, table.sin

    @torch.inference_mode()
    def test_operator_matches_separate_kernels_and_reference(self):
        torch.manual_seed(21)
        for b, hq, hk, c in ((1, 32, 8, 544), (4, 32, 8, 2080), (16, 32, 8, 640),
                              (3, 6, 2, 257), (2, 16, 1, 129), (65, 4, 2, 3), (1, 32, 8, 4097)):
            cos, sin = self.rotary(c)
            plan = self.Plan(b, hq, hk, c, "cuda:0")
            width = (hq + 2 * hk) * 128
            qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
            kw = torch.randn_like(qw)
            weight = (torch.randn(width, 2560, device="cuda") * 0.02).bfloat16()
            for pos in sorted({0, c // 2, c - 1}):
                x = torch.randn(b, 2560, device="cuda", dtype=torch.bfloat16)
                inputs = [F.linear(x, weight)]
                if self.supports(x, weight):
                    inputs.append(self.partials(x, weight))
                for qkv in inputs:
                    with self.subTest(shape=(b, hq, hk, c), pos=pos, splits=qkv.ndim == 3):
                        keys = torch.full((b, hk, c, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
                        keys[:, :, :pos].normal_()
                        values = keys.clone()
                        values[:, :, :pos].normal_()
                        ref_keys, ref_values = keys.clone(), values.clone()
                        position = torch.tensor([pos], device="cuda")
                        before = qkv.clone()
                        out = plan.run_fused(qkv, qw, kw, 1e-6, 3e-6, cos, sin, position, keys, values, 128 ** -0.5)
                        q = self.consumer(qkv, qw, kw, 1e-6, 3e-6, cos, sin, position, ref_keys, ref_values)
                        used = torch.full((b * hk,), pos + 1, dtype=torch.int32, device="cuda")
                        expected = plan.run(q, ref_keys, ref_values, used, 128 ** -0.5)
                        self.assertTrue(bool(torch.isfinite(out).all()))
                        torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)
                        torch.testing.assert_close(keys[:, :, pos], ref_keys[:, :, pos], rtol=0.02, atol=0.02)
                        torch.testing.assert_close(values[:, :, pos], ref_values[:, :, pos], rtol=0, atol=0)
                        others = torch.arange(c, device="cuda") != pos
                        torch.testing.assert_close(keys[:, :, others], ref_keys[:, :, others], rtol=0, atol=0, equal_nan=True)
                        torch.testing.assert_close(values[:, :, others], ref_values[:, :, others], rtol=0, atol=0, equal_nan=True)
                        torch.testing.assert_close(qkv, before, rtol=0, atol=0)
                        self.assertEqual(int(plan.counters.abs().sum()), 0)
                        reference = base.reference(q, ref_keys, ref_values, used, 128 ** -0.5)
                        torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)

    @torch.inference_mode()
    def test_graph_replay_follows_position_and_leaves_counters_zero(self):
        b, hq, hk, c = 3, 12, 3, 257
        cos, sin = self.rotary(c)
        plan = self.Plan(b, hq, hk, c, "cuda:0")
        self.assertGreater(plan.splits, 1)
        width = (hq + 2 * hk) * 128
        qkv = torch.randn(b, width, device="cuda", dtype=torch.bfloat16)
        qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        kw = torch.randn_like(qw)
        keys = torch.zeros(b, hk, c, 128, device="cuda", dtype=torch.bfloat16)
        values = keys.clone()
        position = torch.tensor([0], device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                plan.run_fused(qkv, qw, kw, 1e-6, 1e-6, cos, sin, position, keys, values, 128 ** -0.5)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = plan.run_fused(qkv, qw, kw, 1e-6, 1e-6, cos, sin, position, keys, values, 128 ** -0.5)
        addresses = [t.data_ptr() for t in (plan.partials, plan.stats, plan.counters, out)]
        for pos in (1, 63, 64, 65, 200, 256, 3):
            qkv.normal_()
            keys.normal_(); values.normal_()
            keys[:, :, pos:] = float("nan"); values[:, :, pos:] = float("nan")
            ref_keys, ref_values = keys.clone(), values.clone()
            position.fill_(pos)
            graph.replay()
            q = self.consumer(qkv, qw, kw, 1e-6, 1e-6, cos, sin, position, ref_keys, ref_values)
            used = torch.full((b * hk,), pos + 1, dtype=torch.int32, device="cuda")
            torch.testing.assert_close(out, base.reference(q, ref_keys, ref_values, used, 128 ** -0.5),
                                       rtol=0.02, atol=0.02)
            torch.testing.assert_close(keys[:, :, pos], ref_keys[:, :, pos], rtol=0.02, atol=0.02)
            self.assertEqual(int(plan.counters.abs().sum()), 0)
            self.assertEqual(addresses, [t.data_ptr() for t in (plan.partials, plan.stats, plan.counters, out)])

    @torch.inference_mode()
    def test_real_engine_uses_one_launch_attention_and_matches_native(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        sys.path.insert(0, str(ENGINE))
        try:
            from engine import Engine
            import flash_decode  # the module DecodeState imports, not this file's helper copy
        finally:
            sys.path.pop(0)
        production = flash_decode.FlashDecodeContext
        self.assertIsNot(production, flash.FlashDecodeContext)
        cfg = Qwen3Config(vocab_size=127, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=128, max_position_embeddings=1024, rope_theta=5_000_000,
                          tie_word_embeddings=True, sliding_window=None, eos_token_id=0)
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(271)
        native = Qwen3ForCausalLM(cfg).eval()
        with tempfile.TemporaryDirectory() as path:
            native.save_pretrained(path)
            engine = Engine(path)
            native = native.cuda().bfloat16()
            for batch, length, count in ((1, 7, 5), (1, 7, 5), (3, 13, 4), (2, 3, 1), (2, 3, 0)):
                prompt = torch.randint(0, 127, (batch, length), device="cuda")
                # A same-shape repeat reuses the captured graph: replay issues no
                # Python dispatch, so the spy must see calls only when capturing.
                previous = engine.decode_state
                replaying = (previous is not None and previous.shape == (batch, length, count)
                             and previous.graph is not None)
                with mock.patch.object(production, "attention_from_qkv", autospec=True,
                                       side_effect=production.attention_from_qkv) as fused, \
                        mock.patch.object(production, "attention", autospec=True,
                                          side_effect=production.attention) as legacy:
                    emitted = list(engine.generate(prompt.tolist(), count))
                if count > 1:
                    self.assertIsInstance(engine.decode_state.flash_context, production)
                self.assertEqual(len(emitted), count)
                self.assertTrue(all(len(row) == batch and all(type(t) is int for t in row) for row in emitted))
                if not count:
                    continue
                if count > 1:
                    self.assertIsNotNone(engine.decode_state.flash_context.small_query)
                    if replaying:
                        self.assertIs(engine.decode_state, previous)
                        self.assertIs(engine.decode_state.graph, previous.graph)
                        self.assertEqual(fused.call_count, 0)
                    else:
                        # Eager warmup steps and the capture itself use the fused path.
                        self.assertIsNotNone(engine.decode_state.graph)
                        self.assertGreater(fused.call_count, 0)
                    self.assertEqual(legacy.call_count, 0)
                tokens = torch.tensor(emitted, device="cuda").T
                logits = native(input_ids=torch.cat((prompt, tokens[:, :-1]), 1), use_cache=False).logits[:, length - 1:]
                gap = logits.float().amax(-1) - logits.float().gather(-1, tokens[..., None]).squeeze(-1)
                self.assertTrue(bool(torch.isfinite(gap).all()))
                self.assertLessEqual(float(gap.max()), 2.0)
            engine.model.lm_head.weight.zero_()
            self.assertEqual(list(engine.generate([[3] * 7, [4] * 7], 5)), [[0, 0]] * 5)


if __name__ == "__main__":
    unittest.main()
