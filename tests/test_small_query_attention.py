"""CPU semantic/indexing checks and optional CUDA tests, not GPU timing claims."""

import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

ENGINE = Path(__file__).resolve().parents[1] / "engine"
KERNEL = ENGINE / "kernels/small_query_attention.py"
DIAGNOSTICS = {"kernel_cases": 0, "max_abs_vs_fp64": 0.0, "pipeline_steps": 0}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flash = load("small_query_flash_test", ENGINE / "flash_decode.py")


def bodies(path, names, env):
    """Load actual selected source bodies without importing absent GPU packages."""
    tree = ast.parse(path.read_text())
    result = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            for arg in node.args.args:
                arg.annotation = None
            result.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=result, type_ignores=[])), str(path), "exec"), env)
    return env


partition = bodies(KERNEL, {"partition_keys"}, {"KEY_TILE": 64, "MAX_SPLITS": 32})["partition_keys"]


class Pointer:
    def __init__(self, tensor, offset=0, access=None):
        self.data = tensor.reshape(-1)
        self.offset = offset
        self.access = (torch.zeros(self.data.numel(), dtype=torch.int64),
                       torch.zeros(self.data.numel(), dtype=torch.int64)) if access is None else access
    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset, self.access)


class Ops:
    """Checked CPU math/pointers. NOT Triton interpreter, MMA, warps, or codegen."""
    int64, float32, bfloat16 = torch.int64, torch.float32, torch.bfloat16
    arange, exp2 = staticmethod(torch.arange), staticmethod(torch.exp2)
    trans = staticmethod(torch.t)
    @staticmethod
    def zeros(shape, dtype):
        return torch.zeros(shape, dtype=dtype)
    @staticmethod
    def full(shape, fill, dtype):
        return torch.full(shape, fill, dtype=dtype)
    program = (0, 0)
    def program_id(self, axis):
        return torch.tensor(self.program[axis], dtype=torch.int32)
    @staticmethod
    def sum(t, axis):
        return t.sum(dim=axis)
    @staticmethod
    def max(t, axis):
        return t.amax(dim=axis)
    @staticmethod
    def minimum(a, b):
        return torch.minimum(torch.as_tensor(a), torch.as_tensor(b))
    @staticmethod
    def maximum(a, b):
        return torch.maximum(torch.as_tensor(a), torch.as_tensor(b))
    @staticmethod
    def where(test, a, b):
        return torch.where(torch.as_tensor(test), torch.as_tensor(a), torch.as_tensor(b))
    @staticmethod
    def dot(a, b, acc=None, input_precision=None):
        assert a.dtype == b.dtype == torch.bfloat16
        assert input_precision == "ieee"
        result = a.float() @ b.float()
        return result if acc is None else acc + result
    @staticmethod
    def indices(p, mask):
        index = torch.as_tensor(p.offset, dtype=torch.int64)
        active = torch.broadcast_to(torch.as_tensor(mask, dtype=torch.bool), index.shape)
        if bool((((index < 0) | (index >= p.data.numel())) & active).any()):
            raise AssertionError("out-of-bounds active pointer")
        return index, active
    def load(self, p, mask=True, other=0):
        index, active = self.indices(p, mask)
        out = torch.full(index.shape, other, dtype=p.data.dtype)
        out[active] = p.data[index[active]]
        p.access[0].scatter_add_(0, index[active].flatten(), torch.ones_like(index[active].flatten()))
        return out
    def store(self, p, value, mask=True):
        index, active = self.indices(p, mask)
        value = torch.broadcast_to(value, index.shape)
        p.data[index[active]] = value[active].to(p.data.dtype)
        p.access[1].scatter_add_(0, index[active].flatten(), torch.ones_like(index[active].flatten()))


class KernelBody:
    def __init__(self):
        self.ops = Ops()
        env = bodies(KERNEL, {"_segments", "_merge"}, {"tl": self.ops})
        self.segment, self.merge = env["_segments"], env["_merge"]
    @torch.inference_mode()
    def run(self, q, k, v, used, scale, splits=3, check=True):
        batch, hq, _, width = q.shape
        hk, capacity = k.shape[1:3]
        g, vh = hq // hk, batch * hk
        tiles = (capacity + 63) // 64
        span = ((tiles + splits - 1) // splits) * 64
        partial = torch.full((vh, splits, g, width), float("nan"))
        stats = torch.full((vh, splits, 2, g), float("nan"))
        out = torch.full((batch, 1, hq, width), float("nan"), dtype=q.dtype)
        ptrs = [Pointer(t) for t in (q, k, v, used, partial, stats, out)]
        for head in range(vh):
            for segment in range(splits):
                self.ops.program = (head, segment)
                self.segment(*ptrs, capacity, g, splits, span, scale, width, 16, 64)
        if splits > 1:
            for query in range(batch * hq):
                self.ops.program = (query, 0)
                self.merge(*ptrs[4:], g, splits, width, 1 << (splits - 1).bit_length())
        if check:
            for p in ptrs[:4]:
                assert not bool(p.access[1].any()), "input was mutated"
            assert bool((ptrs[-1].access[1] == 1).all()), "output write coverage"
            if splits > 1:
                assert bool((ptrs[4].access[1] == 1).all()), "partial write coverage"
                assert bool((ptrs[5].access[1] == 1).all()), "statistics write coverage"
            for head in range(vh):
                count = int(used[head])
                for p in ptrs[1:3]:
                    read = p.access[0].view(vh, capacity, width)[head]
                    assert not bool(read[count:].any()), "invalid capacity was loaded"
                    assert bool((read[:count] == 1).all()), "valid K/V not read exactly once per group"
        self.last_partial, self.last_stats, self.last_span = partial, stats, span
        DIAGNOSTICS["kernel_cases"] += 1
        return out


def reference(q, k, v, used, scale):
    """Independent original-batch/head FP64 full-context equations."""
    batch, hq, _, d = q.shape
    hk, capacity = k.shape[1:3]
    out = torch.zeros((batch, 1, hq, d), dtype=q.dtype, device=q.device)
    for b in range(batch):
        for h in range(hk):
            count = int(used[b * hk + h])
            if count:
                lo, hi = h * (hq // hk), (h + 1) * (hq // hk)
                scores = (q[b, lo:hi, 0].double() @ k[b, h, :count].double().T) * scale
                out[b, 0, lo:hi] = (scores.softmax(-1) @ v[b, h, :count].double()).to(q.dtype)
    return out


def norm(x, w, eps):
    f = x.float()
    return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def ref_qkv(packed, qw, kw, qe, ke, cos, sin, position, keys, values):
    # Reference for the unchanged #14 QKV consumer, not an emulation of its MMA.
    if packed.ndim == 3:
        total = torch.zeros_like(packed[0])
        for piece in packed:
            total += piece
        packed = total.bfloat16()
    b, hk, _, d = keys.shape
    hq = packed.shape[-1] // d - 2 * hk
    q, k, v = packed.split((hq*d, hk*d, hk*d), -1)
    q, k = norm(q.reshape(b, hq, 1, d), qw, qe), norm(k.reshape(b, hk, 1, d), kw, ke)
    c, s = cos.index_select(0, position)[None, None], sin.index_select(0, position)[None, None]
    def rotate(x):
        half = torch.cat((-x[..., d//2:], x[..., :d//2]), -1)
        return x * c + half * s
    q, k = rotate(q), rotate(k)
    keys.index_copy_(2, position, k)
    values.index_copy_(2, position, v.reshape(b, hk, 1, d))
    return q.contiguous()


def add_norm(x, branch, w, eps):
    if branch.ndim == 3:
        total = torch.zeros_like(branch[0])
        for piece in branch:
            total += piece
        branch = total.bfloat16()
    total = x + branch
    return total, norm(total, w, eps)


def cpu_forward():
    def silu_mul(p):
        g, u = p.chunk(2, -1)
        return F.silu(g) * u
    return bodies(ENGINE / "decode_step.py", {"_project", "_mlp_hidden", "fused_decode_forward"},
                  {"torch": torch, "F": F, "supports": lambda *a, **k: False,
                   "linear_partials": None, "linear_silu_mul": None,
                   "add_rms_norm": add_norm, "qkv_norm_rope_cache": ref_qkv,
                   "silu_mul": silu_mul})["fused_decode_forward"]


class SmallQueryCPU(unittest.TestCase):
    def test_partition_coverage_alignment_and_limits(self):
        for heads in (1, 8, 24, 128, 512, 4096):
            for capacity in (1, 63, 64, 65, 544, 2080, 8193, 262144):
                for sms in (1, 80, 114, 132, 144):
                    splits, span = partition(heads, capacity, sms)
                    self.assertTrue(1 <= splits <= 32)
                    self.assertEqual(span % 64, 0)
                    self.assertLess((splits - 1) * span, capacity)
                    self.assertGreaterEqual(splits * span, capacity)
        for args in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
            with self.assertRaises(ValueError):
                partition(*args)

    @torch.inference_mode()
    def test_actual_kernels_full_context_splits_tails_groups_and_poison(self):
        torch.manual_seed(67)
        cases = ((1, 4, 1, 1, 1), (1, 4, 1, 63, 1), (1, 4, 1, 65, 3),
                 (1, 32, 8, 544, 9), (2, 6, 2, 257, 3), (3, 8, 2, 129, 8),
                 (1, 16, 1, 193, 2), (1, 8, 1, 641, 4), (1, 4, 1, 1025, 1),
                 (4, 32, 8, 2080, 9), (16, 32, 8, 640, 3), (1, 1, 1, 65, 2))
        kernel = KernelBody()
        for b, hq, hk, cap, splits in cases:
            q = torch.randn(b, hq, 1, 128).bfloat16()
            k = torch.randn(b, hk, cap, 128).bfloat16()
            v = torch.randn_like(k)
            # First pass covers the full capacity; second has empty and partial segments.
            counts = [torch.full((b * hk,), cap, dtype=torch.int32),
                      torch.arange(b * hk, dtype=torch.int32) * 37 % (cap + 1)]
            for used in counts:
                k.normal_(); v.normal_()
                for h in range(b * hk):
                    k.view(b*hk, cap, 128)[h, int(used[h]):] = float("nan")
                    v.view(b*hk, cap, 128)[h, int(used[h]):] = float("nan")
                old = [t.clone() for t in (q, k, v, used)]
                out = kernel.run(q, k, v, used, 128**-0.5, splits)
                expected = reference(q, k, v, used, 128**-0.5)
                torch.testing.assert_close(out, expected, rtol=0.02, atol=0.004)
                self.assertTrue(bool(torch.isfinite(out).all()))
                DIAGNOSTICS["max_abs_vs_fp64"] = max(DIAGNOSTICS["max_abs_vs_fp64"], float((out-expected).abs().max()))
                for new, before in zip((q, k, v, used), old):
                    torch.testing.assert_close(new, before, rtol=0, atol=0, equal_nan=True)

    @torch.inference_mode()
    def test_uniform_and_extreme_scores_not_chunk_average(self):
        kernel = KernelBody()
        q = torch.zeros(1, 4, 1, 128, dtype=torch.bfloat16)
        k = torch.zeros(1, 1, 193, 128, dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        v[:, :, :64] = 2; v[:, :, 64:128] = 6; v[:, :, 128:] = 20
        used = torch.tensor([193], dtype=torch.int32)
        out = kernel.run(q, k, v, used, 128**-0.5, 4)
        torch.testing.assert_close(out, reference(q, k, v, used, 128**-0.5), rtol=0, atol=0)
        # Equal weighting of segment averages is WRONG for unequal segment lengths.
        wrong = torch.tensor((2+6+20+20)/4, dtype=torch.bfloat16)
        self.assertFalse(torch.equal(out, torch.full_like(out, float(wrong))))
        q.fill_(32); k[:, :, :64] = -32; k[:, :, 64:128] = 0; k[:, :, 128:] = 32
        out = kernel.run(q, k, v, used, 128**-0.5, 4)
        torch.testing.assert_close(out, reference(q, k, v, used, 128**-0.5), rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(out).all()))

    @torch.inference_mode()
    def test_merge_requires_fp32_partials_and_valid_lengths(self):
        torch.manual_seed(781)
        q = torch.randn(1, 4, 1, 128).bfloat16()
        k = torch.randn(1, 1, 257, 128).bfloat16()
        v = (torch.randn_like(k).float() * 17).bfloat16()
        used = torch.tensor([201], dtype=torch.int32)
        kernel = KernelBody()
        actual = kernel.run(q, k, v, used, 128**-0.5, 5)
        wrong = torch.empty_like(actual)
        rounded = kernel.last_partial.bfloat16().float()
        for h in range(4):
            kernel.ops.program = (h, 0)
            kernel.merge(Pointer(rounded), Pointer(kernel.last_stats), Pointer(wrong), 4, 5, 128, 8)
        differences = int(torch.count_nonzero(wrong != actual))
        DIAGNOSTICS["wrong_bf16_partial_mismatches"] = differences
        self.assertGreater(differences, 0, "negative control lost its sensitivity")
        incorrect_length = kernel.run(q, k, v, torch.tensor([257], dtype=torch.int32), 128**-0.5, 5)
        self.assertFalse(torch.equal(actual, incorrect_length))

    @torch.inference_mode()
    def test_actual_context_selects_once_preserves_prefix_checks_and_native_fallback(self):
        marker = object()
        plan = SimpleNamespace(run=mock.Mock(return_value=marker))
        constructor = mock.Mock(return_value=plan)
        fake = SimpleNamespace(SmallQueryAttention=constructor)
        # Metadata/control-flow test ONLY; no CUDA tensors or actual kernels here.
        with mock.patch.object(torch.Tensor, "is_cuda", new=property(lambda self: True)), \
                mock.patch.dict(sys.modules, {"kernels.small_query_attention": fake}):
            context = flash.FlashDecodeContext(2, 8, 2, torch.arange(13))
            self.assertIs(context.small_query, plan)
            constructor.assert_called_once_with(2, 8, 2, 13, torch.device("cpu"))
            q = torch.randn(2, 8, 1, 128).bfloat16()
            k = torch.randn(2, 2, 13, 128).bfloat16()
            mask = context.prepare(torch.tensor([7]))
            with mock.patch.object(flash, "_native_flash", side_effect=AssertionError("native path must not run")):
                self.assertIs(context.attention(q, k, k, mask, 0.25), marker)
            args = plan.run.call_args.args
            self.assertIs(args[3], context.used)
            self.assertEqual(args[4], 0.25)
            with self.assertRaises(ValueError):
                context.attention(q, k, k, mask.clone(), 0.25)
            with self.assertRaises(ValueError):
                context.attention(q.float(), k, k, mask, 0.25)
            # No catch-and-fallback of candidate faults.
            plan.run.side_effect = RuntimeError("candidate fault")
            with self.assertRaisesRegex(RuntimeError, "candidate fault"):
                context.attention(q, k, k, mask, 0.25)
            self.assertIsNone(flash.FlashDecodeContext(1, 34, 2, torch.arange(13)).small_query)
        # CPU creation must import no GPU-only module.
        with mock.patch.dict(sys.modules, {"kernels.small_query_attention": None}):
            self.assertIsNone(flash.FlashDecodeContext(1, 4, 1, torch.arange(13)).small_query)

    @torch.inference_mode()
    def test_actual_wrapper_launch_arguments_and_validation(self):
        # Execute the real constructor/run with CPU storage and launch spies.
        # This verifies wiring only; it does not instantiate a CUDA allocation.
        import math
        calls = []
        class Launcher:
            def __init__(self, name):
                self.name = name
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    calls.append((self.name, grid, args, kwargs))
                    if self.name == "segments" and kwargs["S"] == 1:
                        args[6].zero_()
                    if self.name == "merge":
                        args[2].zero_()
                return launch
        proxy = SimpleNamespace(
            device=torch.device, is_grad_enabled=torch.is_grad_enabled,
            float32=torch.float32, bfloat16=torch.bfloat16, int32=torch.int32,
            cuda=SimpleNamespace(get_device_properties=lambda d: SimpleNamespace(multi_processor_count=132)),
            empty=lambda shape, dtype, device: torch.empty(shape, dtype=dtype),
        )
        tree = ast.parse(KERNEL.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SmallQueryAttention")
        env = {"torch": proxy, "math": math, "partition_keys": partition,
               "QUERY_TILE": 16, "KEY_TILE": 64,
               "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length()),
               "_segments": Launcher("segments"), "_merge": Launcher("merge")}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(KERNEL), "exec"), env)
        for batch, capacity in ((2, 257), (200, 1)):
            calls.clear()
            plan = env["SmallQueryAttention"](batch, 4, 1, capacity, "cuda:0")
            q = torch.randn(batch, 4, 1, 128).bfloat16()
            k = torch.randn(batch, 1, capacity, 128).bfloat16()
            used = torch.ones(batch, dtype=torch.int32)
            out = plan.run(q, k, k, used, 0.25)
            self.assertEqual(out.shape, (batch, 1, 4, 128))
            self.assertEqual(calls[0][1], (batch, plan.splits))
            self.assertIs(calls[0][2][3], used)
            self.assertIs(calls[0][2][4], plan.partials)
            self.assertIs(calls[0][2][5], plan.stats)
            self.assertEqual(calls[0][3]["SPAN"], plan.span)
            self.assertEqual(calls[0][3]["QM"], 16)
            self.assertEqual(plan.partials.dtype, torch.float32)
            self.assertEqual(plan.stats.dtype, torch.float32)
            self.assertEqual(len(calls), 1 + (plan.splits > 1))
            with self.assertRaises(ValueError):
                plan.run(q, k, k, used.long(), 0.25)
            with self.assertRaises(ValueError):
                plan.run(q.float(), k, k, used, 0.25)
            with self.assertRaises(ValueError):
                plan.run(q, k, k, used, float("nan"))

    @torch.inference_mode()
    def test_actual_decode_pipeline_with_independent_attention_reference(self):
        # The repository's existing materialized fixture is NOT Transformers.
        ref = load("small_query_equations_fixture", ENGINE.parent / "tests/test_prefill.py")
        forward, kernel = cpu_forward(), KernelBody()
        torch.manual_seed(8)
        for batch, length, steps in ((1, 7, 4), (3, 13, 3)):
            model, cache, (cos, sin) = ref.fixture(batch, length, extra=steps+1)
            old_addresses = [t.data_ptr() for t in cache.key_cache + cache.value_cache]
            for _ in range(2):
                for t in cache.key_cache + cache.value_cache:
                    t.fill_(float("nan"))
                prompt = torch.randint(0, 127, (batch, length))
                initial = ref.materialized(model, cache, prompt, cos, sin)
                oracle_cache = SimpleNamespace(key_cache=[t.clone() for t in cache.key_cache],
                                               value_cache=[t.clone() for t in cache.value_cache])
                context = flash.FlashDecodeContext(batch, 4, 2, torch.arange(length+steps+1))
                oracle = SimpleNamespace(attention=lambda q,k,v,m,s: reference(q,k,v,context.used,s))
                context.attention = lambda q,k,v,m,s: kernel.run(q,k,v,context.used,s,3)
                token = initial[:, -1].argmax(-1, keepdim=True)
                for step in range(steps):
                    pos = torch.tensor([length + step])
                    mask = context.prepare(pos)
                    actual = forward(model, cache, token, pos, mask, context, cos, sin)
                    expected = forward(model, oracle_cache, token, pos, mask, oracle, cos, sin)
                    self.assertTrue(bool(torch.isfinite(actual).all()))
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.125)
                    token = actual[:, -1].argmax(-1, keepdim=True)
                    gap = expected[:, -1].float().amax(-1) - expected[:, -1].float().gather(-1, token).squeeze(-1)
                    self.assertLessEqual(float(gap.max()), 2.0)
                    DIAGNOSTICS["pipeline_max_abs_logits"] = max(
                        DIAGNOSTICS.get("pipeline_max_abs_logits", 0.0), float((actual-expected).abs().max()))
                    DIAGNOSTICS["pipeline_steps"] += 1
                self.assertEqual(old_addresses, [t.data_ptr() for t in cache.key_cache + cache.value_cache])
                for t in cache.key_cache + cache.value_cache:
                    self.assertTrue(bool(torch.isfinite(t[:, :, :length+steps]).all()))
                    self.assertTrue(bool(torch.isnan(t[:, :, length+steps:]).all()))

    @torch.inference_mode()
    def test_real_transformers_own_prefix_cache_reuse(self):
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
        forward = cpu_forward()
        kernel = KernelBody()
        for batch, length, count in ((1, 7, 5), (3, 13, 4)):
            state = DecodeState(model, batch, length, count)
            state.fused_forward = forward
            context = flash.FlashDecodeContext(batch, 4, 2, state.key_positions)
            context.attention = lambda q, k, v, mask, scale: kernel.run(q, k, v, context.used, scale, 3)
            state.flash_context = context
            pointers = [t.data_ptr() for t in state.cache.key_cache + state.cache.value_cache]
            for _ in range(2):
                for t in state.cache.key_cache + state.cache.value_cache:
                    t.fill_(float("nan"))
                prompt = torch.randint(0, 127, (batch, length))
                state.prefill(prompt)
                emitted = list(state.emit(count))
                tokens = torch.tensor(emitted).T
                logits = native(input_ids=torch.cat((prompt, tokens[:, :-1]), 1), use_cache=False).logits[:, length-1:]
                gap = logits.float().amax(-1) - logits.float().gather(-1, tokens[..., None]).squeeze(-1)
                self.assertLessEqual(float(gap.max()), 2.0)
                self.assertEqual(int(state.position), length + count - 1)
                self.assertEqual(pointers, [t.data_ptr() for t in state.cache.key_cache + state.cache.value_cache])
                self.assertTrue(all(type(t) is int for row in emitted for t in row))
                DIAGNOSTICS["pipeline_steps"] += count - 1


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
                     "CUDA/Triton required; CPU shims are not GPU validation")
class SmallQueryCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ENGINE))
        try:
            from kernels.small_query_attention import SmallQueryAttention
        finally:
            sys.path.pop(0)
        cls.Plan = SmallQueryAttention

    @torch.inference_mode()
    def test_actual_operator_shapes_poison_empty_segments_and_native_reference(self):
        torch.manual_seed(18)
        for b, hq, hk, c in ((1, 32, 8, 1), (1, 32, 8, 544), (4, 32, 8, 2080),
                              (16, 32, 8, 640), (3, 6, 2, 257), (2, 16, 1, 129),
                              (65, 4, 2, 3), (1, 32, 8, 4097)):
            ctx = flash.FlashDecodeContext(b, hq, hk, torch.arange(c, device="cuda"))
            self.assertIsNotNone(ctx.small_query)
            q = torch.randn(b, hq, 1, 128, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(b, hk, c, 128, dtype=torch.bfloat16, device="cuda")
            v = torch.randn_like(k)
            for length in sorted({1, max(1, c//2), c}):
                k.normal_(); v.normal_()
                k[:, :, length:] = float("nan"); v[:, :, length:] = float("nan")
                mask = ctx.prepare(torch.tensor([length-1], device="cuda"))
                original = (q.clone(), k.clone(), v.clone(), ctx.used.clone())
                actual = ctx.attention(q, k, v, mask, 128**-0.5)
                expected = reference(q, k, v, ctx.used, 128**-0.5)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)
                # Explicit unchanged native call; no implication it is still the selected path.
                native = ctx._views_and_call(q, k, v, 128**-0.5)
                torch.testing.assert_close(actual, native, rtol=0.03, atol=0.01)
                for new, old in zip((q, k, v, ctx.used), original):
                    torch.testing.assert_close(new, old, rtol=0, atol=0, equal_nan=True)
            ctx.used.zero_()
            actual = ctx.small_query.run(q, k, v, ctx.used, 128**-0.5)
            self.assertEqual(int(torch.count_nonzero(actual)), 0)

    @torch.inference_mode()
    def test_graph_replay_updates_lengths_inputs_and_reuses_scratch(self):
        b, hq, hk, c = 3, 12, 3, 257
        plan = self.Plan(b, hq, hk, c, "cuda:0")
        q = torch.randn(b, hq, 1, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(b, hk, c, 128, dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)
        used = torch.ones(b * hk, dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                plan.run(q, k, v, used, 128**-0.5)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = plan.run(q, k, v, used, 128**-0.5)
        addresses = [t.data_ptr() for t in (plan.partials, plan.stats, out)]
        for length in (1, 63, 64, 65, 256, 257, 3):
            q.normal_(); k.normal_(); v.normal_()
            k[:, :, length:] = float("nan"); v[:, :, length:] = float("nan")
            used.fill_(length)
            graph.replay()
            torch.testing.assert_close(out, reference(q, k, v, used, 128**-0.5), rtol=0.02, atol=0.005)
            self.assertEqual(addresses, [t.data_ptr() for t in (plan.partials, plan.stats, out)])

    @torch.inference_mode()
    def test_real_engine_prefill_decode_own_prefix_and_eos(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        sys.path.insert(0, str(ENGINE))
        try:
            from engine import Engine
        finally:
            sys.path.pop(0)
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
                emitted = list(engine.generate(prompt.tolist(), count))
                self.assertEqual(len(emitted), count)
                self.assertTrue(all(len(row) == batch and all(type(t) is int for t in row) for row in emitted))
                if not count:
                    continue
                if count > 1:
                    self.assertIsNotNone(engine.decode_state.flash_context.small_query)
                tokens = torch.tensor(emitted, device="cuda").T
                logits = native(input_ids=torch.cat((prompt, tokens[:, :-1]), 1), use_cache=False).logits[:, length-1:]
                gap = logits.float().amax(-1) - logits.float().gather(-1, tokens[..., None]).squeeze(-1)
                self.assertTrue(bool(torch.isfinite(gap).all()))
                self.assertLessEqual(float(gap.max()), 2.0)
            # Zero logits force lowest-index argmax, which is EOS=0. No early stop.
            engine.model.lm_head.weight.zero_()
            native.lm_head.weight.zero_()
            self.assertIs(engine.model.lm_head.weight, engine.model.model.embed_tokens.weight)
            self.assertEqual(list(engine.generate([[3]*7, [4]*7], 5)), [[0, 0]]*5)


if __name__ == "__main__":
    unittest.main()
