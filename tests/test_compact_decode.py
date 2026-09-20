"""Compact decode checks. CPU AST execution is NOT Triton/CUDA execution."""

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compact = load("compact_under_test", ENGINE / "compact_decode.py")
mlp = load("compact_test_mlp", ENGINE / "mlp.py")


def actual_forward():
    tree = ast.parse((ENGINE / "decode.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "qwen_forward"]
    env = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ENGINE / "decode.py"), "exec"), env)
    return env["qwen_forward"]


def norm(x, weight, eps):
    f = x.float()
    return (f * torch.rsqrt(f.square().sum(-1, keepdim=True) / f.shape[-1] + eps)).to(x.dtype) * weight


def rotate(x, cos, sin):
    half = torch.cat((-x[..., 64:], x[..., :64]), -1)
    return x * cos[:, None] + half * sin[:, None]


def angles(batch, position, device="cpu"):
    inv = 1 / (5_000_000 ** (torch.arange(0, 128, 2, device=device).float() / 128))
    freqs = (torch.arange(batch, device=device).float()[:, None, None] + position) * inv
    freqs = torch.cat((freqs, freqs), -1)
    return freqs.cos().bfloat16(), freqs.sin().bfloat16()


class Pointer:
    def __init__(self, tensor, offsets=0, writes=None):
        self.data, self.offsets = tensor.reshape(-1), offsets
        self.writes = torch.zeros(self.data.numel(), dtype=torch.int64) if writes is None else writes

    def __add__(self, value):
        return Pointer(self.data, self.offsets + value, self.writes)


class CPUOps:
    """Minimal CPU arithmetic/indexing shim for the ACTUAL three kernel bodies.

    Does not emulate GPU reductions, code generation, warps, libdevice accuracy
    or launch timing. Every active pointer access is bounds checked.
    """
    bfloat16, float32 = torch.bfloat16, torch.float32
    program = 0
    arange = staticmethod(torch.arange)
    rsqrt = staticmethod(torch.rsqrt)

    def program_id(self, axis):
        assert axis == 0
        return self.program

    @staticmethod
    def sum(x, axis):
        return x.sum(dim=axis)

    @staticmethod
    def where(test, yes, no):
        return (yes if test else no) if isinstance(test, bool) else torch.where(test, yes, no)

    @staticmethod
    def indices(pointer, mask):
        index = torch.as_tensor(pointer.offsets, dtype=torch.int64)
        valid = torch.ones_like(index, dtype=torch.bool) if mask is None else torch.as_tensor(mask).expand(index.shape)
        active = index[valid]
        if bool(((active < 0) | (active >= pointer.data.numel())).any()):
            raise AssertionError("out-of-bounds kernel access")
        return index, valid

    def load(self, pointer, mask=None, other=0):
        index, valid = self.indices(pointer, mask)
        out = torch.full(index.shape, other, dtype=pointer.data.dtype)
        out[valid] = pointer.data[index[valid]]
        return out

    def store(self, pointer, value, mask=None):
        index, valid = self.indices(pointer, mask)
        value = torch.as_tensor(value).expand(index.shape)
        pointer.data[index[valid]] = value[valid].to(pointer.data.dtype)
        pointer.writes.scatter_add_(0, index[valid].flatten(), torch.ones_like(index[valid].flatten()))


class KernelCPU:
    def __init__(self):
        self.ops = CPUOps()
        path = ENGINE / "kernels/compact_decode.py"
        tree = ast.parse(path.read_text())
        nodes = []
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and n.name in ("_add_rms", "_qkv_cache", "_swiglu"):
                n.decorator_list = []
                for arg in n.args.args:
                    arg.annotation = None
                nodes.append(n)
        self.env = {"tl": self.ops,
                    "libdevice": SimpleNamespace(exp=torch.exp, div_rn=lambda a, b: a / b)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), self.env)

    def launch(self, name, count, *args):
        for p in range(count):
            self.ops.program = p
            self.env[name](*args)

    def add_rmsnorm(self, r, x, w, eps):
        s, y = torch.empty_like(r), torch.empty_like(r)
        sp, yp = Pointer(s), Pointer(y)
        h = r.shape[-1]
        self.launch("_add_rms", r.shape[0], Pointer(r), Pointer(x), Pointer(w), sp, yp,
                    h, eps, 1 << (h - 1).bit_length())
        assert bool((sp.writes == 1).all()) and bool((yp.writes == 1).all())
        return s, y

    def qkv_rope_cache(self, p, qw, kw, c, s, keys, values, position, nq, nk, qe, ke):
        q = torch.empty(p.shape[0], nq, 1, 128, dtype=p.dtype)
        qp, kp, vp = Pointer(q), Pointer(keys), Pointer(values)
        self.launch("_qkv_cache", p.shape[0] * (nq + nk), Pointer(p), Pointer(qw), Pointer(kw),
                    Pointer(c), Pointer(s), Pointer(position), qp, kp, vp, nq, nk, keys.shape[2],
                    0 if c.shape[0] == 1 else c.stride(0), 0 if s.shape[0] == 1 else s.stride(0), qe, ke)
        expected = torch.zeros_like(keys, dtype=torch.int64)
        expected[:, :, int(position)] = 1
        assert bool((qp.writes == 1).all())
        assert torch.equal(kp.writes.reshape(keys.shape), expected)
        assert torch.equal(vp.writes.reshape(values.shape), expected)
        return q

    def swiglu(self, p):
        i = p.shape[-1] // 2
        y = torch.empty(*p.shape[:-1], i, dtype=p.dtype)
        yp = Pointer(y)
        self.launch("_swiglu", (y.numel() + 255) // 256, Pointer(p), yp, i, y.numel(), 256)
        assert bool((yp.writes == 1).all())
        return y


class Norm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        return norm(x, self.weight, self.variance_epsilon)


class Rotary(nn.Module):
    def forward(self, x, positions):
        inv = 1 / (5_000_000 ** (torch.arange(0, 128, 2, device=x.device).float() / 128))
        f = positions.float()[..., None] * inv
        f = torch.cat((f, f), -1)
        return f.cos().to(x.dtype), f.sin().to(x.dtype)


class ToyLayer(nn.Module):
    """Independent materialized Torch equations; NOT a Transformers execution."""
    def __init__(self, cfg):
        super().__init__()
        h, i, nq, nk = cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads
        self.input_layernorm, self.post_attention_layernorm = Norm(h), Norm(h)
        a = self.self_attn = nn.Module()
        a.q_proj, a.k_proj, a.v_proj = [nn.Linear(h, width * 128, bias=False) for width in (nq, nk, nk)]
        a.o_proj = nn.Linear(nq * 128, h, bias=False)
        a.q_norm, a.k_norm, a.scaling, a.sliding_window = Norm(128), Norm(128), 128**-0.5, None
        self.mlp = mlp.PackedMLP(SimpleNamespace(gate_proj=nn.Linear(h, i, bias=False),
                                    up_proj=nn.Linear(h, i, bias=False), down_proj=nn.Linear(i, h, bias=False),
                                    act_fn=nn.SiLU()))
        self.nq, self.nk = nq, nk

    def forward(self, x, position_embeddings, past_key_value, cache_position, attention_mask, **kwargs):
        a = self.self_attn
        n = self.input_layernorm(x)
        b, t = x.shape[:2]
        q, k, v = [proj(n).view(b, t, heads, 128).transpose(1, 2)
                   for proj, heads in ((a.q_proj, self.nq), (a.k_proj, self.nk), (a.v_proj, self.nk))]
        c, s = position_embeddings
        q, k = rotate(a.q_norm(q), c, s), rotate(a.k_norm(k), c, s)
        k, v = past_key_value.update(k, v, self.index, {"cache_position": cache_position})
        attended = F.scaled_dot_product_attention(q, k.repeat_interleave(self.nq // self.nk, 1),
                    v.repeat_interleave(self.nq // self.nk, 1), attn_mask=attention_mask,
                    is_causal=attention_mask is None and t > 1, scale=a.scaling)
        x = x + a.o_proj(attended.transpose(1, 2).reshape(b, t, -1))
        return (x + self.mlp(self.post_attention_layernorm(x)),)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_act="silu", hidden_size=64, intermediate_size=96,
            num_attention_heads=4, num_key_value_heads=2, head_dim=128, _attn_implementation="starter_decode_gqa")
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(127, 64)
        self.model.layers = nn.ModuleList([ToyLayer(self.config) for _ in range(3)])
        for index, layer in enumerate(self.model.layers):
            layer.index = index
        self.model.norm, self.model.rotary_emb = Norm(64), Rotary()
        self.lm_head = nn.Linear(64, 127, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.eval().bfloat16()

    @property
    def device(self):
        return self.lm_head.weight.device

    @property
    def dtype(self):
        return self.lm_head.weight.dtype


class Cache:
    def __init__(self, batch, capacity):
        self.prefilling = True
        self.key_cache = [torch.full((batch, 2, capacity, 128), 33., dtype=torch.bfloat16) for _ in range(3)]
        self.value_cache = [torch.full_like(t, -41.) for t in self.key_cache]

    def update(self, k, v, layer, kwargs):
        self.key_cache[layer].index_copy_(2, kwargs["cache_position"], k)
        self.value_cache[layer].index_copy_(2, kwargs["cache_position"], v)
        return (k, v) if self.prefilling else (self.key_cache[layer], self.value_cache[layer])


class CPUContext:
    def __init__(self, capacity, position):
        self.mask = (torch.arange(capacity) <= position).view(1, 1, 1, -1)

    def attention(self, q, k, v, mask, scale):
        assert mask is self.mask
        return F.scaled_dot_product_attention(q, k.repeat_interleave(q.shape[1] // k.shape[1], 1),
                v.repeat_interleave(q.shape[1] // v.shape[1], 1), attn_mask=mask, scale=scale).transpose(1, 2)


class CompactCPU(unittest.TestCase):
    @torch.inference_mode()
    def test_pack_preserves_parameters_values_storage_and_prefill_projections(self):
        torch.manual_seed(4)
        model = Toy()
        params = dict(model.named_parameters())
        state = {k: v.clone() for k, v in model.state_dict().items()}
        plan = compact.CompactDecode(model)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        self.assertEqual(set(state), set(model.state_dict()))
        for name, p in model.named_parameters():
            self.assertIs(p, params[name])
            torch.testing.assert_close(p, state[name], rtol=0, atol=0)
        for index, layer in enumerate(model.model.layers):
            packed = plan.packed[index]
            offset = 0
            for name in ("q_proj", "k_proj", "v_proj"):
                p = getattr(layer.self_attn, name).weight
                self.assertEqual(p.untyped_storage().data_ptr(), packed.untyped_storage().data_ptr())
                self.assertEqual(p.storage_offset(), offset)
                self.assertTrue(p.is_contiguous())
                offset += p.numel()
                for tokens in (1, 7):
                    x = torch.randn(3, tokens, 64).bfloat16()
                    old = state[f"model.layers.{index}.self_attn.{name}.weight"]
                    torch.testing.assert_close(F.linear(x, p), F.linear(x, old), rtol=0, atol=0)
            self.assertEqual(offset * 2, packed.untyped_storage().nbytes())

    def test_load_time_relayout_outside_inference_mode(self):
        model = Toy()
        self.assertTrue(torch.is_grad_enabled())
        old = dict(model.named_parameters())
        flags = {name: p.requires_grad for name, p in old.items()}
        plan = compact.CompactDecode(model)
        for name, p in model.named_parameters():
            self.assertIs(p, old[name])
            self.assertEqual(p.requires_grad, flags[name])
        self.assertTrue(all(not t.requires_grad for t in plan.packed))

    def test_fixed_dispatch_rejects_unsupported_layout_without_cpu_install(self):
        model = Toy()
        self.assertTrue(compact.supported_layout(model))
        compact.install_compact_decode(model)
        self.assertFalse(hasattr(model, "compact_decode"))
        model.config.hidden_act = "relu"
        self.assertFalse(compact.supported_layout(model))
        model.config.hidden_act = "silu"; model.train()
        self.assertFalse(compact.supported_layout(model))
        model.eval().float()
        self.assertFalse(compact.supported_layout(model))

    @torch.inference_mode()
    def test_rejects_wrong_prefix_before_any_cache_write(self):
        model = Toy()
        plan = compact.CompactDecode(model)
        cache = Cache(1, 5)
        context = CPUContext(5, 1)
        ids, position = torch.ones(1, 1, dtype=torch.int64), torch.tensor([1])
        blocked = mock.Mock(side_effect=AssertionError("must reject before kernel use"))
        kernels = SimpleNamespace(add_rmsnorm=blocked, qkv_rope_cache=blocked, swiglu=blocked)
        with mock.patch.dict(sys.modules, {"kernels.compact_decode": kernels}):
            with self.assertRaises(ValueError):
                plan.forward(ids, cache, position, context.mask, context)  # still prefill
            cache.prefilling = False
            with self.assertRaises(ValueError):
                plan.forward(ids, cache, position, context.mask.clone(), context)
            with self.assertRaises(ValueError):
                plan.forward(ids.expand(-1, 2), cache, position, context.mask, context)
        blocked.assert_not_called()

    @torch.inference_mode()
    def test_kernel_bodies_residual_swiglu_and_rounding_negative_controls(self):
        torch.manual_seed(9)
        kernel = KernelCPU()
        wrong_residual, wrong_silu = 0, 0
        for batch, h in ((1, 1), (3, 37), (4, 2560), (16, 2560), (2, 8192)):
            r, x = [torch.randn(batch, 1, h).bfloat16() for _ in range(2)]
            w = torch.randn(h).bfloat16()
            before = [t.clone() for t in (r, x, w)]
            summed, result = kernel.add_rmsnorm(r, x, w, 1e-6)
            torch.testing.assert_close(summed, r + x, rtol=0, atol=0)
            torch.testing.assert_close(result, norm(r + x, w, 1e-6), rtol=0, atol=0)
            wrong = norm(r.float() + x.float(), w.float(), 1e-6).bfloat16()
            wrong_residual += int(torch.count_nonzero(wrong != result))
            for t, old in zip((r, x, w), before):
                torch.testing.assert_close(t, old, rtol=0, atol=0)
        for batch, width in ((1, 1), (3, 37), (4, 9728), (16, 9728)):
            p = torch.randn(batch, 1, 2 * width).bfloat16()
            before = p.clone()
            gate, up = p.chunk(2, -1)
            expected = (gate.float() / (1 + torch.exp(-gate.float()))).bfloat16() * up
            actual = kernel.swiglu(p)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            wrong = (gate.float() / (1 + torch.exp(-gate.float())) * up.float()).bfloat16()
            wrong_silu += int(torch.count_nonzero(wrong != actual))
            torch.testing.assert_close(p, before, rtol=0, atol=0)
        self.assertGreater(wrong_residual, 0)
        self.assertGreater(wrong_silu, 0)

    @torch.inference_mode()
    def test_kernel_body_qkv_cache_all_heads_positions_and_exact_write_coverage(self):
        torch.manual_seed(11)
        kernel = KernelCPU()
        for batch, nq, nk, cap in ((1, 32, 8, 1), (3, 4, 2, 129), (4, 32, 8, 257), (16, 32, 8, 641)):
            for position in sorted(set((0, cap // 2, cap - 1))):
                for shared in (True, False):
                    p = torch.randn(batch, 1, (nq + 2 * nk) * 128).bfloat16()
                    qw, kw = torch.randn(128).bfloat16(), torch.randn(128).bfloat16()
                    c, s = angles(1 if shared else batch, position)
                    k = torch.full((batch, nk, cap, 128), float("nan"), dtype=torch.bfloat16)
                    v = torch.full_like(k, -41.)
                    old_p, old_k, old_v = p.clone(), k.clone(), v.clone()
                    q, key, val = p.split((nq * 128, nk * 128, nk * 128), -1)
                    q = q.view(batch, nq, 1, 128)
                    key = key.view(batch, nk, 1, 128)
                    val = val.view(batch, nk, 1, 128)
                    expected_q = rotate(norm(q, qw, 1e-6), c, s)
                    expected_k = rotate(norm(key, kw, 3e-6), c, s)
                    actual = kernel.qkv_rope_cache(p, qw, kw, c, s, k, v, torch.tensor([position]), nq, nk, 1e-6, 3e-6)
                    torch.testing.assert_close(actual, expected_q, rtol=0, atol=0)
                    old_k[:, :, position:position+1] = expected_k
                    old_v[:, :, position:position+1] = val
                    torch.testing.assert_close(k, old_k, rtol=0, atol=0, equal_nan=True)
                    torch.testing.assert_close(v, old_v, rtol=0, atol=0)
                    torch.testing.assert_close(p, old_p, rtol=0, atol=0)

    @torch.inference_mode()
    def test_actual_pipeline_vs_materialized_torch_and_fresh_prompts(self):
        torch.manual_seed(15)
        forward, kernels = actual_forward(), KernelCPU()
        native = Toy(); candidate = deepcopy(native)
        candidate.compact_decode = compact.CompactDecode(candidate)
        for batch, length, count in ((1, 1, 3), (3, 7, 5), (4, 13, 2)):
            cache = Cache(batch, length + count)
            reference_cache = Cache(batch, length + count)
            for attempt in range(2):
                prompt = torch.randint(0, 127, (batch, length))
                for c in (cache, reference_cache): c.prefilling = True
                with mock.patch.object(candidate.compact_decode, "forward", side_effect=AssertionError("prefill bypass")):
                    actual = forward(candidate, prompt, cache, torch.arange(length))
                expected = forward(native, prompt, reference_cache, torch.arange(length))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for c in (cache, reference_cache): c.prefilling = False
                pointers = [t.data_ptr() for t in cache.key_cache + cache.value_cache]
                for step in range(count - 1):
                    ids = actual[:, -1].argmax(-1, keepdim=True)
                    pos = torch.tensor([length + step])
                    context = CPUContext(length + count, int(pos))
                    # Only the kernel launches and native Flash execution are emulated.
                    # The actual new pipeline and its dispatch run unchanged.
                    with mock.patch.dict(sys.modules, {"kernels.compact_decode": kernels}):
                        actual = forward(candidate, ids, cache, pos, context.mask, context)
                    expected = forward(native, ids, reference_cache, pos, context.mask)
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)
                    for a, e in zip(cache.key_cache + cache.value_cache,
                                    reference_cache.key_cache + reference_cache.value_cache):
                        torch.testing.assert_close(a, e, rtol=0.025, atol=0.025)
                self.assertEqual(pointers, [t.data_ptr() for t in cache.key_cache + cache.value_cache])


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
                     "CUDA/Triton required; a skip is not GPU validation")
class CompactCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ENGINE))
        from kernels import compact_decode as kernel
        from kernels.qk_norm_rope import qk_norm_rope
        cls.kernel, cls.old_qk = kernel, staticmethod(qk_norm_rope)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ENGINE))

    @torch.inference_mode()
    def test_cuda_arithmetic_current_shapes_and_cache_mutation(self):
        torch.manual_seed(17)
        for batch in (1, 3, 4, 16, 33):
            r = torch.randn(batch, 1, 2560, device="cuda", dtype=torch.bfloat16)
            x, w = torch.randn_like(r), torch.randn(2560, device="cuda", dtype=torch.bfloat16)
            summed, y = self.kernel.add_rmsnorm(r, x, w, 1e-6)
            torch.testing.assert_close(summed, r + x, rtol=0, atol=0)
            torch.testing.assert_close(y, norm(r + x, w, 1e-6), rtol=0.015, atol=0.02)
            p = torch.randn(batch, 1, 19456, device="cuda", dtype=torch.bfloat16)
            g, u = p.chunk(2, -1)
            torch.testing.assert_close(self.kernel.swiglu(p), F.silu(g) * u, rtol=0.008, atol=1e-5)
            for position in (0, 127, 640):
                p = torch.randn(batch, 1, 6144, device="cuda", dtype=torch.bfloat16)
                c, s = angles(1, position, "cuda")
                qw, kw = [torch.randn(128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
                k = torch.full((batch, 8, 641, 128), 33., device="cuda", dtype=torch.bfloat16)
                v = torch.full_like(k, -41.)
                oq, ok, ov = p.split((4096, 1024, 1024), -1)
                eq, ek = self.old_qk(oq.view(batch, 32, 1, 128), ok.view(batch, 8, 1, 128),
                                     qw, kw, c, s, 1e-6, 1e-6)
                q = self.kernel.qkv_rope_cache(p, qw, kw, c, s, k, v,
                        torch.tensor([position], device="cuda"), 32, 8, 1e-6, 1e-6)
                torch.testing.assert_close(q, eq, rtol=0, atol=0)
                torch.testing.assert_close(k[:, :, position:position+1], ek, rtol=0, atol=0)
                torch.testing.assert_close(v[:, :, position:position+1], ov.view(batch, 8, 1, 128), rtol=0, atol=0)
                k[:, :, position] = 33.; v[:, :, position] = -41.
                self.assertTrue(bool((k == 33).all()) and bool((v == -41).all()))

    @torch.inference_mode()
    def test_cuda_graph_replay_changes_values_and_positions(self):
        p = torch.randn(3, 1, 6144, device="cuda", dtype=torch.bfloat16)
        qw, kw = [torch.randn(128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        c, s = angles(1, 0, "cuda")
        k = torch.zeros(3, 8, 641, 128, device="cuda", dtype=torch.bfloat16)
        v, pos = torch.zeros_like(k), torch.tensor([0], device="cuda")
        def call():
            return self.kernel.qkv_rope_cache(p, qw, kw, c, s, k, v, pos, 32, 8, 1e-6, 1e-6)
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream): q = call()
        pointers = [t.data_ptr() for t in (q, k, v)]
        for position in (1, 127, 128, 640, 6):
            p.normal_(); k.fill_(33.); v.fill_(-41.); pos.fill_(position)
            nc, ns = angles(1, position, "cuda"); c.copy_(nc); s.copy_(ns)
            graph.replay()
            oldq, oldk, oldv = p.split((4096, 1024, 1024), -1)
            eq, ek = self.old_qk(oldq.view(3, 32, 1, 128), oldk.view(3, 8, 1, 128), qw, kw, c, s, 1e-6, 1e-6)
            torch.testing.assert_close(q, eq, rtol=0, atol=0)
            torch.testing.assert_close(k[:, :, position:position+1], ek, rtol=0, atol=0)
            torch.testing.assert_close(v[:, :, position:position+1], oldv.view(3, 8, 1, 128), rtol=0, atol=0)
            self.assertEqual(pointers, [t.data_ptr() for t in (q, k, v)])

    @torch.inference_mode()
    def test_engine_installs_path_prefill_bypasses_and_stream_own_prefix(self):
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("Transformers required")
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from engine import Engine
        cfg = Qwen3Config(vocab_size=127, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=128,
            max_position_embeddings=32768, rope_theta=5_000_000, tie_word_embeddings=True,
            sliding_window=None, eos_token_id=0)
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(25)
        native = Qwen3ForCausalLM(cfg).eval()
        with tempfile.TemporaryDirectory() as path:
            native.save_pretrained(path)
            engine = Engine(path)
            self.assertIsNotNone(getattr(engine.model, "compact_decode", None))
            native = native.cuda().bfloat16()
            for batch, length, count in ((3, 7, 5), (3, 7, 5), (1, 1, 3), (2, 129, 9), (2, 13, 1), (2, 13, 0)):
                prompt = torch.randint(0, 127, (batch, length), device="cuda")
                emitted = list(engine.generate(prompt.tolist(), count))
                self.assertEqual(len(emitted), count)
                self.assertTrue(all(len(row) == batch and all(type(t) is int for t in row) for row in emitted))
                if count:
                    tokens = torch.tensor(emitted, device="cuda").T
                    own_prefix = torch.cat((prompt, tokens[:, :-1]), 1)
                    logits = native(input_ids=own_prefix, use_cache=False).logits[:, length-1:].float()
                    gaps = logits.amax(-1) - logits.gather(-1, tokens[..., None]).squeeze(-1)
                    self.assertTrue(bool(torch.isfinite(gaps).all()))
                    self.assertLessEqual(float(gaps.max()), 2.)
                    plan = engine.model.compact_decode
                    with mock.patch.object(plan, "forward", side_effect=AssertionError("prefill bypass")):
                        engine.decode_state.prefill(prompt)
                    if count > 1:
                        with mock.patch.object(plan, "forward", wraps=plan.forward) as used:
                            engine.decode_state.step()
                        self.assertEqual(used.call_count, 1)
            engine.model.lm_head.weight.zero_()
            self.assertEqual(list(engine.generate([[1, 2, 3]], 5)), [[0]] * 5)


if __name__ == "__main__":
    unittest.main()
