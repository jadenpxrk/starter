"""Prefill checks. CPU kernel-body execution is NOT Triton or CUDA validation."""

import ast
from contextlib import contextmanager
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

ENGINE = Path(__file__).resolve().parents[1] / "engine"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prefill = load("prefill_under_test", ENGINE / "prefill.py")
mlp = load("prefill_test_mlp", ENGINE / "mlp.py")


def norm(x, weight, eps):
    f = x.float()
    return (f * torch.rsqrt(f.square().sum(-1, keepdim=True) / f.shape[-1] + eps)).to(x.dtype) * weight


def add_norm(x, branch, weight, eps):
    total = x + branch
    return total, norm(total, weight, eps)


def silu_mul(p):
    gate, up = p.chunk(2, -1)
    return F.silu(gate) * up


def angles(capacity, device="cpu"):
    inv = 1 / (5_000_000 ** (torch.arange(0, 128, 2, device=device).float() / 128))
    f = torch.arange(capacity, device=device).float()[:, None] * inv
    f = torch.cat((f, f), -1)
    return f.cos().bfloat16(), f.sin().bfloat16()


def reference_qkv(p, qw, kw, qe, ke, cos, sin, keys, values, length):
    b, nk, _, d = keys.shape
    nq = p.shape[-1] // d - 2 * nk
    q, k, v = p.reshape(b, length, -1).split((nq*d, nk*d, nk*d), -1)
    q = norm(q.reshape(b, length, nq, d), qw, qe)
    k = norm(k.reshape(b, length, nk, d), kw, ke)
    def rotate(x):
        half = torch.cat((-x[..., d//2:], x[..., :d//2]), -1)
        return x * cos[None, :length, None, :] + half * sin[None, :length, None, :]
    q, k = rotate(q), rotate(k)
    keys[:, :, :length].copy_(k.transpose(1, 2))
    values[:, :, :length].copy_(v.reshape(b, length, nk, d).transpose(1, 2))
    return q.contiguous()


def reference_attention(q, keys, values, length, scale):
    groups = q.shape[2] // keys.shape[1]
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), keys[:, :, :length].repeat_interleave(groups, 1),
        values[:, :, :length].repeat_interleave(groups, 1), is_causal=True, scale=scale,
    ).transpose(1, 2).contiguous()


class Pointer:
    def __init__(self, data, offsets=0, writes=None):
        self.data, self.offsets = data.reshape(-1), offsets
        self.writes = torch.zeros(self.data.numel(), dtype=torch.int64) if writes is None else writes
    def __add__(self, value):
        return Pointer(self.data, self.offsets + value, self.writes)


class Ops:
    """Checked CPU pointer/math shim; no GPU reduction/codegen/warps emulation."""
    int64, float32, bfloat16 = torch.int64, torch.float32, torch.bfloat16
    arange, rsqrt = staticmethod(torch.arange), staticmethod(torch.rsqrt)
    program = 0
    def program_id(self, axis):
        assert axis == 0
        return torch.tensor(self.program, dtype=torch.int32)
    @staticmethod
    def sum(x, axis):
        return x.sum(dim=axis)
    @staticmethod
    def where(test, a, b):
        return torch.where(torch.as_tensor(test), torch.as_tensor(a), torch.as_tensor(b))
    @staticmethod
    def indices(p):
        i = torch.as_tensor(p.offsets, dtype=torch.int64)
        if bool(((i < 0) | (i >= p.data.numel())).any()):
            raise AssertionError("out-of-bounds pointer access")
        return i
    def load(self, p):
        return p.data[self.indices(p)]
    def store(self, p, value):
        i = self.indices(p)
        p.data[i] = value.to(p.data.dtype)
        p.writes.scatter_add_(0, i.flatten(), torch.ones_like(i.flatten()))


class KernelBody:
    def __init__(self, omit_rotary_rounding=False):
        self.ops = Ops()
        path = ENGINE / "kernels/prefill.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_prefill_qkv")
        node.decorator_list = []
        if omit_rotary_rounding:
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id in ("left", "right")):
                    # Remove only the product's BF16->FP32 round trip.
                    stmt.value = stmt.value.func.value.func.value
        for arg in node.args.args:
            arg.annotation = None
        env = {"tl": self.ops}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), env)
        self.kernel = env[node.name]
    def run(self, p, qw, kw, qe, ke, cos, sin, keys, values, length):
        b, nk, c, d = keys.shape
        nq = p.shape[1] // d - 2 * nk
        q = torch.empty(b, length, nq, d, dtype=p.dtype)
        qp, kp, vp = Pointer(q), Pointer(keys), Pointer(values)
        args = [Pointer(t) for t in (p, qw, kw, cos, sin)]
        for program in range(b * length * (nq + nk)):
            self.ops.program = program
            self.kernel(*args, qp, kp, vp, length, c, nq, nk, d, qe, ke)
        expected = torch.zeros_like(keys, dtype=torch.int64)
        expected[:, :, :length] = 1
        assert torch.equal(kp.writes.view_as(expected), expected)
        assert torch.equal(vp.writes.view_as(expected), expected)
        assert bool((qp.writes == 1).all())
        return q


@contextmanager
def cpu_kernels(qkv):
    # Patch only imports made INSIDE prefill_forward; do not leak fake Triton modules.
    with mock.patch.dict(sys.modules, {
        "kernels.prefill": SimpleNamespace(prefill_qkv=qkv),
        "kernels.decode_fused": SimpleNamespace(add_rms_norm=add_norm, silu_mul=silu_mul),
    }), mock.patch.object(prefill, "_native_attention", side_effect=reference_attention):
        yield


class Norm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = 1e-6
    def forward(self, x):
        return norm(x, self.weight, self.variance_epsilon)


def fixture(batch, length, extra=5):
    """Independent Torch equations fixture; not a Transformers execution."""
    h, nq, nk, d, intermediate = 64, 4, 2, 128, 96
    model = nn.Module()
    model.config = SimpleNamespace(hidden_act="silu", hidden_size=h, num_attention_heads=nq,
        num_key_value_heads=nk, head_dim=d, _attn_implementation="starter_decode_gqa")
    base = model.model = nn.Module()
    base.embed_tokens = nn.Embedding(127, h)
    base.norm = Norm(h)
    base.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.input_layernorm, layer.post_attention_layernorm = Norm(h), Norm(h)
        a = layer.self_attn = nn.Module()
        a.q_proj, a.k_proj, a.v_proj = [nn.Linear(h, n*d, bias=False) for n in (nq, nk, nk)]
        a.o_proj = nn.Linear(nq*d, h, bias=False)
        a.q_norm, a.k_norm, a.scaling, a.sliding_window = Norm(d), Norm(d), d**-0.5, None
        layer.mlp = mlp.PackedMLP(SimpleNamespace(gate_proj=nn.Linear(h, intermediate, bias=False),
            up_proj=nn.Linear(h, intermediate, bias=False), down_proj=nn.Linear(intermediate, h, bias=False),
            act_fn=nn.SiLU()))
        base.layers.append(layer)
    model.lm_head = nn.Linear(h, 127, bias=False)
    model.lm_head.weight = base.embed_tokens.weight
    model.eval().bfloat16()
    for layer in base.layers:
        a = layer.self_attn
        a.qkv_weight = torch.cat([p.weight for p in (a.q_proj, a.k_proj, a.v_proj)])
    model.device, model.dtype = torch.device("cpu"), torch.bfloat16
    c = length + extra
    cache = SimpleNamespace(max_cache_len=c, prefilling=True,
        key_cache=[torch.full((batch, nk, c, d), 33., dtype=torch.bfloat16) for _ in base.layers],
        value_cache=[torch.full((batch, nk, c, d), -41., dtype=torch.bfloat16) for _ in base.layers])
    return model, cache, angles(c)


def materialized(model, cache, ids, cos, sin):
    b, t = ids.shape
    x = model.model.embed_tokens(ids)
    for i, layer in enumerate(model.model.layers):
        a = layer.self_attn
        n = layer.input_layernorm(x)
        p = torch.cat([proj(n) for proj in (a.q_proj, a.k_proj, a.v_proj)], -1).reshape(b*t, -1)
        q = reference_qkv(p, a.q_norm.weight, a.k_norm.weight, a.q_norm.variance_epsilon,
            a.k_norm.variance_epsilon, cos, sin, cache.key_cache[i], cache.value_cache[i], t)
        out = reference_attention(q, cache.key_cache[i], cache.value_cache[i], t, a.scaling)
        x = x + a.o_proj(out.reshape(b, t, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return model.lm_head(model.model.norm(x[:, -1:, :]))


class PrefillCPU(unittest.TestCase):
    @torch.inference_mode()
    def test_kernel_body_values_indexing_and_all_prefix_writes(self):
        torch.manual_seed(41)
        body = KernelBody()
        cases = ((1,1,3,4,2), (2,7,13,4,2), (1,33,39,32,8), (3,3,11,6,2), (1,129,137,4,2))
        for b,t,c,nq,nk in cases:
            with self.subTest(shape=(b,t,c,nq,nk)):
                p = torch.randn(b*t, (nq+2*nk)*128).bfloat16()
                qw, kw = torch.randn(128).bfloat16(), torch.randn(128).bfloat16()
                cos, sin = angles(c)
                for poison in (float("nan"), -41.):
                    k = torch.full((b,nk,c,128), poison, dtype=torch.bfloat16); v = k.clone()
                    rk, rv = k.clone(), v.clone()
                    immutable = [z.clone() for z in (p,qw,kw,cos,sin)]
                    actual = body.run(p,qw,kw,1e-6,3e-6,cos,sin,k,v,t)
                    ref = reference_qkv(p,qw,kw,1e-6,3e-6,cos,sin,rk,rv,t)
                    for a,e in ((actual,ref),(k,rk),(v,rv)):
                        torch.testing.assert_close(a,e,rtol=0,atol=0,equal_nan=True)
                    for a,e in zip((p,qw,kw,cos,sin),immutable):
                        torch.testing.assert_close(a,e,rtol=0,atol=0)

    @torch.inference_mode()
    def test_rotary_product_rounding_negative_control(self):
        torch.manual_seed(48)
        b,t,c,nq,nk = 2,17,25,4,2
        p=torch.randn(b*t,(nq+2*nk)*128).bfloat16()
        qw,kw=torch.randn(128).bfloat16(),torch.randn(128).bfloat16()
        cos,sin=angles(c)
        k=torch.zeros(b,nk,c,128,dtype=torch.bfloat16); v=k.clone()
        correct=KernelBody().run(p,qw,kw,1e-6,1e-6,cos,sin,k,v,t)
        wrong=KernelBody(True).run(p,qw,kw,1e-6,1e-6,cos,sin,k.clone(),v.clone(),t)
        self.assertGreater(int(torch.count_nonzero(correct!=wrong)),0)

    @torch.inference_mode()
    def test_factory_supported_metadata_and_fallback_guards(self):
        # Synthetic CUDA metadata; no allocations or device values are inspected.
        b,t,c,nq,nk,h=3,7,13,4,2,64
        def tensor(shape):
            return SimpleNamespace(shape=shape,device=torch.device('cuda'),dtype=torch.bfloat16,
                                   is_contiguous=lambda:True)
        def norm_meta(width): return SimpleNamespace(weight=tensor((width,)))
        cfg=SimpleNamespace(num_attention_heads=nq,num_key_value_heads=nk,head_dim=128,
            hidden_size=h,hidden_act='silu',_attn_implementation='starter_decode_gqa',rope_scaling=None)
        a=SimpleNamespace(training=False,sliding_window=None,qkv_weight=tensor(((nq+2*nk)*128,h)),
            q_norm=norm_meta(128),k_norm=norm_meta(128),
            q_proj=SimpleNamespace(bias=None),k_proj=SimpleNamespace(bias=None),
            v_proj=SimpleNamespace(bias=None),o_proj=SimpleNamespace(bias=None,weight=tensor((h,nq*128))))
        m=SimpleNamespace(gate_up_weight=tensor((192,h)),down_proj=SimpleNamespace(bias=None,weight=tensor((h,96))))
        layer=SimpleNamespace(training=False,self_attn=a,mlp=m,input_layernorm=norm_meta(h),post_attention_layernorm=norm_meta(h))
        base=SimpleNamespace(layers=[layer],embed_tokens=norm_meta(h),norm=norm_meta(h))
        model=SimpleNamespace(config=cfg,model=base,lm_head=norm_meta(h),training=False,
            device=torch.device('cuda'),dtype=torch.bfloat16)
        cache=SimpleNamespace(max_cache_len=c,key_cache=[tensor((b,nk,c,128))],value_cache=[tensor((b,nk,c,128))])
        cos,sin=tensor((c,128)),tensor((c,128)); marker=object()
        with mock.patch.object(prefill,'PrefillPlan',return_value=marker) as ctor:
            self.assertIs(prefill.make_prefill_plan(model,cache,b,t,cos,sin),marker)
            ctor.assert_called_once_with(model,cache,b,t,cos,sin)
            self.assertIsNone(prefill.make_prefill_plan(model,cache,b,1,cos,sin))
            cfg.rope_scaling={'rope_type':'dynamic'}
            self.assertIsNone(prefill.make_prefill_plan(model,cache,b,t,cos,sin)); cfg.rope_scaling=None
            a.o_proj.bias=torch.zeros(h)
            self.assertIsNone(prefill.make_prefill_plan(model,cache,b,t,cos,sin)); a.o_proj.bias=None
            model.training=True
            self.assertIsNone(prefill.make_prefill_plan(model,cache,b,t,cos,sin)); model.training=False
            cache.key_cache[0].dtype=torch.float32
            self.assertIsNone(prefill.make_prefill_plan(model,cache,b,t,cos,sin))
            self.assertEqual(ctor.call_count,1)

    @torch.inference_mode()
    def test_native_call_schema_views_causality_and_stale_capacity(self):
        torch.manual_seed(42)
        b,t,c,nq,nk = 2,7,13,6,2
        q = torch.randn(b,t,nq,128).bfloat16()
        k = torch.randn(b,nk,c,128).bfloat16(); v = torch.randn_like(k)
        k[:,:,t:] = float("nan"); v[:,:,t:] = float("nan")
        def fake(qv,kv,vv,cq,ck,mq,mk,drop,causal,debug,**kw):
            self.assertIsNone(cq); self.assertIsNone(ck)
            self.assertEqual((mq,mk,drop,causal,debug),(t,t,0.0,True,False))
            self.assertEqual(kv.shape,(b,t,nk,128))
            self.assertEqual(kv.data_ptr(),k.data_ptr()); self.assertEqual(vv.data_ptr(),v.data_ptr())
            self.assertEqual(kv.stride(),(nk*c*128,128,c*128,1))
            self.assertIsNone(kw['seqused_k']); self.assertIsNone(kw['window_size_right'])
            return (reference_attention(qv,kv.transpose(1,2),vv.transpose(1,2),t,kw['scale']),)*5
        with mock.patch.object(torch.ops.aten._flash_attention_forward,"default",side_effect=fake) as call:
            out = prefill._native_attention(q,k,v,t,128**-0.5)
        self.assertEqual(call.call_count,1)
        ref = reference_attention(q,k,v,t,128**-0.5)
        torch.testing.assert_close(out,ref,rtol=0,atol=0)
        # Mutating future prompt positions must not affect outputs for earlier queries.
        kk,vv = k.clone(),v.clone(); kk[:,:,4:t] += 10; vv[:,:,4:t] += 10
        changed = reference_attention(q,kk,vv,t,128**-0.5)
        torch.testing.assert_close(out[:,:4],changed[:,:4],rtol=0,atol=0)
        # Negative controls: no causality or including stale capacity must differ/fail.
        wrong = F.scaled_dot_product_attention(q.transpose(1,2),k[:,:,:t].repeat_interleave(3,1),
            v[:,:,:t].repeat_interleave(3,1),is_causal=False).transpose(1,2)
        self.assertFalse(torch.equal(out,wrong))
        stale = F.scaled_dot_product_attention(q.transpose(1,2),k.repeat_interleave(3,1),
            v.repeat_interleave(3,1),is_causal=True)
        self.assertFalse(bool(torch.isfinite(stale).all()))

    @torch.inference_mode()
    def test_actual_pipeline_fresh_prompts_matches_materialized_equations(self):
        torch.manual_seed(43)
        body = KernelBody()
        for b,t in ((1,1),(2,7),(3,11)):
            model,cache,(c,s) = fixture(b,t)
            pointers = [v.data_ptr() for v in cache.key_cache+cache.value_cache]
            weights = {k:v.clone() for k,v in model.state_dict().items()}
            for attempt in range(2):
                for v in cache.key_cache+cache.value_cache: v.fill_(33 if attempt==0 else -41)
                refcache = deepcopy(cache)
                ids = torch.randint(0,127,(b,t))
                with cpu_kernels(body.run):
                    actual = prefill.prefill_forward(model,cache,ids,c,s)
                expected = materialized(model,refcache,ids,c,s)
                torch.testing.assert_close(actual,expected,rtol=0.02,atol=0.02)
                for a,e in zip(cache.key_cache+cache.value_cache,refcache.key_cache+refcache.value_cache):
                    torch.testing.assert_close(a,e,rtol=0.02,atol=0.02)
                self.assertEqual(pointers,[v.data_ptr() for v in cache.key_cache+cache.value_cache])
                for k,v in model.state_dict().items(): torch.testing.assert_close(v,weights[k],rtol=0,atol=0)
            self.assertIs(model.lm_head.weight,model.model.embed_tokens.weight)
            self.assertIsNone(prefill.make_prefill_plan(model,cache,b,t,c,s))

    @torch.inference_mode()
    def test_graph_lifecycle_fresh_inputs_and_capture_requires_replay(self):
        # Synthetic scheduling test: this validates Python control flow, NOT CUDA.
        model = SimpleNamespace(device=torch.device('cpu'))
        plan = prefill.PrefillPlan(model,object(),2,7,None,None)
        active = [None]; graphs = []; forwards = []
        class Stream:
            def __init__(self,**kw): pass
            def wait_stream(self,other): pass
        class Graph:
            def __init__(self): self.replays=0; graphs.append(self)
            def replay(self): self.replays+=1; self.output.copy_(plan.inputs.sum(1).view(2,1,1).float())
        @contextmanager
        def stream(s): yield
        @contextmanager
        def capture(g,**kw):
            self.assertNotIn('pool',kw)
            active[0]=g
            try: yield
            finally: active[0]=None
        def forward(*args):
            forwards.append(1)
            out = torch.empty(2,1,1)
            if active[0] is not None: active[0].output=out
            else: out.copy_(plan.inputs.sum(1).view(2,1,1).float())
            return out
        with mock.patch.object(torch.cuda,'current_stream',return_value=Stream()), \
             mock.patch.object(torch.cuda,'Stream',Stream), mock.patch.object(torch.cuda,'stream',stream), \
             mock.patch.object(torch.cuda,'CUDAGraph',Graph), mock.patch.object(torch.cuda,'graph',capture), \
             mock.patch.object(prefill,'prefill_forward',side_effect=forward):
            for n in (2,9):
                ids = torch.full((2,7),n,dtype=torch.int64)
                out = plan.run(ids)
                torch.testing.assert_close(out,torch.full((2,1,1),float(n*7)))
                self.assertNotEqual(plan.inputs.data_ptr(),ids.data_ptr())
        self.assertEqual(len(graphs),1); self.assertEqual(graphs[0].replays,2)
        self.assertEqual(len(forwards),3)  # two warmups, one capture; not repeated on next prompt
        with self.assertRaises(ValueError): plan.run(torch.ones(2,8,dtype=torch.int64))

    @torch.inference_mode()
    def test_decode_prefill_integration_and_exception_cleanup(self):
        path = ENGINE/'decode.py'; tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DecodeState')
        method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='prefill')
        old = mock.Mock(side_effect=AssertionError('native path must not run'))
        env = {'qwen_forward':old}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),env)
        logits = torch.tensor([[[1.,3.,2.]],[[4.,1.,0.]]])
        state = SimpleNamespace(cache=SimpleNamespace(prefilling=False),tokens=torch.zeros(2,1,dtype=torch.int64),
            position=torch.zeros(1,dtype=torch.int64),prefill_plan=SimpleNamespace(run=mock.Mock(return_value=logits)))
        inputs = torch.ones(2,7,dtype=torch.int64)
        actual = env['prefill'](state,inputs)
        self.assertIs(actual,logits); self.assertFalse(state.cache.prefilling)
        self.assertEqual(state.tokens.tolist(),[[1],[0]]); self.assertEqual(state.position.item(),7)
        state.prefill_plan.run.side_effect = RuntimeError('injected')
        with self.assertRaisesRegex(RuntimeError,'injected'): env['prefill'](state,inputs)
        self.assertFalse(state.cache.prefilling); old.assert_not_called()
        state.prefill_plan=None; state.model=object(); state.key_positions=torch.arange(9)
        old.side_effect=None; old.return_value=logits
        env['prefill'](state,inputs); self.assertEqual(old.call_count,1)


@unittest.skipUnless(importlib.util.find_spec('transformers') is not None,'Transformers not installed')
class NativeCPU(unittest.TestCase):
    @torch.inference_mode()
    def test_real_transformers_prefill_logits_and_cache(self):
        # Real Transformers when available, but fused ops/Flash still emulated on CPU.
        from transformers import Qwen3Config,Qwen3ForCausalLM,StaticCache
        torch.manual_seed(47)
        cfg=Qwen3Config(vocab_size=127,hidden_size=64,intermediate_size=96,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=128,max_position_embeddings=32768,
            rope_theta=5_000_000,tie_word_embeddings=True,sliding_window=None)
        cfg._attn_implementation='sdpa'
        native=Qwen3ForCausalLM(cfg).eval().bfloat16(); candidate=deepcopy(native)
        for layer in candidate.model.layers:
            a=layer.self_attn; a.qkv_weight=torch.cat([p.weight for p in (a.q_proj,a.k_proj,a.v_proj)])
            layer.mlp=mlp.PackedMLP(layer.mlp)
        body=KernelBody()
        for b,t in ((1,3),(2,7),(3,11)):
            cache=StaticCache(cfg,max_batch_size=b,max_cache_len=t+5,device='cpu',dtype=torch.bfloat16)
            c,s=(a[0] for a in native.model.rotary_emb(torch.empty(0,dtype=torch.bfloat16),torch.arange(t+5)[None]))
            for _ in range(2):
                for a in cache.key_cache+cache.value_cache: a.fill_(33)
                ids=torch.randint(0,127,(b,t))
                with cpu_kernels(body.run): got=prefill.prefill_forward(candidate,cache,ids,c,s)
                ref=native(input_ids=ids,use_cache=True,logits_to_keep=1)
                torch.testing.assert_close(got,ref.logits,rtol=0.03,atol=0.01)
                for k,e in zip(cache.key_cache,ref.past_key_values.key_cache):
                    torch.testing.assert_close(k[:,:,:t],e,rtol=0.03,atol=0.01)
                for v,e in zip(cache.value_cache,ref.past_key_values.value_cache):
                    torch.testing.assert_close(v[:,:,:t],e,rtol=0.03,atol=0.01)


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('triton') is not None,
                     'CUDA/Triton required; CPU is not GPU validation')
class PrefillCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(ENGINE))
        cls.kernel=load('prefill_cuda_kernel',ENGINE/'kernels/prefill.py')

    @torch.inference_mode()
    def test_kernel_and_native_causal_attention(self):
        torch.manual_seed(51)
        for b,t,c in ((1,3,11),(2,33,41),(1,129,137),(3,17,33)):
            p=torch.randn(b*t,6144,device='cuda',dtype=torch.bfloat16)
            qw=torch.randn(128,device='cuda',dtype=torch.bfloat16); kw=torch.randn_like(qw)
            cos,sin=angles(c,'cuda')
            k=torch.full((b,8,c,128),float('nan'),device='cuda',dtype=torch.bfloat16); v=k.clone()
            rk,rv=k.clone(),v.clone(); before=p.clone()
            q=self.kernel.prefill_qkv(p,qw,kw,1e-6,3e-6,cos,sin,k,v,t)
            rq=reference_qkv(p,qw,kw,1e-6,3e-6,cos,sin,rk,rv,t)
            torch.testing.assert_close(q,rq,rtol=0.02,atol=0.02)
            torch.testing.assert_close(k[:,:,:t],rk[:,:,:t],rtol=0.02,atol=0.02)
            torch.testing.assert_close(v[:,:,:t],rv[:,:,:t],rtol=0,atol=0)
            self.assertTrue(bool(torch.isnan(k[:,:,t:]).all() & torch.isnan(v[:,:,t:]).all()))
            out=prefill._native_attention(q,k,v,t,128**-0.5)
            expected=reference_attention(q,k,v,t,128**-0.5)
            torch.testing.assert_close(out,expected,rtol=0.03,atol=0.03)
            torch.testing.assert_close(p,before,rtol=0,atol=0)

    @torch.inference_mode()
    def test_kernel_graph_replay_updates_every_prompt_slot(self):
        b,t,c=2,17,25
        p=torch.randn(b*t,1024,device='cuda',dtype=torch.bfloat16)
        qw=torch.ones(128,device='cuda',dtype=torch.bfloat16); kw=qw.clone()
        cos,sin=angles(c,'cuda')
        k=torch.full((b,2,c,128),33.,device='cuda',dtype=torch.bfloat16); v=k.clone()
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2): self.kernel.prefill_qkv(p,qw,kw,1e-6,1e-6,cos,sin,k,v,t)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            q=self.kernel.prefill_qkv(p,qw,kw,1e-6,1e-6,cos,sin,k,v,t)
        addresses=[a.data_ptr() for a in (q,k,v)]
        for _ in range(3):
            p.normal_(); k.fill_(33); v.fill_(-41); graph.replay()
            rk,rv=k.clone(),v.clone()
            rq=reference_qkv(p,qw,kw,1e-6,1e-6,cos,sin,rk,rv,t)
            for a,e in ((q,rq),(k,rk),(v,rv)): torch.testing.assert_close(a,e,rtol=0.02,atol=0.02)
            self.assertEqual(addresses,[a.data_ptr() for a in (q,k,v)])

    @torch.inference_mode()
    def test_engine_both_graphs_fresh_prompts_and_own_prefix(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from engine import Engine
        torch.manual_seed(53)
        cfg=Qwen3Config(vocab_size=127,hidden_size=256,intermediate_size=512,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=128,max_position_embeddings=32768,
            rope_theta=5_000_000,tie_word_embeddings=True,sliding_window=None,eos_token_id=0)
        cfg._attn_implementation='sdpa'
        native=Qwen3ForCausalLM(cfg).eval()
        with tempfile.TemporaryDirectory() as path:
            native.save_pretrained(path); engine=Engine(path); native=native.cuda().bfloat16()
            for b,t,n in ((2,7,5),(2,7,5),(1,1,3),(3,33,1),(2,129,3),(2,7,0)):
                ids=torch.randint(0,127,(b,t),device='cuda')
                emitted=list(engine.generate(ids.tolist(),n))
                self.assertEqual(len(emitted),n)
                if n==0: continue
                self.assertTrue(all(len(row)==b and all(type(x) is int for x in row) for row in emitted))
                state=engine.decode_state
                self.assertEqual(state.position.item(),t+n-1)
                if t>1: self.assertIsNotNone(state.prefill_plan.graph)
                else: self.assertIsNone(state.prefill_plan)
                out=torch.tensor(emitted,device='cuda').T
                own=torch.cat((ids,out[:,:-1]),1)
                logits=native(input_ids=own,use_cache=False).logits[:,t-1:].float()
                gap=logits.amax(-1)-logits.gather(-1,out[...,None]).squeeze(-1)
                self.assertTrue(bool(torch.isfinite(gap).all())); self.assertLessEqual(float(gap.max()),2.0)
                # Stronger prefill-logit diagnostic than the wide tiny-model token margin.
                got=state.prefill(ids)
                ref=native(input_ids=ids,use_cache=False,logits_to_keep=1).logits
                torch.testing.assert_close(got,ref,rtol=0.03,atol=0.03)
            # Zero tied weights force EOS id 0; it must remain an ordinary token.
            engine.model.model.embed_tokens.weight.zero_()
            result=list(engine.generate([[1,2,3],[3,2,1]],4))
            self.assertEqual(result,[[0,0]]*4)


if __name__=='__main__':
    unittest.main()
