"""Full-vocabulary selection checks. CPU shims are NOT Triton/CUDA execution."""
import ast
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'engine'
sys.path.insert(0, str(ENGINE))
import greedy_head as head
import prefill


def bodies(path, names, env):
    tree = ast.parse(path.read_text())
    picked = []
    for node in tree.body:
        nodes = node.body if isinstance(node, ast.ClassDef) else [node]
        for fn in nodes:
            if isinstance(fn, ast.FunctionDef) and fn.name in names:
                fn.decorator_list = []
                for arg in fn.args.args + fn.args.kwonlyargs:
                    arg.annotation = None
                picked.append(fn)
    exec(compile(ast.fix_missing_locations(ast.Module(body=picked, type_ignores=[])), str(path), 'exec'), env)
    return env


class Pointer:
    def __init__(self, data, offsets=0, writes=None, reads=None):
        self.data, self.offsets = data.reshape(-1), offsets
        self.writes = torch.zeros(self.data.numel(), dtype=torch.int64) if writes is None else writes
        self.reads = torch.zeros(self.data.numel(), dtype=torch.int64) if reads is None else reads
    def __add__(self, offsets):
        return Pointer(self.data, self.offsets + offsets, self.writes, self.reads)


class Ops:
    float32, bfloat16, int64 = torch.float32, torch.bfloat16, torch.int64
    arange = staticmethod(torch.arange)
    @staticmethod
    def trans(x): return x.T
    @staticmethod
    def cdiv(a, b): return (a + b - 1) // b
    @staticmethod
    def zeros(shape, dtype): return torch.zeros(shape, dtype=dtype)
    @staticmethod
    def max(x, axis): return x.max(dim=axis).values
    @staticmethod
    def min(x, axis): return x.min(dim=axis).values
    @staticmethod
    def where(c, a, b): return torch.where(torch.as_tensor(c), torch.as_tensor(a), torch.as_tensor(b))
    @staticmethod
    def dot(a, b, acc, input_precision=None):
        assert a.dtype == b.dtype == torch.bfloat16 and acc.dtype == torch.float32
        assert input_precision == 'ieee'
        return acc + a.float() @ b.float()
    def program_id(self, axis): return self.pid
    @staticmethod
    def indices(p, mask):
        idx = torch.as_tensor(p.offsets, dtype=torch.int64)
        if mask is not None:
            idx, mask = torch.broadcast_tensors(idx, torch.as_tensor(mask))
            checked = idx[mask]
        else:
            checked = idx
        assert not bool(((checked < 0) | (checked >= p.data.numel())).any()), 'OOB access'
        return idx, mask
    def load(self, p, mask=None, other=0):
        idx, mask = self.indices(p, mask)
        enabled = idx if mask is None else idx[mask]
        p.reads.scatter_add_(0, enabled.reshape(-1), torch.ones(enabled.numel(), dtype=torch.int64))
        if mask is None: return p.data[idx]
        safe = torch.where(mask, idx, 0)
        return torch.where(mask, p.data[safe], torch.as_tensor(other, dtype=p.data.dtype))
    def store(self, p, value, mask=None):
        idx, mask = self.indices(p, mask)
        value = torch.broadcast_to(torch.as_tensor(value), idx.shape)
        if mask is not None: idx, value = idx[mask], value[mask]
        p.data[idx] = value.to(p.data.dtype)
        p.writes.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel(), dtype=torch.int64))


class KernelBody:
    def __init__(self):
        self.ops = Ops()
        env = bodies(ENGINE/'kernels/greedy_head.py', {'_project_max', '_merge_max'}, {'tl': self.ops})
        self.project, self.merge = env['_project_max'], env['_merge_max']
    def run(self, x, w):
        m, k = x.shape; n = w.shape[0]; tiles = (n + 127)//128
        scores = torch.full((m, tiles), float('nan'))
        indices = torch.full((m, tiles), -1, dtype=torch.int32)
        out = torch.full((m, 1), -1, dtype=torch.int64)
        xp, wp, sp, ip, op = map(Pointer, (x, w, scores, indices, out))
        for tile in range(tiles):
            self.ops.pid = tile
            self.project(xp, wp, sp, ip, m, n, k, tiles, max(16, 1<<(m-1).bit_length()), 128, 64)
        for row in range(m):
            self.ops.pid = row
            self.merge(sp, ip, op, tiles, 1<<(tiles-1).bit_length())
        assert bool((sp.writes == 1).all()) and bool((ip.writes == 1).all())
        assert bool((op.writes == 1).all())
        assert not bool(xp.writes.any()) and not bool(wp.writes.any())
        assert bool((wp.reads == 1).all()), 'every vocabulary coefficient is read once'
        assert bool((xp.reads == tiles).all()), 'all input features participate in every tile'
        return out


def native_norm(x, w, eps):
    v = x.float()
    return (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def ref_add(x, y, w, eps):
    if y.ndim == 3:
        total = torch.zeros_like(y[0])
        for part in y: total = total + part
        y = total.bfloat16()
    s = x + y
    return s, native_norm(s, w, eps)


def ref_silu(gu):
    g, u = gu.chunk(2, -1)
    return F.silu(g) * u


def ref_qkv(p, qw, kw, qe, ke, cos, sin, keys, values, positions):
    b, nk, cap, d = keys.shape; t = positions.numel(); nq = p.shape[-1]//d-2*nk
    q, k, v = p.reshape(b,t,-1).split((nq*d,nk*d,nk*d), -1)
    q = native_norm(q.reshape(b,t,nq,d), qw, qe)
    k = native_norm(k.reshape(b,t,nk,d), kw, ke)
    c, s = cos[positions][None,:,None,:], sin[positions][None,:,None,:]
    def rope(x):
        rot = torch.cat((-x[...,d//2:],x[...,:d//2]),-1)
        return x*c + rot*s
    q, k = rope(q), rope(k)
    keys.index_copy_(2, positions, k.transpose(1,2))
    values.index_copy_(2, positions, v.reshape(b,t,nk,d).transpose(1,2))
    return q.contiguous()


def ref_prefill_qkv(p,qw,kw,qe,ke,c,s,k,v,t):
    return ref_qkv(p,qw,kw,qe,ke,c,s,k,v,torch.arange(t))


def ref_decode_qkv(p,qw,kw,qe,ke,c,s,pos,k,v):
    return ref_qkv(p,qw,kw,qe,ke,c,s,k,v,pos).transpose(1,2).contiguous()


def ref_attention(q, k, v, t, scale):
    groups = q.shape[2] // k.shape[1]
    return F.scaled_dot_product_attention(q.transpose(1,2),
        k[:,:,:t].repeat_interleave(groups,1), v[:,:,:t].repeat_interleave(groups,1),
        is_causal=True, scale=scale).transpose(1,2).contiguous()


@contextmanager
def reference_kernels():
    with mock.patch.dict(sys.modules, {
        'kernels.decode_fused': SimpleNamespace(add_rms_norm=ref_add, silu_mul=ref_silu),
        'kernels.prefill': SimpleNamespace(prefill_qkv=ref_prefill_qkv),
    }), mock.patch.object(prefill, '_native_attention', side_effect=ref_attention):
        yield


class GreedyCPU(unittest.TestCase):
    @torch.inference_mode()
    def test_fallback_and_metadata_rule(self):
        for dtype in (torch.float32, torch.bfloat16):
            x, w = torch.randn(3,19).to(dtype), torch.randn(257,19).to(dtype)
            with mock.patch.dict(sys.modules, {'kernels.greedy_head': None}):
                torch.testing.assert_close(head.greedy_head(x,w), F.linear(x,w).argmax(-1,keepdim=True))
        def meta(shape, **kw):
            d=dict(ndim=len(shape),shape=shape,is_cuda=True,device=torch.device('cuda'),
                   dtype=torch.bfloat16,is_contiguous=lambda:True);d.update(kw)
            return SimpleNamespace(**d)
        self.assertTrue(head.supports(meta((3,2560)),meta((151936,2560))))
        for x,w in ((meta((65,2560)),meta((151936,2560))),
                    (meta((3,2560)),meta((262145,2560))),
                    (meta((3,2560),dtype=torch.float32),meta((151936,2560))),
                    (meta((3,2560)),meta((151936,2560),is_contiguous=lambda:False))):
            self.assertFalse(head.supports(x,w))
        with torch.enable_grad(): self.assertFalse(head.supports(meta((3,2560)),meta((151936,2560))))

    @torch.inference_mode()
    def test_runtime_wrapper_allocations_launches_and_no_silent_retry(self):
        calls=[];allocations=[]
        class Launch:
            def __init__(self,fn):self.name=fn.__name__
            def __getitem__(self,grid):
                return lambda *args,**kwargs:calls.append((self.name,grid,args,kwargs))
        tr=mock.Mock(jit=Launch,cdiv=lambda a,b:(a+b-1)//b,
                     next_power_of_2=lambda n:1<<(n-1).bit_length())
        spec=importlib.util.spec_from_file_location('head_wrapper_fixture',ENGINE/'kernels/greedy_head.py')
        module=importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules,{'triton':tr,'triton.language':mock.Mock()}):
            spec.loader.exec_module(module)
        def matrix(shape):return SimpleNamespace(ndim=2,shape=shape,is_cuda=True,
            device=torch.device('cuda'),dtype=torch.bfloat16,is_contiguous=lambda:True)
        def allocate(shape,**kwargs):
            result=object();allocations.append((shape,kwargs,result));return result
        with mock.patch.object(torch,'empty',side_effect=allocate):
            out=module.full_vocab_argmax(matrix((3,2560)),matrix((151936,2560)))
        self.assertEqual([a[0] for a in allocations],[(3,1187),(3,1187),(3,1)])
        self.assertEqual([a[1]['dtype'] for a in allocations],[torch.float32,torch.int32,torch.int64])
        self.assertIs(out,allocations[-1][2])
        self.assertEqual([(c[0],c[1]) for c in calls],[('_project_max',(1187,)),('_merge_max',(3,))])
        self.assertEqual(calls[0][3]['BM'],16)
        self.assertEqual(calls[1][3]['BLOCK'],2048)
        broken=SimpleNamespace(full_vocab_argmax=mock.Mock(side_effect=RuntimeError('kernel failure')))
        with mock.patch.object(head,'supports',return_value=True), \
             mock.patch.dict(sys.modules,{'kernels.greedy_head':broken}), \
             mock.patch.object(F,'linear') as fallback:
            with self.assertRaisesRegex(RuntimeError,'kernel failure'):
                head.greedy_head(torch.empty(1,1),torch.empty(1,1))
            fallback.assert_not_called()

    @torch.inference_mode()
    def test_actual_kernel_bodies_cover_features_vocab_tails_and_rows(self):
        torch.manual_seed(22); body = KernelBody()
        cases=((1,1,1),(1,127,63),(3,129,65),(16,257,192),(17,383,128),
               (33,257,256),(64,17,7),(3,257,2560),(1,151936,16))
        for m,n,k in cases:
            with self.subTest(shape=(m,n,k)):
                # Dyadic operands make indexing/cast checks independent of reduction grouping.
                x=(torch.randint(-8,9,(m,k)).float()/8).bfloat16()
                w=(torch.randint(-8,9,(n,k)).float()/16).bfloat16()
                originals=x.clone(),w.clone()
                expected=(x.float()@w.float().T).bfloat16().argmax(-1,keepdim=True)
                torch.testing.assert_close(body.run(x,w),expected,rtol=0,atol=0)
                for a,b in zip((x,w),originals): torch.testing.assert_close(a,b,rtol=0,atol=0)

    @torch.inference_mode()
    def test_bf16_ties_cross_tile_negative_control_and_nan(self):
        body=KernelBody();x=torch.zeros(1,64,dtype=torch.bfloat16);x[0,:2]=1
        w=torch.zeros(257,64,dtype=torch.bfloat16);w[:,0]=-4
        w[127,0]=1;w[128,0]=1;w[128,1]=2**-9
        self.assertEqual(int((x.float()@w.float().T).argmax()),128)
        self.assertEqual(int(body.run(x,w)),127)  # BF16 rounding creates a tie.
        for a,b in ((float('inf'),float('inf')),(float('nan'),float('nan'))):
            w[127,0]=a;w[128,0]=b
            expected=F.linear(x,w).argmax(-1,keepdim=True)
            torch.testing.assert_close(body.run(x,w),expected)
        w.zero_();self.assertEqual(int(body.run(x,w)),0)
        w[:,0]=-float('inf');self.assertEqual(int(body.run(x,w)),0)

    @torch.inference_mode()
    def test_actual_prefill_and_decode_pipelines_with_kernel_body_head(self):
        import test_prefill as fixture
        torch.manual_seed(29)
        body = KernelBody()
        env = bodies(ENGINE/'decode_step.py', {'fused_decode_forward'},
            {'torch':torch, 'F':F, '_project':F.linear,
             '_mlp_hidden':lambda x,w:ref_silu(F.linear(x,w)),
             'qkv_norm_rope_cache':ref_decode_qkv, 'add_rms_norm':ref_add})
        class Context:
            def attention(self,q,k,v,mask,scale):
                g=q.shape[1]//k.shape[1]
                # CPU math SDPA can propagate 0*NaN from masked V. Slice to
                # the initialized prefix to emulate native Flash's bounded loads.
                length=int(mask.sum())
                return F.scaled_dot_product_attention(q,k[:,:,:length].repeat_interleave(g,1),
                    v[:,:,:length].repeat_interleave(g,1),scale=scale).transpose(1,2).contiguous()
        for b,t in ((1,1),(2,7),(3,11)):
            model,cache,(cos,sin)=fixture.fixture(b,t)
            pointers=[z.data_ptr() for z in cache.key_cache+cache.value_cache]
            weights={n:w.clone() for n,w in model.state_dict().items()}
            for repeat in range(2):
                for z in cache.key_cache+cache.value_cache:z.fill_(float('nan'))
                refcache=deepcopy(cache);ids=torch.randint(0,127,(b,t))
                with fixture.cpu_kernels(fixture.KernelBody().run):
                    logits=fixture.prefill.prefill_forward(model,refcache,ids,cos,sin)
                    with mock.patch.object(head,'greedy_head',side_effect=body.run):
                        tokens=fixture.prefill.prefill_forward(model,cache,ids,cos,sin,return_tokens=True)
                for step in range(3):
                    gap=logits[:,0].float().max(-1).values-logits[:,0].float().gather(1,tokens).squeeze(-1)
                    self.assertLessEqual(float(gap.max()),0.125)
                    for a,e in zip(cache.key_cache+cache.value_cache,refcache.key_cache+refcache.value_cache):
                        torch.testing.assert_close(a,e,rtol=0,atol=0,equal_nan=True)
                    if step==2:break
                    position=torch.tensor([t+step]);mask=(torch.arange(cache.max_cache_len)<=position).view(1,1,1,-1)
                    fn=env['fused_decode_forward'];context=Context()
                    logits=fn(model,refcache,tokens,position,mask,context,cos,sin)
                    with mock.patch.object(head,'greedy_head',side_effect=body.run):
                        tokens=fn(model,cache,tokens,position,mask,context,cos,sin,return_tokens=True)
            self.assertEqual(pointers,[z.data_ptr() for z in cache.key_cache+cache.value_cache])
            for name,w in model.state_dict().items():torch.testing.assert_close(w,weights[name],rtol=0,atol=0)
            self.assertIs(model.lm_head.weight,model.model.embed_tokens.weight)

    @torch.inference_mode()
    def test_step_and_prefill_select_ids_without_second_argmax_and_reset(self):
        env=bodies(ENGINE/'decode.py',{'step','prefill'},{'torch':torch,'qwen_forward':mock.Mock()})
        ids=torch.tensor([[2],[3]])
        cache=SimpleNamespace(prefilling=False)
        state=SimpleNamespace(tokens=torch.zeros(2,1,dtype=torch.int64),position=torch.tensor([7]),
            key_positions=torch.arange(20),flash_context=None,cache=cache,model=object(),cos=None,sin=None,
            fused_forward=mock.Mock(return_value=ids),prefill_plan=SimpleNamespace(run=mock.Mock(return_value=ids)))
        env['step'](state,return_tokens=True)
        state.fused_forward.assert_called_once()
        self.assertTrue(state.fused_forward.call_args.kwargs['return_tokens'])
        torch.testing.assert_close(state.tokens,ids);self.assertEqual(int(state.position),8)
        env['prefill'](state,torch.zeros(2,5,dtype=torch.int64),return_tokens=True)
        self.assertEqual(int(state.position),5);self.assertFalse(cache.prefilling)
        state.prefill_plan.run.side_effect=RuntimeError('injected')
        with self.assertRaises(RuntimeError): env['prefill'](state,torch.zeros(2,5,dtype=torch.int64),return_tokens=True)
        self.assertFalse(cache.prefilling)
        # Default diagnostic call still receives and returns logits.
        logits=torch.tensor([[[1.,3.,2.]],[[4.,1.,2.]]]);state.fused_forward=mock.Mock(return_value=logits)
        result=env['step'](state)
        self.assertIs(result,logits);self.assertFalse(state.fused_forward.call_args.kwargs)
        torch.testing.assert_close(state.tokens,logits[:,0].argmax(-1,keepdim=True))

    @torch.inference_mode()
    def test_prefill_two_modes_fresh_inputs_and_independent_graph_results(self):
        plan=prefill.PrefillPlan(SimpleNamespace(device=torch.device('cpu')),None,2,3,None,None)
        graphs=[];capturing=[];calls=[]
        class Stream:
            def wait_stream(self, other): pass
        class Graph:
            def __init__(self): self.work=None;self.replays=0;graphs.append(self)
            def replay(self): self.replays+=1;self.work()
        @contextmanager
        def capture(g,stream=None):
            capturing.append(g)
            try: yield
            finally: capturing.pop()
        def forward(model,cache,inputs,cos,sin,return_tokens=False):
            calls.append(return_tokens)
            def compute():
                ids=inputs.sum(-1,keepdim=True)%5
                return ids if return_tokens else F.one_hot(ids[:,0],5).float()[:,None,:]
            out=torch.empty((2,1) if return_tokens else (2,1,5),dtype=torch.int64 if return_tokens else torch.float32)
            if capturing: capturing[-1].work=lambda:out.copy_(compute())
            else: out.copy_(compute())
            return out
        with mock.patch.object(torch.cuda,'current_stream',return_value=Stream()), \
             mock.patch.object(torch.cuda,'Stream',side_effect=lambda **k:Stream()), \
             mock.patch.object(torch.cuda,'stream',side_effect=lambda s:nullcontext()), \
             mock.patch.object(torch.cuda,'graph',side_effect=capture), \
             mock.patch.object(torch.cuda,'CUDAGraph',Graph),mock.patch.object(prefill,'prefill_forward',side_effect=forward):
            for mode in (True,True,False,True,False):
                inputs=torch.randint(0,9,(2,3));out=plan.run(inputs,return_tokens=mode)
                expected=inputs.sum(-1,keepdim=True)%5
                torch.testing.assert_close(out if mode else out[:,0].argmax(-1,keepdim=True),expected)
        self.assertEqual(len(graphs),2);self.assertEqual(len(calls),6)
        self.assertIsNot(plan.graph,plan.token_graph)
        self.assertNotEqual(plan.logits.data_ptr(),plan.token_ids.data_ptr())

    def test_engine_and_capture_request_token_only_but_emit_does_not_change(self):
        engine_tree=ast.parse((ENGINE/'engine.py').read_text())
        capture_tree=ast.parse((ENGINE/'decode.py').read_text())
        flags=[]
        for tree in (engine_tree,capture_tree):
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    flags += [k.value.value for k in node.keywords
                              if k.arg=='return_tokens' and isinstance(k.value,ast.Constant)]
        self.assertGreaterEqual(flags.count(True),4)
        # Executed source, not just AST: capture must warm and record token-only forwards.
        state=SimpleNamespace(tokens=torch.tensor([[3]]),position=torch.tensor([7]),step=mock.Mock())
        class Stream:
            def wait_stream(self,other):pass
        env=bodies(ENGINE/'decode.py',{'capture'},{'torch':torch})
        with mock.patch.object(torch.cuda,'current_stream',return_value=Stream()), \
             mock.patch.object(torch.cuda,'Stream',return_value=Stream()), \
             mock.patch.object(torch.cuda,'stream',side_effect=lambda s:nullcontext()), \
             mock.patch.object(torch.cuda,'CUDAGraph',return_value=object()), \
             mock.patch.object(torch.cuda,'graph',side_effect=lambda *a,**k:nullcontext()):
            env['capture'](state)
        self.assertEqual(state.step.call_args_list,[mock.call(return_tokens=True)]*4)

    @unittest.skipUnless(importlib.util.find_spec('transformers') is not None,'Transformers required for real CPU model integration')
    @torch.inference_mode()
    def test_real_transformers_prefill_decode_logits_and_token_modes_same_prefix(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from decode import DecodeState
        torch.manual_seed(26)
        cfg=Qwen3Config(vocab_size=257,hidden_size=64,intermediate_size=96,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=128,max_position_embeddings=1024,
            tie_word_embeddings=True,rope_theta=5_000_000,sliding_window=None)
        cfg._attn_implementation='sdpa'
        native=Qwen3ForCausalLM(cfg).eval().bfloat16();candidate=deepcopy(native)
        from mlp import PackedMLP
        for layer in candidate.model.layers:
            a=layer.self_attn;a.qkv_weight=torch.cat((a.q_proj.weight,a.k_proj.weight,a.v_proj.weight))
            layer.mlp=PackedMLP(layer.mlp)
        env=bodies(ENGINE/'decode_step.py',{'fused_decode_forward'},
            {'torch':torch,'F':F,'_project':F.linear,'_mlp_hidden':lambda x,w:ref_silu(F.linear(x,w)),
             'qkv_norm_rope_cache':ref_decode_qkv,'add_rms_norm':ref_add})
        body=KernelBody()
        class Context:
            def prepare(self,p):self.mask=(torch.arange(self.cap)<=p).view(1,1,1,-1);return self.mask
            def attention(self,q,k,v,mask,scale):
                self.assert_mask=mask
                return F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),
                    attn_mask=mask,scale=scale).transpose(1,2).contiguous()
        for batch,length,count in ((1,1,3),(3,7,4),(2,11,3)):
            state=DecodeState(candidate,batch,length,count)
            context=Context();context.cap=length+count
            state.flash_context=context;state.fused_forward=env['fused_decode_forward']
            for repeat in range(2):
                for t in state.cache.key_cache+state.cache.value_cache:t.fill_(33+repeat)
                ids=torch.randint(0,257,(batch,length));prefix=ids
                with reference_kernels(),mock.patch.object(head,'greedy_head',side_effect=body.run):
                    out=prefill.prefill_forward(candidate,state.cache,ids,state.cos,state.sin,return_tokens=True)
                    state.tokens.copy_(out);state.position.fill_(length)
                    for i in range(count):
                        logits=native(input_ids=prefix,use_cache=False,logits_to_keep=1).logits[:,0].float()
                        gap=logits.max(-1).values-logits.gather(1,state.tokens).squeeze(1)
                        self.assertLessEqual(float(gap.max()),2.0)
                        prefix=torch.cat((prefix,state.tokens.clone()),1)
                        if i+1<count:state.step(return_tokens=True)
                self.assertEqual(int(state.position),length+count-1)


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('triton') is not None,
                     'CUDA/Triton required; CPU checks are not GPU validation')
class GreedyCUDA(unittest.TestCase):
    @torch.inference_mode()
    def test_full_vocab_actual_dimensions_ties_tails_and_graph_reuse(self):
        torch.manual_seed(27)
        from kernels.greedy_head import full_vocab_argmax
        for m,n,k in ((1,151936,2560),(4,151936,2560),(16,151936,2560),
                      (33,1025,129),(3,257,65),(64,257,256)):
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)/k**0.5
            old=x.clone(),w.clone()
            got=full_vocab_argmax(x,w);logits=F.linear(x,w)
            gap=logits.float().max(-1).values-logits.float().gather(1,got).squeeze(1)
            self.assertLessEqual(float(gap.max()),0.125) # operator diagnostic; NOT the model gate
            for a,b in zip((x,w),old):torch.testing.assert_close(a,b,rtol=0,atol=0)
            if m==3:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):full_vocab_argmax(x,w)
                torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):out=full_vocab_argmax(x,w)
                for _ in range(3):
                    x.normal_();graph.replay();ref=F.linear(x,w).float()
                    self.assertLessEqual(float((ref.max(-1).values-ref.gather(1,out).squeeze(1)).max()),0.125)
        x=torch.zeros(1,64,device='cuda',dtype=torch.bfloat16);x[0,:2]=1
        w=torch.zeros(257,64,device='cuda',dtype=torch.bfloat16);w[:,0]=-4
        w[127,0]=1;w[128,0]=1;w[128,1]=2**-9
        self.assertEqual(int(full_vocab_argmax(x,w)),127)

    @torch.inference_mode()
    def test_real_engine_both_token_graphs_reuse_own_prefix_and_eos(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from engine import Engine
        from kernels.greedy_head import full_vocab_argmax
        torch.manual_seed(28)
        cfg=Qwen3Config(vocab_size=257,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=128,max_position_embeddings=1024,
            tie_word_embeddings=True,rope_theta=5_000_000,sliding_window=None,eos_token_id=0)
        cfg._attn_implementation='sdpa'
        ref=Qwen3ForCausalLM(cfg).eval()
        with tempfile.TemporaryDirectory() as path:
            ref.save_pretrained(path);engine=Engine(path);ref=ref.cuda().bfloat16()
            self.assertIs(engine.model.lm_head.weight,engine.model.model.embed_tokens.weight)
            for b,t,n in ((3,7,5),(3,7,5),(1,1,3),(2,9,1),(2,9,0)):
                ids=torch.randint(0,257,(b,t),device='cuda')
                if engine.decode_state is not None:
                    for z in engine.decode_state.cache.key_cache+engine.decode_state.cache.value_cache:z.fill_(31)
                with mock.patch('kernels.greedy_head.full_vocab_argmax',wraps=full_vocab_argmax) as used:
                    emitted=list(engine.generate(ids.tolist(),n))
                self.assertEqual(len(emitted),n)
                self.assertTrue(all(len(row)==b and all(type(v) is int for v in row) for row in emitted))
                if not n:continue
                if t>1:self.assertIsNotNone(engine.decode_state.prefill_plan.token_graph)
                if used.call_count==0:
                    # Only reused graphs may execute no Python wrapper calls.
                    self.assertEqual((b,t,n),(3,7,5))
                tokens=torch.tensor(emitted,device='cuda').T
                prefix=torch.cat((ids,tokens[:,:-1]),1)
                logits=ref(input_ids=prefix,use_cache=False).logits[:,t-1:].float()
                gap=logits.max(-1).values-logits.gather(-1,tokens[...,None]).squeeze(-1)
                self.assertLessEqual(float(gap.max()),2.0)
            engine.model.lm_head.weight.zero_()  # tied embedding zero; every logit ties => EOS 0
            self.assertEqual(list(engine.generate([[1,2,3],[4,5,6]],4)),[[0,0]]*4)


if __name__=='__main__':unittest.main()
