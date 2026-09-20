"""Split SwiGLU checks. CPU source-body execution is NOT Triton/CUDA validation."""

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
from torch.nn import functional as F

ENGINE = Path(__file__).resolve().parents[1] / "engine"
HAS_TRITON = importlib.util.find_spec("triton") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Launch:
    calls = []
    def __init__(self, fn):
        self.name = fn.__name__
    def __getitem__(self, grid):
        def record(*args, **kwargs):
            self.calls.append((self.name, grid, args, kwargs))
        return record


def fake_modules():
    language = SimpleNamespace(constexpr=int)
    fake_triton = SimpleNamespace(language=language, jit=Launch, cdiv=lambda a, b: (a+b-1)//b,
                                 next_power_of_2=lambda n: 1 << (n-1).bit_length())
    with mock.patch.dict(sys.modules, {"triton": fake_triton,
                                      "triton.language": language}):
        old = load("split_swiglu_old_test", ENGINE / "kernels/skinny_gemm.py")
        with mock.patch.dict(sys.modules, {"kernels.skinny_gemm": old}):
            new = load("split_swiglu_new_test", ENGINE / "kernels/split_swiglu.py")
    return old, new


old, new = fake_modules()


def functions(path, names, env):
    picked = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            for arg in node.args.args:
                arg.annotation = None
            picked.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=picked, type_ignores=[])),
                 str(path), 'exec'), env)
    return env


class Pointer:
    def __init__(self, tensor, offsets=0, access=None, writable=False):
        self.data, self.offsets = tensor.reshape(-1), offsets
        self.writable = writable
        self.access = (dict(reads=torch.zeros(tensor.numel(), dtype=torch.int32),
                            writes=torch.zeros(tensor.numel(), dtype=torch.int32))
                       if access is None else access)
    def __add__(self, amount):
        return Pointer(self.data, self.offsets + amount, self.access, self.writable)


class Ops:
    float32, bfloat16, int64 = torch.float32, torch.bfloat16, torch.int64
    arange, exp = staticmethod(torch.arange), staticmethod(torch.exp)
    static_range = staticmethod(range)
    program = (0, 0)
    def program_id(self, axis):
        return torch.tensor(self.program[axis], dtype=torch.int64)
    @staticmethod
    def zeros(shape, dtype): return torch.zeros(shape, dtype=dtype)
    @staticmethod
    def where(test, a, b):
        return torch.where(torch.as_tensor(test), torch.as_tensor(a), torch.as_tensor(b))
    @staticmethod
    def trans(x): return x.T
    @staticmethod
    def dot(a, b, acc, input_precision):
        assert a.dtype == b.dtype == torch.bfloat16
        assert acc.dtype == torch.float32 and input_precision == 'ieee'
        return acc + a.float() @ b.float()
    @staticmethod
    def indices(p, mask):
        i, mask = torch.broadcast_tensors(torch.as_tensor(p.offsets, dtype=torch.int64),
                                          torch.as_tensor(mask, dtype=torch.bool))
        safe = torch.where(mask, i, 0)
        if bool(((safe < 0) | (safe >= p.data.numel())).any()):
            raise AssertionError('out-of-bounds pointer')
        return safe, mask
    def load(self, p, mask=True, other=0.):
        i, mask = self.indices(p, mask)
        p.access['reads'].scatter_add_(0, i[mask].flatten(),
                                      torch.ones_like(i[mask].flatten(), dtype=torch.int32))
        return torch.where(mask, p.data[i], torch.full_like(p.data[i], other))
    def store(self, p, value, mask=True):
        assert p.writable, 'write to read-only input'
        i, mask = self.indices(p, mask)
        value, _ = torch.broadcast_tensors(torch.as_tensor(value), mask)
        p.data[i[mask]] = value[mask].to(p.data.dtype)
        p.access['writes'].scatter_add_(0, i[mask].flatten(),
                                       torch.ones_like(i[mask].flatten(), dtype=torch.int32))


class Bodies:
    """Actual kernel bodies with checked CPU indexing; no GPU code generation."""
    def __init__(self):
        self.ops = Ops()
        self.fn = functions(ENGINE / 'kernels/split_swiglu.py',
                            {'_paired_partials', '_reduce_silu'}, {'tl': self.ops})
    def partials(self, x, w):
        m, k = x.shape; n = w.shape[0] // 2; s = 4
        assert n % 32 == 0 and k % 256 == 0
        p = torch.full((s, m, 2*n), float('nan'))
        xp, wp, pp = Pointer(x), Pointer(w), Pointer(p, writable=True)
        bm = max(16, 1 << (m-1).bit_length())
        for tile in range(n // 32):
            for split in range(s):
                self.ops.program = (tile, split)
                self.fn['_paired_partials'](xp, wp, pp, m, n, k, s, bm, 32, 64)
        assert bool((pp.access['writes'] == 1).all())
        assert bool((wp.access['reads'] == 1).all()), 'every gate/up weight read once'
        assert not bool(xp.access['writes'].any()) and not bool(wp.access['writes'].any())
        return p
    def finish(self, p):
        s, m, width = p.shape; n = width // 2
        y = torch.full((m, n), float('nan'), dtype=torch.bfloat16)
        pp, yp = Pointer(p), Pointer(y, writable=True)
        for tile in range((m*n+255)//256):
            self.ops.program = (tile, 0)
            self.fn['_reduce_silu'](pp, yp, m, n, s, 256)
        assert bool((yp.access['writes'] == 1).all())
        assert bool((pp.access['reads'] == 1).all()), 'all FP32 partials consumed exactly once'
        return y
    def run(self, x, w):
        return self.finish(self.partials(x, w))


def finish_reference(p):
    total = torch.zeros_like(p[0])
    for part in p:
        total = total + part
    gate, up = total.bfloat16().float().chunk(2, -1)
    act = (gate / (1 + torch.exp(-gate))).bfloat16().float()
    return (act * up).bfloat16()


def metadata(shape, dtype=torch.bfloat16, cuda=True):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype, is_cuda=cuda,
                           device=torch.device('cuda' if cuda else 'cpu'),
                           is_contiguous=lambda: True)


class SplitCPU(unittest.TestCase):
    @torch.inference_mode()
    def test_fixed_rule_wrapper_dispatch_and_scratch(self):
        for k, s in ((256, 1), (768, 1), (1024, 4), (1088, 1), (1280, 4), (2560, 4)):
            self.assertEqual(new.split_count(k), s)
        for m in (1, 3, 16, 17, 33, 64):
            x, w = metadata((m, 2560)), metadata((19456, 2560))
            partial, out = object(), object()
            Launch.calls.clear()
            with mock.patch.object(torch, 'empty', side_effect=[partial, out]) as alloc:
                self.assertIs(new.linear_silu_mul(x, w), out)
            self.assertEqual(alloc.call_args_list[0].args[0], (4, m, 19456))
            self.assertEqual(alloc.call_args_list[0].kwargs['dtype'], torch.float32)
            self.assertEqual(Launch.calls[0][1], (304, 4))
            self.assertEqual(Launch.calls[0][3]['BM'], max(16, 1 << (m-1).bit_length()))
            self.assertIs(Launch.calls[0][2][0], x); self.assertIs(Launch.calls[0][2][1], w)
            self.assertIs(Launch.calls[1][2][0], partial)
            self.assertFalse(Launch.calls[1][3]['enable_fp_fusion'])
        with mock.patch.object(new, 'unsplit_silu_mul', return_value='original') as fallback:
            x, w = metadata((3, 768)), metadata((128, 768))
            self.assertEqual(new.linear_silu_mul(x, w), 'original')
            fallback.assert_called_once_with(x, w)
        for x, w in ((metadata((65, 1024)), metadata((128, 1024))),
                     (metadata((3, 1024), cuda=False), metadata((128, 1024), cuda=False)),
                     (metadata((3, 1024)), metadata((130, 1024))),
                     (metadata((3, 1024)), metadata((0, 1024)))):
            with self.assertRaises(ValueError): new.linear_silu_mul(x, w)

    @torch.inference_mode()
    def test_actual_kernel_coverage_values_pair_order_and_immutable_inputs(self):
        torch.manual_seed(180)
        body = Bodies()
        # Actual hidden width and actual MLP width exercised separately. The
        # reduced-K large-N case tests addressing, not the wrapper's K threshold.
        cases = ((1,64,2560), (3,96,1280), (16,32,1024), (17,64,1024),
                 (33,32,1024), (64,32,1024), (2,9728,256))
        for m,n,k in cases:
            with self.subTest(m=m,n=n,k=k):
                x = (torch.randn(m,k)*0.2).bfloat16()
                w = (torch.randn(2*n,k)*0.2).bfloat16()
                before = (x.clone(),w.clone())
                p = body.partials(x,w)
                for split in range(4):
                    left,right = split*(k//4),(split+1)*(k//4)
                    expected = x[:,left:right].float() @ w[:,left:right].float().T
                    torch.testing.assert_close(p[split],expected,rtol=2e-5,atol=2e-5)
                actual = body.finish(p)
                torch.testing.assert_close(actual,finish_reference(p),rtol=0,atol=0)
                for t,b in zip((x,w),before): torch.testing.assert_close(t,b,rtol=0,atol=0)

    @torch.inference_mode()
    def test_partial_and_projection_and_activation_rounding_negative_controls(self):
        torch.manual_seed(181)
        p = torch.randn(4,3,256)*2
        good = Bodies().finish(p)
        self.assertGreater(int((good != Bodies().finish(p.bfloat16().float())).sum()),0)
        total = torch.zeros_like(p[0])
        for part in p: total = total + part
        g,u = total.chunk(2,-1)
        wrong_projection = ((g/(1+torch.exp(-g))).bfloat16().float()*u).bfloat16()
        g,u = total.bfloat16().float().chunk(2,-1)
        wrong_activation = ((g/(1+torch.exp(-g)))*u).bfloat16()
        self.assertGreater(int((good != wrong_projection).sum()),0)
        self.assertGreater(int((good != wrong_activation).sum()),0)

    @torch.inference_mode()
    def test_decode_dispatch_is_lazy_and_retains_native_and_original_paths(self):
        env = functions(ENGINE/'decode_step.py',
                        {'linear_silu_mul','_mlp_hidden','_project'},
                        {'F':F,'supports':old.supports, 'silu_mul':lambda p:F.silu(p.chunk(2,-1)[0])*p.chunk(2,-1)[1]})
        x,w = torch.randn(3,128).bfloat16(),torch.randn(192,128).bfloat16()
        expected = F.silu(F.linear(x,w).chunk(2,-1)[0])*F.linear(x,w).chunk(2,-1)[1]
        with mock.patch.dict(sys.modules, {'kernels.split_swiglu':None}):
            torch.testing.assert_close(env['_mlp_hidden'](x,w),expected,rtol=0,atol=0)
        marker = object(); called = mock.Mock(return_value=marker)
        env['supports'] = lambda *a,**k:True
        with mock.patch.dict(sys.modules, {'kernels.split_swiglu':SimpleNamespace(linear_silu_mul=called)}):
            self.assertIs(env['_mlp_hidden'](x,w),marker)
        called.assert_called_once_with(x,w)
        env['linear_partials'] = mock.Mock(return_value='unchanged projection')
        self.assertEqual(env['_project'](x,w),'unchanged projection')

    @torch.inference_mode()
    def test_actual_mlp_callsite_with_source_body_kernel(self):
        torch.manual_seed(182)
        body = Bodies()
        env = functions(ENGINE/'decode_step.py', {'linear_silu_mul','_mlp_hidden'},
                        {'F':F, 'supports':lambda *a,**k:True})
        for m in (1,3,17):
            x = (torch.randn(m,1024)*0.2).bfloat16()
            w = (torch.randn(256,1024)*0.2).bfloat16()
            fn = mock.Mock(side_effect=body.run)
            with mock.patch.dict(sys.modules, {'kernels.split_swiglu':SimpleNamespace(linear_silu_mul=fn)}):
                actual = env['_mlp_hidden'](x,w)
            fn.assert_called_once_with(x,w)
            pieces = torch.stack([x[:,s*256:(s+1)*256].float() @ w[:,s*256:(s+1)*256].float().T
                                  for s in range(4)])
            torch.testing.assert_close(actual,finish_reference(pieces),rtol=1e-2,atol=1e-3)

    def test_selected_failure_propagates_without_vendor_retry(self):
        with torch.inference_mode(), mock.patch.object(new,'_paired_partials') as kernel, \
                mock.patch.object(new,'unsplit_silu_mul') as old_impl, \
                mock.patch.object(torch,'empty',side_effect=[object(),object()]):
            kernel.__getitem__.return_value.side_effect=RuntimeError('device failure')
            with self.assertRaisesRegex(RuntimeError,'device failure'):
                new.linear_silu_mul(metadata((1,1024)),metadata((256,1024)))
            old_impl.assert_not_called()


@contextmanager
def module_path():
    sys.path.insert(0,str(ENGINE))
    try: yield
    finally: sys.path.remove(str(ENGINE))


def tiny_model():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    cfg=Qwen3Config(vocab_size=127,hidden_size=1024,intermediate_size=128,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=128,max_position_embeddings=256,
        rope_theta=5_000_000,tie_word_embeddings=True,sliding_window=None,eos_token_id=0)
    cfg._attn_implementation='sdpa'
    return Qwen3ForCausalLM(cfg).eval().bfloat16()


def ref_norm(x,w,eps):
    f=x.float()
    return (f*torch.rsqrt(f.square().mean(-1,keepdim=True)+eps)).to(x.dtype)*w


def ref_qkv(p,qw,kw,qe,ke,cos,sin,pos,keys,values):
    b,nk,_,d=keys.shape; nq=p.shape[-1]//d-2*nk
    q,k,v=p.split((nq*d,nk*d,nk*d),-1)
    q=ref_norm(q.view(b,nq,1,d),qw,qe); k=ref_norm(k.view(b,nk,1,d),kw,ke)
    c=cos.index_select(0,pos)[None,None,:,:]; s=sin.index_select(0,pos)[None,None,:,:]
    def rotate(t):return t*c+torch.cat((-t[...,d//2:],t[...,:d//2]),-1)*s
    q,k=rotate(q),rotate(k)
    keys.index_copy_(2,pos,k);values.index_copy_(2,pos,v.view(b,nk,1,d))
    return q.contiguous()


class TransformersCPU(unittest.TestCase):
    @unittest.skipUnless(HAS_TRANSFORMERS,'Transformers required; arithmetic fixtures are not HF integration')
    @torch.inference_mode()
    def test_real_transformers_source_body_decode_own_prefix_and_cache_reset(self):
        torch.manual_seed(183)
        native=tiny_model();candidate=deepcopy(native)
        packed=load('split_test_mlp',ENGINE/'mlp.py')
        for layer in candidate.model.layers:
            a=layer.self_attn
            a.qkv_weight=torch.cat([p.weight for p in (a.q_proj,a.k_proj,a.v_proj)])
            layer.mlp=packed.PackedMLP(layer.mlp)
        env=functions(ENGINE/'decode_step.py',{'fused_decode_forward','linear_silu_mul','_mlp_hidden'},
            {'F':F,'torch':torch,'supports':lambda *a,**k:True,
             '_project':F.linear,'qkv_norm_rope_cache':ref_qkv,
             'add_rms_norm':lambda x,y,w,e:(x+y,ref_norm(x+y,w,e))})
        state_module=load('split_cpu_decode',ENGINE/'decode.py')
        flash=load('split_cpu_flash',ENGINE/'flash_decode.py')
        class Context(flash.FlashDecodeContext):
            def attention(self,q,k,v,mask,scale):
                assert mask is self.mask
                g=q.shape[1]//k.shape[1]
                return F.scaled_dot_product_attention(q,k.repeat_interleave(g,1),v.repeat_interleave(g,1),
                    attn_mask=mask,scale=scale).transpose(1,2).contiguous()
        body=Bodies();calls=mock.Mock(side_effect=body.run)
        for b,length,count in ((1,1,3),(3,5,4)):
            state=state_module.DecodeState(candidate,b,length,count)
            state.flash_context=Context(b,4,2,state.key_positions)
            state.fused_forward=env['fused_decode_forward']
            ptrs=[t.data_ptr() for t in state.cache.key_cache+state.cache.value_cache]
            for repeat in range(2):
                for t in state.cache.key_cache+state.cache.value_cache:t.fill_(19+repeat)
                ids=torch.randint(0,127,(b,length));actual=state.prefill(ids)
                cache=None;current=ids
                for step in range(count):
                    expected=native(input_ids=current,past_key_values=cache,use_cache=True,logits_to_keep=1)
                    torch.testing.assert_close(actual,expected.logits,rtol=2e-2,atol=2e-2)
                    token=state.tokens.clone()
                    gap=expected.logits[:,-1].float().amax(-1)-expected.logits[:,-1].float().gather(1,token).squeeze(1)
                    self.assertLessEqual(float(gap.max()),0.125)
                    current=token;cache=expected.past_key_values
                    if step+1<count:
                        with mock.patch.dict(sys.modules,{'kernels.split_swiglu':SimpleNamespace(linear_silu_mul=calls)}):
                            actual=state.step()
                self.assertEqual(int(state.position),length+count-1)
                self.assertEqual(ptrs,[t.data_ptr() for t in state.cache.key_cache+state.cache.value_cache])
        self.assertEqual(calls.call_count,20)


@unittest.skipUnless(torch.cuda.is_available() and HAS_TRITON,'CUDA and Triton required; CPU checks are not GPU validation')
class SplitCUDA(unittest.TestCase):
    @torch.inference_mode()
    def test_real_dimensions_and_changed_input_graph_replay(self):
        with module_path():
            from kernels import split_swiglu as gpu
            from kernels.skinny_gemm import linear_silu_mul as original
        torch.manual_seed(184)
        w=(torch.randn(19456,2560,device='cuda')/2560**0.5).bfloat16()
        before=w.clone()
        for m in (1,3,16,17,64):
            x=torch.randn(m,2560,device='cuda',dtype=torch.bfloat16)
            out=gpu.linear_silu_mul(x,w)
            gate,up=F.linear(x,w).chunk(2,-1)
            torch.testing.assert_close(out,F.silu(gate)*up,rtol=2e-2,atol=3e-3)
        x=torch.randn(3,2560,device='cuda',dtype=torch.bfloat16)
        current=torch.cuda.current_stream();stream=torch.cuda.Stream();stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):gpu.linear_silu_mul(x,w)
        current.wait_stream(stream);graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):out=gpu.linear_silu_mul(x,w)
        current.wait_stream(stream)
        for _ in range(4):
            x.normal_();graph.replay();gate,up=F.linear(x,w).chunk(2,-1)
            torch.testing.assert_close(out,F.silu(gate)*up,rtol=2e-2,atol=3e-3)
        torch.testing.assert_close(w,before,rtol=0,atol=0)
        # The short-K route is exactly the already-passing kernel, not F.linear.
        x=torch.randn(3,768,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(256,768,device='cuda',dtype=torch.bfloat16)
        torch.testing.assert_close(gpu.linear_silu_mul(x,w),original(x,w),rtol=0,atol=0)

    @unittest.skipUnless(HAS_TRANSFORMERS,'Transformers required')
    @torch.inference_mode()
    def test_real_engine_forces_new_kernel_own_prefix_reuse_and_eos(self):
        with module_path():
            from engine import Engine
            from kernels import split_swiglu as gpu
        torch.manual_seed(185);native=tiny_model()
        with tempfile.TemporaryDirectory() as path:
            native.save_pretrained(path);engine=Engine(path);native=native.cuda()
            with mock.patch.object(gpu,'linear_silu_mul',wraps=gpu.linear_silu_mul) as used:
                for b,p,n in ((1,5,7),(1,5,7),(3,7,9),(2,1,2),(1,7,1),(1,5,0)):
                    if engine.decode_state is not None:
                        for t in engine.decode_state.cache.key_cache+engine.decode_state.cache.value_cache:t.fill_(37)
                    ids=torch.randint(0,127,(b,p),device='cuda')
                    outputs=list(engine.generate(ids.tolist(),n))
                    self.assertEqual(len(outputs),n)
                    self.assertTrue(all(len(row)==b and all(type(t) is int for t in row) for row in outputs))
                    if n:
                        selected=torch.tensor(outputs,device='cuda').T.contiguous()
                        prefix=torch.cat((ids,selected[:,:-1]),1)
                        logits=native(input_ids=prefix,use_cache=False).logits[:,p-1:p+n-1].float()
                        gap=logits.amax(-1)-logits.gather(-1,selected.unsqueeze(-1)).squeeze(-1)
                        self.assertLessEqual(float(gap.max()),2.0)
                    self.assertIs(engine.model.lm_head.weight,engine.model.model.embed_tokens.weight)
                self.assertGreater(used.call_count,0)
            engine.model.lm_head.weight.zero_()  # tied embedding makes a deterministic all-zero/EOS fixture.
            self.assertEqual(list(engine.generate([[3,4]],7)),[[0]]*7)


if __name__=='__main__':unittest.main()
