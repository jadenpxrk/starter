"""Tile-major checks. CPU source-body execution is NOT Triton or GPU validation."""
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
from torch import nn
from torch.nn import functional as F

ENGINE=Path(__file__).resolve().parents[1]/'engine'
HAS_TRITON=importlib.util.find_spec('triton') is not None
HAS_TRANSFORMERS=importlib.util.find_spec('transformers') is not None


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


layout=load('weight_layout_under_test',ENGINE/'weight_layout.py')


def functions(path,names,env):
    tree=ast.parse(path.read_text());nodes=[]
    for parent in [tree]+[n for n in tree.body if isinstance(n,ast.ClassDef)]:
        for n in parent.body:
            if isinstance(n,ast.FunctionDef) and n.name in names:
                n.decorator_list=[]
                for a in n.args.args:a.annotation=None
                nodes.append(n)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),env)
    return env


class Launch:
    calls=[]
    def __init__(self,fn):self.name=fn.__name__
    def __getitem__(self,grid):
        def record(*args,**kwargs):self.calls.append((grid,args,kwargs))
        return record


def fake_kernels():
    language=SimpleNamespace(constexpr=int)
    triton=SimpleNamespace(language=language,jit=Launch,next_power_of_2=lambda n:1<<(n-1).bit_length())
    with mock.patch.dict(sys.modules,{'triton':triton,'triton.language':language}):
        old=load('layout_original_kernels',ENGINE/'kernels/skinny_gemm.py')
        with mock.patch.dict(sys.modules,{'kernels.skinny_gemm':old}):
            new=load('layout_tiled_kernels',ENGINE/'kernels/tiled_gemm.py')
    return old,new


old,new=fake_kernels()


class Pointer:
    def __init__(self,tensor,offsets=0,access=None,writable=False):
        self.data,self.offsets=tensor.reshape(-1),offsets;self.writable=writable
        self.access=(dict(reads=torch.zeros(tensor.numel(),dtype=torch.int32),
                          writes=torch.zeros(tensor.numel(),dtype=torch.int32)) if access is None else access)
    def __add__(self,value):return Pointer(self.data,self.offsets+value,self.access,self.writable)


class Ops:
    float32=torch.float32
    arange=staticmethod(torch.arange)
    program=(0,0)
    def program_id(self,axis):return self.program[axis]
    @staticmethod
    def zeros(shape,dtype):return torch.zeros(shape,dtype=dtype)
    @staticmethod
    def trans(t):return t.T
    @staticmethod
    def dot(a,b,acc):
        assert a.dtype==b.dtype==torch.bfloat16 and acc.dtype==torch.float32
        return acc+a.float()@b.float()
    @staticmethod
    def indices(p,mask):
        i,mask=torch.broadcast_tensors(torch.as_tensor(p.offsets,dtype=torch.int64),torch.as_tensor(mask,dtype=torch.bool))
        safe=torch.where(mask,i,0)
        assert not bool(((safe<0)|(safe>=p.data.numel())).any()),'out of bounds'
        return safe,mask
    def load(self,p,mask=True,other=0.):
        i,mask=self.indices(p,mask)
        p.access['reads'].scatter_add_(0,i[mask].flatten(),torch.ones_like(i[mask].flatten(),dtype=torch.int32))
        return torch.where(mask,p.data[i],torch.full_like(p.data[i],other))
    def store(self,p,value,mask=True):
        assert p.writable,'input mutation'
        i,mask=self.indices(p,mask);value,_=torch.broadcast_tensors(torch.as_tensor(value),mask)
        p.data[i[mask]]=value[mask].to(p.data.dtype)
        p.access['writes'].scatter_add_(0,i[mask].flatten(),torch.ones_like(i[mask].flatten(),dtype=torch.int32))


class Bodies:
    """Run BOTH actual projection bodies with identical checked CPU operations."""
    def __init__(self):
        self.ops=Ops();self.calls=0
        self.original=functions(ENGINE/'kernels/skinny_gemm.py',{'_partials_kernel'},{'tl':self.ops})['_partials_kernel']
        self.tiled=functions(ENGINE/'kernels/tiled_gemm.py',{'_tiled_partials_kernel'},{'tl':self.ops})['_tiled_partials_kernel']
    def run(self,x,weight,tiles=None):
        m,k=x.shape;n=weight.shape[0];s=old.split_count(n,k)
        out=torch.full((s,m,n),float('nan'))
        xp,wp,pp=Pointer(x),Pointer(weight if tiles is None else tiles),Pointer(out,writable=True)
        fn=self.original if tiles is None else self.tiled
        for col in range(n//32):
            for split in range(s):
                self.ops.program=(col,split)
                fn(xp,wp,pp,m,n,k,k//s,max(16,1<<(m-1).bit_length()),32,64)
        assert bool((wp.access['reads']==1).all()),'weight coverage'
        assert bool((pp.access['writes']==1).all()),'partial coverage'
        assert not bool(xp.access['writes'].any()) and not bool(wp.access['writes'].any())
        if tiles is not None:self.calls+=1
        return out


def meta(shape,dtype=torch.bfloat16,cuda=True):
    return SimpleNamespace(shape=shape,ndim=len(shape),dtype=dtype,is_cuda=cuda,
        device=torch.device('cuda' if cuda else 'cpu'),is_contiguous=lambda:True)


def norm(x,w,eps):
    f=x.float()
    return (f*torch.rsqrt(f.square().mean(-1,keepdim=True)+eps)).to(x.dtype)*w


class Norm(nn.Module):
    def __init__(self,width):
        super().__init__();self.weight=nn.Parameter(torch.ones(width));self.variance_epsilon=1e-6
    def forward(self,x):return norm(x,self.weight,self.variance_epsilon)


def tiny_fixture():
    model=nn.Module();base=model.model=nn.Module()
    base.embed_tokens=nn.Embedding(127,64);base.norm=Norm(64);base.layers=nn.ModuleList()
    mlp=load('layout_mlp',ENGINE/'mlp.py')
    for _ in range(2):
        layer=nn.Module();layer.input_layernorm=Norm(64);layer.post_attention_layernorm=Norm(64)
        a=layer.self_attn=nn.Module()
        a.q_proj,a.k_proj,a.v_proj=[nn.Linear(64,n*128,bias=False) for n in (4,2,2)]
        a.o_proj,a.q_norm,a.k_norm,a.scaling=nn.Linear(512,64,bias=False),Norm(128),Norm(128),128**-0.5
        layer.mlp=mlp.PackedMLP(SimpleNamespace(gate_proj=nn.Linear(64,128,bias=False),
            up_proj=nn.Linear(64,128,bias=False),down_proj=nn.Linear(128,64,bias=False),act_fn=nn.SiLU()))
        base.layers.append(layer)
    model.lm_head=nn.Linear(64,127,bias=False);model.lm_head.weight=base.embed_tokens.weight
    model.eval().bfloat16()
    for layer in base.layers:
        a=layer.self_attn;a.qkv_weight=torch.cat([p.weight for p in (a.q_proj,a.k_proj,a.v_proj)])
    return model


def fold(p):
    if p.ndim!=3:return p
    out=torch.zeros_like(p[0])
    for piece in p:out=out+piece
    return out.bfloat16()


def add_norm(x,p,w,eps):
    total=x+fold(p)
    return total,norm(total,w,eps)


def qkv_reference(p,qw,kw,qe,ke,cos,sin,pos,keys,values):
    p=fold(p);b,nk,_,d=keys.shape;nq=p.shape[-1]//d-2*nk
    q,k,v=p.split((nq*d,nk*d,nk*d),-1)
    q=norm(q.view(b,nq,1,d),qw,qe);k=norm(k.view(b,nk,1,d),kw,ke)
    c,s=cos.index_select(0,pos)[None,None],sin.index_select(0,pos)[None,None]
    def rotate(t):return t*c+torch.cat((-t[...,d//2:],t[...,:d//2]),-1)*s
    q,k=rotate(q),rotate(k)
    keys.index_copy_(2,pos,k);values.index_copy_(2,pos,v.view(b,nk,1,d))
    return q.contiguous()


def silu_reference(x,w):
    g,u=F.linear(x,w).chunk(2,-1)
    return F.silu(g)*u


def forward_with_bodies(body):
    return functions(ENGINE/'decode_step.py',{'linear_partials','_project','_mlp_hidden','fused_decode_forward'},
        {'torch':torch,'F':F,'supports':lambda *a,**k:True,'_row_major_partials':body.run,
         'linear_silu_mul':silu_reference,'add_rms_norm':add_norm,'qkv_norm_rope_cache':qkv_reference})['fused_decode_forward']


class LayoutCPU(unittest.TestCase):
    @torch.inference_mode()
    def test_pack_all_bf16_bit_patterns_and_tile_mapping(self):
        w=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).view(256,256)
        before=w.view(torch.int16).clone();tiles=layout.pack_weight(w)
        restored=tiles.permute(0,2,1,3).contiguous().view_as(w)
        torch.testing.assert_close(restored.view(torch.int16),before,rtol=0,atol=0)
        torch.testing.assert_close(w.view(torch.int16),before,rtol=0,atol=0)
        self.assertEqual(tiles.shape,(8,4,32,64));self.assertTrue(tiles.is_contiguous())
        for ni in range(8):
            for ki in range(4):
                torch.testing.assert_close(tiles[ni,ki].view(torch.int16),before[ni*32:(ni+1)*32,ki*64:(ki+1)*64],rtol=0,atol=0)
        for bad in (torch.empty(33,64,dtype=torch.bfloat16),torch.empty(32,65,dtype=torch.bfloat16),
                    torch.empty(0,64,dtype=torch.bfloat16),torch.empty(32,64),torch.empty(64,32,dtype=torch.bfloat16).T):
            with self.assertRaises(ValueError):layout.pack_weight(bad)

    @torch.inference_mode()
    def test_install_preserves_parameters_prefill_weights_and_state_dict(self):
        model=tiny_fixture();parameters=dict(model.named_parameters());state=deepcopy(model.state_dict())
        pointers={n:p.data_ptr() for n,p in parameters.items()};bytes_=layout.install_tiled_weights(model)
        self.assertEqual(set(model.state_dict()),set(state));self.assertIs(model.lm_head.weight,model.model.embed_tokens.weight)
        for n,p in model.named_parameters():
            self.assertIs(p,parameters[n]);self.assertEqual(p.data_ptr(),pointers[n]);torch.testing.assert_close(p,state[n],rtol=0,atol=0)
        buffers=[t for n,t in model.named_buffers() if n.endswith('_tiles')]
        self.assertEqual(len(buffers),6);self.assertEqual(bytes_,sum(t.numel()*2 for t in buffers))
        for layer in model.model.layers:
            a=layer.self_attn
            torch.testing.assert_close(a._qkv_weight_tiles.permute(0,2,1,3).contiguous().view_as(a.qkv_weight),a.qkv_weight,rtol=0,atol=0)
            self.assertFalse(hasattr(layer.mlp,'_weight_tiles'))
        self.assertEqual(layout.install_tiled_weights(model),bytes_)
        model.train()
        with self.assertRaises(ValueError):layout.install_tiled_weights(model)

    @torch.inference_mode()
    def test_actual_kernel_bitwise_partials_and_complete_weight_coverage(self):
        torch.manual_seed(210);body=Bodies()
        for m,n,k in ((1,64,2560),(3,64,4096),(16,64,9728),(17,96,256),(33,32,128),(64,64,64),(2,6144,64),(4,2560,64)):
            with self.subTest(m=m,n=n,k=k):
                x=torch.randn(m,k).bfloat16();w=(torch.randn(n,k)/k**0.5).bfloat16()
                tiles=layout.pack_weight(w);before=(x.clone(),w.clone(),tiles.clone())
                torch.testing.assert_close(body.run(x,w,tiles),body.run(x,w),rtol=0,atol=0)
                for a,b in zip((x,w,tiles),before):torch.testing.assert_close(a,b,rtol=0,atol=0)

    @torch.inference_mode()
    def test_wrong_layout_negative_control(self):
        x=torch.arange(128).remainder(13).view(1,128).bfloat16();w=torch.arange(64*128).remainder(31).view(64,128).bfloat16()
        good=layout.pack_weight(w);bad=w.view_as(good);body=Bodies()
        self.assertFalse(torch.equal(body.run(x,w,good),body.run(x,w,bad)))

    @torch.inference_mode()
    def test_launch_and_validation_keep_original_split_geometry(self):
        for m in (1,3,16,17,33,64):
            for n,k in ((6144,2560),(2560,4096),(2560,9728)):
                x,w,t=meta((m,k)),meta((n,k)),meta((n//32,k//64,32,64));marker=object();Launch.calls.clear()
                with mock.patch.object(torch,'empty',return_value=marker) as alloc:self.assertIs(new.tiled_partials(x,w,t),marker)
                s=old.split_count(n,k);alloc.assert_called_once_with((s,m,n),dtype=torch.float32,device=x.device)
                (grid,args,kw),=Launch.calls
                self.assertEqual(grid,(n//32,s));self.assertIs(args[1],t);self.assertEqual(args[-1],k//s)
                self.assertEqual(kw,dict(BM=max(16,1<<(m-1).bit_length()),BN=32,BK=64,num_warps=4,num_stages=4))
        for x,w,t in ((meta((65,64)),meta((64,64)),meta((2,1,32,64))),
                      (meta((1,64)),meta((64,64)),meta((2,2,32,64))),
                      (meta((1,64)),meta((64,64)),meta((2,1,32,64),torch.float32)),
                      (meta((1,64)),meta((0,64)),meta((0,1,32,64)))):
            with self.assertRaises(ValueError):new.tiled_partials(x,w,t)

    @torch.inference_mode()
    def test_dispatch_fallback_and_selected_errors_propagate(self):
        original=mock.Mock(return_value='row-major')
        env=functions(ENGINE/'decode_step.py',{'linear_partials','_project'},
            {'F':F,'supports':lambda *a,**k:True,'_row_major_partials':original})
        x,w=torch.randn(2,64).bfloat16(),torch.randn(32,64).bfloat16();tiles=layout.pack_weight(w)
        self.assertEqual(env['_project'](x,w),'row-major');original.assert_called_once_with(x,w)
        called=mock.Mock(return_value='tiled')
        with mock.patch.dict(sys.modules,{'kernels.tiled_gemm':SimpleNamespace(tiled_partials=called)}):
            self.assertEqual(env['_project'](x,w,tiles),'tiled');called.assert_called_once_with(x,w,tiles)
            called.side_effect=RuntimeError('injected selected failure')
            with self.assertRaisesRegex(RuntimeError,'injected selected failure'):env['_project'](x,w,tiles)
        self.assertEqual(original.call_count,1);env['supports']=lambda *a,**k:False
        with mock.patch.dict(sys.modules,{'kernels.tiled_gemm':None}):
            torch.testing.assert_close(env['_project'](x,w,tiles),F.linear(x,w),rtol=0,atol=0)

    @torch.inference_mode()
    def test_actual_capture_prepares_before_steps_once_and_skips_large_rows(self):
        # Real capture BODY, but synthetic streams/graphs: NOT a CUDA test.
        class Stream:
            def wait_stream(self, other): pass
        cuda = SimpleNamespace(current_stream=Stream, Stream=Stream,
            stream=lambda stream: nullcontext(), CUDAGraph=object,
            graph=lambda *a, **k: nullcontext())
        capture = functions(ENGINE/'decode.py', {'capture'},
                            {'torch': SimpleNamespace(cuda=cuda)})['capture']
        for rows, fused, expected_packs in ((3, object(), 1), (65, object(), 0), (3, None, 0)):
            model = tiny_fixture()
            state = SimpleNamespace(model=model, shape=(rows, 7, 5), fused_forward=fused,
                tokens=torch.zeros(rows, 1, dtype=torch.int64), position=torch.tensor([7]))
            def step():
                if expected_packs:
                    self.assertTrue(model._decode_tiles_ready)
                    self.assertIsNotNone(model.model.layers[0].self_attn._qkv_weight_tiles)
                state.tokens.add_(1); state.position.add_(1)
            state.step = step
            with mock.patch.dict(sys.modules, {'weight_layout': layout}), \
                    mock.patch.object(layout, 'install_tiled_weights', wraps=layout.install_tiled_weights) as installed:
                capture(state); capture(state)
                self.assertEqual(installed.call_count, expected_packs)
            self.assertEqual(int(state.position), 7)
            self.assertEqual(int(state.tokens.sum()), 0)

    @torch.inference_mode()
    def test_actual_decode_forward_matches_row_major_and_cache_reuse(self):
        torch.manual_seed(211);row_model=tiny_fixture();tile_model=deepcopy(row_model)
        layout.install_tiled_weights(tile_model);body=Bodies();forward=forward_with_bodies(body)
        flash=load('layout_flash',ENGINE/'flash_decode.py')
        for batch in (1,3):
            cap=9;freq=torch.arange(cap).float()[:,None]*torch.arange(64).float()[None,:]/200
            phase=torch.cat((freq,freq),1);cos,sin=phase.cos().bfloat16(),phase.sin().bfloat16()
            caches=[SimpleNamespace(key_cache=[torch.empty(batch,2,cap,128,dtype=torch.bfloat16) for _ in range(2)],
                value_cache=[torch.empty(batch,2,cap,128,dtype=torch.bfloat16) for _ in range(2)]) for _ in range(2)]
            ctx=flash.FlashDecodeContext(batch,4,2,torch.arange(cap))
            ctx.attention=lambda q,k,v,mask,scale:F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=mask,scale=scale).transpose(1,2).contiguous()
            for repeat in range(2):
                for ca in caches:
                    for t in ca.key_cache+ca.value_cache:t.fill_(37+repeat)
                ptrs=[[t.data_ptr() for t in ca.key_cache+ca.value_cache] for ca in caches]
                for pos in range(4):
                    token=torch.randint(0,127,(batch,1));position=torch.tensor([pos]);mask=ctx.prepare(position)
                    expected=forward(row_model,caches[0],token,position,mask,ctx,cos,sin)
                    with mock.patch.dict(sys.modules,{'kernels.tiled_gemm':SimpleNamespace(tiled_partials=body.run)}):
                        actual=forward(tile_model,caches[1],token,position,mask,ctx,cos,sin)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    for a,b in zip(caches[0].key_cache+caches[0].value_cache,caches[1].key_cache+caches[1].value_cache):
                        torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertTrue(bool((a[:,:,pos+1:]==37+repeat).all()))
                self.assertEqual(ptrs,[[t.data_ptr() for t in ca.key_cache+ca.value_cache] for ca in caches])
        self.assertEqual(body.calls,16*6)


@contextmanager
def module_path():
    sys.path.insert(0,str(ENGINE))
    try:yield
    finally:sys.path.remove(str(ENGINE))


def tiny_hf():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    cfg=Qwen3Config(vocab_size=127,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=128,max_position_embeddings=1024,
        rope_theta=5_000_000,tie_word_embeddings=True,sliding_window=None,eos_token_id=0)
    cfg._attn_implementation='sdpa'
    return Qwen3ForCausalLM(cfg).eval().bfloat16()


class TransformersCPU(unittest.TestCase):
    @unittest.skipUnless(HAS_TRANSFORMERS,'Transformers required; fixtures are not native model validation')
    @torch.inference_mode()
    def test_real_cpu_own_prefix_and_streaming(self):
        torch.manual_seed(212);native=tiny_hf();candidate=deepcopy(native)
        mlp=load('layout_hf_mlp',ENGINE/'mlp.py');state_module=load('layout_hf_decode',ENGINE/'decode.py')
        flash=load('layout_hf_flash',ENGINE/'flash_decode.py')
        for layer in candidate.model.layers:
            a=layer.self_attn;a.qkv_weight=torch.cat([p.weight for p in (a.q_proj,a.k_proj,a.v_proj)])
            layer.mlp=mlp.PackedMLP(layer.mlp)
        layout.install_tiled_weights(candidate);body=Bodies();forward=forward_with_bodies(body)
        class Context(flash.FlashDecodeContext):
            def attention(self,q,k,v,mask,scale):
                assert mask is self.mask
                return F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=mask,scale=scale).transpose(1,2).contiguous()
        for b,p,n in ((1,1,5),(3,7,6),(2,5,1)):
            state=state_module.DecodeState(candidate,b,p,n);state.fused_forward=forward
            state.flash_context=Context(b,4,2,state.key_positions)
            for repeat in range(2):
                for t in state.cache.key_cache+state.cache.value_cache:t.fill_(19+repeat)
                ids=torch.randint(0,127,(b,p));state.prefill(ids)
                with mock.patch.dict(sys.modules,{'kernels.tiled_gemm':SimpleNamespace(tiled_partials=body.run)}):emitted=list(state.emit(n))
                self.assertEqual(len(emitted),n);self.assertTrue(all(len(row)==b and all(type(t) is int for t in row) for row in emitted))
                selected=torch.tensor(emitted).T
                logits=native(input_ids=torch.cat((ids,selected[:,:-1]),1),use_cache=False).logits[:,p-1:].float()
                gap=logits.amax(-1)-logits.gather(-1,selected[...,None]).squeeze(-1)
                self.assertLessEqual(float(gap.max()),2.0);self.assertEqual(int(state.position),p+n-1)
        self.assertGreater(body.calls,0)


@unittest.skipUnless(torch.cuda.is_available() and HAS_TRITON,'CUDA/Triton required; CPU results are not GPU validation')
class TiledCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import triton
        if (sys.version_info[:2]!=(3,11) or torch.__version__.split('+')[0]!='2.5.1'
                or triton.__version__!='3.1.0' or torch.version.cuda!='12.4'):
            raise RuntimeError('Run CUDA acceptance checks in the pinned runtime')
        torch.backends.cuda.matmul.allow_tf32=False

    @torch.inference_mode()
    def test_real_matrix_shapes_partials_and_changed_input_graph_replay(self):
        with module_path():
            from kernels.tiled_gemm import tiled_partials
            from kernels.skinny_gemm import linear_partials
        torch.manual_seed(213)
        for n,k in ((6144,2560),(2560,4096),(2560,9728)):
            weight=(torch.randn(n,k,device='cuda')/k**0.5).bfloat16();tiles=layout.pack_weight(weight);before=weight.clone()
            for m in (1,3,16,33,64):
                x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                actual=tiled_partials(x,weight,tiles);expected=linear_partials(x,weight)
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6);self.assertEqual(actual.dtype,torch.float32)
            x=torch.randn(3,k,device='cuda',dtype=torch.bfloat16)
            stream=torch.cuda.Stream();current=torch.cuda.current_stream();stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):tiled_partials(x,weight,tiles)
            current.wait_stream(stream);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):actual=tiled_partials(x,weight,tiles)
            current.wait_stream(stream)
            for _ in range(3):
                x.normal_();graph.replay();torch.testing.assert_close(actual,linear_partials(x,weight),rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(weight,before,rtol=0,atol=0)

    @unittest.skipUnless(HAS_TRANSFORMERS,'Transformers required')
    @torch.inference_mode()
    def test_real_engine_dispatch_fresh_prompts_own_prefix_eos_and_zero(self):
        import transformers
        self.assertEqual(transformers.__version__,'4.51.3')
        with module_path():
            from engine import Engine
            from kernels import tiled_gemm
        torch.manual_seed(214);native=tiny_hf()
        with tempfile.TemporaryDirectory() as path:
            native.save_pretrained(path);engine=Engine(path);native=native.cuda()
            for b,p,n in ((1,7,5),(1,7,5),(3,13,6),(2,1,2),(2,7,1),(1,7,0),(65,3,2)):
                if engine.decode_state is not None:
                    for t in engine.decode_state.cache.key_cache+engine.decode_state.cache.value_cache:t.fill_(33)
                ids=torch.randint(0,127,(b,p),device='cuda')
                with mock.patch.object(tiled_gemm,'tiled_partials',wraps=tiled_gemm.tiled_partials) as used:
                    emitted=list(engine.generate(ids.tolist(),n))
                    if n>1:
                        engine.decode_state.prefill(ids);engine.decode_state.step()
                        if b<=64:self.assertGreater(used.call_count,0)
                        else:self.assertEqual(used.call_count,0)
                self.assertEqual(len(emitted),n);self.assertTrue(all(len(row)==b and all(type(t) is int for t in row) for row in emitted))
                if n:
                    selected=torch.tensor(emitted,device='cuda').T
                    logits=native(input_ids=torch.cat((ids,selected[:,:-1]),1),use_cache=False).logits[:,p-1:].float()
                    gap=logits.amax(-1)-logits.gather(-1,selected[...,None]).squeeze(-1)
                    self.assertLessEqual(float(gap.max()),2.0)
                self.assertIs(engine.model.lm_head.weight,engine.model.model.embed_tokens.weight)
            engine.model.lm_head.weight.zero_()  # Only untiled tied embedding/head changes.
            self.assertEqual(list(engine.generate([[3,4]],5)),[[0]]*5)


if __name__=='__main__':unittest.main()
