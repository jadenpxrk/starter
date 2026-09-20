"""cuBLASLt dispatch checks. Mocked CPU dispatch is NOT CUDA validation."""

import ast
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
sys.path.insert(0, str(ENGINE))
import blas_backend as blas


class Preference:
    """CPU-only stand-in for the process-wide preference API."""
    def __init__(self):
        self.initial = object()
        self.current = self.initial
        self.calls = []

    def __call__(self, backend=None):
        self.calls.append(backend)
        if backend is not None:
            self.current = backend
        return self.current


class BlasCPU(unittest.TestCase):
    def test_workspace_units_and_no_precision_changes(self):
        names = ("allow_tf32", "allow_bf16_reduced_precision_reduction",
                 "allow_fp16_reduced_precision_reduction")
        before = [getattr(torch.backends.cuda.matmul, name) for name in names]
        with mock.patch.dict(os.environ, {"CUBLASLT_WORKSPACE_SIZE": "1024",
                                         "BLAS_TEST_SENTINEL": "preserve"}), \
                mock.patch.object(torch.backends.cuda, "preferred_blas_library") as api:
            blas.configure_workspace()
            self.assertEqual(os.environ["CUBLASLT_WORKSPACE_SIZE"], "32768")
            self.assertEqual(blas.WORKSPACE_KIB * 1024, 32 * 1024**2)
            self.assertEqual(os.environ["BLAS_TEST_SENTINEL"], "preserve")
            api.assert_not_called()
        self.assertEqual(before, [getattr(torch.backends.cuda.matmul, name) for name in names])

    @torch.inference_mode()
    def test_actual_eligibility_metadata_and_cpu_fallback(self):
        x = SimpleNamespace(is_cuda=True, ndim=2, device=torch.device("cuda:0"),
                            dtype=torch.bfloat16)
        w = SimpleNamespace(ndim=2, device=x.device, dtype=x.dtype)
        self.assertTrue(blas._eligible(x, w))
        for field, value in (("ndim", 3), ("dtype", torch.float32), ("is_cuda", False)):
            old = getattr(x, field)
            setattr(x, field, value)
            self.assertFalse(blas._eligible(x, w))
            setattr(x, field, old)
        w.device = torch.device("cuda:1")
        self.assertFalse(blas._eligible(x, w))
        w.device = x.device
        with torch.enable_grad():
            self.assertFalse(blas._eligible(x, w))
        torch.manual_seed(14)
        with mock.patch.object(torch.backends.cuda, "preferred_blas_library",
                               side_effect=AssertionError("CPU must not select CUDA BLAS")):
            for dtype in (torch.float32, torch.bfloat16):
                for rows in (1, 3, 33):
                    inp = torch.randn(rows, 130).to(dtype)[:, ::2]
                    weight = torch.randn(39, 65).to(dtype)
                    old = [t.clone() for t in (inp, weight)]
                    result = blas.cublaslt_linear(inp, weight)
                    torch.testing.assert_close(result, F.linear(inp, weight), rtol=0, atol=0)
                    self.assertEqual(result.dtype, dtype)
                    for t, saved in zip((inp, weight), old):
                        torch.testing.assert_close(t, saved, rtol=0, atol=0)

    @torch.inference_mode()
    def test_scoped_dispatch_and_failure_restoration_without_gpu(self):
        torch.manual_seed(7)
        x, w = torch.randn(3, 17).bfloat16(), torch.randn(29, 17).bfloat16()
        ref = F.linear(x, w)
        native = F.linear
        pref = Preference()
        addresses = (x.data_ptr(), w.data_ptr())
        def linear(inp, weight):
            self.assertEqual(pref.current, "cublaslt")
            self.assertIs(inp, x)
            self.assertIs(weight, w)
            return native(inp, weight)
        # Explicitly force routing with CPU tensors and a MOCK preference API.
        # This validates Python control/data flow, not the vendor implementation.
        with mock.patch.object(blas, "_eligible", return_value=True), \
                mock.patch.object(torch.backends.cuda, "preferred_blas_library", side_effect=pref), \
                mock.patch.object(blas.F, "linear", side_effect=linear) as call:
            actual = blas.cublaslt_linear(x, w)
            call.assert_called_once_with(x, w)
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
        self.assertEqual(addresses, (x.data_ptr(), w.data_ptr()))
        self.assertIs(pref.current, pref.initial)
        self.assertEqual(pref.calls, [None, "cublaslt", pref.initial])
        pref.calls.clear()
        failure = RuntimeError("synthetic GEMM failure")
        with mock.patch.object(blas, "_eligible", return_value=True), \
                mock.patch.object(torch.backends.cuda, "preferred_blas_library", side_effect=pref), \
                mock.patch.object(blas.F, "linear", side_effect=failure) as call:
            with self.assertRaisesRegex(RuntimeError, "synthetic GEMM failure"):
                blas.cublaslt_linear(x, w)
            self.assertEqual(call.call_count, 1)  # No retry with another vendor.
        self.assertIs(pref.current, pref.initial)

    @torch.inference_mode()
    def test_unavailable_backend_is_not_silently_retried(self):
        pref = Preference()
        def unavailable(backend=None):
            if backend == "cublaslt":
                raise RuntimeError("synthetic unsupported backend")
            return pref(backend)
        with mock.patch.object(blas, "_eligible", return_value=True), \
                mock.patch.object(torch.backends.cuda, "preferred_blas_library", side_effect=unavailable), \
                mock.patch.object(blas.F, "linear") as call:
            with self.assertRaisesRegex(RuntimeError, "synthetic unsupported backend"):
                blas.cublaslt_linear(torch.empty(1, 1), torch.empty(1, 1))
            call.assert_not_called()
        self.assertIs(pref.current, pref.initial)

    def test_workspace_configuration_precedes_checkpoint_loading(self):
        # Execute the actual constructor up to a deliberately failing loader;
        # no CUDA, model allocation, or mock Transformer execution is claimed.
        tree = ast.parse((ENGINE / "engine.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Engine")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        events = []
        fake_torch = SimpleNamespace(backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True)), bfloat16=torch.bfloat16)
        def config():
            self.assertFalse(fake_torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(fake_torch.backends.cudnn.allow_tf32)
            events.append("workspace")
        def load(*args, **kwargs):
            events.append("load")
            self.assertEqual(args, ("checkpoint",))
            self.assertTrue(kwargs["local_files_only"])
            raise RuntimeError("stop before GPU allocation")
        namespace = dict(torch=fake_torch, configure_workspace=config,
                         AutoModelForCausalLM=SimpleNamespace(from_pretrained=load))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[init], type_ignores=[])),
                     "engine-constructor-check", "exec"), namespace)
        with self.assertRaisesRegex(RuntimeError, "stop before GPU allocation"):
            namespace["__init__"](SimpleNamespace(), "checkpoint")
        self.assertEqual(events, ["workspace", "load"])

    @torch.inference_mode()
    def test_actual_prefill_and_decode_route_every_projection(self):
        # Existing equation fixtures; actual forward bodies and actual wrapper.
        # Arithmetic is CPU Torch; all CUDA preference calls are mocked.
        import test_prefill as fixture
        pref = Preference()
        original_linear = F.linear
        selected = []
        def linear(x, w):
            selected.append((pref.current, tuple(x.shape), tuple(w.shape)))
            self.assertEqual(pref.current, "cublaslt")
            return original_linear(x, w)
        torch.manual_seed(37)
        model, cache, (cos, sin) = fixture.fixture(2, 7)
        ids = torch.randint(0, 127, (2, 7))
        parameters = {name: (id(p), p.data_ptr(), p.clone()) for name, p in model.named_parameters()}
        ref_cache = deepcopy(cache)
        expected = fixture.materialized(model, ref_cache, ids, cos, sin)
        with fixture.cpu_kernels(fixture.reference_qkv), \
                mock.patch.object(blas, "_eligible", return_value=True), \
                mock.patch.object(torch.backends.cuda, "preferred_blas_library", side_effect=pref), \
                mock.patch.object(blas.F, "linear", side_effect=linear):
            actual = fixture.prefill.prefill_forward(model, cache, ids, cos, sin)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(selected), 4 * len(model.model.layers) + 1)
        self.assertEqual(selected[-1][1][0], 2)  # Last position only, not B*T.
        self.assertIs(pref.current, pref.initial)

        def qkv(p, qw, kw, qe, ke, cos, sin, position, keys, values):
            batch, hk, _, d = keys.shape
            hq = p.shape[1] // d - 2 * hk
            q, k, v = p.split((hq*d, hk*d, hk*d), -1)
            q = fixture.norm(q.view(batch, hq, 1, d), qw, qe)
            k = fixture.norm(k.view(batch, hk, 1, d), kw, ke)
            pos = int(position)
            def rotate(z):
                half = torch.cat((-z[..., d//2:], z[..., :d//2]), -1)
                return z*cos[pos] + half*sin[pos]
            q, k = rotate(q), rotate(k)
            keys[:, :, pos:pos+1].copy_(k)
            values[:, :, pos:pos+1].copy_(v.view(batch, hk, 1, d))
            return q.contiguous()
        kernels = SimpleNamespace(add_rms_norm=fixture.add_norm,
                                  silu_mul=fixture.silu_mul, qkv_norm_rope_cache=qkv)
        spec = importlib.util.spec_from_file_location("lt_decode_fixture", ENGINE / "decode_step.py")
        step = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"kernels.decode_fused": kernels}):
            spec.loader.exec_module(step)
        pos = torch.tensor([7])
        mask = (torch.arange(cache.max_cache_len) <= pos).view(1, 1, 1, -1)
        def attention(q, k, v, mask, scale):
            groups = q.shape[1] // k.shape[1]
            return F.scaled_dot_product_attention(q, k.repeat_interleave(groups, 1),
                v.repeat_interleave(groups, 1), attn_mask=mask, scale=scale).transpose(1, 2).contiguous()
        context = SimpleNamespace(attention=attention)
        token = actual[:, -1].argmax(-1, keepdim=True)
        baseline_cache = deepcopy(cache)
        expected = step.fused_decode_forward(model, baseline_cache, token, pos, mask, context, cos, sin)
        selected.clear()
        with mock.patch.object(blas, "_eligible", return_value=True), \
                mock.patch.object(torch.backends.cuda, "preferred_blas_library", side_effect=pref), \
                mock.patch.object(blas.F, "linear", side_effect=linear):
            actual = step.fused_decode_forward(model, cache, token, pos, mask, context, cos, sin)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(selected), 4 * len(model.model.layers) + 1)
        self.assertIs(pref.current, pref.initial)
        for a, b in zip(cache.key_cache + cache.value_cache,
                        baseline_cache.key_cache + baseline_cache.value_cache):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for name, p in model.named_parameters():
            old_id, ptr, data = parameters[name]
            self.assertEqual((id(p), p.data_ptr()), (old_id, ptr))
            torch.testing.assert_close(p, data, rtol=0, atol=0)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)


@contextmanager
def vendor(name):
    old = torch.backends.cuda.preferred_blas_library()
    try:
        torch.backends.cuda.preferred_blas_library(name)
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(old)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required; mocked CPU calls are not GPU validation")
class BlasCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Must precede the first Lt GEMM in the test process, just as in Engine.
        cls.old_workspace = os.environ.get("CUBLASLT_WORKSPACE_SIZE")
        cls.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        blas.configure_workspace()

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.old_tf32
        if cls.old_workspace is None:
            os.environ.pop("CUBLASLT_WORKSPACE_SIZE", None)
        else:
            os.environ["CUBLASLT_WORKSPACE_SIZE"] = cls.old_workspace
        # The pinned C++ workspace-size cache cannot be reset from Python.

    @torch.inference_mode()
    def test_actual_projection_dimensions_and_general_rows(self):
        torch.manual_seed(43)
        # All five distinct packed projection shapes, including the tied head.
        for n, k in ((6144, 2560), (2560, 4096), (19456, 2560),
                     (2560, 9728), (151936, 2560)):
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
            saved_w = w.clone()
            for m in (1, 3, 16, 33, 257):
                with self.subTest(m=m, n=n, k=k):
                    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                    saved_x = x.clone()
                    with vendor("cublas"):
                        expected = F.linear(x, w)
                        actual = blas.cublaslt_linear(x, w)
                        self.assertEqual(torch.backends.cuda.preferred_blas_library(),
                                         torch._C._BlasBackend.Cublas)
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    self.assertEqual(actual.shape, (m, n))
                    self.assertTrue(bool(torch.isfinite(actual).all()))
                    # Operator diagnostic only, NOT the official 2.0-logit gate.
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.03)
                    torch.testing.assert_close(x, saved_x, rtol=0, atol=0)
            torch.testing.assert_close(w, saved_w, rtol=0, atol=0)

    @torch.inference_mode()
    def test_lt_capture_replay_with_changed_inputs(self):
        x = torch.randn(3, 2560, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(6144, 2560, device="cuda", dtype=torch.bfloat16) / 2560**0.5
        with vendor("cublas"):
            current = torch.cuda.current_stream()
            stream = torch.cuda.Stream()
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    blas.cublaslt_linear(x, w)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = blas.cublaslt_linear(x, w)
            current.wait_stream(stream)
            pointer = result.data_ptr()
            for _ in range(4):
                x.normal_()
                graph.replay()
                # Host preference was restored to cuBLAS; captured Lt work
                # still executes. This comparison does not profile its kernel.
                expected = F.linear(x, w)
                torch.testing.assert_close(result, expected, rtol=0.02, atol=0.03)
                self.assertEqual(result.data_ptr(), pointer)

    @torch.inference_mode()
    def test_engine_graphs_reuse_and_native_own_prefix(self):
        if (importlib.util.find_spec("transformers") is None
                or importlib.util.find_spec("triton") is None):
            self.skipTest("Transformers and Triton required for engine integration")
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from engine import Engine
        torch.manual_seed(61)
        cfg = Qwen3Config(vocab_size=127, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=128, max_position_embeddings=32768, rope_theta=5_000_000,
            tie_word_embeddings=True, sliding_window=None, bos_token_id=125, eos_token_id=0)
        cfg._attn_implementation = "sdpa"
        native = Qwen3ForCausalLM(cfg).eval()
        with tempfile.TemporaryDirectory() as path, vendor("cublas"):
            native.save_pretrained(path)
            engine = Engine(path)
            native = native.cuda().bfloat16()
            for b, t, n in ((3, 7, 5), (3, 7, 5), (1, 1, 3), (2, 13, 1), (2, 13, 0)):
                with self.subTest(batch=b, prompt=t, output=n):
                    fresh = (engine.decode_state is None
                             or engine.decode_state.shape != (b, t, n))
                    previous_pointers = None
                    if not fresh:
                        buffers = (engine.decode_state.cache.key_cache
                                   + engine.decode_state.cache.value_cache)
                        previous_pointers = [z.data_ptr() for z in buffers]
                        for z in buffers:
                            z.fill_(float("nan"))
                    prompt = torch.randint(0, 127, (b, t), device="cuda")
                    prompt[:, -1] = cfg.eos_token_id  # EOS is an ordinary input ID.
                    # Count real wrapper invocations during fresh captures;
                    # replays correctly make no Python projection calls.
                    real_api = torch.backends.cuda.preferred_blas_library
                    with mock.patch.object(torch.backends.cuda, "preferred_blas_library", wraps=real_api) as api:
                        emitted = list(engine.generate(prompt.tolist(), n))
                    self.assertEqual(len(emitted), n)
                    self.assertTrue(all(len(row) == b and all(type(x) is int for x in row)
                                        for row in emitted))
                    lt_calls = sum(call.args == ("cublaslt",) for call in api.call_args_list)
                    if not n:
                        self.assertEqual(lt_calls, 0)
                        continue
                    if fresh and (t > 1 or n > 1):
                        self.assertGreater(lt_calls, 0)
                    ids = torch.tensor(emitted, device="cuda").T
                    own = torch.cat((prompt, ids[:, :-1]), 1)
                    # Untouched reference model, explicitly under original cuBLAS.
                    logits = native(input_ids=own, use_cache=False).logits[:, t-1:].float()
                    gap = logits.amax(-1) - logits.gather(-1, ids[..., None]).squeeze(-1)
                    self.assertTrue(bool(torch.isfinite(gap).all()))
                    self.assertLessEqual(float(gap.max()), 2.0)
                    state = engine.decode_state
                    if t > 1:
                        self.assertIsNotNone(state.prefill_plan)
                        self.assertIsNotNone(state.prefill_plan.graph)
                    if n > 1:
                        self.assertIsNotNone(state.fused_forward)
                        self.assertIsNotNone(state.graph)
                    self.assertEqual(int(state.position), t + n - 1)
                    self.assertIs(engine.model.lm_head.weight, engine.model.model.embed_tokens.weight)
                    if previous_pointers is not None:
                        self.assertEqual(previous_pointers, [z.data_ptr() for z in
                            state.cache.key_cache + state.cache.value_cache])
            # Force argmax 0 == EOS by zeroing the live tied head. Generation
            # must still emit N lists; this also exercises live weight reads.
            engine.model.lm_head.weight.zero_()
            self.assertEqual(list(engine.generate([[0, 0], [0, 0]], 4)), [[0, 0]] * 4)


if __name__ == "__main__":
    unittest.main()
