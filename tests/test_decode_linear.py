"""CPU integration guards and optional pinned-CUDA tests for dense decode.

Run this file on CPU without importing Triton; skipped CUDA tests are NOT
validation. Full-model H100 authority remains the official own-prefix replay.
"""

from copy import deepcopy
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
from decode_linear import DecodeLinear, DecodeLinearPlan, install_decode_linears, split_k_for, wins
from mlp import PackedMLP


def fake_model(dtype):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(16, 32, bias=False)
            self.up_proj = nn.Linear(16, 32, bias=False)
            self.down_proj = nn.Linear(32, 16, bias=False)
            self.act_fn = nn.SiLU()

    model = nn.Module()
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(37, 16)
    model.lm_head = nn.Linear(16, 37, bias=False)
    model.lm_head.weight = model.model.embed_tokens.weight
    layers = []
    for _ in range(2):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, name, nn.Linear(16, 16, bias=False))
        layer.mlp = MLP()
        layers.append(layer)
    model.model.layers = nn.ModuleList(layers)
    model.eval().to(dtype=dtype)
    for layer in layers:
        layer.mlp = PackedMLP(layer.mlp)
    return model


class DenseCPU(unittest.TestCase):
    @torch.inference_mode()
    def test_install_reuses_all_parameters_and_tied_head(self):
        for dtype in (torch.float32, torch.bfloat16):
            model = fake_model(dtype)
            parameters = dict(model.named_parameters())
            shapes = {n: p.shape for n, p in model.state_dict().items()}
            plan = install_decode_linears(model)
            self.assertEqual(shapes, {n: p.shape for n, p in model.state_dict().items()})
            for name, parameter in parameters.items():
                self.assertIs(dict(model.named_parameters())[name], parameter)
            self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
            self.assertIs(model.decode_linear_plan, plan)
            self.assertTrue(all(layer.mlp.linear_plan is plan for layer in model.model.layers))

    @torch.inference_mode()
    def test_cpu_fallback_packed_mlp_and_native_outputs_are_exact(self):
        torch.manual_seed(43)
        with mock.patch.dict(sys.modules, {"kernels.decode_linear": None}):
            for dtype in (torch.float32, torch.bfloat16):
                model = fake_model(dtype)
                reference = deepcopy(model)
                plan = install_decode_linears(model)
                for policy in ("torch", "simt", "tensorcore"):
                    plan.active, plan.policy = True, policy
                    for shape in ((1, 1, 16), (3, 1, 16), (2, 7, 16), (35, 1, 16)):
                        x = torch.randn(shape, dtype=dtype)
                        for actual, native in zip(model.model.layers, reference.model.layers):
                            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                                torch.testing.assert_close(getattr(actual.self_attn, name)(x),
                                                           getattr(native.self_attn, name)(x), rtol=0, atol=0)
                            torch.testing.assert_close(actual.mlp(x), native.mlp(x), rtol=0, atol=0)
                        torch.testing.assert_close(model.lm_head(x), reference.lm_head(x), rtol=0, atol=0)

    @torch.inference_mode()
    def test_bias_and_noncontiguous_fallbacks(self):
        reference = nn.Linear(16, 19).eval()
        plan = DecodeLinearPlan()
        plan.active, plan.policy = True, "tensorcore"
        actual = DecodeLinear(reference, plan)
        self.assertIs(actual.bias, reference.bias)
        x = torch.randn(3, 1, 32)[..., ::2]
        torch.testing.assert_close(actual(x), reference(x), rtol=0, atol=0)

    def test_shape_planner_and_conservative_selection(self):
        for m in (1, 3, 16, 17, 32):
            for n in (7, 1024, 2560, 19456, 151936):
                for k in (3, 257, 2560, 9728):
                    for sms in (1, 80, 132, 144):
                        self.assertIn(split_k_for(m, n, k, sms), (1, 2, 4, 8, 16))
        # Synthetic fixtures for the decision rule, NOT GPU measurements.
        self.assertTrue(wins([10., 11., 10.], [8., 8., 8.]))
        self.assertFalse(wins([10., 10., 10.], [9.5, 9.5, 9.5]))
        self.assertFalse(wins([10., 10., 7.], [8., 8., 8.]))
        self.assertFalse(wins([10., 10., 10.], [8., float("nan"), 8.]))
        self.assertFalse(wins([10.], [8.]))


    @torch.inference_mode()
    def test_real_selector_control_flow_with_mock_gpu_clock(self):
        # Execute the actual selector, but with CPU state and a synthetic clock.
        # The clock values are fixtures, not measured GPU performance.
        for reject_simt in (False, True):
            plan = DecodeLinearPlan()
            clock, graphs = [0.0], []
            class Graph:
                def __init__(self, state, policy):
                    self.state, self.policy, self.deleted = state, policy, False
                    graphs.append(self)
                def replay(self):
                    if self.deleted:
                        raise AssertionError("replayed a deleted graph")
                    if self.state.tokens.item() != 42 or self.state.position.item() != 7:
                        raise AssertionError("each replay must restore its original inputs")
                    self.state.tokens.add_(1)
                    self.state.position.add_(1)
                    clock[0] += {"torch": 10., "simt": 8., "tensorcore": 8.5}[self.policy]
                def reset(self):
                    self.deleted = True
            class Event:
                def __init__(self, **kwargs):
                    self.at = None
                def record(self):
                    self.at = clock[0]
                def synchronize(self):
                    pass
                def elapsed_time(self, other):
                    return other.at - self.at
            class State:
                shape = (1, 7, 5)
                linear_policy = "torch"
                def __init__(self):
                    self.tokens = torch.tensor([[42]])
                    self.position = torch.tensor([7])
                def _capture_graph(self, tokens, position):
                    graph = Graph(self, self.linear_policy)
                    graph.replay()
                    self.tokens.copy_(tokens); self.position.copy_(position)
                    return graph
                def step(self):
                    self.tokens.add_(1); self.position.add_(1)
                    if reject_simt and self.linear_policy == "simt":
                        raise AssertionError("synthetic arithmetic rejection")
                    plan.audit_count = 13
            state = State()
            first, pos = state.tokens.clone(), state.position.clone()
            with mock.patch("decode_linear.torch.cuda.Event", Event), mock.patch.object(plan, "_log"), \
                    mock.patch("decode_linear.time.monotonic", return_value=0.):
                graph, policy = plan.capture(state, first, pos)
            self.assertEqual(policy, "tensorcore" if reject_simt else "simt")
            self.assertEqual(state.linear_policy, policy)
            torch.testing.assert_close(state.tokens, first)
            torch.testing.assert_close(state.position, pos)
            self.assertFalse(plan.audit)
            self.assertFalse(graph.deleted)
            self.assertTrue(all(g.deleted for g in graphs if g is not graph))

    @torch.inference_mode()
    def test_actual_step_restores_policy_after_forward_exception(self):
        # Run the authored DecodeState.step body with a stub forward on CPU.
        # This tests exception/state handling, not Transformers or GPU compute.
        path = Path(__file__).resolve().parents[1] / "engine/decode.py"
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "DecodeState")
        plan = DecodeLinearPlan()
        plan.policy = "previous"
        def forward(*args, **kwargs):
            self.assertTrue(plan.active)
            self.assertEqual(plan.policy, "simt")
            raise RuntimeError("injected forward failure")
        ns = {"torch": torch, "qwen_forward": forward}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), ns)
        state = ns["DecodeState"].__new__(ns["DecodeState"])
        state.model = SimpleNamespace(decode_linear_plan=plan)
        state.tokens, state.position = torch.tensor([[3]]), torch.tensor([7])
        state.key_positions, state.cache = torch.arange(12), None
        state.linear_policy = "simt"
        with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
            state.step()
        self.assertFalse(plan.active)
        self.assertEqual(plan.policy, "previous")
        self.assertEqual(state.tokens.item(), 3)
        self.assertEqual(state.position.item(), 7)


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
                     "CUDA and Triton required; CPU is not H100 validation")
class DenseCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import triton
        import transformers
        if (sys.version_info[:2] != (3, 11) or torch.__version__.split("+")[0] != "2.5.1"
                or torch.version.cuda != "12.4" or triton.__version__ != "3.1.0"
                or transformers.__version__ != "4.51.3"):
            raise RuntimeError("Run GPU acceptance checks in the pinned runtime")
        torch.backends.cuda.matmul.allow_tf32 = False
        from kernels.decode_linear import small_linear
        cls.kernel = staticmethod(small_linear)

    @torch.inference_mode()
    def test_kernels_actual_projection_dimensions_and_tail_masks(self):
        torch.manual_seed(103)
        dimensions = ((4096, 2560), (1024, 2560), (2560, 4096),
                      (19456, 2560), (2560, 9728), (151936, 2560), (71, 257))
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        for n, k in dimensions:
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 32
            for m in (1, 3, 16, 17, 32):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                xold = x.clone()
                expected = F.linear(x, w)
                for policy in (("simt", "tensorcore") if m == 1 else ("tensorcore",)):
                    with self.subTest(m=m, n=n, k=k, policy=policy):
                        actual = self.kernel(x, w, policy, split_k_for(m, n, k, sms))
                        self.assertEqual(actual.dtype, torch.bfloat16)
                        self.assertTrue(actual.is_contiguous())
                        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
                        # Extra FP32 oracle for a small output slice, not a second
                        # full-sized FP32 checkpoint allocation.
                        oracle = F.linear(x.float(), w[:min(71, n)].float()).bfloat16()
                        torch.testing.assert_close(actual[:, :min(71, n)], oracle, rtol=0.01, atol=0.002)
                        torch.testing.assert_close(x, xold, rtol=0, atol=0)
            del w

    @torch.inference_mode()
    def test_kernel_graph_replay_reads_new_inputs(self):
        x = torch.randn(3, 257, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(71, 257, dtype=torch.bfloat16, device="cuda") / 32
        wold = w.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.kernel(x, w, "tensorcore", 4)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y = self.kernel(x, w, "tensorcore", 4)
        torch.cuda.current_stream().wait_stream(stream)
        address = y.data_ptr()
        for _ in range(5):
            x.normal_()
            graph.replay()
            torch.testing.assert_close(y, F.linear(x, w), rtol=0.01, atol=0.002)
            self.assertEqual(address, y.data_ptr())
        torch.testing.assert_close(w, wold, rtol=0, atol=0)

    @staticmethod
    def tiny():
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(vocab_size=127, hidden_size=256, intermediate_size=512,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=128, max_position_embeddings=4096, rope_theta=5_000_000,
                             tie_word_embeddings=True, sliding_window=None, eos_token_id=0)
        config._attn_implementation = "sdpa"
        return Qwen3ForCausalLM(config).eval()

    @torch.inference_mode()
    def test_forced_backend_cache_reuse_and_own_prefix_logits(self):
        from decode import DecodeState
        from engine import Engine
        torch.manual_seed(104)
        native = self.tiny()
        with tempfile.TemporaryDirectory() as path:
            native.save_pretrained(path)
            engine = Engine(path)
            native = native.cuda().bfloat16()
            plan = engine.model.decode_linear_plan
            for batch in (1, 3, 17):
                for policy in (("simt", "tensorcore") if batch == 1 else ("tensorcore",)):
                    state = DecodeState(engine.model, batch, 7, 5)
                    state.linear_policy = policy  # Force coverage even if tiny-model tuning prefers Torch.
                    buffers = state.cache.key_cache + state.cache.value_cache
                    pointers = [t.data_ptr() for t in buffers]
                    for attempt in range(2):
                        for t in buffers:
                            t.fill_(33 if attempt == 0 else -41)
                        prompt = torch.randint(0, 127, (batch, 7), device="cuda")
                        with mock.patch("kernels.decode_linear.small_linear", side_effect=AssertionError("prefill must bypass")):
                            prefill_logits = state.prefill(prompt)
                        reference_prefill = native(input_ids=prompt, use_cache=False, logits_to_keep=1).logits
                        torch.testing.assert_close(prefill_logits, reference_prefill, rtol=0.03, atol=0.04)
                        first, pos = state.tokens.clone(), state.position.clone()
                        prefix_cache = [t[:, :, :7].clone() for t in buffers]
                        plan.audit = True
                        plan.audit_count = 0
                        actual_logits = state.step()
                        plan.audit = False
                        reference_logits = native(input_ids=torch.cat((prompt, first), dim=1),
                                                  use_cache=False, logits_to_keep=1).logits
                        torch.testing.assert_close(actual_logits, reference_logits, rtol=0.03, atol=0.04)
                        self.assertEqual(plan.audit_count, 13)  # 6 per layer + tied head.
                        state.tokens.copy_(first); state.position.copy_(pos)
                        graph = state._capture_graph(first, pos)
                        for t, before in zip(buffers, prefix_cache):
                            torch.testing.assert_close(t[:, :, :7], before, rtol=0, atol=0)
                            self.assertTrue(bool((t[:, :, 8:] == (33 if attempt == 0 else -41)).all()))
                        prefix = prompt
                        for step in range(5):
                            token = state.tokens.clone()
                            logits = native(input_ids=prefix, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                            gap = logits.amax(-1) - logits.gather(-1, token).squeeze(-1)
                            self.assertLessEqual(float(gap.max()), 2.0)
                            prefix = torch.cat((prefix, token), dim=1)
                            if step < 4:
                                graph.replay()
                        self.assertEqual(int(state.position), 11)
                        self.assertEqual(pointers, [t.data_ptr() for t in buffers])
                        self.assertFalse(plan.active)
                        graph.reset()

    @torch.inference_mode()
    def test_real_selector_stream_counts_and_eos_is_ordinary(self):
        from engine import Engine
        torch.manual_seed(107)
        model = self.tiny()
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            engine = Engine(path)
            native = model.cuda().bfloat16()
            for batch, length, count in ((3, 7, 5), (3, 7, 5), (1, 1, 3), (2, 13, 1), (2, 13, 0)):
                prompt = torch.randint(0, 127, (batch, length), device="cuda")
                emitted = list(engine.generate(prompt.tolist(), count))
                self.assertEqual(len(emitted), count)
                self.assertTrue(all(len(row) == batch and all(type(t) is int for t in row) for row in emitted))
                if count:
                    tokens = torch.tensor(emitted, device="cuda").T
                    prefix = torch.cat((prompt, tokens[:, :-1]), dim=1)
                    logits = native(input_ids=prefix, use_cache=False).logits[:, length - 1:].float()
                    gap = logits.amax(-1) - logits.gather(-1, tokens[..., None]).squeeze(-1)
                    self.assertLessEqual(float(gap.max()), 2.0)
            # Tied zero embedding/head makes token 0 (also configured EOS) the
            # exact lowest-index argmax at EVERY step. Do not stop at EOS.
            engine.model.lm_head.weight.zero_()
            self.assertEqual(list(engine.generate([[1] * 7], 5)), [[0]] * 5)


if __name__ == "__main__":
    unittest.main()
