"""Native Flash decode integration checks; CPU emulation is NOT GPU execution."""

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

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
spec = importlib.util.spec_from_file_location("flash_decode_test_target", ENGINE / "flash_decode.py")
flash = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flash)


def functions_from(path, names, namespace):
    """Load actual selected bodies without requiring the absent HF package."""
    tree = ast.parse(path.read_text())
    result = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            result.append(node)
        if isinstance(node, ast.ClassDef):
            result.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names)
    module = ast.Module(body=result, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def reference(q, k, v, used, scale):
    # Independent per-original-batch/head reference, with no virtual-batch views.
    batch, hq, _, width = q.shape
    hk = k.shape[1]
    result = torch.empty(batch, 1, hq, width, dtype=q.dtype, device=q.device)
    for b in range(batch):
        for h in range(hq):
            kh = h // (hq // hk)
            count = int(used[b * hk + kh])
            scores = (k[b, kh, :count].double() * q[b, h, 0].double()).sum(-1) * scale
            result[b, 0, h] = (scores.softmax(-1)[:, None] * v[b, kh, :count].double()).sum(0)
    return result


class FlashCPU(unittest.TestCase):
    @torch.inference_mode()
    def test_metadata_boundaries_reset_and_mask_identity(self):
        context = flash.FlashDecodeContext(3, 12, 3, torch.arange(257))
        torch.testing.assert_close(context.cu_q, torch.arange(10, dtype=torch.int32))
        torch.testing.assert_close(context.cu_k, torch.arange(10, dtype=torch.int32) * 257)
        pointers = [t.data_ptr() for t in (context.cu_q, context.cu_k, context.used)]
        q = torch.randn(3, 12, 1, 128).bfloat16()
        k = torch.randn(3, 3, 257, 128).bfloat16()
        for position in (0, 127, 128, 256, 6):
            pos = torch.tensor([position])
            mask = context.prepare(pos)
            self.assertTrue(bool((context.used == position + 1).all()))
            torch.testing.assert_close(mask.flatten(), torch.arange(257) <= position)
            self.assertTrue(context.matches(q, k, k, mask))
            self.assertFalse(context.matches(q, k, k, mask.clone()))
            self.assertEqual(pointers, [t.data_ptr() for t in (context.cu_q, context.cu_k, context.used)])
        # It must not silently treat an arbitrary mask, strided cache or dtype as supported.
        self.assertFalse(context.matches(q.float(), k, k, context.mask))
        self.assertFalse(context.matches(q, k.transpose(1, 2), k, context.mask))
        with self.assertRaises(ValueError):
            context.attention(q, k, k, context.mask, 128**-0.5)  # CPU is rejected.
        with self.assertRaises(ValueError):
            context.prepare(torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            flash.FlashDecodeContext(1, 7, 2, torch.arange(3))
        with self.assertRaises(ValueError):
            flash.FlashDecodeContext(2**30, 4, 2, torch.arange(2))

    @torch.inference_mode()
    def test_view_mapping_all_heads_offsets_no_copies_and_stale_slots(self):
        torch.manual_seed(61)
        cases = ((1, 32, 8, 1), (1, 32, 8, 544), (3, 12, 3, 129),
                 (4, 32, 8, 257), (16, 32, 8, 640))
        for batch, hq, hk, capacity in cases:
            context = flash.FlashDecodeContext(batch, hq, hk, torch.arange(capacity))
            q = torch.randn(batch, hq, 1, 128).bfloat16()
            k = torch.randn(batch, hk, capacity, 128).bfloat16()
            v = torch.randn_like(k)
            for count in sorted(set((1, max(1, capacity // 2), capacity))):
                with self.subTest(batch=batch, capacity=capacity, count=count):
                    k.normal_(); v.normal_()
                    k[:, :, count:] = float("nan")
                    v[:, :, count:] = float("nan")
                    originals = [t.clone() for t in (q, k, v)]
                    context.prepare(torch.tensor([count - 1]))

                    def fake_operator(qf, kf, vf, cuq, cuk, max_k, used, scale):
                        # Emulates the documented varlen contract; NOT Flash arithmetic.
                        self.assertEqual(tuple(qf.shape), (batch * hk, hq // hk, 128))
                        self.assertEqual(tuple(kf.shape), (batch * hk * capacity, 1, 128))
                        self.assertEqual(qf.data_ptr(), q.data_ptr())
                        self.assertEqual(kf.data_ptr(), k.data_ptr())
                        self.assertEqual(vf.data_ptr(), v.data_ptr())
                        self.assertEqual(max_k, capacity)
                        out = torch.empty_like(qf)
                        for b in range(batch * hk):
                            left, length = int(cuk[b]), int(used[b])
                            self.assertEqual(int(cuq[b + 1]) - int(cuq[b]), 1)
                            scores = qf[b].double() @ kf[left:left+length, 0].double().T * scale
                            out[b] = (scores.softmax(-1) @ vf[left:left+length, 0].double()).to(q.dtype)
                        return out

                    with mock.patch.object(flash, "_native_flash", side_effect=fake_operator) as called:
                        actual = context._views_and_call(q, k, v, 128**-0.5)
                    self.assertEqual(called.call_count, 1)
                    expected = reference(q, k, v, context.used, 128**-0.5)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(bool(torch.isfinite(actual).all()))
                    for new, old in zip((q, k, v), originals):
                        torch.testing.assert_close(new, old, rtol=0, atol=0, equal_nan=True)

    @torch.inference_mode()
    def test_negative_controls_wrong_offsets_and_ignored_length(self):
        context = flash.FlashDecodeContext(2, 8, 2, torch.arange(5))
        context.prepare(torch.tensor([1]))
        q = torch.zeros(2, 8, 1, 128, dtype=torch.bfloat16)
        k = torch.zeros(2, 2, 5, 128, dtype=torch.bfloat16)
        v = (torch.arange(4).view(2, 2, 1, 1) * 20
             + torch.arange(5).view(1, 1, 5, 1)).expand(2, 2, 5, 128).to(torch.bfloat16).contiguous()
        expected = reference(q, k, v, context.used, 128**-0.5)
        for wrong_offset in (True, False):
            def broken(qf, kf, vf, cuq, cuk, cap, used, scale):
                out = torch.empty_like(qf)
                for b in range(context.virtual_batch):
                    start = 0 if wrong_offset else int(cuk[b])
                    count = int(used[b]) if wrong_offset else cap
                    out[b] = vf[start:start+count, 0].float().mean(0).to(qf.dtype)
                return out
            with mock.patch.object(flash, "_native_flash", side_effect=broken):
                actual = context._views_and_call(q, k, v, 128**-0.5)
            self.assertFalse(torch.equal(actual, expected))

    def test_native_schema_and_call_arguments(self):
        schema = torch.ops.aten._flash_attention_forward.default._schema
        expected = ["query", "key", "value", "cum_seq_q", "cum_seq_k", "max_q", "max_k",
                    "dropout_p", "is_causal", "return_debug_mask", "scale", "window_size_left",
                    "window_size_right", "seqused_k", "alibi_slopes"]
        self.assertEqual([a.name for a in schema.arguments], expected)
        tensors = [torch.empty(1) for _ in range(6)]
        marker = object()
        with mock.patch.object(torch.ops.aten._flash_attention_forward, "default",
                               return_value=(marker, None, None, None, None)) as op:
            actual = flash._native_flash(*tensors[:5], 544, tensors[5], 128**-0.5)
        self.assertIs(actual, marker)
        args, kwargs = op.call_args
        self.assertEqual(args[5:], (1, 544, 0.0, False, False))
        self.assertIs(kwargs["seqused_k"], tensors[5])
        self.assertIsNone(kwargs["window_size_left"])
        self.assertIsNone(kwargs["window_size_right"])
        self.assertIsNone(kwargs["alibi_slopes"])

    def test_factory_is_metadata_only_and_prefill_only_bypasses(self):
        config = SimpleNamespace(num_attention_heads=32, num_key_value_heads=8,
                                 head_dim=128, hidden_size=2560, _attn_implementation="starter_decode_gqa")
        model = SimpleNamespace(config=config, device=torch.device("cuda"), dtype=torch.bfloat16)
        marker = object()
        with mock.patch.object(flash, "FlashDecodeContext", return_value=marker) as constructor:
            self.assertIs(flash.make_flash_context(model, 3, 9, torch.arange(43)), marker)
            self.assertEqual(constructor.call_args.args[:3], (3, 32, 8))
        self.assertIsNone(flash.make_flash_context(model, 3, 1, torch.arange(43)))
        model.device = torch.device("cpu")
        self.assertIsNone(flash.make_flash_context(model, 3, 9, torch.arange(43)))
        model.device = torch.device("cuda"); model.dtype = torch.float32
        self.assertIsNone(flash.make_flash_context(model, 3, 9, torch.arange(43)))
        model.dtype = torch.bfloat16; config.head_dim = 16
        self.assertIsNone(flash.make_flash_context(model, 3, 9, torch.arange(43)))

    @torch.inference_mode()
    def test_forward_threads_context_only_when_requested(self):
        calls = []
        class Layer:
            def __call__(self, x, **kwargs):
                calls.append(kwargs)
                return (x,)
        base = SimpleNamespace(embed_tokens=lambda ids: torch.zeros(*ids.shape, 4),
                               rotary_emb=lambda x, p: (p, p), layers=[Layer(), Layer()], norm=lambda x: x)
        model = SimpleNamespace(model=base, lm_head=lambda x: x)
        env = functions_from(ENGINE / "decode.py", {"qwen_forward"}, {"torch": torch})
        ids, positions = torch.ones(2, 7, dtype=torch.long), torch.arange(7)
        env["qwen_forward"](model, ids, None, positions)
        self.assertTrue(all("_flash_decode_context" not in c for c in calls))
        calls.clear(); marker = object()
        env["qwen_forward"](model, ids[:, :1], None, positions[:1], flash_context=marker)
        self.assertTrue(all(c["_flash_decode_context"] is marker for c in calls))

    @torch.inference_mode()
    def test_step_updates_lengths_then_forwards_and_advances(self):
        context = flash.FlashDecodeContext(2, 4, 2, torch.arange(13))
        state = SimpleNamespace(flash_context=context, position=torch.tensor([7]),
                                key_positions=torch.arange(13), tokens=torch.zeros(2, 1, dtype=torch.long),
                                model=object(), cache=object(), fused_forward=None,
                                cos=object(), sin=object())
        def forward(model, tokens, cache, position, mask, flash_context):
            self.assertIs(flash_context, context)
            self.assertTrue(bool((context.used == 8).all()))
            self.assertIs(mask, context.mask)
            self.assertEqual(int(position), 7)
            return torch.tensor([[[0., 3., 1.]], [[5., 0., 1.]]])
        env = functions_from(ENGINE / "decode.py", {"step"}, {"torch": torch, "qwen_forward": forward})
        env["step"](state)
        torch.testing.assert_close(state.tokens, torch.tensor([[1], [0]]))
        self.assertEqual(int(state.position), 8)
        # With a fused forward installed, step routes there with the same prepared prefix.
        def fused(model, cache, tokens, position, mask, flash_context, cos, sin):
            self.assertIs(flash_context, context)
            self.assertIs(mask, context.mask)
            self.assertTrue(bool((context.used == 9).all()))
            self.assertIs(cos, state.cos); self.assertIs(sin, state.sin)
            return torch.tensor([[[9., 3., 1.]], [[5., 0., 7.]]])
        state.fused_forward = fused
        env = functions_from(ENGINE / "decode.py", {"step"}, {"torch": torch, "qwen_forward": mock.Mock(side_effect=AssertionError("module path must not run"))})
        env["step"](state)
        torch.testing.assert_close(state.tokens, torch.tensor([[0], [2]]))
        self.assertEqual(int(state.position), 9)

    @torch.inference_mode()
    def test_attention_dispatch_context_and_unchanged_fallback(self):
        native = mock.Mock(return_value=("native", None))
        env = functions_from(ENGINE / "decode_attention.py", {"grouped_decode_attention"},
                             {"torch": torch, "sdpa_attention_forward": native})
        fn = env["grouped_decode_attention"]
        module = SimpleNamespace(num_key_value_groups=4)
        q = torch.randn(2, 8, 1, 128); k = torch.randn(2, 2, 19, 128); v = torch.randn_like(k)
        mask = (torch.arange(19) < 7).view(1, 1, 1, -1)
        scale = 128**-0.5
        # The unchanged folded path, with NO private context, still honors the mask.
        expected = F.scaled_dot_product_attention(q, k.repeat_interleave(4, 1),
                                                  v.repeat_interleave(4, 1), attn_mask=mask, scale=scale)
        actual, _ = fn(module, q, k, v, mask, scaling=scale)
        torch.testing.assert_close(actual, expected.transpose(1, 2), rtol=1e-5, atol=1e-6)
        marker = object(); context = SimpleNamespace(attention=mock.Mock(return_value=marker))
        result, _ = fn(module, q, k, v, mask, scaling=scale, _flash_decode_context=context)
        self.assertIs(result, marker)
        context.attention.assert_called_once_with(q, k, v, mask, scale)
        with self.assertRaises(ValueError):
            fn(module, q, k, v, mask, dropout=0.1, _flash_decode_context=context)
        with self.assertRaises(ValueError):
            fn(module, q, k, v, mask, is_causal=True, _flash_decode_context=context)
        fn(module, q.expand(-1, -1, 3, -1), k, v, None)
        native.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required; CPU checks do not validate Flash execution")
class FlashCUDA(unittest.TestCase):
    @torch.inference_mode()
    def test_native_operator_all_heads_masks_and_used_lengths(self):
        torch.manual_seed(18)
        for batch, cap in ((1, 1), (1, 544), (3, 257), (4, 2080), (16, 640), (2, 8193)):
            context = flash.FlashDecodeContext(batch, 32, 8, torch.arange(cap, device="cuda"))
            q = torch.randn(batch, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(batch, 8, cap, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            for length in sorted(set((1, max(1, cap // 2), cap))):
                with self.subTest(batch=batch, cap=cap, length=length):
                    k.normal_(); v.normal_()
                    k[:, :, length:] = float("nan"); v[:, :, length:] = float("nan")
                    before = [t.clone() for t in (q, k, v)]
                    mask = context.prepare(torch.tensor([length - 1], device="cuda"))
                    out = context.attention(q, k, v, mask, 128**-0.5)
                    expected = reference(q, k, v, context.used, 128**-0.5)
                    self.assertEqual(out.dtype, torch.bfloat16)
                    self.assertTrue(bool(torch.isfinite(out).all()))
                    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.01)
                    for a, b in zip((q, k, v), before):
                        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)

    @torch.inference_mode()
    def test_cuda_graph_replay_changes_lengths_inputs_and_cache(self):
        cap = 641
        context = flash.FlashDecodeContext(3, 32, 8, torch.arange(cap, device="cuda"))
        q = torch.randn(3, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(3, 8, cap, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k); position = torch.tensor([0], device="cuda")
        def call():
            return context.attention(q, k, v, context.prepare(position), 128**-0.5)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = call()
        pointers = [t.data_ptr() for t in (context.used, context.cu_q, context.cu_k, output)]
        for p in (1, 127, 128, 512, 640, 6):
            q.normal_(); k.normal_(); v.normal_()
            k[:, :, p+1:] = float("nan"); v[:, :, p+1:] = float("nan")
            position.fill_(p)
            graph.replay()
            torch.testing.assert_close(output, reference(q, k, v, context.used, 128**-0.5),
                                       rtol=0.02, atol=0.01)
            self.assertEqual(pointers, [t.data_ptr() for t in (context.used, context.cu_q, context.cu_k, output)])

    @torch.inference_mode()
    def test_full_engine_own_prefix_prefill_cache_reuse_and_eos(self):
        if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("triton") is None:
            self.skipTest("Transformers and Triton required for existing engine")
        sys.path.insert(0, str(ENGINE))
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
            from engine import Engine as RealEngine
            import flash_decode as runtime_flash
        finally:
            sys.path.pop(0)
        config = Qwen3Config(vocab_size=127, hidden_size=256, intermediate_size=512,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=128, max_position_embeddings=32768, rope_theta=5_000_000,
                            tie_word_embeddings=True, sliding_window=None, eos_token_id=0)
        config._attn_implementation = "sdpa"
        torch.manual_seed(29)
        native = Qwen3ForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as path:
            native.save_pretrained(path)
            engine = RealEngine(path)
            native = native.cuda().bfloat16()
            for batch, length, count in ((3, 7, 5), (3, 7, 5), (1, 1, 3), (2, 129, 9), (2, 13, 1), (2, 13, 0)):
                prompt = torch.randint(0, 127, (batch, length), device="cuda")
                with mock.patch.object(runtime_flash, "_native_flash", wraps=runtime_flash._native_flash) as op:
                    emitted = list(engine.generate(prompt.tolist(), count))
                self.assertEqual(len(emitted), count)
                self.assertTrue(all(len(row) == batch and all(type(t) is int for t in row) for row in emitted))
                if count > 1:
                    # Confirm installation independent of any performance selector.
                    self.assertIsNotNone(engine.decode_state.flash_context)
                    if op.call_count == 0:  # Existing graph: prove eager dispatch, too.
                        engine.decode_state.prefill(prompt)
                        with mock.patch.object(runtime_flash, "_native_flash", wraps=runtime_flash._native_flash) as eager:
                            engine.decode_state.step()
                        self.assertEqual(eager.call_count, config.num_hidden_layers)
                    else:
                        self.assertGreaterEqual(op.call_count, config.num_hidden_layers)
                if count:
                    tokens = torch.tensor(emitted, device="cuda").T
                    prefix = torch.cat((prompt, tokens[:, :-1]), dim=1)
                    logits = native(input_ids=prefix, use_cache=False).logits[:, length-1:].float()
                    gaps = logits.amax(-1) - logits.gather(-1, tokens[..., None]).squeeze(-1)
                    self.assertTrue(bool(torch.isfinite(gaps).all()))
                    self.assertLessEqual(float(gaps.max()), 2.0)
            # Poison old capacity; prefill must not call Flash decode or reuse old content.
            state = engine.decode_state
            prompt = torch.randint(0, 127, (2, 13), device="cuda")
            for tensor in state.cache.key_cache + state.cache.value_cache: tensor.fill_(33)
            with mock.patch.object(runtime_flash, "_native_flash", side_effect=AssertionError("prefill bypass")):
                state.prefill(prompt)
            # Zero tied LM-head -> greedy token 0, which is also EOS; do not stop.
            engine.model.lm_head.weight.zero_()
            self.assertEqual(list(engine.generate([[1, 2, 3]], 5)), [[0]] * 5)


if __name__ == "__main__":
    unittest.main()
