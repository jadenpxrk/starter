"""Focused checks for the decode Q/K patch. GPU tests require the pinned runtime."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
if importlib.util.find_spec("triton") is None:
    # Only the native CPU fallback is exercised without Triton.
    unavailable = mock.Mock(qk_norm_rope=mock.Mock(
        side_effect=AssertionError("CPU fallback must not call the CUDA kernel"),
    ))
    with mock.patch.dict(sys.modules, {"kernels.qk_norm_rope": unavailable}):
        import qk_norm_rope as adapter
else:
    import qk_norm_rope as adapter
from decode import DecodeState
from decode_attention import install_decode_attention
from mlp import PackedMLP


def tiny():
    cfg = Qwen3Config(
        vocab_size=127, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=128, max_position_embeddings=32768, rope_theta=5_000_000,
        tie_word_embeddings=True, sliding_window=None,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(cfg).eval()


def native_norm(x, weight, eps):
    f = x.float()
    return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def angles(batch, position, shared=True):
    # Actual Qwen3 default RoPE formula: absolute positions and duplicated halves.
    inv = 1.0 / (5_000_000 ** (torch.arange(0, 128, 2, device="cuda").float() / 128))
    p = torch.full((1 if shared else batch, 1), position, device="cuda")
    if not shared:
        p = p + torch.arange(batch, device="cuda")[:, None]
    f = p.float()[:, :, None] * inv[None, None, :]
    f = torch.cat((f, f), dim=-1)
    return f.cos().bfloat16(), f.sin().bfloat16()


class AdapterFallbackTests(unittest.TestCase):
    @torch.inference_mode()
    def test_cpu_fallback_and_weight_identity(self):
        torch.manual_seed(0)
        model = tiny()
        native = model.model.layers[0].self_attn
        fused = adapter.DecodeQKNormRoPE(native)
        self.assertFalse(fused.training)
        self.assertEqual(set(native.state_dict()), set(fused.state_dict()))
        for name, param in native.named_parameters():
            self.assertIs(dict(fused.named_parameters())[name], param)
        for length in (1, 7):
            x = torch.randn(2, length, 256)
            pos = torch.arange(length).unsqueeze(0)
            emb = model.model.rotary_emb(x, pos)
            for mask in (None, torch.ones(1, 1, length, length, dtype=torch.bool)):
                expected = native(x, emb, mask)[0]
                with mock.patch.object(adapter, "qk_norm_rope", side_effect=AssertionError("must fall back")):
                    actual = fused(x, emb, mask)[0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
    "CUDA and Triton are required; CPU is not a GPU validation",
)
class QKCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global qk_norm_rope, rms_norm, Engine, FusedRMSNorm, decode_step
        from kernels.qk_norm_rope import qk_norm_rope
        from kernels.rmsnorm import rms_norm
        from engine import Engine, FusedRMSNorm
        import decode_step

    @torch.inference_mode()
    def test_kernel_values_strides_positions_and_rounding(self):
        torch.manual_seed(19)
        for batch in (1, 3, 4, 16):
            for shared in (True, False):
                for position in (0, 511, 2047, 16383):
                    for strided in (False, True):
                        with self.subTest(batch=batch, shared=shared, position=position, strided=strided):
                            width = 256 if strided else 128
                            q = torch.randn(batch, 32, 1, width, device="cuda", dtype=torch.bfloat16)
                            k = torch.randn(batch, 8, 1, width, device="cuda", dtype=torch.bfloat16)
                            if strided:
                                q, k = q[..., ::2], k[..., ::2]
                            qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
                            kw = torch.randn_like(qw)
                            c, s = angles(batch, position, shared)
                            before = [t.clone() for t in (q, k, qw, kw, c, s)]
                            actual = qk_norm_rope(q, k, qw, kw, c, s, 1e-6, 3e-6)
                            # Compare both native Torch and the already-passing fused norm.
                            for norm in (native_norm, rms_norm):
                                expected = apply_rotary_pos_emb(norm(q, qw, 1e-6), norm(k, kw, 3e-6), c, s)
                                for a, e in zip(actual, expected):
                                    self.assertEqual(a.dtype, torch.bfloat16)
                                    self.assertTrue(a.is_contiguous())
                                    self.assertTrue(bool(torch.isfinite(a).all()))
                                    # Operator diagnostic, NOT the 2.0-logit token gate.
                                    torch.testing.assert_close(a, e, rtol=0.02, atol=0.02)
                            for t, old in zip((q, k, qw, kw, c, s), before):
                                torch.testing.assert_close(t, old, rtol=0, atol=0)
        # Exact normalization removes reduction uncertainty: rotary rounding must be exact.
        q = torch.ones(3, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.full((3, 8, 1, 128), 2., device="cuda", dtype=torch.bfloat16)
        qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        kw = torch.randn_like(qw)
        c, s = angles(3, 8191, False)
        actual = qk_norm_rope(q, k, qw, kw, c, s, 0., 0.)
        expected = apply_rotary_pos_emb(q * qw, (k / 2) * kw, c, s)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        q.zero_(); k.zero_()
        for a in qk_norm_rope(q, k, qw, kw, c, s, 1e-6, 1e-6):
            self.assertEqual(int(torch.count_nonzero(a)), 0)

    @torch.inference_mode()
    def test_kernel_cuda_graph_changes_inputs_and_positions(self):
        q = torch.randn(3, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(3, 8, 1, 128, device="cuda", dtype=torch.bfloat16)
        qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        kw = torch.randn_like(qw)
        c, s = angles(3, 0, False)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                qk_norm_rope(q, k, qw, kw, c, s, 1e-6, 1e-6)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = qk_norm_rope(q, k, qw, kw, c, s, 1e-6, 1e-6)
        pointers = [a.data_ptr() for a in outputs]
        for position in (1, 512, 2048, 16384):
            q.normal_(); k.normal_()
            nc, ns = angles(3, position, False)
            c.copy_(nc); s.copy_(ns)
            graph.replay()
            expected = apply_rotary_pos_emb(native_norm(q, qw, 1e-6), native_norm(k, kw, 1e-6), c, s)
            for a, e in zip(outputs, expected):
                torch.testing.assert_close(a, e, rtol=0.02, atol=0.02)
            self.assertEqual(pointers, [a.data_ptr() for a in outputs])

    @torch.inference_mode()
    def test_cached_logits_prefill_bypass_and_cache_reuse(self):
        torch.manual_seed(7)
        native = tiny().cuda().bfloat16()
        candidate = deepcopy(native)
        install_decode_attention(candidate)
        candidate.model.norm = FusedRMSNorm(candidate.model.norm)
        for layer in candidate.model.layers:
            layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
            layer.post_attention_layernorm = FusedRMSNorm(layer.post_attention_layernorm)
            layer.self_attn.q_norm = FusedRMSNorm(layer.self_attn.q_norm)
            layer.self_attn.k_norm = FusedRMSNorm(layer.self_attn.k_norm)
            layer.mlp = PackedMLP(layer.mlp)
            layer.self_attn = adapter.DecodeQKNormRoPE(layer.self_attn)
        for batch, length, count in ((3, 7, 5), (1, 1, 3), (2, 13, 1)):
            state = DecodeState(candidate, batch, length, count)
            buffers = state.cache.key_cache + state.cache.value_cache
            pointers = [t.data_ptr() for t in buffers]
            for attempt in range(2):
                for t in buffers:
                    t.fill_(33 if attempt == 0 else -41)
                prompt = torch.randint(0, 127, (batch, length), device="cuda")
                with mock.patch.object(adapter, "qk_norm_rope", side_effect=AssertionError("prefill must bypass")):
                    actual = state.prefill(prompt)
                current, cache = prompt, None
                for step in range(count):
                    ref = native(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    torch.testing.assert_close(actual, ref.logits, rtol=0.03, atol=0.04)
                    # Follow candidate's prefix, not the reference's independent continuation.
                    current = state.tokens.clone()
                    cache = ref.past_key_values
                    if step + 1 < count:
                        # Decode runs the fused step: packed Q/K/V per layer, no module path.
                        with mock.patch.object(adapter, "qk_norm_rope", side_effect=AssertionError("fused step must not use the module path")), \
                                mock.patch.object(decode_step, "qkv_norm_rope_cache", wraps=decode_step.qkv_norm_rope_cache) as call:
                            actual = state.step()
                            self.assertEqual(call.call_count, 2)
                self.assertEqual(pointers, [t.data_ptr() for t in buffers])
                self.assertEqual(state.position.item(), length + count - 1)

    @torch.inference_mode()
    def test_engine_graph_stream_and_own_prefix_tokens(self):
        torch.manual_seed(31)
        reference = tiny()
        with tempfile.TemporaryDirectory() as path:
            reference.save_pretrained(path)
            engine = Engine(path)
            reference = reference.cuda().bfloat16()
            for layer in engine.model.model.layers:
                self.assertIsInstance(layer.self_attn, adapter.DecodeQKNormRoPE)
            for batch, length, count in ((3, 7, 5), (3, 7, 5), (1, 1, 3), (2, 13, 1), (2, 13, 0)):
                prompt = torch.randint(0, 127, (batch, length), device="cuda")
                emitted = list(engine.generate(prompt.tolist(), count))
                self.assertEqual(len(emitted), count)
                if count == 0:
                    continue
                self.assertTrue(all(len(row) == batch for row in emitted))
                tokens = torch.tensor(emitted, device="cuda").T
                own_prefix = torch.cat((prompt, tokens[:, :-1]), dim=1)
                logits = reference(input_ids=own_prefix, use_cache=False).logits[:, length-1:]
                gap = logits.float().amax(-1) - logits.float().gather(-1, tokens[..., None]).squeeze(-1)
                self.assertTrue(bool(torch.isfinite(gap).all()))
                self.assertLessEqual(float(gap.max()), 2.0)
