"""Fused decode step checks. CPU runs verify orchestration with Torch references
standing in for the Triton kernels; kernel arithmetic needs the CUDA class."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

ENGINE = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(ENGINE))
HAS_TRITON = importlib.util.find_spec("triton") is not None
if HAS_TRITON:
    import decode_step
    import qk_norm_rope as adapter
else:
    # Only the CPU orchestration is exercised without Triton; kernels are replaced below.
    with mock.patch.dict(sys.modules, {
            "kernels.qk_norm_rope": mock.Mock(), "kernels.decode_fused": mock.Mock(),
            # The static shape rule keeps F.linear here, exactly as on a CPU device.
            "kernels.skinny_gemm": mock.Mock(supports=mock.Mock(return_value=False),
                                             supports_qkv=mock.Mock(return_value=False))}):
        import decode_step
        import qk_norm_rope as adapter
import decode
from decode import DecodeState
from decode_attention import install_decode_attention
from flash_decode import FlashDecodeContext
from mlp import PackedMLP


def tiny_model(dtype=torch.float32, head_dim=16):
    config = Qwen3Config(
        vocab_size=127, hidden_size=64, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=head_dim, max_position_embeddings=32768, rope_theta=5_000_000,
        tie_word_embeddings=True, sliding_window=None,
    )
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval().to(dtype=dtype)


def install(model):
    install_decode_attention(model)
    for layer in model.model.layers:
        layer.mlp = PackedMLP(layer.mlp)
        layer.self_attn = adapter.DecodeQKNormRoPE(layer.self_attn)
    return model


def native_norm(x, weight, eps):
    f = x.float()
    return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def ref_add_rms_norm(residual, branch, weight, eps):
    total = residual + branch
    return total, native_norm(total, weight, eps)


def ref_silu_mul(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def ref_qkv_norm_rope_cache(qkv, qw, kw, q_eps, k_eps, cos, sin, position, key_cache, value_cache):
    batch, kv_heads, _, width = key_cache.shape
    heads = qkv.shape[1] // width - 2 * kv_heads
    q, k, v = qkv.split([heads * width, kv_heads * width, kv_heads * width], dim=-1)
    q = native_norm(q.reshape(batch, heads, 1, width), qw, q_eps)
    k = native_norm(k.reshape(batch, kv_heads, 1, width), kw, k_eps)
    c, s = cos.index_select(0, position).unsqueeze(0), sin.index_select(0, position).unsqueeze(0)
    q, k = apply_rotary_pos_emb(q, k, c, s)
    key_cache.index_copy_(2, position, k)
    value_cache.index_copy_(2, position, v.reshape(batch, kv_heads, 1, width))
    return q.contiguous()


class CPUContext(FlashDecodeContext):
    """Varlen semantics of the native call, in Torch, driven by the prepared lengths."""

    def attention(self, query, key, value, mask, scale):
        assert mask is self.mask
        batch, heads, _, width = query.shape
        kv_heads = key.shape[1]
        visible = torch.arange(key.shape[2]).view(1, 1, 1, -1) < self.used.view(batch, kv_heads, 1, 1)
        out = F.scaled_dot_product_attention(
            query.view(batch, kv_heads, heads // kv_heads, width), key, value,
            attn_mask=visible, scale=scale,
        )
        return out.reshape(batch, 1, heads, width)


def patched_kernels():
    return (
        mock.patch.object(decode_step, "qkv_norm_rope_cache", side_effect=ref_qkv_norm_rope_cache),
        mock.patch.object(decode_step, "add_rms_norm", side_effect=ref_add_rms_norm),
        mock.patch.object(decode_step, "silu_mul", side_effect=ref_silu_mul),
    )


class FusedStepCPU(unittest.TestCase):
    @torch.inference_mode()
    def test_fused_step_matches_native_across_prompts(self):
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            model = tiny_model(dtype)
            candidate = install(deepcopy(model))
            for batch, length, count in ((1, 1, 2), (2, 7, 5), (4, 13, 3)):
                state = DecodeState(candidate, batch, length, count)
                self.assertIsNone(state.fused_forward)  # CPU keeps the module path by default.
                state.flash_context = CPUContext(batch, 4, 2, state.key_positions)
                state.fused_forward = decode_step.fused_decode_forward
                buffers = state.cache.key_cache + state.cache.value_cache
                addresses = [t.data_ptr() for t in buffers]
                for attempt in range(2):
                    with self.subTest(dtype=dtype, batch=batch, length=length, count=count, attempt=attempt):
                        for t in buffers:
                            t.fill_(33)  # Stale capacity must stay invisible after reuse.
                        prompt = torch.randint(0, 127, (batch, length))
                        actual = state.prefill(prompt)
                        current, cache = prompt, None
                        for step in range(count):
                            expected = model(input_ids=current, past_key_values=cache,
                                             use_cache=True, logits_to_keep=1)
                            tolerance = (dict(rtol=2e-2, atol=2e-3) if dtype == torch.bfloat16
                                         else dict(rtol=1e-4, atol=1e-5))
                            torch.testing.assert_close(actual, expected.logits, **tolerance)
                            torch.testing.assert_close(
                                state.tokens, expected.logits[:, -1].argmax(-1, keepdim=True))
                            current = state.tokens.clone()
                            cache = expected.past_key_values
                            if step + 1 < count:
                                qkv, add, act = patched_kernels()
                                with qkv as qkv_calls, add as add_calls, act as act_calls, \
                                        mock.patch.object(decode, "qwen_forward",
                                                          side_effect=AssertionError("module path must not decode")):
                                    actual = state.step()
                                self.assertEqual(qkv_calls.call_count, 2)
                                self.assertEqual(add_calls.call_count, 4)
                                self.assertEqual(act_calls.call_count, 2)
                                self.assertEqual(int(state.position), length + step + 1)
                        self.assertEqual(int(state.position), length + count - 1)
                        self.assertEqual(addresses, [t.data_ptr() for t in buffers])

    @torch.inference_mode()
    def test_rotary_tables_equal_per_step_rotary_values(self):
        model = tiny_model(torch.bfloat16)
        state = DecodeState(model, 2, 5, 4)
        self.assertEqual(tuple(state.cos.shape), (9, 16))
        self.assertTrue(state.cos.is_contiguous() and state.sin.is_contiguous())
        self.assertEqual(state.cos.dtype, torch.bfloat16)
        probe = torch.empty(0, dtype=torch.bfloat16)
        for position in range(9):
            cos, sin = model.model.rotary_emb(probe, torch.tensor([[position]]))
            torch.testing.assert_close(state.cos[position], cos[0, 0], rtol=0, atol=0)
            torch.testing.assert_close(state.sin[position], sin[0, 0], rtol=0, atol=0)

    @torch.inference_mode()
    def test_packed_qkv_weight_shares_storage_with_projections(self):
        torch.manual_seed(5)
        native = tiny_model().model.layers[0].self_attn
        originals = [m.weight.clone() for m in (native.q_proj, native.k_proj, native.v_proj)]
        keys = set(native.state_dict())
        fused = adapter.DecodeQKNormRoPE(native)
        torch.testing.assert_close(fused.qkv_weight, torch.cat(originals), rtol=0, atol=0)
        self.assertEqual(set(fused.state_dict()), keys)
        self.assertNotIn("qkv_weight", dict(fused.named_parameters()))
        self.assertNotIn("qkv_weight", dict(fused.named_buffers()))
        offset = 0
        for module, original in zip((fused.q_proj, fused.k_proj, fused.v_proj), originals):
            self.assertEqual(module.weight.untyped_storage().data_ptr(),
                             fused.qkv_weight.untyped_storage().data_ptr())
            self.assertEqual(module.weight.storage_offset(), offset)
            self.assertTrue(module.weight.is_contiguous())
            self.assertFalse(module.weight.requires_grad)
            torch.testing.assert_close(module.weight, original, rtol=0, atol=0)
            offset += original.numel()
        x = torch.randn(2, 3, 64)
        torch.testing.assert_close(
            F.linear(x, fused.qkv_weight), torch.cat([F.linear(x, w) for w in originals], dim=-1),
            rtol=1e-5, atol=1e-6)

    @torch.inference_mode()
    def test_emit_yields_exact_count_in_order_and_runs_one_step_fewer(self):
        torch.manual_seed(3)
        model = tiny_model()
        candidate = install(deepcopy(model))
        for batch, length, count in ((2, 7, 5), (1, 3, 1), (3, 5, 2)):
            with self.subTest(batch=batch, length=length, count=count):
                state = DecodeState(candidate, batch, length, count)
                self.assertIsNone(state.ready)
                self.assertEqual(tuple(state.host.shape), (batch, 1))
                prompt = torch.randint(0, 127, (batch, length))
                current, cache, expected = prompt, None, []
                for _ in range(count):
                    result = model(input_ids=current, past_key_values=cache,
                                   use_cache=True, logits_to_keep=1)
                    current = result.logits[:, -1].argmax(-1, keepdim=True)
                    cache = result.past_key_values
                    expected.append(current[:, 0].tolist())
                state.prefill(prompt)
                with mock.patch.object(DecodeState, "step", autospec=True,
                                       side_effect=DecodeState.step) as steps:
                    emitted = list(state.emit(count))
                self.assertEqual(emitted, expected)
                self.assertEqual(steps.call_count, count - 1)
                self.assertEqual(int(state.position), length + count - 1)
                self.assertTrue(all(type(t) is int for row in emitted for t in row))


@unittest.skipUnless(torch.cuda.is_available() and HAS_TRITON,
                     "CUDA and Triton are required; CPU checks do not validate the kernels")
class FusedKernelsCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global kernels, FusedRMSNorm
        from kernels import decode_fused as kernels
        from engine import FusedRMSNorm

    @torch.inference_mode()
    def test_add_rms_norm_matches_native(self):
        torch.manual_seed(11)
        for rows, cols in ((1, 2560), (4, 2560), (16, 2560), (3, 256), (5, 128)):
            with self.subTest(rows=rows, cols=cols):
                x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
                y = torch.randn_like(x) * 3
                w = torch.randn(cols, device="cuda", dtype=torch.bfloat16)
                before = (x.clone(), y.clone())
                total, normed = kernels.add_rms_norm(x, y, w, 1e-6)
                ref_total, ref_normed = ref_add_rms_norm(x, y, w, 1e-6)
                torch.testing.assert_close(total, ref_total, rtol=0, atol=0)
                torch.testing.assert_close(normed, ref_normed, rtol=8e-3, atol=1e-5)
                # The shipped norm kernel on the same sum: allow one BF16 ulp for reduction grouping.
                norm = torch.nn.Module(); norm.weight = w; norm.variance_epsilon = 1e-6
                torch.testing.assert_close(normed, FusedRMSNorm(norm)(total), rtol=4e-3, atol=0)
                for t, old in zip((x, y), before):
                    torch.testing.assert_close(t, old, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            kernels.add_rms_norm(x.float(), y.float(), w.float(), 1e-6)
        with self.assertRaises(ValueError):
            kernels.add_rms_norm(x, y[:, :64].contiguous(), w, 1e-6)

    @torch.inference_mode()
    def test_silu_mul_matches_native(self):
        torch.manual_seed(12)
        for rows, cols in ((1, 9728), (4, 9728), (16, 9728), (3, 512), (2, 96)):
            with self.subTest(rows=rows, cols=cols):
                gate_up = torch.randn(rows, 2 * cols, device="cuda", dtype=torch.bfloat16) * 4
                gate_up[0, :8] = torch.tensor([-40., -12., -1., 0., 1., 12., 40., 88.], device="cuda")
                out = kernels.silu_mul(gate_up)
                self.assertEqual(tuple(out.shape), (rows, cols))
                self.assertTrue(bool(torch.isfinite(out).all()))
                torch.testing.assert_close(out, ref_silu_mul(gate_up), rtol=1e-2, atol=1e-3)
        # Odd packed width cannot split into equal gate/up halves (contiguous, so only width fails).
        with self.assertRaises(ValueError):
            kernels.silu_mul(gate_up[:, :-1].contiguous())

    @torch.inference_mode()
    def test_qkv_norm_rope_cache_values_slots_and_untouched_capacity(self):
        torch.manual_seed(13)
        # Tables must be 128 wide: the runtime builds them from the model's own rotary module.
        model = tiny_model(torch.bfloat16, head_dim=128).cuda()
        for batch, heads, kv_heads, capacity in ((1, 32, 8, 544), (3, 32, 8, 640), (16, 32, 8, 640), (2, 4, 2, 129)):
            table = DecodeState(model, 1, 1, capacity - 1)
            self.assertEqual(tuple(table.cos.shape), (capacity, 128))
            for position in (0, capacity // 2, capacity - 1):
                with self.subTest(batch=batch, heads=heads, capacity=capacity, position=position):
                    qkv = torch.randn(batch, (heads + 2 * kv_heads) * 128, device="cuda", dtype=torch.bfloat16)
                    qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
                    kw = torch.randn_like(qw)
                    pos = torch.tensor([position], device="cuda")
                    keys = torch.full((batch, kv_heads, capacity, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
                    values = keys.clone()
                    ref_keys, ref_values = keys.clone(), values.clone()
                    before = qkv.clone()
                    q = kernels.qkv_norm_rope_cache(qkv, qw, kw, 1e-6, 3e-6, table.cos, table.sin, pos, keys, values)
                    ref_q = ref_qkv_norm_rope_cache(qkv, qw, kw, 1e-6, 3e-6, table.cos, table.sin, pos, ref_keys, ref_values)
                    self.assertEqual(tuple(q.shape), (batch, heads, 1, 128))
                    self.assertTrue(q.is_contiguous() and q.dtype == torch.bfloat16)
                    torch.testing.assert_close(q, ref_q, rtol=0.02, atol=0.02)
                    torch.testing.assert_close(keys[:, :, position], ref_keys[:, :, position], rtol=0.02, atol=0.02)
                    torch.testing.assert_close(values[:, :, position], ref_values[:, :, position], rtol=0, atol=0)
                    untouched = torch.arange(capacity, device="cuda") != position
                    self.assertTrue(bool(keys[:, :, untouched].isnan().all()))
                    self.assertTrue(bool(values[:, :, untouched].isnan().all()))
                    self.assertTrue(bool(torch.isfinite(q).all()))
                    torch.testing.assert_close(qkv, before, rtol=0, atol=0)

    @torch.inference_mode()
    def test_qkv_exact_rounding_at_absolute_position_and_contract_violations(self):
        torch.manual_seed(15)
        # Self-consistent fixture: capacity 640, slot 613, tables built for exactly that capacity.
        model = tiny_model(torch.bfloat16, head_dim=128).cuda()
        capacity, position = 640, 613
        table = DecodeState(model, 1, 1, capacity - 1)
        pos = torch.tensor([position], device="cuda")
        qw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        kw = torch.randn_like(qw)
        v_rows = torch.randn(3, 8 * 128, device="cuda", dtype=torch.bfloat16)
        qkv = torch.cat((torch.ones(3, 32 * 128, device="cuda"), torch.full((3, 8 * 128), 2., device="cuda"),
                         v_rows.float()), dim=-1).bfloat16()
        keys = torch.zeros(3, 8, capacity, 128, device="cuda", dtype=torch.bfloat16)
        values = keys.clone()
        q = kernels.qkv_norm_rope_cache(qkv, qw, kw, 0., 0., table.cos, table.sin, pos, keys, values)
        # Exact normalization (eps 0, constant rows) removes reduction uncertainty, so RoPE
        # rounding must match bit for bit. Angles come from the native rotary module at the
        # absolute position, independently of the table, proving the kernel used slot 613.
        cos, sin = model.model.rotary_emb(torch.empty(0, dtype=torch.bfloat16, device="cuda"),
                                          torch.tensor([[position]], device="cuda"))
        ones_q = torch.ones(3, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
        ones_k = torch.ones(3, 8, 1, 128, device="cuda", dtype=torch.bfloat16)
        ref_q, ref_k = apply_rotary_pos_emb(ones_q * qw, ones_k * kw, cos, sin)
        torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
        torch.testing.assert_close(keys[:, :, position], ref_k[:, :, 0], rtol=0, atol=0)
        torch.testing.assert_close(values[:, :, position], v_rows.view(3, 8, 128), rtol=0, atol=0)
        others = torch.arange(capacity, device="cuda") != position
        self.assertEqual(int(keys[:, :, others].count_nonzero()), 0)
        self.assertEqual(int(values[:, :, others].count_nonzero()), 0)
        # Contract violations the wrapper must reject before any launch:
        with self.assertRaises(ValueError):  # position must be one int64 device value
            kernels.qkv_norm_rope_cache(qkv, qw, kw, 0., 0., table.cos, table.sin, pos.int(), keys, values)
        with self.assertRaises(ValueError):  # a partial head: width is not a multiple of 128
            kernels.qkv_norm_rope_cache(qkv[:, :-64].contiguous(), qw, kw, 0., 0., table.cos, table.sin, pos, keys, values)
        with self.assertRaises(ValueError):  # tables built for another capacity than the cache
            short = DecodeState(model, 1, 1, capacity - 2)
            kernels.qkv_norm_rope_cache(qkv, qw, kw, 0., 0., short.cos, short.sin, pos, keys, values)
        with self.assertRaises(ValueError):  # cache batch differs from the packed rows
            kernels.qkv_norm_rope_cache(qkv, qw, kw, 0., 0., table.cos, table.sin, pos, keys[:1].contiguous(), values[:1].contiguous())

    @torch.inference_mode()
    def test_cuda_graph_replay_follows_position_and_inputs(self):
        torch.manual_seed(14)
        model = tiny_model(torch.bfloat16, head_dim=128).cuda()
        capacity = 641
        table = DecodeState(model, 1, 1, capacity - 1)
        self.assertEqual(tuple(table.cos.shape), (capacity, 128))
        qkv = torch.randn(3, 48 * 128, device="cuda", dtype=torch.bfloat16)
        qw = torch.randn(128, device="cuda", dtype=torch.bfloat16); kw = torch.randn_like(qw)
        pos = torch.tensor([0], device="cuda")
        keys = torch.zeros(3, 8, capacity, 128, device="cuda", dtype=torch.bfloat16); values = keys.clone()
        x = torch.randn(3, 2560, device="cuda", dtype=torch.bfloat16); y = torch.randn_like(x)
        w = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
        gate_up = torch.randn(3, 2 * 9728, device="cuda", dtype=torch.bfloat16)

        def call():
            return (kernels.qkv_norm_rope_cache(qkv, qw, kw, 1e-6, 1e-6, table.cos, table.sin, pos, keys, values),
                    *kernels.add_rms_norm(x, y, w, 1e-6), kernels.silu_mul(gate_up))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = call()
        pointers = [t.data_ptr() for t in outputs]
        for position in (1, 300, 640, 7):
            qkv.normal_(); x.normal_(); y.normal_(); gate_up.normal_()
            keys.zero_(); values.zero_()
            pos.fill_(position)
            graph.replay()
            ref_keys, ref_values = keys.clone().zero_(), values.clone().zero_()
            ref_q = ref_qkv_norm_rope_cache(qkv, qw, kw, 1e-6, 1e-6, table.cos, table.sin, pos, ref_keys, ref_values)
            torch.testing.assert_close(outputs[0], ref_q, rtol=0.02, atol=0.02)
            torch.testing.assert_close(keys, ref_keys, rtol=0.02, atol=0.02)
            torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
            ref_total, ref_normed = ref_add_rms_norm(x, y, w, 1e-6)
            torch.testing.assert_close(outputs[1], ref_total, rtol=0, atol=0)
            torch.testing.assert_close(outputs[2], ref_normed, rtol=8e-3, atol=1e-5)
            torch.testing.assert_close(outputs[3], ref_silu_mul(gate_up), rtol=1e-2, atol=1e-3)
            self.assertEqual(pointers, [t.data_ptr() for t in outputs])


if __name__ == "__main__":
    unittest.main()
