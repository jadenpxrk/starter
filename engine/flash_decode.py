"""PyTorch 2.5.1 native variable-length FlashAttention for static-cache decode.

A (sequence, KV-head) pair is a virtual batch element. This makes the existing
head-major cache a flat varlen input by VIEW, with no KV relayout or copies.
Only DecodeState's uniform, initialized prefix mask may use this interface.
There is no timing selector, custom floating-point kernel, or telemetry.
"""

import torch


_INT32_MAX = 2**31 - 1


def _native_flash(query, key, value, cu_q, cu_k, capacity, used, scale):
    # Exact schema: aten/src/ATen/native/native_functions.yaml at v2.5.1.
    return torch.ops.aten._flash_attention_forward.default(
        query, key, value, cu_q, cu_k, 1, capacity, 0.0, False, False,
        scale=scale, window_size_left=None, window_size_right=None,
        seqused_k=used, alibi_slopes=None,
    )[0]


class FlashDecodeContext:
    def __init__(self, batch, query_heads, kv_heads, key_positions):
        capacity = key_positions.numel()
        virtual_batch = batch * kv_heads
        if (batch < 1 or kv_heads < 1 or query_heads <= kv_heads
                or query_heads % kv_heads or capacity < 1
                or virtual_batch * capacity > _INT32_MAX
                or key_positions.ndim != 1 or key_positions.dtype != torch.int64):
            raise ValueError("unsupported Flash decode metadata")
        self.batch = batch
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.groups = query_heads // kv_heads
        self.capacity = capacity
        self.virtual_batch = virtual_batch
        self.key_positions = key_positions
        self.cu_q = torch.arange(virtual_batch + 1, dtype=torch.int32,
                                 device=key_positions.device)
        self.cu_k = self.cu_q * capacity
        self.used = torch.zeros(virtual_batch, dtype=torch.int32,
                                device=key_positions.device)
        self.mask = None

    def prepare(self, position):
        """Called once per step, before cache update/attention in all layers.

        Caller invariant: 0 <= position < capacity. DecodeState guarantees it.
        The used length includes the K/V slot the current forward will write.
        Contents never cross to the host; graph replays update these buffers.
        """
        if (position.shape != (1,) or position.dtype != torch.int64
                or position.device != self.used.device):
            raise ValueError("position must be one int64 value on the cache device")
        self.used.copy_(position)  # Broadcast and cast integer metadata only.
        self.used.add_(1)
        self.mask = (self.key_positions <= position).view(1, 1, 1, -1)
        return self.mask

    def matches(self, query, key, value, mask):
        """Validate layout and the exact mask issued by prepare(); no value reads."""
        return (mask is self.mask and mask is not None
                and query.shape == (self.batch, self.query_heads, 1, 128)
                and key.shape == (self.batch, self.kv_heads, self.capacity, 128)
                and value.shape == key.shape
                and all(t.dtype == torch.bfloat16 and t.device == self.used.device
                        and t.is_contiguous() for t in (query, key, value)))

    def attention(self, query, key, value, mask, scale):
        if torch.is_grad_enabled() or not query.is_cuda or not self.matches(query, key, value, mask):
            raise ValueError("Flash decode requires its prepared prefix and contiguous BF16 CUDA Q/K/V")
        return self._views_and_call(query, key, value, scale)

    def _views_and_call(self, query, key, value, scale):
        # virtual index = batch_index * Hkv + kv_head; query heads for it are
        # kv_head*G ... kv_head*G+G-1. No query ever sees another KV head/batch.
        q = query.view(self.virtual_batch, self.groups, 128)
        k = key.view(self.virtual_batch * self.capacity, 1, 128)
        v = value.view(self.virtual_batch * self.capacity, 1, 128)
        output = _native_flash(q, k, v, self.cu_q, self.cu_k,
                               self.capacity, self.used, scale)
        return output.reshape(self.batch, 1, self.query_heads, 128)


def make_flash_context(model, batch, max_new_tokens, key_positions):
    """Fixed metadata dispatch; no performance or prompt-content selection.

    Keep the exact old path for CPU, non-BF16, non-GQA, other head widths, and
    offset sizes the pinned int32 varlen interface cannot represent.
    """
    config = model.config
    hq, hk = config.num_attention_heads, config.num_key_value_heads
    width = getattr(config, "head_dim", config.hidden_size // hq)
    if (model.device.type != "cuda" or model.dtype != torch.bfloat16
            or config._attn_implementation != "starter_decode_gqa"
            or width != 128 or hk < 1 or hq <= hk or hq % hk
            or max_new_tokens <= 1 or batch * hk * key_positions.numel() > _INT32_MAX):
        return None
    return FlashDecodeContext(batch, hq, hk, key_positions)
