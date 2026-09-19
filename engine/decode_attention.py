"""Single-token GQA without materializing repeated K/V heads (HF 4.51.3)."""

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def grouped_decode_attention(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None,
    is_causal=None, **kwargs,
):
    # This is only valid when every query head has the same visibility mask.
    # In particular, never fold the heads of a multi-token prefill into time.
    if (
        query.shape[2] == 1
        and query.shape[1] > key.shape[1]
        and query.shape[1] == key.shape[1] * module.num_key_value_groups
        and attention_mask is not None
        and attention_mask.ndim == 4
        and attention_mask.shape[1:3] == (1, 1)
        and attention_mask.shape[-1] == key.shape[-2]
        and not is_causal
        and dropout == 0.0
    ):
        batch, heads, _, width = query.shape
        kv_heads = key.shape[1]
        # [B, Hkv * G, 1, D] -> [B, Hkv, G, D]. Each SDPA row is an
        # independent query, so row g uses exactly KV head h // G.
        grouped_query = query.reshape(batch, kv_heads, heads // kv_heads, width)
        output = torch.nn.functional.scaled_dot_product_attention(
            grouped_query.contiguous(), key.contiguous(), value.contiguous(),
            attn_mask=attention_mask, dropout_p=0.0, scale=scaling,
            # The G rows are heads, NOT successive causal positions.
            is_causal=False,
        )
        return output.reshape(batch, 1, heads, width).contiguous(), None

    return sdpa_attention_forward(
        module, query, key, value, attention_mask, dropout=dropout,
        scaling=scaling, is_causal=is_causal, **kwargs,
    )


def install_decode_attention(model):
    # Use a new key; do not replace the native "sdpa" reference globally.
    # Qwen3's layers share this config and consult the registry in forward.
    ALL_ATTENTION_FUNCTIONS["starter_decode_gqa"] = grouped_decode_attention
    model.config._attn_implementation = "starter_decode_gqa"
