"""One captured Qwen3 decode step that walks the loaded layers directly.

Same formula as Qwen3DecoderLayer in Transformers 4.51.3 with decode-only
fusions: packed Q/K/V projection, head norm + RoPE + cache write in one
launch, residual add folded into the following RMSNorm, SiLU * up in one
launch. Prefill keeps the module path; DecodeState.step chooses this only on
the CUDA/BF16 static-cache configuration that owns a Flash decode context.
"""

import torch
from torch.nn import functional as F

from kernels.decode_fused import add_rms_norm, qkv_norm_rope_cache, silu_mul


def fused_decode_forward(model, cache, tokens, position, mask, context, cos, sin):
    """Logits [batch, 1, vocab] for one token per sequence at ``position``."""
    base = model.model
    layers = base.layers
    batch = tokens.shape[0]
    x = base.embed_tokens(tokens).view(batch, -1)
    normed = layers[0].input_layernorm(x)
    for index, layer in enumerate(layers):
        attn = layer.self_attn
        keys, values = cache.key_cache[index], cache.value_cache[index]
        q = qkv_norm_rope_cache(
            F.linear(normed, attn.qkv_weight), attn.q_norm.weight, attn.k_norm.weight,
            attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
            cos, sin, position, keys, values,
        )
        attended = context.attention(q, keys, values, mask, attn.scaling)
        post = layer.post_attention_layernorm
        x, normed = add_rms_norm(
            x, F.linear(attended.view(batch, -1), attn.o_proj.weight),
            post.weight, post.variance_epsilon,
        )
        hidden = silu_mul(F.linear(normed, layer.mlp.gate_up_weight))
        following = layers[index + 1].input_layernorm if index + 1 < len(layers) else base.norm
        x, normed = add_rms_norm(
            x, F.linear(hidden, layer.mlp.down_proj.weight),
            following.weight, following.variance_epsilon,
        )
    return F.linear(normed, model.lm_head.weight).unsqueeze(1)
