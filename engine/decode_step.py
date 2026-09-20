"""One captured Qwen3 decode step that walks the loaded layers directly.

Same formula as Qwen3DecoderLayer in Transformers 4.51.3 with decode-only
fusions: packed Q/K/V projection, head norm + RoPE + cache write in one
launch, residual add folded into the following RMSNorm, SiLU * up in one
launch. Prefill keeps the module path; DecodeState.step chooses this only on
the CUDA/BF16 static-cache configuration that owns a Flash decode context.

Decode-sized projections (at most kernels.skinny_gemm.MAX_ROWS rows) run as
split-K Triton weight streams whose FP32 partials the fused consumers add and
round; gate/up carries SiLU * up in its epilogue. Larger row counts and
tile-misaligned shapes keep F.linear. The rule is static per shape.
"""

import torch
from torch.nn import functional as F

from kernels.decode_fused import add_rms_norm, qkv_norm_rope_cache, silu_mul
from kernels.skinny_gemm import linear_partials as _row_major_partials, linear_silu_mul, supports


def linear_partials(x, weight, tiles=None):
    """Preserve the projection entry point; optional immutable tile-major storage."""
    if tiles is None:
        return _row_major_partials(x, weight)
    from kernels.tiled_gemm import tiled_partials
    return tiled_partials(x, weight, tiles)


def _project(x, weight, tiles=None):
    """Same partial sums/consumers; installed tile copies alter weight addressing."""
    if supports(x, weight):
        return (linear_partials(x, weight) if tiles is None
                else linear_partials(x, weight, tiles))
    return F.linear(x, weight)


def _mlp_hidden(normed, gate_up_weight):
    if supports(normed, gate_up_weight, pairs=True):
        return linear_silu_mul(normed, gate_up_weight)
    return silu_mul(F.linear(normed, gate_up_weight))


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
            _project(normed, attn.qkv_weight, getattr(attn, "_qkv_weight_tiles", None)),
            attn.q_norm.weight, attn.k_norm.weight,
            attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
            cos, sin, position, keys, values,
        )
        attended = context.attention(q, keys, values, mask, attn.scaling)
        post = layer.post_attention_layernorm
        x, normed = add_rms_norm(
            x, _project(attended.view(batch, -1), attn.o_proj.weight,
                        getattr(attn.o_proj, "_weight_tiles", None)),
            post.weight, post.variance_epsilon,
        )
        hidden = _mlp_hidden(normed, layer.mlp.gate_up_weight)
        following = layers[index + 1].input_layernorm if index + 1 < len(layers) else base.norm
        x, normed = add_rms_norm(
            x, _project(hidden, layer.mlp.down_proj.weight,
                        getattr(layer.mlp.down_proj, "_weight_tiles", None)),
            following.weight, following.variance_epsilon,
        )
    return F.linear(normed, model.lm_head.weight).unsqueeze(1)
