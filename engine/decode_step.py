"""One captured Qwen3 decode step that walks the loaded layers directly.

Same formula as Qwen3DecoderLayer in Transformers 4.51.3 with decode-only
fusions: packed Q/K/V projection, head norm + RoPE + cache write in one
launch, residual add folded into the following RMSNorm, SiLU * up in one
launch. Prefill keeps the module path; DecodeState.step chooses this only on
the CUDA/BF16 static-cache configuration that owns a Flash decode context.

Decode-sized projections (at most kernels.skinny_gemm.MAX_ROWS rows) run as
split-K Triton weight streams whose FP32 partials the fused consumers add and
round. Gate/up uses the paired split-K experiment in split_swiglu for long
reductions, otherwise the original fused SiLU/product kernel. Larger rows and
tile-misaligned shapes keep F.linear. The rule is static per shape.
"""

import torch
from torch.nn import functional as F

from kernels.decode_fused import add_rms_norm, qkv_norm_rope_cache, silu_mul
from kernels.skinny_gemm import linear_partials, supports


def linear_silu_mul(x, weight):
    # Import only on the CUDA skinny path; CPU/native fallbacks need no Triton.
    from kernels.split_swiglu import linear_silu_mul as split_silu_mul
    return split_silu_mul(x, weight)


def _project(x, weight):
    """x @ weight.T as FP32 split-K partials when decode-sized, else BF16 output."""
    return linear_partials(x, weight) if supports(x, weight) else F.linear(x, weight)


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
            _project(normed, attn.qkv_weight), attn.q_norm.weight, attn.k_norm.weight,
            attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
            cos, sin, position, keys, values,
        )
        attended = context.attention(q, keys, values, mask, attn.scaling)
        post = layer.post_attention_layernorm
        x, normed = add_rms_norm(
            x, _project(attended.view(batch, -1), attn.o_proj.weight),
            post.weight, post.variance_epsilon,
        )
        hidden = _mlp_hidden(normed, layer.mlp.gate_up_weight)
        following = layers[index + 1].input_layernorm if index + 1 < len(layers) else base.norm
        x, normed = add_rms_norm(
            x, _project(hidden, layer.mlp.down_proj.weight),
            following.weight, following.variance_epsilon,
        )
    return F.linear(normed, model.lm_head.weight).unsqueeze(1)
