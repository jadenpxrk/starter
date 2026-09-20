"""One captured Qwen3 decode step that walks the loaded layers directly.

Same formula as Qwen3DecoderLayer in Transformers 4.51.3 with decode-only
fusions: packed Q/K/V projection, head norm + RoPE + cache write in one
launch, residual add folded into the projection that produces the branch,
SiLU * up in one launch. Prefill keeps the module path; DecodeState.step
chooses this only on the CUDA/BF16 static-cache configuration that owns a
Flash decode context.

Decode-sized projections (at most kernels.skinny_gemm.MAX_ROWS rows) run as
split-K Triton weight streams. The packed Q/K/V stream leaves FP32 partials
for the norm/RoPE/cache consumer. The o_proj and down_proj streams finish the
residual add in their epilogue and publish per-tile sums of squares; the
following gate/up or Q/K/V stream applies that RMSNorm to its input tiles on
the way in, so no separate residual-norm launch runs between them. The final
norm before the LM head is materialized as before. Larger row counts and
tile-misaligned shapes keep F.linear plus add_rms_norm. Every rule is static
per shape.
"""

import torch
from torch.nn import functional as F

from kernels.decode_fused import add_rms_norm, qkv_norm_rope_cache, silu_mul
from kernels.skinny_gemm import linear_add_stats, linear_partials, linear_silu_mul, supports


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
    # (stats, gain, eps) of a residual norm still to be applied to ``x`` by the
    # next weight stream; None when ``normed`` already holds the normalized rows.
    pending = None
    for index, layer in enumerate(layers):
        attn, mlp = layer.self_attn, layer.mlp
        keys, values = cache.key_cache[index], cache.value_cache[index]
        projected = (linear_partials(x, attn.qkv_weight, pending) if pending is not None
                     else _project(normed, attn.qkv_weight))
        q = qkv_norm_rope_cache(
            projected, attn.q_norm.weight, attn.k_norm.weight,
            attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
            cos, sin, position, keys, values,
        )
        attended = context.attention(q, keys, values, mask, attn.scaling).view(batch, -1)
        post = layer.post_attention_layernorm
        if supports(attended, attn.o_proj.weight) and supports(x, mlp.gate_up_weight, pairs=True):
            x, stats = linear_add_stats(attended, attn.o_proj.weight, x)
            hidden = linear_silu_mul(x, mlp.gate_up_weight, (stats, post.weight, post.variance_epsilon))
        else:
            x, normed = add_rms_norm(
                x, _project(attended, attn.o_proj.weight), post.weight, post.variance_epsilon,
            )
            hidden = _mlp_hidden(normed, mlp.gate_up_weight)
        following = layers[index + 1] if index + 1 < len(layers) else None
        if (following is not None and supports(hidden, mlp.down_proj.weight)
                and supports(x, following.self_attn.qkv_weight)):
            x, stats = linear_add_stats(hidden, mlp.down_proj.weight, x)
            norm = following.input_layernorm
            pending = (stats, norm.weight, norm.variance_epsilon)
        else:
            norm = base.norm if following is None else following.input_layernorm
            x, normed = add_rms_norm(
                x, _project(hidden, mlp.down_proj.weight), norm.weight, norm.variance_epsilon,
            )
            pending = None
    return F.linear(normed, model.lm_head.weight).unsqueeze(1)
