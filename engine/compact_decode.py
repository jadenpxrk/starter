"""Inference-only decode pipeline; native GEMMs and #9 FlashAttention remain.

Install AFTER loading/eval/device conversion. Packing relayouts each original
Q/K/V Parameter into disjoint views of one allocation; values, shapes, identities
and the prefill module path are preserved. Do not move/train the model afterward.
No timing selector, prompt-dependent policy, extra weights, or graph telemetry.
"""

import torch
from torch.nn import functional as F


def supported_layout(model):
    c = model.config
    if (model.training or c.hidden_act != "silu" or c.hidden_size > 8192
            or getattr(c, "head_dim", c.hidden_size // c.num_attention_heads) != 128
            or c.num_key_value_heads < 1 or c.num_attention_heads <= c.num_key_value_heads
            or c.num_attention_heads % c.num_key_value_heads
            or c._attn_implementation != "starter_decode_gqa"):
        return False
    h, i, nq, nk = c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads
    if h < 1 or i < 1 or len(model.model.layers) == 0:
        return False
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        if (layer.training or a.training or a.sliding_window is not None
                or not hasattr(m, "gate_up_weight")
                or m.gate_up_weight.shape != (2 * i, h)):
            return False
        projections = ((a.q_proj, (nq * 128, h)), (a.k_proj, (nk * 128, h)),
                       (a.v_proj, (nk * 128, h)), (a.o_proj, (h, nq * 128)),
                       (m.down_proj, (h, i)))
        if any(p.bias is not None or p.weight.shape != shape for p, shape in projections):
            return False
        norms = ((layer.input_layernorm, h), (layer.post_attention_layernorm, h),
                 (a.q_norm, 128), (a.k_norm, 128))
        if any(n.weight.shape != (width,) for n, width in norms):
            return False
        tensors = [p.weight for p, _ in projections] + [n.weight for n, _ in norms] + [m.gate_up_weight]
        if any(t.dtype != torch.bfloat16 or t.device != model.device or not t.is_contiguous()
               for t in tensors):
            return False
    final = model.model.norm.weight
    return (final.shape == (h,) and final.dtype == torch.bfloat16
            and final.device == model.device and final.is_contiguous())


@torch.no_grad()
def pack_qkv(attention):
    """One allocation, unchanged Parameter objects and unchanged BF16 values.

    This is a load-time storage relayout, not a new model or a training adapter.
    set_(Tensor) shares the source's storage, offset, shape and strides in 2.5.1.
    """
    weights = (attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight)
    packed = torch.cat(weights, dim=0)
    offset = 0
    for weight in weights:
        rows = weight.shape[0]
        weight.set_(packed.narrow(0, offset, rows))
        offset += rows
    return packed


class CompactDecode:
    def __init__(self, model):
        if not supported_layout(model):
            raise ValueError("unsupported compact decode model layout")
        self.base = model.model
        self.head = model.lm_head
        self.nq = model.config.num_attention_heads
        self.nk = model.config.num_key_value_heads
        self.packed = tuple(pack_qkv(layer.self_attn) for layer in self.base.layers)

    def forward(self, input_ids, cache, positions, mask, context):
        from kernels.compact_decode import add_rmsnorm, qkv_rope_cache, swiglu

        if (torch.is_grad_enabled() or input_ids.ndim != 2 or input_ids.shape[1] != 1 or cache.prefilling
                or context is None or mask is not context.mask or mask is None
                or positions.shape != (1,) or positions.dtype != torch.int64):
            raise ValueError("compact decode requires one token and its prepared prefix")
        x = self.base.embed_tokens(input_ids)
        cos, sin = self.base.rotary_emb(x, positions.unsqueeze(0))
        residual, branch = x, None
        for index, layer in enumerate(self.base.layers):
            if branch is None:
                normalized = layer.input_layernorm(residual)
            else:
                # The preceding MLP's residual sum is still rounded to BF16
                # BEFORE the next layer's norm. Only its launch is deferred.
                residual, normalized = add_rmsnorm(
                    residual, branch, layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon,
                )
            a = layer.self_attn
            qkv = F.linear(normalized, self.packed[index])
            keys, values = cache.key_cache[index], cache.value_cache[index]
            q = qkv_rope_cache(qkv, a.q_norm.weight, a.k_norm.weight, cos, sin,
                               keys, values, positions, self.nq, self.nk,
                               a.q_norm.variance_epsilon, a.k_norm.variance_epsilon)
            # Same full-context native attention implementation as #9.
            attention = context.attention(q, keys, values, mask, a.scaling)
            branch = a.o_proj(attention.reshape(x.shape[0], 1, self.nq * 128))
            residual, normalized = add_rmsnorm(
                residual, branch, layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.variance_epsilon,
            )
            gate_up = F.linear(normalized, layer.mlp.gate_up_weight)
            branch = layer.mlp.down_proj(swiglu(gate_up))
        _, normalized = add_rmsnorm(residual, branch, self.base.norm.weight,
                                     self.base.norm.variance_epsilon)
        return self.head(normalized)


def install_compact_decode(model):
    # No public-batch table. Unsupported model layouts retain the exact #9 path.
    # A missing Flash context also bypasses this plan at the call site.
    if (getattr(model, "compact_decode", None) is None
            and model.device.type == "cuda" and model.dtype == torch.bfloat16
            and supported_layout(model)):
        model.compact_decode = CompactDecode(model)
