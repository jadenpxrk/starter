"""Captured full-prompt prefill; no changes to the passing decode/emit path.

Reuse #10's packed weights, residual/norm and SiLU/product kernels on B*T
rows, and invoke PyTorch 2.5.1's dense causal FlashAttention with native GQA.
No repeated K/V heads, prompt-sized mask, weight copy, or performance selector.
CPU/unsupported configurations retain the original qwen_forward path.
"""

import torch
from torch.nn import functional as F


def _native_attention(q, keys, values, length, scale):
    # Physical caches are head-major with capacity C. The sliced, transposed
    # views keep their actual strides and expose exactly T prompt positions.
    # v2.5.1 mha_fwd accepts these strides (only the last stride must be one).
    k = keys[:, :, :length, :].transpose(1, 2)
    v = values[:, :, :length, :].transpose(1, 2)
    return torch.ops.aten._flash_attention_forward.default(
        q, k, v, None, None, length, length, 0.0, True, False,
        scale=scale, window_size_left=None, window_size_right=None,
        seqused_k=None, alibi_slopes=None,
    )[0]


def prefill_forward(model, cache, inputs, cos, sin, *, return_tokens=False):
    """Full fresh prefill; return last-position logits, or int64 IDs[B,1]."""
    from kernels.decode_fused import add_rms_norm, silu_mul
    from kernels.prefill import prefill_qkv

    batch, length = inputs.shape
    base = model.model
    layers = base.layers
    x = base.embed_tokens(inputs).reshape(batch * length, -1)
    normed = layers[0].input_layernorm(x)
    for index, layer in enumerate(layers):
        attn = layer.self_attn
        keys, values = cache.key_cache[index], cache.value_cache[index]
        q = prefill_qkv(
            F.linear(normed, attn.qkv_weight), attn.q_norm.weight, attn.k_norm.weight,
            attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
            cos, sin, keys, values, length,
        )
        attended = _native_attention(q, keys, values, length, attn.scaling)
        del q
        projected = F.linear(attended.reshape(batch * length, -1), attn.o_proj.weight)
        del attended
        post = layer.post_attention_layernorm
        x, normed = add_rms_norm(x, projected, post.weight, post.variance_epsilon)
        del projected
        hidden = silu_mul(F.linear(normed, layer.mlp.gate_up_weight))
        branch = F.linear(hidden, layer.mlp.down_proj.weight)
        del hidden
        last = index + 1 == len(layers)
        following = base.norm if last else layers[index + 1].input_layernorm
        if last:
            # Preserve the baseline's last-position-only final normalization.
            x = x.view(batch, length, -1)[:, -1, :].contiguous()
            branch = branch.view(batch, length, -1)[:, -1, :].contiguous()
        x, normed = add_rms_norm(x, branch, following.weight, following.variance_epsilon)
        del branch
    if return_tokens:
        from greedy_head import greedy_head
        return greedy_head(normed, model.lm_head.weight)
    return F.linear(normed, model.lm_head.weight).unsqueeze(1)


class PrefillPlan:
    def __init__(self, model, cache, batch, length, cos, sin):
        self.model, self.cache = model, cache
        self.cos, self.sin = cos, sin
        self.inputs = torch.empty((batch, length), dtype=torch.int64, device=model.device)
        self.graph = None
        self.logits = None
        self.logit_graph = None
        # Normal Engine use captures only the token graph. The separate logits
        # graph remains available for diagnostics without changing output types.
        self.token_graph = None
        self.token_ids = None

    def run(self, inputs, *, return_tokens=False):
        if (torch.is_grad_enabled() or inputs.shape != self.inputs.shape
                or inputs.dtype != torch.int64 or inputs.device != self.inputs.device):
            raise ValueError("prefill requires fixed-shape int64 inputs under inference mode")
        self.inputs.copy_(inputs)
        graph = self.token_graph if return_tokens else self.logit_graph

        def forward():
            if return_tokens:
                return prefill_forward(self.model, self.cache, self.inputs,
                                       self.cos, self.sin, return_tokens=True)
            return prefill_forward(self.model, self.cache, self.inputs, self.cos, self.sin)

        if graph is None:
            current = torch.cuda.current_stream(self.inputs.device)
            stream = torch.cuda.Stream(device=self.inputs.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    forward()
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            # Each graph owns a private pool. Engine captures only token mode;
            # optional diagnostic-logits calls must not alias its live results.
            with torch.cuda.graph(graph, stream=stream):
                result = forward()
            current.wait_stream(stream)
            if return_tokens:
                self.token_graph, self.token_ids = graph, result
            else:
                self.logit_graph, self.logits = graph, result
        self.graph = graph  # Most recently used graph; diagnostic compatibility.
        graph.replay()
        # Graph-owned storage until the next run of that mode. Caller copies
        # IDs or consumes logits before decode; it must not mutate the result.
        return self.token_ids if return_tokens else self.logits


def make_prefill_plan(model, cache, batch, length, cos, sin):
    """Fixed model/layout dispatch, evaluated once per shape; no GPU value reads."""
    cfg, base = model.config, model.model
    nq, nk = cfg.num_attention_heads, cfg.num_key_value_heads
    width = getattr(cfg, "head_dim", cfg.hidden_size // nq)
    if (model.device.type != "cuda" or model.dtype != torch.bfloat16 or model.training
            or torch.is_grad_enabled() or batch < 1 or length <= 1
            or width != 128 or nk < 1 or nq <= nk or nq % nk
            or not 1 <= cfg.hidden_size <= 8192 or cfg.hidden_act != "silu"
            or cfg._attn_implementation != "starter_decode_gqa"
            or getattr(cfg, "rope_scaling", None) is not None
            or getattr(cfg, "use_sliding_window", False) or not len(base.layers)):
        return None
    capacity = cache.max_cache_len
    if (length > capacity or cos.shape != (capacity, width) or sin.shape != cos.shape
            or len(cache.key_cache) != len(base.layers)
            or len(cache.value_cache) != len(base.layers)):
        return None
    tensors = [cos, sin, base.embed_tokens.weight, base.norm.weight, model.lm_head.weight]
    for i, layer in enumerate(base.layers):
        a, m = layer.self_attn, layer.mlp
        if (layer.training or a.training or a.sliding_window is not None
                or not hasattr(a, "qkv_weight") or not hasattr(m, "gate_up_weight")
                or a.qkv_weight.shape != ((nq + 2 * nk) * width, cfg.hidden_size)
                or any(proj.bias is not None for proj in (a.q_proj, a.k_proj, a.v_proj, a.o_proj,
                                                          m.down_proj))):
            return None
        k, v = cache.key_cache[i], cache.value_cache[i]
        if k.shape != (batch, nk, capacity, width) or v.shape != k.shape:
            return None
        tensors.extend((a.qkv_weight, a.q_norm.weight, a.k_norm.weight, a.o_proj.weight,
                        m.gate_up_weight, m.down_proj.weight, k, v,
                        layer.input_layernorm.weight, layer.post_attention_layernorm.weight))
    if any(t.device != model.device or t.dtype != torch.bfloat16 or not t.is_contiguous()
           for t in tensors):
        return None
    return PrefillPlan(model, cache, batch, length, cos, sin)
