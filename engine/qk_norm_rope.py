"""Decode-only Q/K fusion; Qwen3 4.51.3 owns every other operation."""

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from kernels.qk_norm_rope import qk_norm_rope


class DecodeQKNormRoPE(Qwen3Attention):
    @torch.no_grad()
    def __init__(self, reference):
        # Reuse all modules and metadata; do not allocate or initialize weights.
        torch.nn.Module.__init__(self)
        for name in (
            "config", "layer_idx", "head_dim", "num_key_value_groups", "scaling",
            "attention_dropout", "is_causal", "sliding_window", "q_proj", "k_proj",
            "v_proj", "o_proj", "q_norm", "k_norm",
        ):
            setattr(self, name, getattr(reference, name))
        self.train(reference.training)
        # One [q | k | v] projection for the fused decode step. The three modules
        # keep row views of it, so prefill and the state dict see the same values
        # without a second copy of the weights.
        self.qkv_weight = torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight))
        start = 0
        for module in (self.q_proj, self.k_proj, self.v_proj):
            stop = start + module.weight.shape[0]
            module.weight = torch.nn.Parameter(self.qkv_weight[start:stop], requires_grad=False)
            start = stop

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_value=None, cache_position=None, **kwargs):
        cos, sin = position_embeddings
        # Shape/type checks only: no device-value reads during CUDA capture.
        # Prefill, including length-one prefill, retains the original path.
        if (
            hidden_states.shape[1] != 1 or not hidden_states.is_cuda
            or hidden_states.dtype != torch.bfloat16 or self.head_dim != 128
            or self.training or attention_mask is None
            or getattr(past_key_value, "prefilling", False)
            or kwargs.get("output_attentions", False)
            or self.config._attn_implementation not in ("sdpa", "starter_decode_gqa")
            or cos.ndim != 3 or cos.shape != sin.shape
            or cos.shape[0] not in (1, hidden_states.shape[0])
            or cos.shape[1:] != (1, 128)
            or cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16
        ):
            return super().forward(
                hidden_states, position_embeddings, attention_mask,
                past_key_value=past_key_value, cache_position=cache_position, **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        head_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(head_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(head_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        q, k = qk_norm_rope(
            q, k, self.q_norm.weight, self.k_norm.weight, cos, sin,
            self.q_norm.variance_epsilon, self.k_norm.variance_epsilon,
        )
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx, {
                "sin": sin, "cos": cos, "cache_position": cache_position,
            })
        output, weights = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation](
            self, q, k, v, attention_mask, dropout=0.0, scaling=self.scaling,
            sliding_window=self.sliding_window, **kwargs,
        )
        return self.o_proj(output.reshape(*input_shape, -1).contiguous()), weights
