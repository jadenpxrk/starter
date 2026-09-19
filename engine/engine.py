"""Qwen3 4B greedy decoding with fused RMSNorm and CUDA graph replay."""

import torch
from transformers import AutoModelForCausalLM

from decode import DecodeState
from decode_attention import install_decode_attention
from kernels.rmsnorm import rms_norm
from mlp import PackedMLP


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight = reference.weight
        self.variance_epsilon = reference.variance_epsilon

    def forward(self, x):
        return rms_norm(x, self.weight, self.variance_epsilon)


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )
        install_decode_attention(self.model)
        base = self.model.model
        base.norm = FusedRMSNorm(base.norm)
        for layer in base.layers:
            layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
            layer.post_attention_layernorm = FusedRMSNorm(layer.post_attention_layernorm)
            layer.self_attn.q_norm = FusedRMSNorm(layer.self_attn.q_norm)
            layer.self_attn.k_norm = FusedRMSNorm(layer.self_attn.k_norm)
            layer.mlp = PackedMLP(layer.mlp)
        self.decode_state = None

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        if max_new_tokens <= 0:
            return
        with torch.inference_mode():
            prompt = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
            shape = (prompt.shape[0], prompt.shape[1], max_new_tokens)
            if self.decode_state is None or self.decode_state.shape != shape:
                self.decode_state = None
                self.decode_state = DecodeState(self.model, *shape)
            state = self.decode_state
            state.prefill(prompt)
            if max_new_tokens > 1 and state.graph is None:
                state.capture()
            yield state.tokens[:, 0].tolist()
            for _ in range(max_new_tokens - 1):
                state.graph.replay()
                yield state.tokens[:, 0].tolist()
