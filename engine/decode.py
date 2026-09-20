"""Fixed-address KV storage and CUDA graph replay for Qwen3 decode."""

import torch
from transformers import StaticCache


class StaticKVCache(StaticCache):
    prefilling = False

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cached = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if self.prefilling:
            return key_states, value_states
        return cached


def qwen_forward(model, input_ids, cache, positions, attention_mask=None):
    base = model.model
    x = base.embed_tokens(input_ids)
    position_ids = positions.unsqueeze(0)
    position_embeddings = base.rotary_emb(x, position_ids)
    for layer in base.layers:
        x = layer(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=positions,
            position_embeddings=position_embeddings,
        )[0]
    return model.lm_head(base.norm(x[:, -1:, :]))


class DecodeState:
    def __init__(self, model, batch_size, prompt_length, max_new_tokens):
        self.model = model
        self.shape = (batch_size, prompt_length, max_new_tokens)
        capacity = prompt_length + max_new_tokens
        self.cache = StaticKVCache(
            config=model.config,
            max_batch_size=batch_size,
            max_cache_len=capacity,
            device=model.device,
            dtype=model.dtype,
        )
        self.tokens = torch.zeros((batch_size, 1), dtype=torch.int64, device=model.device)
        self.position = torch.zeros(1, dtype=torch.int64, device=model.device)
        self.key_positions = torch.arange(capacity, device=model.device)
        self.graph = None
        self.linear_policy = "torch"

    def prefill(self, input_ids):
        prompt_length = input_ids.shape[1]
        # Overwrite every prompt slot; prefill attends only to the new K/V.
        # Decode masks the remaining slots, so old prompt data is never reused.
        self.cache.prefilling = True
        logits = qwen_forward(
            self.model, input_ids, self.cache, self.key_positions[:prompt_length],
        )
        self.cache.prefilling = False
        self.tokens.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        self.position.fill_(prompt_length)
        return logits

    def step(self):
        mask = (self.key_positions <= self.position).view(1, 1, 1, -1)
        plan = getattr(self.model, "decode_linear_plan", None)
        if plan is None:
            logits = qwen_forward(self.model, self.tokens, self.cache, self.position, mask)
        else:
            old_active, old_policy = plan.active, plan.policy
            plan.active, plan.policy = True, self.linear_policy
            try:
                logits = qwen_forward(self.model, self.tokens, self.cache, self.position, mask)
            finally:
                plan.active, plan.policy = old_active, old_policy
        self.tokens.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        self.position.add_(1)
        return logits

    def capture(self):
        tokens = self.tokens.clone()
        position = self.position.clone()
        plan = getattr(self.model, "decode_linear_plan", None)
        if plan is None:
            self.graph = self._capture_graph(tokens, position)
        else:
            self.graph, self.linear_policy = plan.capture(self, tokens, position)
        self.tokens.copy_(tokens)
        self.position.copy_(position)

    def _capture_graph(self, tokens, position):
        current_stream = torch.cuda.current_stream()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self.tokens.copy_(tokens)
                self.position.copy_(position)
                self.step()
        current_stream.wait_stream(warmup_stream)
        self.tokens.copy_(tokens)
        self.position.copy_(position)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=warmup_stream):
            self.step()
        current_stream.wait_stream(warmup_stream)
        self.tokens.copy_(tokens)
        self.position.copy_(position)
        return graph
