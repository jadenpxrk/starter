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


def qwen_forward(model, input_ids, cache, positions, attention_mask=None, flash_context=None):
    base = model.model
    x = base.embed_tokens(input_ids)
    position_ids = positions.unsqueeze(0)
    position_embeddings = base.rotary_emb(x, position_ids)
    extra = {} if flash_context is None else {"_flash_decode_context": flash_context}
    for layer in base.layers:
        x = layer(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=positions,
            position_embeddings=position_embeddings,
            **extra,
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
        # Cos/sin for every slot, from the model's own rotary module: one product,
        # cosine and cast per element, so a row equals the per-step computation.
        self.cos, self.sin = (
            table[0].contiguous() for table in model.model.rotary_emb(
                torch.empty(0, dtype=model.dtype, device=model.device),
                self.key_positions.unsqueeze(0),
            )
        )
        self.graph = None
        self.flash_context = None
        self.fused_forward = None
        cuda = model.device.type == "cuda"
        if cuda:
            from flash_decode import make_flash_context
            self.flash_context = make_flash_context(model, batch_size, max_new_tokens, self.key_positions)
            if self.flash_context is not None:
                from decode_step import fused_decode_forward
                self.fused_forward = fused_decode_forward
        self.host = torch.empty((batch_size, 1), dtype=torch.int64, pin_memory=cuda)
        self.ready = torch.cuda.Event() if cuda else None
        self.prefill_plan = None
        if cuda:
            from prefill import make_prefill_plan
            self.prefill_plan = make_prefill_plan(
                model, self.cache, batch_size, prompt_length, self.cos, self.sin,
            )

    def prefill(self, input_ids):
        prompt_length = input_ids.shape[1]
        # Overwrite every prompt slot; prefill attends only to the new K/V.
        # Decode masks the remaining slots, so old prompt data is never reused.
        self.cache.prefilling = True
        try:
            if self.prefill_plan is None:
                logits = qwen_forward(
                    self.model, input_ids, self.cache, self.key_positions[:prompt_length],
                )
            else:
                logits = self.prefill_plan.run(input_ids)
        finally:
            self.cache.prefilling = False
        self.tokens.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        self.position.fill_(prompt_length)
        return logits

    def step(self):
        mask = ((self.key_positions <= self.position).view(1, 1, 1, -1)
                if self.flash_context is None else self.flash_context.prepare(self.position))
        if self.fused_forward is None:
            logits = qwen_forward(self.model, self.tokens, self.cache, self.position, mask,
                                  flash_context=self.flash_context)
        else:
            logits = self.fused_forward(self.model, self.cache, self.tokens, self.position,
                                        mask, self.flash_context, self.cos, self.sin)
        self.tokens.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        self.position.add_(1)
        return logits

    def emit(self, max_new_tokens):
        """Yield one host list per output step after prefill.

        The device-to-host copy of step k is enqueued before step k+1 is
        launched, intended to let the pipe write and Python resume overlap
        GPU work. Exactly max_new_tokens - 1 steps run.
        """
        for step in range(max_new_tokens):
            self.host.copy_(self.tokens, non_blocking=True)
            if self.ready is not None:
                self.ready.record()
            if step + 1 < max_new_tokens:
                if self.graph is None:
                    self.step()
                else:
                    self.graph.replay()
            if self.ready is not None:
                self.ready.synchronize()
            yield self.host[:, 0].tolist()

    def capture(self):
        # Shape is known in warmup. Avoid allocating duplicate weights for
        # prefill-only requests or shapes that use the native projection path.
        if self.fused_forward is not None and 1 <= self.shape[0] <= 64:
            from weight_layout import ensure_tiled_weights
            ensure_tiled_weights(self.model)
        tokens = self.tokens.clone()
        position = self.position.clone()
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
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=warmup_stream):
            self.step()
        self.tokens.copy_(tokens)
        self.position.copy_(position)
