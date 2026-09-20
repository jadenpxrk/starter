"""Full-model causal verification of a fixed block; no draft model or quantization.

K/V keep #12's physical layout. A virtual batch item is (sequence, KV head),
but unlike single-token decode its query length is the actual block length.
PyTorch 2.5.1 native Flash uses bottom-right causality: row i sees [0,start+i].
"""

import torch
from torch.nn import functional as F


def _native_block(q, k, v, cu_q, cu_k, width, capacity, used, scale):
    return torch.ops.aten._flash_attention_forward.default(
        q, k, v, cu_q, cu_k, width, capacity, 0.0, True, False,
        scale=scale, window_size_left=None, window_size_right=None,
        seqused_k=used, alibi_slopes=None,
    )[0]


class BlockContext:
    def __init__(self, batch, nq, nk, capacity, width, device):
        if (batch < 1 or nk < 1 or nq <= nk or nq % nk or not 1 < width <= capacity
                or batch * nk * capacity > 2**31 - 1):
            raise ValueError("unsupported block attention metadata")
        self.batch, self.nq, self.nk = batch, nq, nk
        self.capacity, self.width = capacity, width
        self.virtual = batch * nk
        offsets = torch.arange(self.virtual + 1, dtype=torch.int32, device=device)
        self.cu_q, self.cu_k = offsets * width, offsets * capacity
        self.used = torch.empty(self.virtual, dtype=torch.int32, device=device)

    def prepare(self, start):
        if start.shape != (1,) or start.dtype != torch.int64 or start.device != self.used.device:
            raise ValueError("one device-side int64 start is required")
        self.used.copy_(start)
        self.used.add_(self.width)

    def attention(self, q, k, v, scale):
        b, w, nq, nk, c = self.batch, self.width, self.nq, self.nk, self.capacity
        if (q.shape != (b, w, nq, 128) or k.shape != (b, nk, c, 128)
                or v.shape != k.shape or torch.is_grad_enabled() or not q.is_cuda
                or any(t.dtype != torch.bfloat16 or t.device != self.used.device
                       or not t.is_contiguous() for t in (q, k, v))):
            raise ValueError("block attention requires contiguous BF16 Q/K/V and inference mode")
        return self._views_and_call(q, k, v, scale)

    def _views_and_call(self, q, k, v, scale):
        b, w, nq, nk, c = self.batch, self.width, self.nq, self.nk, self.capacity
        # This small Q relayout is NOT a K/V relayout or head expansion.
        qv = q.view(b, w, nk, nq // nk, 128).permute(0, 2, 1, 3, 4).contiguous()
        qv = qv.view(self.virtual * w, nq // nk, 128)
        kv = k.view(self.virtual * c, 1, 128)
        vv = v.view_as(kv)
        out = _native_block(qv, kv, vv, self.cu_q, self.cu_k, w, c, self.used, scale)
        return out.view(b, nk, w, nq // nk, 128).permute(0, 2, 1, 3, 4).contiguous().view(b, w, nq, 128)


def verify_forward(model, cache, inputs, start, context, cos, sin):
    """Return logits for EVERY block position; same full weights and casts as #12."""
    from kernels.decode_fused import add_rms_norm, silu_mul
    from kernels.verify import verify_qkv

    b, width = inputs.shape
    base, layers = model.model, model.model.layers
    context.prepare(start)
    x = base.embed_tokens(inputs).reshape(b * width, -1)
    normed = layers[0].input_layernorm(x)
    for i, layer in enumerate(layers):
        a, m = layer.self_attn, layer.mlp
        keys, values = cache.key_cache[i], cache.value_cache[i]
        q = verify_qkv(F.linear(normed, a.qkv_weight), a.q_norm.weight, a.k_norm.weight,
                       a.q_norm.variance_epsilon, a.k_norm.variance_epsilon,
                       cos, sin, keys, values, width, start)
        attended = context.attention(q, keys, values, a.scaling)
        post = layer.post_attention_layernorm
        x, normed = add_rms_norm(x, F.linear(attended.view(b * width, -1), a.o_proj.weight),
                                 post.weight, post.variance_epsilon)
        hidden = silu_mul(F.linear(normed, m.gate_up_weight))
        following = layers[i + 1].input_layernorm if i + 1 < len(layers) else base.norm
        x, normed = add_rms_norm(x, F.linear(hidden, m.down_proj.weight),
                                 following.weight, following.variance_epsilon)
    return F.linear(normed, model.lm_head.weight).view(b, width, -1)


class VerificationPlan:
    def __init__(self, state, width):
        self.state, self.width = state, width
        cfg = state.model.config
        b, _, _ = state.shape
        self.context = BlockContext(b, cfg.num_attention_heads, cfg.num_key_value_heads,
                                     state.cache.max_cache_len, width, state.tokens.device)
        self.inputs = torch.empty((b, width), device=state.tokens.device, dtype=torch.int64)
        self.input_host = torch.empty((b, width), dtype=torch.int64, pin_memory=True)
        self.host = torch.empty((b, width), dtype=torch.int64, pin_memory=True)
        self.ready = torch.cuda.Event()
        self.graph, self.predictions = None, None
        self.pending = False

    def _call(self):
        s = self.state
        return verify_forward(s.model, s.cache, self.inputs, s.position,
                              self.context, s.cos, s.sin).argmax(dim=-1)

    def capture(self):
        if self.graph is not None:
            return
        # Capture ONLY during the existing shape warmup. Warmup writes beyond
        # the valid prompt, never into it; it does not advance tokens/position.
        self.inputs.copy_(self.state.tokens.expand_as(self.inputs))
        current = torch.cuda.current_stream(self.inputs.device)
        stream = torch.cuda.Stream(device=self.inputs.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._call()
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # Private pool; do not alias prefill or one-token decode graph storage.
        with torch.cuda.graph(graph, stream=stream):
            self.predictions = self._call()
        current.wait_stream(stream)
        self.graph = graph

    def enqueue(self, inputs):
        if self.graph is None or self.pending:
            raise RuntimeError("verifier must be warmed and idle")
        source = torch.tensor(inputs, dtype=torch.int64)
        if source.shape != self.inputs.shape:
            raise ValueError("fixed verifier input shape required")
        self.input_host.copy_(source)
        self.inputs.copy_(self.input_host, non_blocking=True)
        self.graph.replay()
        self.host.copy_(self.predictions, non_blocking=True)
        self.ready.record()
        self.pending = True

    def finish(self):
        if not self.pending:
            raise RuntimeError("no verification is pending")
        self.ready.synchronize()
        self.pending = False
        return self.host.tolist()

    def commit(self, count):
        if self.pending or not 1 <= count <= self.width:
            raise ValueError("verification must finish before committing its accepted prefix")
        # The last emitted prediction is NOT in the cache. The accepted input
        # block prefix is cached, hence next position increases by count.
        # Rejected physical suffix slots remain invisible and are overwritten
        # before they can become valid in the next full-model forward.
        self.state.tokens.copy_(self.predictions[:, count - 1:count])
        self.state.position.add_(count)
