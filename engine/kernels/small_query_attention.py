"""Full-context BF16 GQA decode with a 16-row query tile (Triton 3.1.0).

One program owns one original (batch, KV head) and a disjoint key segment.
All G query heads reuse that segment's K/V. Softmax statistics and weighted
value partials remain FP32; only probabilities supplied to the BF16 P@V dot
and the final attention output are BF16, as in native FlashAttention.

``_fused_attention`` is the same attention in one launch per layer: each
program first completes this KV head's packed Q/K/V rows (split-K sum, one
BF16 rounding, head RMSNorm, RoPE) with the exact cast sequence of
kernels/decode_fused's consumer, the program whose key segment holds the
current slot writes the new K/V row into the cache before it reads it back,
and the last segment program to arrive at a KV head merges the FP32
partials. Arrival is a release/acquire counter that nobody waits on, so no
program depends on another being resident. The legacy split kernels stay
for the module path. No sparsity, telemetry, autotuning or runtime timing.
The scratch buffers belong to one context and require serialized calls.
"""

import math

import torch
import triton
import triton.language as tl


QUERY_TILE = 16
KEY_TILE = 64
MAX_SPLITS = 32


def partition_keys(virtual_heads, capacity, multiprocessors):
    """Metadata-only scheduling heuristic, NOT measured optimal occupancy.

    Aim for two programs per SM, subject to >=one key tile per segment and
    a 32-segment cap. Cover capacity with nonoverlapping KEY_TILE-aligned
    segments. Actual valid length is read on device, independently per head.
    """
    if min(virtual_heads, capacity, multiprocessors) < 1:
        raise ValueError("positive head count, capacity and SM count required")
    tiles = (capacity + KEY_TILE - 1) // KEY_TILE
    desired = (2 * multiprocessors + virtual_heads - 1) // virtual_heads
    splits = min(MAX_SPLITS, tiles, max(1, desired))
    tiles_per_split = (tiles + splits - 1) // splits
    splits = (tiles + tiles_per_split - 1) // tiles_per_split
    return splits, tiles_per_split * KEY_TILE


@triton.jit
def _segments(Q, K, V, USED, PART, STATS, OUT,
              C: tl.constexpr, G: tl.constexpr, S: tl.constexpr,
              SPAN: tl.constexpr, SCALE: tl.constexpr,
              D: tl.constexpr, QM: tl.constexpr, KN: tl.constexpr):
    vh = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(1).to(tl.int64)
    g = tl.arange(0, QM)
    d = tl.arange(0, D)
    n = tl.arange(0, KN)
    # Flattening the original batch/KV-head dimensions preserves their order.
    q = tl.load(Q + (vh * G + g[:, None]) * D + d[None, :],
                mask=g[:, None] < G, other=0.0)
    end = tl.minimum(tl.load(USED + vh), tl.minimum((segment + 1) * SPAN, C))
    maximum = tl.full((QM,), -float("inf"), tl.float32)
    denominator = tl.zeros((QM,), tl.float32)
    numerator = tl.zeros((QM, D), tl.float32)
    # Empty segments take zero iterations and still write neutral statistics.
    for start in range(segment * SPAN, end, KN):
        keys = start + n
        offsets = (vh * C + keys[:, None]) * D + d[None, :]
        # Do not even load invalid capacity: stale NaNs must remain invisible.
        k = tl.load(K + offsets, mask=keys[:, None] < end, other=0.0)
        v = tl.load(V + offsets, mask=keys[:, None] < end, other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee")
        scores = scores * (SCALE * 1.4426950408889634)
        scores = tl.where(keys[None, :] < end, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        probability = tl.exp2(scores - next_maximum[:, None])
        rescale = tl.exp2(maximum - next_maximum)
        denominator = denominator * rescale + tl.sum(probability, axis=1)
        numerator = numerator * rescale[:, None]
        # No FP16/FP8/INT casts; native Flash also feeds BF16 P and V to this dot.
        numerator = tl.dot(probability.to(tl.bfloat16), v, numerator,
                           input_precision="ieee")
        maximum = next_maximum
    if S == 1:
        result = numerator / tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(OUT + (vh * G + g[:, None]) * D + d[None, :],
                 result.to(tl.bfloat16), mask=g[:, None] < G)
    else:
        tl.store(PART + ((vh * S + segment) * G + g[:, None]) * D + d[None, :],
                 numerator, mask=g[:, None] < G)
        stat = (vh * S + segment) * 2 * G + g
        tl.store(STATS + stat, maximum, mask=g < G)
        tl.store(STATS + stat + G, denominator, mask=g < G)


@triton.jit
def _merge(PART, STATS, OUT, G: tl.constexpr, S: tl.constexpr,
           D: tl.constexpr, SS: tl.constexpr):
    query = tl.program_id(0).to(tl.int64)
    vh, g = query // G, query % G
    s = tl.arange(0, SS)
    d = tl.arange(0, D)
    stat = (vh * S + s) * 2 * G + g
    maximum = tl.load(STATS + stat, mask=s < S, other=-float("inf"))
    denominator = tl.load(STATS + stat + G, mask=s < S, other=0.0)
    global_maximum = tl.max(maximum, axis=0)
    # Covers a zero-length standalone operator test without -inf - -inf.
    global_maximum = tl.where(global_maximum == -float("inf"), 0.0, global_maximum)
    weight = tl.where(denominator > 0, tl.exp2(maximum - global_maximum), 0.0)
    numerator = tl.load(
        PART + ((vh * S + s[:, None]) * G + g) * D + d[None, :],
        mask=s[:, None] < S, other=0.0,
    )
    total = tl.sum(denominator * weight, axis=0)
    result = tl.sum(numerator * weight[:, None], axis=0) / tl.where(total > 0, total, 1.0)
    # No BF16 partial outputs: round only after all segments are combined.
    tl.store(OUT + query * D + d, result.to(tl.bfloat16))


@triton.jit
def _packed_rows(QKV, offsets, mask, split_stride, SPLITS: tl.constexpr):
    """BF16 rows as FP32, or FP32 split-K partials added in split order and rounded once."""
    if SPLITS == 0:
        return tl.load(QKV + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(QKV + offsets, mask=mask, other=0.0)
    for split in tl.static_range(1, SPLITS):
        x += tl.load(QKV + split * split_stride + offsets, mask=mask, other=0.0)
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _norm_rope(x, xp, w, wp, eps, cos, sin, D: tl.constexpr):
    """Head RMSNorm, gain and RoPE for [R, D] rows with the decode consumer's casts."""
    inv = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    # Round normalized values BEFORE multiplying learned gains.
    n = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    np = (xp * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    n = (n * w[None, :]).to(tl.bfloat16).to(tl.float32)
    np = (np * wp[None, :]).to(tl.bfloat16).to(tl.float32)
    d = tl.arange(0, D)
    rotated = tl.where(d[None, :] < D // 2, -np, np)
    # Native RoPE materializes TWO BF16 products before their BF16 sum.
    left = (n * cos[None, :]).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sin[None, :]).to(tl.bfloat16).to(tl.float32)
    return left + right


@triton.jit
def _fused_attention(QKV, QW, KW, COS, SIN, POS, K, V, PART, STATS, COUNT, OUT,
                     row_stride, split_stride,
                     C: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr, G: tl.constexpr,
                     S: tl.constexpr, SPAN: tl.constexpr, SCALE: tl.constexpr,
                     D: tl.constexpr, QM: tl.constexpr, KN: tl.constexpr, SS: tl.constexpr,
                     SPLITS: tl.constexpr, QEPS: tl.constexpr, KEPS: tl.constexpr):
    vh = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(1).to(tl.int64)
    batch = vh // NK
    kv_head = vh % NK
    pos = tl.load(POS)
    g = tl.arange(0, QM)
    d = tl.arange(0, D)
    n = tl.arange(0, KN)
    one = tl.arange(0, 1)
    partner = (d + D // 2) % D
    # This KV head's G query heads and its key/value head in the packed
    # [q heads | k heads | v heads] projection. Every segment program repeats
    # this small completion instead of a separate launch.
    is_q = g < G
    q_source = batch * row_stride + (kv_head * G + g) * D
    k_source = batch * row_stride + (NQ + kv_head) * D + one
    v_source = batch * row_stride + (NQ + NK + kv_head) * D + d
    qw = tl.load(QW + d).to(tl.float32)
    kw = tl.load(KW + d).to(tl.float32)
    qwp = tl.load(QW + partner).to(tl.float32)
    kwp = tl.load(KW + partner).to(tl.float32)
    cos = tl.load(COS + pos * D + d).to(tl.float32)
    sin = tl.load(SIN + pos * D + d).to(tl.float32)
    x = _packed_rows(QKV, q_source[:, None] + d[None, :], is_q[:, None], split_stride, SPLITS)
    xp = _packed_rows(QKV, q_source[:, None] + partner[None, :], is_q[:, None], split_stride, SPLITS)
    q = tl.where(is_q[:, None], _norm_rope(x, xp, qw, qwp, QEPS, cos, sin, D), 0.0).to(tl.bfloat16)
    # Exactly one program per KV head owns slot ``pos``: it writes the new K/V
    # row, and after the block barrier its own segment loop reads it back.
    # No other program's segment contains that slot, so nothing else reads it.
    owns = (segment * SPAN <= pos) & (pos < (segment + 1) * SPAN) & (pos < C)
    slot = (vh * C + pos) * D
    row_mask = owns & (d < D)
    kx = _packed_rows(QKV, k_source[:, None] + d[None, :], one[:, None] == 0, split_stride, SPLITS)
    kxp = _packed_rows(QKV, k_source[:, None] + partner[None, :], one[:, None] == 0, split_stride, SPLITS)
    key_row = tl.sum(_norm_rope(kx, kxp, kw, kwp, KEPS, cos, sin, D), axis=0)
    tl.store(K + slot + d, key_row.to(tl.bfloat16), mask=row_mask)
    value_row = _packed_rows(QKV, v_source, row_mask, split_stride, SPLITS)
    tl.store(V + slot + d, value_row.to(tl.bfloat16), mask=row_mask)
    tl.debug_barrier()
    end = tl.minimum(pos + 1, tl.minimum((segment + 1) * SPAN, C))
    maximum = tl.full((QM,), -float("inf"), tl.float32)
    denominator = tl.zeros((QM,), tl.float32)
    numerator = tl.zeros((QM, D), tl.float32)
    # Same segment scan as _segments: empty segments write neutral statistics.
    for start in range(segment * SPAN, end, KN):
        keys = start + n
        offsets = (vh * C + keys[:, None]) * D + d[None, :]
        k = tl.load(K + offsets, mask=keys[:, None] < end, other=0.0)
        v = tl.load(V + offsets, mask=keys[:, None] < end, other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee")
        scores = scores * (SCALE * 1.4426950408889634)
        scores = tl.where(keys[None, :] < end, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        probability = tl.exp2(scores - next_maximum[:, None])
        rescale = tl.exp2(maximum - next_maximum)
        denominator = denominator * rescale + tl.sum(probability, axis=1)
        numerator = numerator * rescale[:, None]
        numerator = tl.dot(probability.to(tl.bfloat16), v, numerator,
                           input_precision="ieee")
        maximum = next_maximum
    if S == 1:
        result = numerator / tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(OUT + (vh * G + g[:, None]) * D + d[None, :],
                 result.to(tl.bfloat16), mask=g[:, None] < G)
    else:
        tl.store(PART + ((vh * S + segment) * G + g[:, None]) * D + d[None, :],
                 numerator, mask=g[:, None] < G)
        stat = (vh * S + segment) * 2 * G + g
        tl.store(STATS + stat, maximum, mask=g < G)
        tl.store(STATS + stat + G, denominator, mask=g < G)
        # Every thread's partial store precedes thread 0's release; the arrival
        # count is the only cross-program handshake and nobody waits on it.
        tl.debug_barrier()
        arrived = tl.atomic_add(COUNT + vh, 1, sem="acq_rel")
        if arrived == S - 1:
            # Leave the counter zero for the next launch or graph replay.
            tl.store(COUNT + vh, 0)
            s = tl.arange(0, SS)
            for query in range(G):
                # Same combination as _merge, one query head at a time.
                stat = (vh * S + s) * 2 * G + query
                peak = tl.load(STATS + stat, mask=s < S, other=-float("inf"),
                               cache_modifier=".cg")
                mass = tl.load(STATS + stat + G, mask=s < S, other=0.0, cache_modifier=".cg")
                global_maximum = tl.max(peak, axis=0)
                global_maximum = tl.where(global_maximum == -float("inf"), 0.0, global_maximum)
                weight = tl.where(mass > 0, tl.exp2(peak - global_maximum), 0.0)
                partial = tl.load(PART + ((vh * S + s[:, None]) * G + query) * D + d[None, :],
                                  mask=s[:, None] < S, other=0.0, cache_modifier=".cg")
                total = tl.sum(mass * weight, axis=0)
                result = tl.sum(partial * weight[:, None], axis=0) / tl.where(total > 0, total, 1.0)
                tl.store(OUT + (vh * G + query) * D + d, result.to(tl.bfloat16))


class SmallQueryAttention:
    """Fixed-shape, read-only attention; returns fresh [B,1,Hq,128] BF16 output.

    USED is contiguous int32 [B*Hkv], with 0 <= each value <= capacity.
    DecodeState's prepared prefix satisfies this invariant. Calls must be
    ordered on one stream; concurrent calls cannot share this instance.
    """
    def __init__(self, batch, query_heads, kv_heads, capacity, device):
        device = torch.device(device)
        if (device.type != "cuda" or min(batch, query_heads, kv_heads, capacity) < 1
                or query_heads % kv_heads or not 1 <= query_heads // kv_heads <= QUERY_TILE):
            raise ValueError("unsupported small-query attention metadata")
        self.batch, self.hq, self.hk, self.capacity = batch, query_heads, kv_heads, capacity
        self.groups = query_heads // kv_heads
        self.virtual_heads = batch * kv_heads
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.splits, self.span = partition_keys(self.virtual_heads, capacity, sms)
        self.partials = torch.empty((self.virtual_heads, self.splits, self.groups, 128),
                                    dtype=torch.float32, device=device)
        self.stats = torch.empty((self.virtual_heads, self.splits, 2, self.groups),
                                 dtype=torch.float32, device=device)
        # Arrival counts for the one-launch merge; zero between completed launches.
        self.counters = torch.zeros(self.virtual_heads, dtype=torch.int32, device=device)

    def run(self, query, key, value, used, scale):
        device = self.partials.device
        if (torch.is_grad_enabled() or not math.isfinite(scale)
                or query.shape != (self.batch, self.hq, 1, 128)
                or key.shape != (self.batch, self.hk, self.capacity, 128)
                or value.shape != key.shape
                or used.shape != (self.virtual_heads,) or used.dtype != torch.int32
                or used.device != device or not used.is_contiguous()
                or any(t.dtype != torch.bfloat16 or t.device != device or not t.is_contiguous()
                       for t in (query, key, value))):
            raise ValueError("expected fixed-shape contiguous BF16 CUDA Q/K/V and int32 lengths")
        out = torch.empty((self.batch, 1, self.hq, 128), dtype=torch.bfloat16, device=device)
        _segments[(self.virtual_heads, self.splits)](
            query, key, value, used, self.partials, self.stats, out,
            C=self.capacity, G=self.groups, S=self.splits, SPAN=self.span,
            SCALE=scale, D=128, QM=QUERY_TILE, KN=KEY_TILE,
            num_warps=4, num_stages=2,
        )
        if self.splits > 1:
            _merge[(self.batch * self.hq,)](
                self.partials, self.stats, out, G=self.groups, S=self.splits,
                D=128, SS=triton.next_power_of_2(self.splits), num_warps=4,
            )
        return out

    def run_fused(self, qkv, q_weight, k_weight, q_eps, k_eps, cos, sin, position,
                  key, value, scale):
        """One launch: complete packed Q/K/V, write slot ``position``, attend.

        ``qkv`` is BF16 [B, (Hq + 2 Hkv) * 128] from one packed projection or
        FP32 split-K partials [S, B, (Hq + 2 Hkv) * 128] of it. ``cos``/``sin``
        are [capacity, 128] tables; ``position`` is one int64 device value in
        [0, capacity) and is the only slot of ``key``/``value`` ([B, Hkv,
        capacity, 128]) that is mutated. Attention covers slots [0, position].
        Returns fresh [B, 1, Hq, 128] BF16 output. Calls must be ordered on one
        stream; the counters and scratch belong to this instance.
        """
        device = self.partials.device
        width = (self.hq + 2 * self.hk) * 128
        if qkv.ndim == 3 and qkv.dtype == torch.float32:
            splits, rows = qkv.shape[0], tuple(qkv.shape[1:])
        else:
            splits, rows = 0, tuple(qkv.shape)
        if (torch.is_grad_enabled() or not math.isfinite(scale) or splits < 0
                or rows != (self.batch, width) or (splits == 0 and qkv.dtype != torch.bfloat16)
                or qkv.device != device or not qkv.is_contiguous()
                or key.shape != (self.batch, self.hk, self.capacity, 128)
                or value.shape != key.shape
                or q_weight.shape != (128,) or k_weight.shape != (128,)
                or cos.shape != (self.capacity, 128) or sin.shape != cos.shape
                or position.shape != (1,) or position.dtype != torch.int64
                or position.device != device
                or any(t.dtype != torch.bfloat16 or t.device != device or not t.is_contiguous()
                       for t in (q_weight, k_weight, cos, sin, key, value))):
            raise ValueError("expected packed BF16 rows or FP32 partials, BF16 gains/tables/caches "
                             "and one int64 position on this plan's device")
        out = torch.empty((self.batch, 1, self.hq, 128), dtype=torch.bfloat16, device=device)
        _fused_attention[(self.virtual_heads, self.splits)](
            qkv, q_weight, k_weight, cos, sin, position, key, value,
            self.partials, self.stats, self.counters, out, width, self.batch * width,
            C=self.capacity, NQ=self.hq, NK=self.hk, G=self.groups, S=self.splits,
            SPAN=self.span, SCALE=scale, D=128, QM=QUERY_TILE, KN=KEY_TILE,
            SS=triton.next_power_of_2(self.splits), SPLITS=splits, QEPS=q_eps, KEPS=k_eps,
            num_warps=4, num_stages=2, enable_fp_fusion=False,
        )
        return out
