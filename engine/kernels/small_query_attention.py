"""Full-context BF16 GQA decode with a 16-row query tile (Triton 3.1.0).

One program owns one original (batch, KV head) and a disjoint key segment.
All G query heads reuse that segment's K/V. Softmax statistics and weighted
value partials remain FP32; only probabilities supplied to the BF16 P@V dot
and the final attention output are BF16, as in native FlashAttention.

No sparsity, atomics, cache mutation, telemetry, autotuning or runtime timing.
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
