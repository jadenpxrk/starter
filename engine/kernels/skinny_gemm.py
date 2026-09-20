"""Bandwidth-oriented BF16 projections for decode rows (Triton 3.1.0).

A decode step multiplies at most a few rows by every layer weight, so each
projection is a weight stream. One program owns BN output columns and one
K range, walks it with BF16 tensor-core dots and an FP32 accumulator, and
never touches more than its tile. Three epilogues, no timing selector:

* ``linear_partials``: FP32 split-K partial sums ``[S, M, N]``. Consumers
  (residual add + RMSNorm, packed Q/K/V norm/RoPE/cache write) add the S
  partials and round once to BF16, where cuBLAS rounds its output. There is
  no extra reduction launch, no atomic, and no BF16 partial storage.
* ``linear_add_stats``: the same split-K stream for o_proj and down_proj,
  whose consumer is a residual add followed by RMSNorm. The last program to
  finish a column tile (counted with a never-waiting acquire/release atomic)
  adds that tile's partials in split order, rounds once, adds the residual in
  BF16, stores the sum and that tile's FP32 sum of squares. No program waits
  for another, no single program walks the whole row, and the arithmetic
  order is fixed whichever program arrives last.
* ``linear_silu_mul``: gate and up rows of the packed weight in one program,
  so ``silu(gate) * up`` runs in the epilogue with the reference's casts.

``linear_partials`` and ``linear_silu_mul`` accept ``norm=(stats, gain, eps)``
from ``linear_add_stats``: each program then sums the per-tile statistics in
tile order, forms the row's rsqrt, and applies the RMSNorm cast chain to its
BF16 input tiles on the way into the dot. That removes the separate
residual-norm launch and its BF16 round trip.

Prefill (thousands of rows) keeps cuBLAS; ``supports`` is a static shape rule.
"""

import torch
import triton
import triton.language as tl

MAX_ROWS = 64
BN = 32
BK = 64
TARGET_PROGRAMS = 512  # about four per H100 SM


@triton.jit
def _row_rstd(SQ, m, K, eps, BM: tl.constexpr, TILES: tl.constexpr, TP: tl.constexpr):
    """rsqrt(mean of squares + eps) per row from [TILES, BM] tile sums, fixed order."""
    t = tl.arange(0, TP)
    sq = tl.load(SQ + t[None, :] * BM + m[:, None], mask=t[None, :] < TILES, other=0.0)
    return tl.math.rsqrt(tl.sum(sq, axis=1) / K + eps)


@triton.jit
def _normalize(x, inv, g_ptrs):
    """kernels/rmsnorm.py cast placement: round the normalized value, then multiply the gain."""
    g = tl.load(g_ptrs)
    return (x.to(tl.float32) * inv[:, None]).to(tl.bfloat16) * g[None, :]


@triton.jit
def _partials_kernel(X, W, P, SQ, G, M, N, K, k_per_split, eps,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     NORM: tl.constexpr, TILES: tl.constexpr, TP: tl.constexpr):
    n0 = tl.program_id(0) * BN
    split = tl.program_id(1)
    m = tl.arange(0, BM)
    n = n0 + tl.arange(0, BN)
    k = tl.arange(0, BK)
    x_ptrs = X + m[:, None] * K + k[None, :]
    w_ptrs = W + n[:, None] * K + k[None, :]
    x_mask = (m[:, None] < M) & (k[None, :] < BK)
    k_start = split * k_per_split
    if NORM:
        inv = _row_rstd(SQ, m, K, eps, BM, TILES, TP)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(k_start, k_start + k_per_split, BK):
        x = tl.load(x_ptrs + k0, mask=x_mask, other=0.0)
        if NORM:
            x = _normalize(x, inv, G + k0 + k)
        w = tl.load(w_ptrs + k0)
        acc = tl.dot(x, tl.trans(w), acc)
    out_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(P + (split * M + m[:, None]) * N + n[None, :], acc, mask=out_mask)


@triton.jit
def _partials_stats_kernel(X, W, P, R, T, SQ, CNT, M, N, K, k_per_split,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           SPLITS: tl.constexpr):
    block = tl.program_id(0)
    split = tl.program_id(1)
    m = tl.arange(0, BM)
    n = block * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    x_ptrs = X + m[:, None] * K + k[None, :]
    w_ptrs = W + n[:, None] * K + k[None, :]
    x_mask = (m[:, None] < M) & (k[None, :] < BK)
    k_start = split * k_per_split
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(k_start, k_start + k_per_split, BK):
        x = tl.load(x_ptrs + k0, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs + k0)
        acc = tl.dot(x, tl.trans(w), acc)
    tile = m[:, None] * N + n[None, :]
    tile_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(P + split * M * N + tile, acc, mask=tile_mask)
    # Every thread's partial store precedes thread 0's release; the arrival
    # count is the only cross-program handshake and nobody waits on it.
    tl.debug_barrier()
    arrived = tl.atomic_add(CNT + block, 1, sem="acq_rel")
    if arrived == SPLITS - 1:
        # Same order and casts as add_rms_norm: add the S partials in split
        # order, round once to BF16, add the residual in BF16.
        y = tl.zeros((BM, BN), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            y += tl.load(P + s * M * N + tile, mask=tile_mask, other=0.0, cache_modifier=".cg")
        y = y.to(tl.bfloat16).to(tl.float32)
        r = tl.load(R + tile, mask=tile_mask, other=0.0).to(tl.float32)
        total = (r + y).to(tl.bfloat16)
        tl.store(T + tile, total, mask=tile_mask)
        total = total.to(tl.float32)
        # Masked rows are zero here, so padded statistics stay finite.
        tl.store(SQ + block * BM + m, tl.sum(total * total, axis=1))
        tl.store(CNT + block, 0)


@triton.jit
def _silu_mul_kernel(X, W, Y, SQ, G, M, N, K, eps,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     NORM: tl.constexpr, TILES: tl.constexpr, TP: tl.constexpr):
    n0 = tl.program_id(0) * BN
    m = tl.arange(0, BM)
    n = n0 + tl.arange(0, BN)
    k = tl.arange(0, BK)
    x_ptrs = X + m[:, None] * K + k[None, :]
    gate_ptrs = W + n[:, None] * K + k[None, :]
    up_ptrs = gate_ptrs + N * K
    x_mask = (m[:, None] < M) & (k[None, :] < BK)
    if NORM:
        inv = _row_rstd(SQ, m, K, eps, BM, TILES, TP)
    gate = tl.zeros((BM, BN), dtype=tl.float32)
    up = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(x_ptrs + k0, mask=x_mask, other=0.0)
        if NORM:
            x = _normalize(x, inv, G + k0 + k)
        gate = tl.dot(x, tl.trans(tl.load(gate_ptrs + k0)), gate)
        up = tl.dot(x, tl.trans(tl.load(up_ptrs + k0)), up)
    # Each projection rounds to BF16 as a GEMM output; native then rounds
    # SiLU to BF16 before the BF16 product (same order as silu_mul).
    g = gate.to(tl.bfloat16).to(tl.float32)
    u = up.to(tl.bfloat16).to(tl.float32)
    act = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    out_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(Y + m[:, None] * N + n[None, :], (act * u).to(tl.bfloat16), mask=out_mask)


def supports(x, weight, pairs=False):
    """Static shape/dtype/layout rule; identical for every call of one shape."""
    columns = 2 * BN if pairs else BN
    return (x.ndim == 2 and weight.ndim == 2 and x.shape[1] == weight.shape[1]
            and x.is_cuda and weight.device == x.device
            and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
            and x.is_contiguous() and weight.is_contiguous()
            and 1 <= x.shape[0] <= MAX_ROWS
            and weight.shape[0] % columns == 0 and x.shape[1] % BK == 0)


def split_count(n, k):
    """Power-of-two K splits toward TARGET_PROGRAMS; every split is BK-aligned."""
    splits = 1
    while n // BN * splits < TARGET_PROGRAMS and k % (2 * splits * BK) == 0:
        splits *= 2
    return splits


def _dims(x, weight, pairs=False):
    if not supports(x, weight, pairs):
        raise ValueError("expected contiguous BF16 CUDA X[M<=64,K] and W[N,K] with tile-aligned N, K")
    return x.shape[0], weight.shape[0], x.shape[1]


def _block_rows(m):
    return max(16, triton.next_power_of_2(m))


def _norm_args(x, norm):
    """(stats, gain, eps, TILES, TP) for the on-the-fly RMSNorm, or inert values."""
    m, k = x.shape
    if norm is None:
        return x, x, 0.0, 1, 1
    stats, gain, eps = norm
    tiles = k // BN
    if (stats.shape != (tiles, _block_rows(m)) or stats.dtype != torch.float32
            or gain.shape != (k,) or gain.dtype != torch.bfloat16
            or any(t.device != x.device or not t.is_contiguous() for t in (stats, gain))):
        raise ValueError("expected contiguous FP32 stats [K // BN, BM] and BF16 gain [K] on X's device")
    return stats, gain, float(eps), tiles, triton.next_power_of_2(tiles)


_counters = {}


def _arrival_counters(device, tiles):
    """Zeroed int32 [tiles], one per device and width; every completed launch leaves it zeroed.

    Launches on one stream never overlap, so all layers share one buffer.
    """
    key = (device, tiles)
    if key not in _counters:
        _counters[key] = torch.zeros(tiles, dtype=torch.int32, device=device)
    return _counters[key]


def linear_partials(x, weight, norm=None):
    """FP32 split-K partials [S, M, N] of x @ weight.T; inputs are read-only.

    With ``norm=(stats, gain, eps)`` from ``linear_add_stats``, ``x`` is the
    BF16 residual sum and each program normalizes its tiles on the way in.
    """
    m, n, k = _dims(x, weight)
    splits = split_count(n, k)
    stats, gain, eps, tiles, tp = _norm_args(x, norm)
    partials = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    _partials_kernel[(n // BN, splits)](
        x, weight, partials, stats, gain, m, n, k, k // splits, eps,
        BM=_block_rows(m), BN=BN, BK=BK, NORM=norm is not None, TILES=tiles, TP=tp,
        num_warps=4, num_stages=4,
    )
    return partials


def linear_add_stats(x, weight, residual):
    """(residual + bf16(x @ weight.T), per-tile FP32 sums of squares of it).

    ``residual`` is BF16 [M, N] on ``x``'s device. Returns a fresh BF16 [M, N]
    sum and fresh FP32 [N // BN, BM] statistics whose padded rows are zero.
    The sum equals ``add_rms_norm(residual, linear_partials(x, weight), ...)[0]``
    exactly; the statistics only regroup that kernel's FP32 sum of squares.
    """
    m, n, k = _dims(x, weight)
    if (residual.shape != (m, n) or residual.dtype != torch.bfloat16
            or residual.device != x.device or not residual.is_contiguous()):
        raise ValueError("expected a contiguous BF16 residual [M, N] on X's device")
    splits = split_count(n, k)
    tiles, bm = n // BN, _block_rows(m)
    partials = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    total = torch.empty_like(residual)
    stats = torch.empty((tiles, bm), dtype=torch.float32, device=x.device)
    _partials_stats_kernel[(tiles, splits)](
        x, weight, partials, residual, total, stats, _arrival_counters(x.device, tiles),
        m, n, k, k // splits, BM=bm, BN=BN, BK=BK, SPLITS=splits, num_warps=4, num_stages=4,
    )
    return total, stats


def linear_silu_mul(x, gate_up_weight, norm=None):
    """BF16 silu(x @ gate.T) * (x @ up.T) for a packed [2N, K] weight; returns [M, N].

    ``norm`` is as for ``linear_partials``.
    """
    m, width, k = _dims(x, gate_up_weight, pairs=True)
    n = width // 2
    stats, gain, eps, tiles, tp = _norm_args(x, norm)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _silu_mul_kernel[(n // BN,)](
        x, gate_up_weight, out, stats, gain, m, n, k, eps,
        BM=_block_rows(m), BN=BN, BK=BK, NORM=norm is not None, TILES=tiles, TP=tp,
        num_warps=4, num_stages=4,
    )
    return out
