"""Bandwidth-oriented BF16 projections for decode rows (Triton 3.1.0).

A decode step multiplies at most a few rows by every layer weight, so each
projection is a weight stream. One program owns BN output columns and one
K range, walks it with BF16 tensor-core dots and an FP32 accumulator, and
never touches more than its tile. Three epilogues, no timing selector:

* ``linear_partials``: FP32 split-K partial sums ``[S, M, N]``. The packed
  Q/K/V consumer adds the S partials and rounds once to BF16, where cuBLAS
  rounds its output. There is no extra reduction launch and no BF16 partials.
* ``linear_add_rms_norm``: the same weight stream for o_proj and down_proj,
  with the residual add and the following RMSNorm finished inside the launch.
  The last split to arrive at a column block sums that block's partials in
  split order, rounds once, adds the residual, and publishes the BF16 sum and
  its FP32 sum of squares; the last column block to finish normalizes every
  row. Arrival is counted with acquire/release atomics that never wait, so no
  program depends on another being resident, and the arithmetic order is fixed
  whichever program arrives last.
* ``linear_silu_mul``: gate and up rows of the packed weight in one program,
  so ``silu(gate) * up`` runs in the epilogue with the reference's casts.

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
def _partials_kernel(X, W, P, M, N, K, k_per_split,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n0 = tl.program_id(0) * BN
    split = tl.program_id(1)
    m = tl.arange(0, BM)
    n = n0 + tl.arange(0, BN)
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
    out_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(P + (split * M + m[:, None]) * N + n[None, :], acc, mask=out_mask)


@triton.jit
def _add_norm_kernel(X, W, R, G, P, SQ, CNT, T, O, M, N, K, k_per_split, eps,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     SPLITS: tl.constexpr, BLOCKS: tl.constexpr, CB: tl.constexpr):
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
        # Same order and casts as the separate consumer: add the S partials in
        # split order, round once to BF16, add the residual in BF16.
        y = tl.zeros((BM, BN), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            y += tl.load(P + s * M * N + tile, mask=tile_mask, other=0.0, cache_modifier=".cg")
        y = y.to(tl.bfloat16).to(tl.float32)
        r = tl.load(R + tile, mask=tile_mask, other=0.0).to(tl.float32)
        total = (r + y).to(tl.bfloat16)
        tl.store(T + tile, total, mask=tile_mask)
        total = total.to(tl.float32)
        tl.store(SQ + block * BM + m, tl.sum(total * total, axis=1))
        tl.store(CNT + block, 0)
        tl.debug_barrier()
        done = tl.atomic_add(CNT + BLOCKS, 1, sem="acq_rel")
        if done == BLOCKS - 1:
            sq = tl.zeros((BM,), dtype=tl.float32)
            for j in range(BLOCKS):
                sq += tl.load(SQ + j * BM + m, cache_modifier=".cg")
            rstd = tl.math.rsqrt(sq / N + eps)
            for c0 in range(0, N, CB):
                cols = c0 + tl.arange(0, CB)
                cmask = (m[:, None] < M) & (cols[None, :] < N)
                t = tl.load(T + m[:, None] * N + cols[None, :], mask=cmask, other=0.0,
                            cache_modifier=".cg").to(tl.float32)
                g = tl.load(G + cols, mask=cols < N, other=0.0)
                # kernels/rmsnorm.py cast placement: round, then multiply the gain.
                tl.store(O + m[:, None] * N + cols[None, :],
                         (t * rstd[:, None]).to(tl.bfloat16) * g, mask=cmask)
            tl.store(CNT + BLOCKS, 0)


@triton.jit
def _silu_mul_kernel(X, W, Y, M, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n0 = tl.program_id(0) * BN
    m = tl.arange(0, BM)
    n = n0 + tl.arange(0, BN)
    k = tl.arange(0, BK)
    x_ptrs = X + m[:, None] * K + k[None, :]
    gate_ptrs = W + n[:, None] * K + k[None, :]
    up_ptrs = gate_ptrs + N * K
    x_mask = (m[:, None] < M) & (k[None, :] < BK)
    gate = tl.zeros((BM, BN), dtype=tl.float32)
    up = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(x_ptrs + k0, mask=x_mask, other=0.0)
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


def linear_partials(x, weight):
    """FP32 split-K partials [S, M, N] of x @ weight.T; inputs are read-only."""
    m, n, k = _dims(x, weight)
    splits = split_count(n, k)
    partials = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    _partials_kernel[(n // BN, splits)](
        x, weight, partials, m, n, k, k // splits,
        BM=max(16, triton.next_power_of_2(m)), BN=BN, BK=BK, num_warps=4, num_stages=4,
    )
    return partials


_counters = {}


def _arrival_counters(device, blocks):
    """Zeroed int32 [blocks + 1], one per device and width; every launch leaves it zeroed.

    Launches on one stream never overlap, so all layers share one buffer.
    """
    key = (device, blocks)
    if key not in _counters:
        _counters[key] = torch.zeros(blocks + 1, dtype=torch.int32, device=device)
    return _counters[key]


def linear_add_rms_norm(x, weight, residual, gain, eps):
    """(residual + bf16(x @ weight.T), RMSNorm of that sum): fresh BF16 [M, N] pair.

    ``residual`` is BF16 [M, N] and ``gain`` BF16 [N], all on ``x``'s device.
    Inputs are read-only. Equals add_rms_norm(residual, linear_partials(x,
    weight), gain, eps) up to the FP32 grouping of the sum of squares.
    """
    m, n, k = _dims(x, weight)
    if (residual.shape != (m, n) or gain.shape != (n,)
            or any(t.dtype != torch.bfloat16 or t.device != x.device or not t.is_contiguous()
                   for t in (residual, gain))):
        raise ValueError("expected contiguous BF16 residual [M, N] and gain [N] on the input device")
    splits = split_count(n, k)
    blocks = n // BN
    bm = max(16, triton.next_power_of_2(m))
    partials = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    sumsq = torch.empty((blocks, bm), dtype=torch.float32, device=x.device)
    total, normed = torch.empty_like(residual), torch.empty_like(residual)
    _add_norm_kernel[(blocks, splits)](
        x, weight, residual, gain, partials, sumsq, _arrival_counters(x.device, blocks),
        total, normed, m, n, k, k // splits, eps,
        BM=bm, BN=BN, BK=BK, SPLITS=splits, BLOCKS=blocks, CB=max(32, 4096 // bm),
        num_warps=4, num_stages=4,
    )
    return total, normed


def linear_silu_mul(x, gate_up_weight):
    """BF16 silu(x @ gate.T) * (x @ up.T) for a packed [2N, K] weight; returns [M, N]."""
    m, width, k = _dims(x, gate_up_weight, pairs=True)
    n = width // 2
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _silu_mul_kernel[(n // BN,)](
        x, gate_up_weight, out, m, n, k,
        BM=max(16, triton.next_power_of_2(m)), BN=BN, BK=BK, num_warps=4, num_stages=4,
    )
    return out
