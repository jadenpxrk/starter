"""Bandwidth-oriented BF16 projections for decode rows (Triton 3.1.0).

A decode step multiplies at most a few rows by every layer weight, so each
projection is a weight stream. One program owns BN output columns and one
K range, walks it with BF16 tensor-core dots and an FP32 accumulator, and
never touches more than its tile. Two epilogues, no timing selector:

* ``linear_partials``: FP32 split-K partial sums ``[S, M, N]``. Consumers
  (residual add + RMSNorm, packed Q/K/V norm/RoPE/cache write) add the S
  partials and round once to BF16, where cuBLAS rounds its output. There is
  no extra reduction launch, no atomic, and no BF16 partial storage.
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
