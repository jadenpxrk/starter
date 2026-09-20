"""Small-row BF16 linear kernels; no weight conversion or approximate math.

Inputs are contiguous X[M,K] and W[N,K], BF16 on one CUDA device.
The result is a fresh contiguous BF16 Y[M,N]. Inputs are never mutated.
Split-K partials are FP32, reduced deterministically before the only BF16
output cast. No atomic reduction, TF32, or BF16 partial-sum storage is used.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    lane = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + lane
        x = tl.load(X + k, k < K, other=0).to(tl.float32)
        w = tl.load(W + n[:, None] * K + k[None, :],
                    (n[:, None] < N) & (k[None, :] < K), other=0).to(tl.float32)
        acc = acc + w * x[None, :]
    tl.store(Y + n, tl.sum(acc, axis=1).to(tl.bfloat16), n < N)


@triton.jit
def _small_mm(X, W, Y, PARTIAL, M: tl.constexpr, N: tl.constexpr,
              K: tl.constexpr, S: tl.constexpr, BM: tl.constexpr,
              BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    split = tl.program_id(2)
    lane = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    # Cyclic K tiles cover every input feature exactly once across splits.
    for block in range(tl.cdiv(K, BK * S)):
        k = (block * S + split) * BK + lane
        a = tl.load(X + m[:, None] * K + k[None, :],
                    (m[:, None] < M) & (k[None, :] < K), other=0)
        wt = tl.load(W + n[:, None] * K + k[None, :],
                     (n[:, None] < N) & (k[None, :] < K), other=0)
        # Both dot operands remain BF16. The accumulator remains FP32.
        acc = tl.dot(a, tl.trans(wt), acc, input_precision="ieee")
    offsets = m[:, None] * N + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    if S == 1:
        tl.store(Y + offsets, acc.to(tl.bfloat16), mask)
    else:
        tl.store(PARTIAL + split * M * N + offsets, acc, mask)


@triton.jit
def _finish(PARTIAL, Y, TOTAL: tl.constexpr, S: tl.constexpr,
            BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, S)
    p = tl.load(PARTIAL + split[:, None] * TOTAL + col[None, :],
                col[None, :] < TOTAL, other=0)
    # Round only after ALL FP32 partial sums have been combined.
    y = tl.sum(p, axis=0).to(tl.bfloat16)
    tl.store(Y + col, y, col < TOTAL)


def small_linear(x, weight, policy, split_k=1):
    """Launch one GEMV or a BF16 tensor-core matmul plus optional FP32 reduction."""
    if (x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
            or not x.is_contiguous() or not weight.is_contiguous()
            or not x.is_cuda or weight.device != x.device
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16):
        raise ValueError("small_linear requires contiguous BF16 CUDA matrices X[M,K], W[N,K]")
    m, k = x.shape
    n = weight.shape[0]
    if not (1 <= m <= 32 and n > 0 and k > 0):
        raise ValueError("unsupported matrix dimensions")
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if policy == "simt":
        if m != 1:
            raise ValueError("SIMT GEMV requires exactly one row")
        _gemv[(triton.cdiv(n, 8),)](
            x, weight, out, n, k, BN=8, BK=512,
            num_warps=4, enable_fp_fusion=False,
        )
    elif policy == "tensorcore":
        if split_k not in (1, 2, 4, 8, 16):
            raise ValueError("split_k must be a supported power of two")
        partial = (torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
                   if split_k > 1 else out)
        _small_mm[(triton.cdiv(n, 64), triton.cdiv(m, 16), split_k)](
            x, weight, out, partial, m, n, k, split_k,
            BM=16, BN=64, BK=64, num_warps=4, num_stages=3,
        )
        if split_k > 1:
            _finish[(triton.cdiv(m * n, 256),)](
                partial, out, m * n, split_k, BLOCK=256, num_warps=4,
            )
    else:
        raise ValueError(f"unknown linear policy: {policy}")
    return out
