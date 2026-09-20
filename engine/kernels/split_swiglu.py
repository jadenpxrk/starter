"""Paired split-K gate/up projection for decode, with FP32 partials.

This replaces only skinny_gemm.linear_silu_mul on eligible long reductions.
Four independent K ranges shorten each producer's loop. One dot computes
32 gate and the corresponding 32 up channels; no weight relayout is needed.
A second kernel combines ALL FP32 partials before the BF16 projection cast,
then preserves the existing BF16 SiLU-result and product boundaries.

Extra launch/scratch traffic versus #19 is intentional and UNMEASURED. There
are no atomics, counters, tuning, telemetry, or changes to cache/attention.
"""

import torch
import triton
import triton.language as tl

from kernels.skinny_gemm import linear_silu_mul as unsplit_silu_mul, supports

PAIR_TILE = 32
K_TILE = 64
SPLITS = 4
MIN_K = 16 * K_TILE


def split_count(k):
    """Fixed experimental rule, not a measured optimum or a workload table."""
    return SPLITS if k >= MIN_K and k % (SPLITS * K_TILE) == 0 else 1


@triton.jit
def _paired_partials(X, W, P, M: tl.constexpr, N: tl.constexpr,
                     K: tl.constexpr, S: tl.constexpr,
                     BM: tl.constexpr, PN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    m = tl.arange(0, BM)
    lane = tl.arange(0, 2 * PN)
    # Logical W is still [all gate rows | all up rows]. Pairing is addressing,
    # NOT concatenation, a cast, or a persistent weight copy.
    channel = tile * PN + lane % PN
    wrow = channel + tl.where(lane < PN, 0, N)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, 2 * PN), dtype=tl.float32)
    for start in range(split * (K // S), (split + 1) * (K // S), BK):
        k = start + kk
        x = tl.load(X + m[:, None] * K + k[None, :],
                    mask=m[:, None] < M, other=0.0)
        w = tl.load(W + wrow[:, None] * K + k[None, :])
        acc = tl.dot(x, tl.trans(w), acc, input_precision="ieee")
    # P[S,M,2N] retains separate gate/up planes in each row. Each element has
    # one producer; no partial BF16 cast and no atomic reduction.
    offset = (split * M + m[:, None]) * (2 * N) + wrow[None, :]
    tl.store(P + offset, acc, mask=m[:, None] < M)


@triton.jit
def _reduce_silu(P, Y, M: tl.constexpr, N: tl.constexpr,
                 S: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = i < M * N
    row, col = i // N, i % N
    gate = tl.zeros((BLOCK,), dtype=tl.float32)
    up = tl.zeros((BLOCK,), dtype=tl.float32)
    for split in tl.static_range(S):
        offset = (split * M + row) * (2 * N) + col
        gate = gate + tl.load(P + offset, mask=live, other=0.0)
        up = up + tl.load(P + offset + N, mask=live, other=0.0)
    # Never apply SiLU to partial sums. Complete both projections first.
    g = gate.to(tl.bfloat16).to(tl.float32)
    u = up.to(tl.bfloat16).to(tl.float32)
    act = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + i, (act * u).to(tl.bfloat16), mask=live)


def linear_silu_mul(x, gate_up_weight):
    """Read-only X[M,K], W[2N,K]; fresh BF16 Y[M,N], same interface as #19.

    Inference only. Existing skinny layout/dtype limits apply (M<=64,
    contiguous BF16 CUDA, N multiple of 32, K multiple of 64). K not meeting
    split_count's stricter rule uses the EXACT original fused implementation.
    Scratch is FP32 [4,M,2N], owned by this call/its captured graph. No cached
    activations or model-dependent state is retained by the Python wrapper.
    """
    if (torch.is_grad_enabled() or not supports(x, gate_up_weight, pairs=True)
            or min(gate_up_weight.shape) < 1):
        raise ValueError("split SwiGLU requires supported BF16 CUDA matrices under inference mode")
    m, k = x.shape
    n = gate_up_weight.shape[0] // 2
    splits = split_count(k)
    if splits == 1:
        return unsplit_silu_mul(x, gate_up_weight)
    partial = torch.empty((splits, m, 2 * n), dtype=torch.float32, device=x.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _paired_partials[(n // PAIR_TILE, splits)](
        x, gate_up_weight, partial, M=m, N=n, K=k, S=splits,
        BM=max(16, triton.next_power_of_2(m)), PN=PAIR_TILE, BK=K_TILE,
        num_warps=4, num_stages=4,
    )
    _reduce_silu[(triton.cdiv(m * n, 256),)](
        partial, out, M=m, N=n, S=splits, BLOCK=256,
        num_warps=4, enable_fp_fusion=False,
    )
    return out
