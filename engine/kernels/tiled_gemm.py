"""The #19 split-K projection with tile-major weight addresses, Triton 3.1.0.

Dot geometry, K split count/ranges/order, launch configuration, FP32 partial
layout, and consumers are unchanged. Only W addressing differs. Inputs are
read-only, output is fresh FP32 [S,M,N], and all weights/features are used.
No activation fusion, extra output cast, atomics, counters, or timing selector.
"""

import torch
import triton
import triton.language as tl

from kernels.skinny_gemm import BN, BK, split_count, supports


@triton.jit
def _tiled_partials_kernel(X, WT, P, M, N, K, k_per_split,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    m = tl.arange(0, BM)
    inner_n = tl.arange(0, BN)
    n = tile * BN + inner_n
    k = tl.arange(0, BK)
    x_ptrs = X + m[:, None] * K + k[None, :]
    # WT[tile, k_tile, output_lane, feature_lane]. Every loaded weight tile
    # occupies BN*BK consecutive BF16 elements; dot operands keep [BN,BK].
    w_ptrs = WT + tile * K * BN + inner_n[:, None] * BK + k[None, :]
    x_mask = (m[:, None] < M) & (k[None, :] < BK)
    k_start = split * k_per_split
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(k_start, k_start + k_per_split, BK):
        x = tl.load(x_ptrs + k0, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs + k0 * BN)
        acc = tl.dot(x, tl.trans(w), acc)
    out_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(P + (split * M + m[:, None]) * N + n[None, :], acc, mask=out_mask)


def tiled_partials(x, weight, tiles):
    """Use the original W only for metadata; multiply by its immutable tiled copy.

    Caller owns source/copy coherence. Engine installs the copy once after its
    last weight mutation. This function never repacks during capture or replay.
    """
    if (torch.is_grad_enabled() or not supports(x, weight)
            or weight.shape[0] < 1 or x.shape[1] < 1):
        raise ValueError("tiled projection requires inference-mode skinny BF16 CUDA matrices")
    m, k = x.shape
    n = weight.shape[0]
    if (tiles.shape != (n // BN, k // BK, BN, BK)
            or tiles.dtype != torch.bfloat16 or tiles.device != x.device
            or not tiles.is_contiguous()):
        raise ValueError("tile buffer does not match projection shape, dtype, layout, or device")
    splits = split_count(n, k)
    partials = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    _tiled_partials_kernel[(n // BN, splits)](
        x, tiles, partials, m, n, k, k // splits,
        BM=max(16, triton.next_power_of_2(m)), BN=BN, BK=BK,
        num_warps=4, num_stages=4,
    )
    return partials
