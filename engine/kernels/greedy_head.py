"""BF16 full-vocabulary projection + hierarchical argmax, Triton 3.1.0.

All K features and all V vocabulary rows participate. Each tile finishes its
FP32 dot products and rounds them to BF16 BEFORE comparison. Tile winners hold
exact FP32 expansions of those BF16 logits, never partial projection sums.
Ties (including BF16-induced ties) choose the lowest global vocabulary index.
NaNs follow PyTorch ArgMaxOps: first NaN wins. Inputs are never mutated.
"""

import torch
import triton
import triton.language as tl


BN = 128
BK = 64


@triton.jit
def _project_max(X, W, SCORES, INDICES,
                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 TILES: tl.constexpr, BM: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    m = tl.arange(0, BM)
    n = tile * BN + tl.arange(0, BN)
    lane = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + lane
        x = tl.load(X + m[:, None].to(tl.int64) * K + k[None, :],
                    mask=(m[:, None] < M) & (k[None, :] < K), other=0.0)
        w = tl.load(W + n[:, None].to(tl.int64) * K + k[None, :],
                    mask=(n[:, None] < N) & (k[None, :] < K), other=0.0)
        # BF16 tensor-core operands, FP32 accumulator. No TF32 or FP8.
        acc = tl.dot(x, tl.trans(w), acc, input_precision="ieee")
    logits = acc.to(tl.bfloat16).to(tl.float32)
    valid = n[None, :] < N
    isnan = logits != logits
    nan_index = tl.min(tl.where(valid & isnan, n[None, :], 2147483647), axis=1)
    best = tl.max(tl.where(valid & ~isnan, logits, -float("inf")), axis=1)
    index = tl.min(tl.where(valid & (logits == best[:, None]),
                            n[None, :], 2147483647), axis=1)
    has_nan = nan_index != 2147483647
    index = tl.where(has_nan, nan_index, index)
    best = tl.where(has_nan, float("nan"), best)
    offsets = m * TILES + tile
    tl.store(SCORES + offsets, best, mask=m < M)
    tl.store(INDICES + offsets, index, mask=m < M)


@triton.jit
def _merge_max(SCORES, INDICES, TOKENS,
               TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < TILES
    scores = tl.load(SCORES + row * TILES + cols, mask=valid, other=-float("inf"))
    indices = tl.load(INDICES + row * TILES + cols, mask=valid, other=2147483647)
    isnan = scores != scores
    nan_index = tl.min(tl.where(valid & isnan, indices, 2147483647), axis=0)
    best = tl.max(tl.where(valid & ~isnan, scores, -float("inf")), axis=0)
    index = tl.min(tl.where(valid & (scores == best), indices, 2147483647), axis=0)
    tl.store(TOKENS + row, tl.where(nan_index != 2147483647, nan_index, index).to(tl.int64))


def full_vocab_argmax(x, weight):
    """Return fresh int64 [B,1]; allocate only O(B*ceil(V/128)) winner scratch."""
    if (torch.is_grad_enabled() or x.ndim != 2 or weight.ndim != 2
            or not x.is_cuda or weight.device != x.device
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not x.is_contiguous() or not weight.is_contiguous()
            or not 1 <= x.shape[0] <= 64 or not 1 <= weight.shape[0] <= 262144
            or not 1 <= x.shape[1] == weight.shape[1]):
        raise ValueError("expected contiguous BF16 CUDA X[1<=B<=64,H] and W[1<=V<=262144,H]")
    m, k = x.shape
    n = weight.shape[0]
    tiles = triton.cdiv(n, BN)
    scores = torch.empty((m, tiles), device=x.device, dtype=torch.float32)
    indices = torch.empty((m, tiles), device=x.device, dtype=torch.int32)
    tokens = torch.empty((m, 1), device=x.device, dtype=torch.int64)
    bm = max(16, triton.next_power_of_2(m))
    _project_max[(tiles,)](
        x, weight, scores, indices, M=m, N=n, K=k, TILES=tiles,
        BM=bm, BN=BN, BK=BK, num_warps=8 if bm == 64 else 4, num_stages=3,
    )
    _merge_max[(m,)](scores, indices, tokens, TILES=tiles,
                      BLOCK=triton.next_power_of_2(tiles), num_warps=4)
    return tokens
