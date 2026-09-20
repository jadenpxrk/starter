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
* ``linear_qkv_norm_rope_cache``: one program owns one complete 128-wide Q, K
  or V head over the full K range, so the head RMSNorm, RoPE and static-cache
  write run in its epilogue. No partial buffer and no consumer launch; the
  head's two 64-column halves are the rotary partners of each other.

Prefill (thousands of rows) keeps cuBLAS; ``supports`` is a static shape rule.
"""

import torch
import triton
import triton.language as tl

MAX_ROWS = 64
BN = 32
BK = 64
TARGET_PROGRAMS = 512  # about four per H100 SM
HEAD = 128
QKV_STAGES = 6  # few programs per launch, so keep more weight tiles in flight each


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


@triton.jit
def _qkv_head_kernel(X, W, QW, KW, COS, SIN, POS, OQ, KC, VC, M, K, capacity,
                     NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
                     QEPS: tl.constexpr, KEPS: tl.constexpr,
                     BM: tl.constexpr, BK: tl.constexpr):
    head = tl.program_id(0)
    m = tl.arange(0, BM)
    h = tl.arange(0, D // 2)
    k = tl.arange(0, BK)
    x_ptrs = X + m[:, None] * K + k[None, :]
    # Packed rows are [q heads | k heads | v heads]; rows head*D .. head*D+D-1
    # are this head's columns, loaded as two D/2 halves (rotary partners).
    left_ptrs = W + (head * D + h)[:, None] * K + k[None, :]
    right_ptrs = left_ptrs + (D // 2) * K
    x_mask = (m[:, None] < M) & (k[None, :] < BK)
    left = tl.zeros((BM, D // 2), dtype=tl.float32)
    right = tl.zeros((BM, D // 2), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(x_ptrs + k0, mask=x_mask, other=0.0)
        left = tl.dot(x, tl.trans(tl.load(left_ptrs + k0)), left)
        right = tl.dot(x, tl.trans(tl.load(right_ptrs + k0)), right)
    # The completed projection rounds to BF16 where the GEMM output rounds.
    left = left.to(tl.bfloat16).to(tl.float32)
    right = right.to(tl.bfloat16).to(tl.float32)
    pos = tl.load(POS)
    rows = m.to(tl.int64)
    row_mask = m[:, None] < M
    # Row r of X is sequence r. Offsets are only addressed for the matching role.
    q_out = (rows * NQ + head) * D
    k_slot = ((rows * NK + head - NQ) * capacity + pos) * D
    v_slot = ((rows * NK + head - NQ - NK) * capacity + pos) * D
    out = tl.where(head < NQ, q_out, k_slot)
    if head < NQ:
        G, Y = QW, OQ
    else:
        G, Y = KW, KC
    if head < NQ + NK:
        eps = tl.where(head < NQ, QEPS, KEPS)
        inv = tl.rsqrt((tl.sum(left * left, axis=1) + tl.sum(right * right, axis=1)) / D + eps)
        # Round normalized values BEFORE multiplying learned gains.
        nl = (left * inv[:, None]).to(tl.bfloat16).to(tl.float32)
        nr = (right * inv[:, None]).to(tl.bfloat16).to(tl.float32)
        gl = tl.load(G + h).to(tl.float32)
        gr = tl.load(G + D // 2 + h).to(tl.float32)
        nl = (nl * gl[None, :]).to(tl.bfloat16).to(tl.float32)
        nr = (nr * gr[None, :]).to(tl.bfloat16).to(tl.float32)
        cl = tl.load(COS + pos * D + h).to(tl.float32)
        cr = tl.load(COS + pos * D + D // 2 + h).to(tl.float32)
        sl = tl.load(SIN + pos * D + h).to(tl.float32)
        sr = tl.load(SIN + pos * D + D // 2 + h).to(tl.float32)
        # rotate_half is [-right, left]. Native materializes TWO BF16 products
        # before their BF16 sum, per output column.
        ol = (nl * cl[None, :]).to(tl.bfloat16).to(tl.float32)
        ol += (-nr * sl[None, :]).to(tl.bfloat16).to(tl.float32)
        orr = (nr * cr[None, :]).to(tl.bfloat16).to(tl.float32)
        orr += (nl * sr[None, :]).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + out[:, None] + h[None, :], ol.to(tl.bfloat16), mask=row_mask)
        tl.store(Y + out[:, None] + D // 2 + h[None, :], orr.to(tl.bfloat16), mask=row_mask)
    else:
        tl.store(VC + v_slot[:, None] + h[None, :], left.to(tl.bfloat16), mask=row_mask)
        tl.store(VC + v_slot[:, None] + D // 2 + h[None, :], right.to(tl.bfloat16), mask=row_mask)


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


def supports_qkv(x, weight, key_cache):
    """Static rule for the fused head path: decode rows, packed [(Nq+2Nkv)*128, K]."""
    return (supports(x, weight) and key_cache.ndim == 4 and key_cache.shape[-1] == HEAD
            and key_cache.shape[0] == x.shape[0] and weight.shape[0] % HEAD == 0
            and weight.shape[0] // HEAD > 2 * key_cache.shape[1])


def linear_qkv_norm_rope_cache(x, weight, q_weight, k_weight, q_eps, k_eps, cos, sin,
                               position, key_cache, value_cache):
    """Fused ``qkv_norm_rope_cache(x @ weight.T)`` with the same casts and writes.

    ``x`` is BF16 [batch, K] and ``weight`` the packed [(Nq + 2 Nkv) * 128, K]
    projection; ``cos``/``sin`` are [capacity, 128] tables for every absolute
    slot and ``position`` one int64 device value, the only slot of
    ``key_cache``/``value_cache`` ([batch, Nkv, capacity, 128]) that is
    mutated. Returns fresh Q [batch, Nq, 1, 128]. Inputs are read-only.
    """
    m, n, k = _dims(x, weight)
    if not supports_qkv(x, weight, key_cache) or value_cache.shape != key_cache.shape:
        raise ValueError("expected packed head rows and [batch, Nkv, capacity, 128] caches")
    _, kv_heads, capacity, width = key_cache.shape
    heads = n // width - 2 * kv_heads
    if (q_weight.shape != (width,) or k_weight.shape != (width,)
            or cos.shape != (capacity, width) or sin.shape != cos.shape
            or position.shape != (1,) or position.dtype != torch.int64):
        raise ValueError("unsupported gain, rotary table or position shape")
    tensors = (q_weight, k_weight, cos, sin, key_cache, value_cache)
    if (position.device != x.device or any(
            t.device != x.device or t.dtype != torch.bfloat16 or not t.is_contiguous()
            for t in tensors)):
        raise ValueError("fused Q/K/V requires contiguous BF16 tensors on one CUDA device")
    q = torch.empty((m, heads, 1, width), dtype=torch.bfloat16, device=x.device)
    _qkv_head_kernel[(heads + 2 * kv_heads,)](
        x, weight, q_weight, k_weight, cos, sin, position, q, key_cache, value_cache,
        m, k, capacity, NQ=heads, NK=kv_heads, D=width, QEPS=q_eps, KEPS=k_eps,
        BM=max(16, triton.next_power_of_2(m)), BK=BK,
        num_warps=4, num_stages=QKV_STAGES, enable_fp_fusion=False,
    )
    return q
