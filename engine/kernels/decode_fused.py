"""Decode-only fusions that keep Qwen3's BF16 cast boundaries.

Three kernels replace runs of small per-layer launches in the captured step:
residual add + RMSNorm, SiLU * up, and packed Q/K/V head-norm + RoPE with the
K/V rows written straight into their static cache slot. Each BF16 rounding
point is placed where the Transformers 4.51.3 reference places it. Results
are not bit-identical to the reference: reduction grouping, the packed GEMM
geometry and the exp/rsqrt implementations differ at FP32 level before those
roundings, the same class of difference as the engine's earlier fusions.
"""

import torch
import triton
import triton.language as tl

MAX_BLOCK = 8192


@triton.jit
def _add_rms_norm_kernel(X, Y, W, S, O, n_cols, eps, split_stride,
                         BLOCK: tl.constexpr, SPLITS: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offsets = row * n_cols + cols
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    if SPLITS == 0:
        y = tl.load(Y + offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        # FP32 split-K partials of the branch GEMM: add them all, then round
        # once to BF16, which is where the GEMM output itself rounds.
        y = tl.zeros((BLOCK,), dtype=tl.float32)
        for split in tl.static_range(SPLITS):
            y += tl.load(Y + split * split_stride + offsets, mask=mask, other=0.0)
        y = y.to(S.dtype.element_ty).to(tl.float32)
    # Native adds the residual in BF16; the norm then reads that rounded sum.
    total = (x + y).to(S.dtype.element_ty)
    tl.store(S + offsets, total, mask=mask)
    total = total.to(tl.float32)
    variance = tl.sum(total * total, axis=0) / n_cols
    normed = total * tl.math.rsqrt(variance + eps)
    # Same cast placement as kernels/rmsnorm.py: round, then multiply the gain.
    weight = tl.load(W + cols, mask=mask, other=0.0)
    tl.store(O + offsets, normed.to(O.dtype.element_ty) * weight, mask=mask)


@triton.jit
def _silu_mul_kernel(GU, O, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    gate = tl.load(GU + row * 2 * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GU + row * 2 * n_cols + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    # Native rounds SiLU to BF16 before the BF16 product.
    act = (gate / (1.0 + tl.exp(-gate))).to(O.dtype.element_ty).to(tl.float32)
    tl.store(O + row * n_cols + cols, (act * up).to(O.dtype.element_ty), mask=mask)


@triton.jit
def _qkv_norm_rope_cache_kernel(
    QKV, QW, KW, COS, SIN, POS, OQ, KC, VC, row_stride, capacity, split_stride,
    NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
    QEPS: tl.constexpr, KEPS: tl.constexpr, SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // (NQ + NK)
    head = row % (NQ + NK)
    pos = tl.load(POS)
    # Packed columns are [q heads | k heads | v heads], so head*D addresses q or k.
    source = batch * row_stride + head * D
    q_out = (batch * NQ + head) * D
    kv_slot = ((batch * NK + head - NQ) * capacity + pos) * D
    out = tl.where(head < NQ, q_out, kv_slot)
    if head < NQ:
        W, Y = QW, OQ
    else:
        W, Y = KW, KC

    eps = tl.where(head < NQ, QEPS, KEPS)
    cols = tl.arange(0, D)
    partner = (cols + D // 2) % D
    if SPLITS == 0:
        x = tl.load(QKV + source + cols).to(tl.float32)
        xp = tl.load(QKV + source + partner).to(tl.float32)
    else:
        # FP32 split-K partials of the packed projection, rounded once to BF16.
        x = tl.zeros((D,), dtype=tl.float32)
        xp = tl.zeros((D,), dtype=tl.float32)
        for split in tl.static_range(SPLITS):
            x += tl.load(QKV + split * split_stride + source + cols)
            xp += tl.load(QKV + split * split_stride + source + partner)
        x = x.to(tl.bfloat16).to(tl.float32)
        xp = xp.to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    # Round normalized values BEFORE multiplying learned gains.
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    np = (xp * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    wp = tl.load(W + partner).to(tl.float32)
    n = (n * w).to(tl.bfloat16).to(tl.float32)
    np = (np * wp).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(cols < D // 2, -np, np)
    c = tl.load(COS + pos * D + cols).to(tl.float32)
    s = tl.load(SIN + pos * D + cols).to(tl.float32)
    # Native RoPE materializes TWO BF16 products before their BF16 sum.
    left = (n * c).to(tl.bfloat16).to(tl.float32)
    right = (rotated * s).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + out + cols, (left + right).to(tl.bfloat16))
    if head >= NQ:
        v_source = batch * row_stride + (NK + head) * D + cols
        if SPLITS == 0:
            v = tl.load(QKV + v_source)
        else:
            v = tl.zeros((D,), dtype=tl.float32)
            for split in tl.static_range(SPLITS):
                v += tl.load(QKV + split * split_stride + v_source)
            v = v.to(tl.bfloat16)
        tl.store(VC + kv_slot + cols, v)


def _rows(x):
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("expected a contiguous 2-D BF16 CUDA tensor")
    return x.shape


def _splits(branch, shape, device):
    """0 for a BF16 [rows, cols] operand; S for FP32 split-K partials [S, rows, cols]."""
    if branch.ndim == 3 and branch.dtype == torch.float32:
        if (tuple(branch.shape[1:]) != tuple(shape) or branch.shape[0] < 1
                or branch.device != device or not branch.is_contiguous()):
            raise ValueError("split-K partials must be contiguous FP32 [splits, rows, cols]")
        return branch.shape[0]
    if tuple(branch.shape) != tuple(shape) or branch.device != device:
        raise ValueError("operand shape or device disagrees")
    _rows(branch)
    return 0


def add_rms_norm(residual, branch, weight, eps):
    """Return (residual + branch, RMSNorm of that sum); fresh [rows, cols] BF16 tensors.

    ``branch`` is BF16 [rows, cols], or FP32 split-K partials [S, rows, cols]
    that are added and rounded once to BF16 inside the kernel.
    """
    n_rows, n_cols = _rows(residual)
    if weight.shape != (n_cols,):
        raise ValueError("residual and gain shapes disagree")
    splits = _splits(branch, residual.shape, residual.device)
    block = triton.next_power_of_2(n_cols)
    if block > MAX_BLOCK:
        raise ValueError(f"a row must fit in one block; {n_cols} columns does not")
    total, normed = torch.empty_like(residual), torch.empty_like(residual)
    _add_rms_norm_kernel[(n_rows,)](
        residual, branch, weight, total, normed, n_cols, eps, n_rows * n_cols,
        BLOCK=block, SPLITS=splits, num_warps=max(4, min(16, block // 256)),
    )
    return total, normed


def silu_mul(gate_up, block=1024):
    """silu(gate) * up for a packed [rows, 2 * cols] projection; returns [rows, cols]."""
    n_rows, width = _rows(gate_up)
    if width % 2:
        raise ValueError("packed gate/up width must be even")
    n_cols = width // 2
    out = torch.empty((n_rows, n_cols), dtype=gate_up.dtype, device=gate_up.device)
    _silu_mul_kernel[(n_rows, triton.cdiv(n_cols, block))](gate_up, out, n_cols, BLOCK=block)
    return out


def qkv_norm_rope_cache(qkv, q_weight, k_weight, q_eps, k_eps, cos, sin, position,
                        key_cache, value_cache):
    """Normalize and rotate packed Q/K, write K/V rows into slot ``position``.

    ``qkv`` is BF16 [batch, (Nq + 2 Nkv) * D] from one packed projection, or
    FP32 split-K partials [S, batch, (Nq + 2 Nkv) * D] of it, added and rounded
    once to BF16 inside the kernel. ``cos`` and ``sin`` are [capacity, D] tables
    for every absolute slot; ``position`` is one int64 device value and is the
    only slot of ``key_cache``/``value_cache`` ([batch, Nkv, capacity, D]) that
    is mutated. Returns fresh Q [batch, Nq, 1, D].
    """
    if qkv.ndim == 3 and qkv.dtype == torch.float32:
        if not qkv.is_cuda or not qkv.is_contiguous() or qkv.shape[0] < 1:
            raise ValueError("split-K partials must be contiguous FP32 CUDA [splits, batch, width]")
        splits, (batch, width) = qkv.shape[0], qkv.shape[1:]
    else:
        splits, (batch, width) = 0, _rows(qkv)
    if (key_cache.ndim != 4 or key_cache.shape != value_cache.shape
            or key_cache.shape[0] != batch):
        raise ValueError("K/V caches must be [batch, Nkv, capacity, D]")
    _, kv_heads, capacity, head = key_cache.shape
    heads = width // head - 2 * kv_heads
    if (head & (head - 1) or heads < 1 or width != (heads + 2 * kv_heads) * head
            or q_weight.shape != (head,) or k_weight.shape != (head,)
            or cos.shape != (capacity, head) or sin.shape != cos.shape
            or position.shape != (1,) or position.dtype != torch.int64):
        raise ValueError("unsupported packed Q/K/V, rotary table or position shape")
    tensors = (q_weight, k_weight, cos, sin, key_cache, value_cache)
    if (position.device != qkv.device or any(
            t.device != qkv.device or t.dtype != torch.bfloat16 or not t.is_contiguous()
            for t in tensors)):
        raise ValueError("Q/K/V fusion requires contiguous BF16 tensors on one CUDA device")
    q = torch.empty((batch, heads, 1, head), dtype=torch.bfloat16, device=qkv.device)
    _qkv_norm_rope_cache_kernel[(batch * (heads + kv_heads),)](
        qkv, q_weight, k_weight, cos, sin, position, q, key_cache, value_cache,
        width, capacity, batch * width, NQ=heads, NK=kv_heads, D=head,
        QEPS=q_eps, KEPS=k_eps, SPLITS=splits, num_warps=4, enable_fp_fusion=False,
    )
    return q
