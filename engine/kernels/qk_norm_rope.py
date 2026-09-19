"""BF16 Q/K head RMSNorm plus RoPE for a single decode token."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope_kernel(
    Q, K, QW, KW, COS, SIN, OQ, OK,
    NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
    COS_BATCH: tl.constexpr, SIN_BATCH: tl.constexpr,
    QEPS: tl.constexpr, KEPS: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // (NQ + NK)
    head = row % (NQ + NK)
    if head < NQ:
        X, W, Y = Q, QW, OQ
        offset = (batch * NQ + head) * D
    else:
        X, W, Y = K, KW, OK
        offset = (batch * NK + head - NQ) * D

    eps = tl.where(head < NQ, QEPS, KEPS)
    cols = tl.arange(0, D)
    partner = (cols + D // 2) % D
    x = tl.load(X + offset + cols).to(tl.float32)
    xp = tl.load(X + offset + partner).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    # Round normalized values BEFORE multiplying learned gains.
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    np = (xp * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    wp = tl.load(W + partner).to(tl.float32)
    n = (n * w).to(tl.bfloat16).to(tl.float32)
    np = (np * wp).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(cols < D // 2, -np, np)
    c = tl.load(COS + batch * COS_BATCH + cols).to(tl.float32)
    s = tl.load(SIN + batch * SIN_BATCH + cols).to(tl.float32)
    # Native RoPE materializes TWO BF16 products before their BF16 sum.
    left = (n * c).to(tl.bfloat16).to(tl.float32)
    right = (rotated * s).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + offset + cols, (left + right).to(tl.bfloat16))


def qk_norm_rope(q, k, q_weight, k_weight, cos, sin, q_eps, k_eps):
    """Read-only BF16 inputs; return fresh contiguous Q/K, [B,H,1,128].

    Q/K may be strided; gains must be [128]. Cos/sin are [1 or B,1,128]
    from the existing rotary embedding, NOT recomputed or cached here.
    """
    batch = q.shape[0]
    if (q.ndim != 4 or k.ndim != 4 or q.shape[2:] != (1, 128)
            or k.shape[2:] != (1, 128) or k.shape[0] != batch
            or cos.shape != sin.shape or cos.ndim != 3
            or cos.shape[0] not in (1, batch) or cos.shape[1:] != (1, 128)
            or q_weight.shape != (128,) or k_weight.shape != (128,)):
        raise ValueError("Unsupported Q/K or rotary shape")
    inputs = (q, k, q_weight, k_weight, cos, sin)
    if not q.is_cuda or any(t.device != q.device or t.dtype != torch.bfloat16 for t in inputs):
        raise ValueError("Q/K norm-RoPE requires BF16 tensors on one CUDA device")
    q, k, q_weight, k_weight, cos, sin = (t.contiguous() for t in inputs)
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    if batch:
        _qk_norm_rope_kernel[(batch * (q.shape[1] + k.shape[1]),)](
            q, k, q_weight, k_weight, cos, sin, oq, ok,
            NQ=q.shape[1], NK=k.shape[1], D=128,
            COS_BATCH=0 if cos.shape[0] == 1 else cos.stride(0),
            SIN_BATCH=0 if sin.shape[0] == 1 else sin.stride(0),
            QEPS=q_eps, KEPS=k_eps, num_warps=4, enable_fp_fusion=False,
        )
    return oq, ok
