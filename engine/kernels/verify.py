"""Four-position verifier QKV; same arithmetic as the passing prefill kernel.

Write only [start, start + length). Earlier valid K/V are immutable. The
caller rolls back LOGICAL length after rejection; no valid context is evicted.
All BF16 rounding boundaries of kernels/prefill.py are retained.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _verify_qkv(
    QKV, QW, KW, COS, SIN, START, Q, KC, VC,
    T: tl.constexpr, C: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
    D: tl.constexpr, QEPS: tl.constexpr, KEPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token = row // (NQ + NK)
    head = row % (NQ + NK)
    batch = token // T
    pos = tl.load(START).to(tl.int64) + token % T
    source = token * (NQ + 2 * NK) * D + head * D
    if head < NQ:
        W, OUT = QW, Q
        offset = (token * NQ + head) * D
    else:
        W, OUT = KW, KC
        offset = ((batch * NK + head - NQ) * C + pos) * D
    cols = tl.arange(0, D)
    partner = (cols + D // 2) % D
    x = tl.load(QKV + source + cols).to(tl.float32)
    xp = tl.load(QKV + source + partner).to(tl.float32)
    eps = tl.where(head < NQ, QEPS, KEPS)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    np = (xp * inv).to(tl.bfloat16).to(tl.float32)
    n = (n * tl.load(W + cols).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    np = (np * tl.load(W + partner).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(cols < D // 2, -np, np)
    cos = tl.load(COS + pos * D + cols).to(tl.float32)
    sin = tl.load(SIN + pos * D + cols).to(tl.float32)
    left = (n * cos).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sin).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + offset + cols, (left + right).to(tl.bfloat16))
    if head >= NQ:
        v = tl.load(QKV + token * (NQ + 2 * NK) * D + (NK + head) * D + cols)
        tl.store(VC + offset + cols, v)


def verify_qkv(qkv, qw, kw, qeps, keps, cos, sin, keys, values, length, start):
    """Return Q[B,T,Hq,128], mutate ONLY cache slots start ... start+T-1.

    Caller guarantees 0 <= start and start+T <= C. No device value is read
    on the host. start is a fixed-address int64 CUDA tensor of shape [1].
    """
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("K/V must have matching [B,Hkv,C,D] shapes")
    batch, nk, capacity, width = keys.shape
    if (batch < 1 or nk < 1 or width != 128 or not 1 <= length <= capacity
            or qkv.ndim != 2 or qkv.shape[0] != batch * length):
        raise ValueError("unsupported verification dimensions")
    nq = qkv.shape[1] // width - 2 * nk
    if (nq < 1 or nq % nk or qkv.shape[1] != (nq + 2 * nk) * width
            or qw.shape != (width,) or kw.shape != qw.shape
            or cos.shape != (capacity, width) or sin.shape != cos.shape
            or start.shape != (1,) or start.dtype != torch.int64
            or start.device != qkv.device):
        raise ValueError("packed QKV, gains, angles, or position disagree")
    tensors = (qkv, qw, kw, cos, sin, keys, values)
    if (not qkv.is_cuda or any(t.device != qkv.device or t.dtype != torch.bfloat16
                              or not t.is_contiguous() for t in tensors)):
        raise ValueError("verification requires contiguous BF16 CUDA tensors")
    q = torch.empty((batch, length, nq, width), device=qkv.device, dtype=qkv.dtype)
    _verify_qkv[(batch * length * (nq + nk),)](
        qkv, qw, kw, cos, sin, start, q, keys, values,
        T=length, C=capacity, NQ=nq, NK=nk, D=width,
        QEPS=qeps, KEPS=keps, num_warps=4, enable_fp_fusion=False,
    )
    return q
