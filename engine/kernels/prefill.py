"""Packed prefill Q/K head norm, RoPE, and full-prefix static-cache writes.

The arithmetic/cast sequence follows the passing decode kernel. Only indexing
changes: one program owns HEADS consecutive Q-or-K head rows of one prompt
token as a [HEADS, D] tile, so an 8192-token prompt launches 40,960 programs
instead of 327,680 single-head ones. Every head still reduces over its own
128 values and rounds at the same points. Q is [B,T,Hq,128]; K/V retain the
existing [B,Hkv,C,128] storage. No cache slot >= T is read or written here.
Inputs and the angle tables are read-only.
"""

import torch
import triton
import triton.language as tl

HEADS = 8


@triton.jit
def _prefill_qkv(
    QKV, QW, KW, COS, SIN, Q, KC, VC,
    T: tl.constexpr, C: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
    D: tl.constexpr, QEPS: tl.constexpr, KEPS: tl.constexpr,
    HEADS: tl.constexpr, GROUPS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    token = pid // GROUPS
    heads = (pid % GROUPS) * HEADS + tl.arange(0, HEADS)
    batch, pos = token // T, token % T
    is_q = heads < NQ
    is_k = (heads >= NQ) & (heads < NQ + NK)
    live = heads < NQ + NK
    cols = tl.arange(0, D)
    partner = (cols + D // 2) % D
    source = token * (NQ + 2 * NK) * D + heads * D
    x = tl.load(QKV + source[:, None] + cols[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    xp = tl.load(QKV + source[:, None] + partner[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    eps = tl.where(is_q, QEPS, KEPS)
    inv = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    n = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    np = (xp * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    qw = tl.load(QW + cols).to(tl.float32)
    kw = tl.load(KW + cols).to(tl.float32)
    qwp = tl.load(QW + partner).to(tl.float32)
    kwp = tl.load(KW + partner).to(tl.float32)
    w = tl.where(is_q[:, None], qw[None, :], kw[None, :])
    wp = tl.where(is_q[:, None], qwp[None, :], kwp[None, :])
    n = (n * w).to(tl.bfloat16).to(tl.float32)
    np = (np * wp).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(cols[None, :] < D // 2, -np, np)
    cos = tl.load(COS + pos * D + cols).to(tl.float32)
    sin = tl.load(SIN + pos * D + cols).to(tl.float32)
    left = (n * cos[None, :]).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sin[None, :]).to(tl.bfloat16).to(tl.float32)
    out = (left + right).to(tl.bfloat16)
    q_offset = (token * NQ + heads) * D
    # Negative for Q rows; those lanes are masked off and never addressed.
    kv_offset = ((batch * NK + heads - NQ) * C + pos) * D
    tl.store(Q + q_offset[:, None] + cols[None, :], out, mask=is_q[:, None])
    tl.store(KC + kv_offset[:, None] + cols[None, :], out, mask=is_k[:, None])
    v_source = token * (NQ + 2 * NK) * D + (heads + NK) * D
    v = tl.load(QKV + v_source[:, None] + cols[None, :], mask=is_k[:, None], other=0.0)
    tl.store(VC + kv_offset[:, None] + cols[None, :], v, mask=is_k[:, None])


def prefill_qkv(qkv, qw, kw, qeps, keps, cos, sin, keys, values, length):
    """Read packed BF16 [B*T,(Hq+2Hkv)*128]; overwrite only cache [:,:,:T].

    Angles are the model-generated [C,128] tables already owned by DecodeState.
    Return a fresh, contiguous BF16 Q [B,T,Hq,128]. All tensors share a device.
    This is a full fresh prefill at position zero, not chunked prefill.
    """
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("K/V must have matching [B,Hkv,C,D] shapes")
    batch, nk, capacity, width = keys.shape
    if (batch < 1 or nk < 1 or width != 128 or not 1 <= length <= capacity
            or qkv.ndim != 2 or qkv.shape[0] != batch * length):
        raise ValueError("unsupported full-prefill dimensions")
    nq = qkv.shape[1] // width - 2 * nk
    if (nq < 1 or nq % nk or qkv.shape[1] != (nq + 2 * nk) * width
            or qw.shape != (width,) or kw.shape != qw.shape
            or cos.shape != (capacity, width) or sin.shape != cos.shape):
        raise ValueError("packed QKV, gains, or rotary tables disagree")
    tensors = (qkv, qw, kw, cos, sin, keys, values)
    if (not qkv.is_cuda or any(t.device != qkv.device or t.dtype != torch.bfloat16
                              or not t.is_contiguous() for t in tensors)):
        raise ValueError("prefill requires contiguous BF16 tensors on one CUDA device")
    q = torch.empty((batch, length, nq, width), device=qkv.device, dtype=qkv.dtype)
    groups = triton.cdiv(nq + nk, HEADS)
    _prefill_qkv[(batch * length * groups,)](
        qkv, qw, kw, cos, sin, q, keys, values,
        T=length, C=capacity, NQ=nq, NK=nk, D=width,
        QEPS=qeps, KEPS=keps, HEADS=HEADS, GROUPS=groups,
        num_warps=4, enable_fp_fusion=False,
    )
    return q
