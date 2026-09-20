"""Decode pipeline kernels. BF16 boundaries match the materialized operations.

All floating inputs/outputs are contiguous BF16 on one CUDA device. Only
qkv_rope_cache mutates inputs: exactly the current K/V slot. No host reads of
position, temporary reduced-precision cache, approximated context or atomics.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _add_rms(R, X, W, SUM, Y, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offset = row * H + cols
    r = tl.load(R + offset, cols < H, other=0).to(tl.float32)
    x = tl.load(X + offset, cols < H, other=0).to(tl.float32)
    # Do NOT norm the FP32 residual sum: native materializes this as BF16.
    summed = (r + x).to(tl.bfloat16)
    s = summed.to(tl.float32)
    inv = tl.rsqrt(tl.sum(s * s, axis=0) / H + EPS)
    normed = (s * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, cols < H, other=0).to(tl.float32)
    tl.store(SUM + offset, summed, cols < H)
    tl.store(Y + offset, (normed * w).to(tl.bfloat16), cols < H)


@triton.jit
def _qkv_cache(P, QW, KW, COS, SIN, POS, Q, KC, VC,
               NQ: tl.constexpr, NK: tl.constexpr, CAP: tl.constexpr,
               CB: tl.constexpr, SB: tl.constexpr, QE: tl.constexpr, KE: tl.constexpr):
    row = tl.program_id(0)
    batch = row // (NQ + NK)
    head = row % (NQ + NK)
    cols = tl.arange(0, 128)
    partner = (cols + 64) % 128
    offset = (batch * (NQ + 2 * NK) + head) * 128
    if head < NQ:
        W = QW
    else:
        W = KW
    x = tl.load(P + offset + cols).to(tl.float32)
    xp = tl.load(P + offset + partner).to(tl.float32)
    eps = tl.where(head < NQ, QE, KE)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / 128 + eps)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    np = (xp * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    wp = tl.load(W + partner).to(tl.float32)
    n = (n * w).to(tl.bfloat16).to(tl.float32)
    np = (np * wp).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(cols < 64, -np, np)
    c = tl.load(COS + batch * CB + cols).to(tl.float32)
    s = tl.load(SIN + batch * SB + cols).to(tl.float32)
    left = (n * c).to(tl.bfloat16).to(tl.float32)
    right = (rotated * s).to(tl.bfloat16).to(tl.float32)
    output = (left + right).to(tl.bfloat16)
    if head < NQ:
        tl.store(Q + (batch * NQ + head) * 128 + cols, output)
    else:
        # Native StaticCache.update only writes these two indexed slices.
        # Position stays device-side and changes on every graph replay.
        position = tl.load(POS)
        dst = ((batch * NK + head - NQ) * CAP + position) * 128 + cols
        tl.store(KC + dst, output)
        v = tl.load(P + offset + NK * 128 + cols)
        tl.store(VC + dst, v)


@triton.jit
def _swiglu(P, Y, I: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // I, index % I
    gate = tl.load(P + row * (2 * I) + col, index < TOTAL, other=0).to(tl.float32)
    up = tl.load(P + row * (2 * I) + I + col, index < TOTAL, other=0).to(tl.float32)
    # PyTorch 2.5.1 SiLU: x / (1 + exp(-x)), in opmath float for BF16.
    # Use libdevice exp, not tl.exp's exp2 approximation; keep SiLU's BF16 result.
    activation = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate))
    rounded = activation.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + index, (rounded * up).to(tl.bfloat16), index < TOTAL)


def _bf16_cuda(*tensors):
    first = tensors[0]
    if (not first.is_cuda or any(t.device != first.device or t.dtype != torch.bfloat16
                                or not t.is_contiguous() for t in tensors)):
        raise ValueError("expected contiguous BF16 tensors on one CUDA device")


def add_rmsnorm(residual, branch, weight, eps):
    """Return fresh (BF16 residual sum, BF16 normalized/gained sum); inputs read-only."""
    _bf16_cuda(residual, branch, weight)
    if (residual.ndim != 3 or residual.shape[1] != 1 or branch.shape != residual.shape
            or weight.shape != (residual.shape[-1],) or not 1 <= weight.numel() <= 8192):
        raise ValueError("expected residual/branch [B,1,H] and gain [H]")
    h, batch = residual.shape[-1], residual.shape[0]
    summed, out = torch.empty_like(residual), torch.empty_like(residual)
    block = triton.next_power_of_2(h)
    if batch:
        _add_rms[(batch,)](residual, branch, weight, summed, out, h, eps, block,
                           num_warps=max(4, min(16, block // 256)), enable_fp_fusion=False)
    return summed, out


def qkv_rope_cache(packed, qw, kw, cos, sin, keys, values, position, nq, nk, qe, ke):
    """[B,1,(Nq+2Nk)*128] -> fresh Q[B,Nq,1,128] plus one in-place K/V slot.

    Cache [B,Nk,C,128] retains every other slot byte-for-byte. The caller must
    guarantee 0 <= position < C, as DecodeState does; no host synchronization.
    """
    _bf16_cuda(packed, qw, kw, cos, sin, keys, values)
    batch = packed.shape[0]
    if (packed.shape != (batch, 1, (nq + 2 * nk) * 128) or nq < 1 or nk < 1
            or keys.ndim != 4 or keys.shape[:2] != (batch, nk) or keys.shape[-1] != 128
            or keys.shape[2] < 1 or values.shape != keys.shape
            or qw.shape != (128,) or kw.shape != (128,)
            or cos.ndim != 3 or cos.shape != sin.shape
            or cos.shape[0] not in (1, batch) or cos.shape[1:] != (1, 128)
            or position.shape != (1,) or position.dtype != torch.int64
            or position.device != packed.device):
        raise ValueError("unsupported packed QKV/cache/position layout")
    q = torch.empty((batch, nq, 1, 128), device=packed.device, dtype=torch.bfloat16)
    if batch:
        _qkv_cache[(batch * (nq + nk),)](
            packed, qw, kw, cos, sin, position, q, keys, values,
            nq, nk, keys.shape[2], 0 if cos.shape[0] == 1 else cos.stride(0),
            0 if sin.shape[0] == 1 else sin.stride(0), qe, ke,
            num_warps=4, enable_fp_fusion=False,
        )
    return q


def swiglu(gate_up):
    """Fresh BF16 product; round SiLU to BF16 BEFORE multiplying the BF16 up input."""
    _bf16_cuda(gate_up)
    if gate_up.ndim != 3 or gate_up.shape[1] != 1 or gate_up.shape[2] < 2 or gate_up.shape[2] % 2:
        raise ValueError("expected packed gate/up [B,1,2I]")
    batch, _, width = gate_up.shape
    intermediate = width // 2
    out = torch.empty((batch, 1, intermediate), device=gate_up.device, dtype=torch.bfloat16)
    if batch:
        _swiglu[(triton.cdiv(batch * intermediate, 256),)](
            gate_up, out, intermediate, batch * intermediate, 256,
            num_warps=4, enable_fp_fusion=False,
        )
    return out
