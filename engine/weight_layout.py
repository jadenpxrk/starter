"""Warmup-time, lossless tile-major copies for immutable decode projection weights.

The original row-major weights remain the authority for prefill/fallbacks.
These non-persistent buffers change addresses only: no dtype conversion,
parameter replacement, padding, or reordering within a 32x64 dot tile.
Install after loading/conversion and before graph capture. Weights must not
be trained, replaced, or mutated afterward; rebuild before capturing if they
change. This engine never updates its loaded projection weights.
"""

import torch

TILE_N = 32
TILE_K = 64


def can_pack(weight):
    return (weight.ndim == 2 and weight.dtype == torch.bfloat16
            and weight.is_contiguous() and weight.shape[0] > 0 and weight.shape[1] > 0
            and weight.shape[0] % TILE_N == 0 and weight.shape[1] % TILE_K == 0)


@torch.no_grad()
def pack_weight(weight):
    """W[N,K] -> Wt[N/32,K/64,32,64], with exactly the same BF16 bits.

    CPU is supported for layout tests. Runtime installation uses loaded CUDA
    tensors. A contiguous copy is required except for already equivalent layouts.
    """
    if not can_pack(weight):
        raise ValueError("tile weights require contiguous BF16 W[N%32=0,K%64=0]")
    n, k = weight.shape
    return weight.detach().view(n // TILE_N, TILE_N, k // TILE_K, TILE_K).permute(
        0, 2, 1, 3).contiguous()


@torch.no_grad()
def install_tiled_weights(model):
    """Add three read-only decode buffers per supported layer; return payload bytes.

    No model Parameter is rebound. Non-persistent buffers keep state_dict keys
    unchanged. Calling again rebuilds copies; do not do so with live graphs.
    Unsupported weight shapes/dtypes retain their existing projection path.
    """
    if model.training:
        raise ValueError("weight tiling is inference-only; call eval() first")
    total = 0
    for layer in model.model.layers:
        a, m = layer.self_attn, layer.mlp
        for owner, name, weight in (
            (a, "_qkv_weight_tiles", a.qkv_weight),
            (a.o_proj, "_weight_tiles", a.o_proj.weight),
            (m.down_proj, "_weight_tiles", m.down_proj.weight),
        ):
            tiles = pack_weight(weight) if can_pack(weight) else None
            owner.register_buffer(name, tiles, persistent=False)
            if tiles is not None:
                total += tiles.numel() * tiles.element_size()
    return total


def ensure_tiled_weights(model):
    """Prepare once per engine, before the first eligible decode graph capture."""
    if not getattr(model, "_decode_tiles_ready", False):
        install_tiled_weights(model)
        model._decode_tiles_ready = True
