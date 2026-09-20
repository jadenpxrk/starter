"""Qwen3 MLP with a shared gate/up projection."""

import torch
from torch import nn
from torch.nn import functional as F


class PackedMLP(nn.Module):
    @torch.no_grad()
    def __init__(self, reference):
        super().__init__()
        self.gate_up_weight = nn.Parameter(
            torch.cat((reference.gate_proj.weight, reference.up_proj.weight)),
            requires_grad=False,
        )
        self.down_proj = reference.down_proj
        self.act_fn = reference.act_fn

    def forward(self, x):
        gate, up = F.linear(x, self.gate_up_weight).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)
