import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    Fused SiLU + Mul (SwiGLU)

    Input shape:
        [..., 2 * hidden_dim]

    Output shape:
        [..., hidden_dim]
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_size = x.size(-1) // 2

        x1 = x[..., :hidden_size]
        x2 = x[..., hidden_size:]

        return F.silu(x1) * x2
