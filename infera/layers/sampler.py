import torch
from torch import nn
from dataclasses import dataclass


# ================================
# Infera Sampling Config
# ================================

@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-5, "temperature must be > 0"


# ================================
# Sampler
# ================================

class Sampler(nn.Module):
    """
    Infera Sampler

    - Gumbel-Max sampling
    - Logits-domain (no softmax)
    - Greedy-friendly
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(
        self,
        logits: torch.Tensor,        # [B, V]
        temperatures: torch.Tensor,  # [B]
    ) -> torch.Tensor:

        # ---------- Greedy fast path ----------
        if torch.all(temperatures <= 1.0):
            return logits.argmax(dim=-1)

        # ---------- Temperature scaling ----------
        logits = logits.float().div_(
            temperatures.unsqueeze(1).clamp_min_(1e-5)
        )

        # ---------- Gumbel noise ----------
        gumbel = torch.empty_like(logits).exponential_().log_().neg_()

        # ---------- Gumbel-Max ----------
        return torch.argmax(logits + gumbel, dim=-1)
