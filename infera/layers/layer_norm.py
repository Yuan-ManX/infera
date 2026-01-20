import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Inference-optimized RMSNorm.

    Supports:
      - rms_norm(x)
      - rms_norm(x, residual)  (fused add + rms)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # --------------------------------------------------
    # RMSNorm (no residual)
    # --------------------------------------------------

    @torch.compile
    def _rms_norm(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype

        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_f.mul_(torch.rsqrt(var + self.eps))

        return x_f.to(orig_dtype).mul_(self.weight)

    # --------------------------------------------------
    # RMSNorm + Residual (fused)
    # --------------------------------------------------

    @torch.compile
    def _rms_norm_with_residual(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype

        # fused add
        x_f = x.float()
        x_f.add_(residual.float())

        # update residual (Infera-style)
        new_residual = x_f.to(orig_dtype)

        # rms
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_f.mul_(torch.rsqrt(var + self.eps))

        out = x_f.to(orig_dtype).mul_(self.weight)
        return out, new_residual

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._rms_norm(x)
        return self._rms_norm_with_residual(x, residual)
