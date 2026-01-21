from functools import lru_cache
import torch
from torch import nn


# =========================
# Rotary Apply (Infera-style)
# =========================

@torch.compile
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    x: [..., head_dim]
    cos/sin: [..., head_dim // 2]
    """
    # 保持 dtype，不强制 float()
    x1, x2 = x.chunk(2, dim=-1)

    # RoPE rotation
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

    return torch.cat((y1, y2), dim=-1)


# =========================
# Rotary Embedding Module
# =========================

class RotaryEmbedding(nn.Module):
    """
    Infera-style RoPE:
    - cos / sin cached separately
    - no runtime chunk
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()

        assert rotary_dim == head_size
        self.rotary_dim = rotary_dim

        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )

        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # [max_pos, rotary_dim // 2]
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,  # [seq_len]
        query: torch.Tensor,      # [B, H, T, D]
        key: torch.Tensor,        # [B, H, T, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        cos = self.cos_cache[positions]  # [T, D//2]
        sin = self.sin_cache[positions]

        # broadcast: [1, 1, T, D//2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        return query, key


# =========================
# RoPE Factory (cached)
# =========================

@lru_cache(maxsize=1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    # Infera 当前默认不在这里处理 scaling
    assert rope_scaling is None

    return RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
    )
