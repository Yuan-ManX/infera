import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

from infera.utils.context import get_context


# ============================================================
# Triton KV Cache Store Kernel (Infera-style)
# ============================================================

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    stride_k,
    stride_v,
    stride_cache,
    N,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    slots = tl.load(slot_mapping_ptr + offs, mask=mask, other=-1)
    valid = slots != -1

    d = tl.arange(0, D)

    for i in range(BLOCK):
        if valid[i]:
            k = tl.load(key_ptr + offs[i] * stride_k + d)
            v = tl.load(value_ptr + offs[i] * stride_v + d)
            cache_off = slots[i] * stride_cache + d
            tl.store(k_cache_ptr + cache_off, k)
            tl.store(v_cache_ptr + cache_off, v)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    key, value: [N, num_heads, head_dim]
    k_cache, v_cache: [num_blocks * block_size, num_heads * head_dim]
    slot_mapping: [N]
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    assert key.is_cuda and value.is_cuda
    assert k_cache.is_cuda and v_cache.is_cuda
    assert slot_mapping.is_cuda

    BLOCK = 4
    grid = (triton.cdiv(N, BLOCK),)

    store_kvcache_kernel[grid](
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        key.stride(0),
        value.stride(0),
        k_cache.stride(1),
        N,
        D=D,
        BLOCK=BLOCK,
    )


# ============================================================
# Attention Module (Inference-only, Infera-aligned)
# ============================================================

class Attention(nn.Module):
    """
    Inference-only Attention module with:
    - FlashAttention v2
    - Prefix cache (block table)
    - KV cache decode
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

        # Assigned by ModelRunner.allocate_kv_cache
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def forward(
        self,
        q: torch.Tensor,  # [Nq, num_heads, head_dim]
        k: torch.Tensor,  # [Nk, num_kv_heads, head_dim]
        v: torch.Tensor,  # [Nk, num_kv_heads, head_dim]
    ):
        ctx = get_context()

        # ----------------------------------------------------
        # KV cache write (deterministic, graph-friendly)
        # ----------------------------------------------------
        if self.k_cache is not None and ctx.slot_mapping is not None:
            store_kvcache(
                k,
                v,
                self.k_cache,
                self.v_cache,
                ctx.slot_mapping,
            )

        # ----------------------------------------------------
        # Prefill (prompt / chunk)
        # ----------------------------------------------------
        if ctx.is_prefill:
            # Prefix cache path
            if ctx.block_tables is not None:
                k, v = self.k_cache, self.v_cache

            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=ctx.cu_seqlens_q,
                cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=ctx.block_tables,
            )

        # ----------------------------------------------------
        # Decode (single-token hot path)
        # ----------------------------------------------------
        return flash_attn_with_kvcache(
            q.unsqueeze(1),          # [B, 1, H, D]
            self.k_cache,
            self.v_cache,
            cache_seqlens=ctx.context_lens,
            block_table=ctx.block_tables,
            softmax_scale=self.scale,
            causal=True,
        )
