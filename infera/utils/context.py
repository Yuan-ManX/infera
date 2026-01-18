from dataclasses import dataclass
from typing import Optional

import torch


# ============================================================
# Execution Context
# ============================================================

@dataclass(frozen=True)
class Context:
    """
    Runtime execution context for a single model step.

    This context is:
      - Step-scoped
      - Read-only after creation
      - Consumed by attention / CUDA graph kernels
    """

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------
    is_prefill: bool

    # --------------------------------------------------------
    # Prefill-only (FlashAttention)
    # --------------------------------------------------------
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # --------------------------------------------------------
    # Decode-only
    # --------------------------------------------------------
    context_lens: Optional[torch.Tensor] = None

    # --------------------------------------------------------
    # Shared
    # --------------------------------------------------------
    slot_mapping: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None


# Global step context (single active step)
_CONTEXT: Optional[Context] = None


# ============================================================
# Context API
# ============================================================

def get_context() -> Context:
    assert _CONTEXT is not None, "Context is not set"
    return _CONTEXT


def set_prefill_context(
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    slot_mapping: torch.Tensor,
    block_tables: Optional[torch.Tensor] = None,
) -> None:
    """
    Set execution context for prefill step.
    """
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
    )


def set_decode_context(
    *,
    slot_mapping: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
) -> None:
    """
    Set execution context for decode step.
    """
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=False,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )


def reset_context() -> None:
    """
    Clear execution context.

    Must be called after each model step.
    """
    global _CONTEXT
    _CONTEXT = None
