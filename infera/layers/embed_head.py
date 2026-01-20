import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from infera.utils.context import get_context


# ============================================================
# Vocab Parallel Embedding (TP)
# ============================================================

class VocabParallelEmbedding(nn.Module):
    """
    Tensor-parallel vocabulary embedding.

    - vocab dimension is sharded
    - embedding dim is replicated
    - output is all-reduced
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // self.tp_size

        self.vocab_start_idx = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )

        # Infera-style loader hook
        self.weight.weight_loader = self.weight_loader

    # --------------------------------------------------
    # Weight loading (TP shard)
    # --------------------------------------------------

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        shard_size = param.size(0)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start, shard_size))

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [N]
        return: [N, hidden_dim]
        """
        if self.tp_size > 1:
            mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)
            local_ids = input_ids - self.vocab_start_idx
            local_ids = torch.where(mask, local_ids, 0)
        else:
            mask = None
            local_ids = input_ids

        output = F.embedding(local_ids, self.weight)

        if self.tp_size > 1:
            output = output * mask.unsqueeze(-1)
            dist.all_reduce(output)

        return output


# ============================================================
# Parallel LM Head (TP)
# ============================================================

class ParallelLMHead(nn.Module):
    """
    Tensor-parallel LM Head.

    - hidden dim replicated
    - vocab dim sharded
    - gather logits on rank 0
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__()

        assert not bias, "Bias is not supported in ParallelLMHead"

        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // self.tp_size

        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )

        self.weight.weight_loader = self.weight_loader

    # --------------------------------------------------
    # Weight loading
    # --------------------------------------------------

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        shard_size = param.size(0)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start, shard_size))

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states:
          - prefill: [T, H]
          - decode:  [B, H]

        returns:
          - rank 0: [*, vocab]
          - others: None
        """
        ctx = get_context()

        # Prefill: only compute logits for last token of each sequence
        if ctx.is_prefill:
            last_indices = ctx.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()

        # Local shard logits
        local_logits = F.linear(hidden_states, self.weight)

        if self.tp_size == 1:
            return local_logits

        # Gather vocab shards to rank 0
        if self.tp_rank == 0:
            gathered = [
                torch.empty_like(local_logits)
                for _ in range(self.tp_size)
            ]
        else:
            gathered = None

        dist.gather(local_logits, gathered, dst=0)

        if self.tp_rank == 0:
            return torch.cat(gathered, dim=-1)

        return None
