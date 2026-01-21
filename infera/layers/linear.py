import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, List


# =========================
# Utils
# =========================

def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0
    return numerator // denominator


def get_tp_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


# =========================
# Base Linear
# =========================

class LinearBase(nn.Module):
    """
    Base class for tensor-parallel linear layers.

    - tp_dim: which dimension is sharded (0 = output, 1 = input)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: Optional[int] = None,
    ):
        super().__init__()

        self.tp_dim = tp_dim
        self.tp_rank, self.tp_size = get_tp_info()

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# =========================
# Replicated Linear
# =========================

class ReplicatedLinear(LinearBase):
    """
    Fully replicated linear (no sharding).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# =========================
# Column Parallel Linear
# =========================

class ColumnParallelLinear(LinearBase):
    """
    Column-parallel linear:
    - shard output dimension
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        _, tp_size = get_tp_info()
        super().__init__(
            input_size,
            divide(output_size, tp_size),
            bias,
            tp_dim=0,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.size(self.tp_dim)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(self.tp_dim, start, shard_size))

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# =========================
# Merged Column Parallel Linear
# =========================

class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    Used for merged projections like [gate, up] or similar.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ):
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        param_slice = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_slice = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]

        param_slice.copy_(loaded_slice)


# =========================
# QKV Parallel Linear
# =========================

class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV packed column-parallel projection.
    Layout: [Q | K | V]
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = False,
    ):
        tp_rank, tp_size = get_tp_info()

        total_num_kv_heads = total_num_kv_heads or total_num_heads

        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)

        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ):
        assert loaded_shard_id in ("q", "k", "v")

        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + shard_size

        param_slice = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_slice = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]

        param_slice.copy_(loaded_slice)


# =========================
# Row Parallel Linear
# =========================

class RowParallelLinear(LinearBase):
    """
    Row-parallel linear:
    - shard input dimension
    - all-reduce on output
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        _, tp_size = get_tp_info()
        super().__init__(
            divide(input_size, tp_size),
            output_size,
            bias,
            tp_dim=1,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.size(self.tp_dim)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(self.tp_dim, start, shard_size))

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)

        if self.tp_size > 1:
            dist.all_reduce(y, op=dist.ReduceOp.SUM)

        return y
