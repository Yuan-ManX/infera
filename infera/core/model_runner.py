import pickle
from typing import List, Optional

import torch
import torch.distributed as dist
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

from infera.config import InferaConfig
from infera.core.sequence import Sequence
from infera.models.qwen3 import Qwen3ForCausalLM
from infera.layers.sampler import Sampler
from infera.utils.context import (
    set_context,
    get_context,
    reset_context,
)
from infera.utils.loader import load_model


class ModelRunner:
    """
    Model execution worker.

    Responsibilities:
      - Owns a shard of the model (Tensor Parallel)
      - Manages KV cache memory
      - Executes prefill / decode steps
      - Provides IPC interface for rank-0 driver
    """

    def __init__(
        self,
        config: InferaConfig,
        rank: int,
        event: Event | List[Event],
    ):
        self.config = config
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        self.event = event
        self.is_driver_rank = rank == 0

        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager

        # -----------------------------
        # Distributed & CUDA setup
        # -----------------------------
        self._init_distributed()
        self._init_model()

        # -----------------------------
        # Runtime helpers
        # -----------------------------
        self.sampler = Sampler()

        self._warmup_model()
        self._allocate_kv_cache()

        if not self.enforce_eager:
            self._capture_cuda_graphs()

        # -----------------------------
        # IPC setup (TP > 1)
        # -----------------------------
        if self.world_size > 1:
            self._init_ipc()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_distributed(self) -> None:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:2333",
            world_size=self.world_size,
            rank=self.rank,
        )
        torch.cuda.set_device(self.rank)

    def _init_model(self) -> None:
        hf_config = self.config.hf_config

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, self.config.model)

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def _init_ipc(self) -> None:
        if self.is_driver_rank:
            self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
            dist.barrier()
        else:
            dist.barrier()
            self.shm = SharedMemory(name="nanovllm")
            self._ipc_loop()

    # ------------------------------------------------------------------
    # IPC
    # ------------------------------------------------------------------

    def _ipc_loop(self) -> None:
        """Worker loop (non-rank0)."""
        while True:
            method, args = self._read_shm()
            self.call(method, *args)
            if method == "exit":
                break

    def _read_shm(self):
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], "little")
        method, *args = pickle.loads(self.shm.buf[4 : 4 + n])
        self.event.clear()
        return method, args

    def _write_shm(self, method: str, *args) -> None:
        data = pickle.dumps([method, *args])
        self.shm.buf[:4] = len(data).to_bytes(4, "little")
        self.shm.buf[4 : 4 + len(data)] = data
        for e in self.event:
            e.set()

    def call(self, method: str, *args):
        """Driver dispatch or local execution."""
        if self.world_size > 1 and self.is_driver_rank:
            self._write_shm(method, *args)
        return getattr(self, method)(*args)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def exit(self) -> None:
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.is_driver_rank:
                self.shm.unlink()

        if not self.enforce_eager:
            del self.graphs, self.graph_pool

        torch.cuda.synchronize()
        dist.destroy_process_group()

    # ------------------------------------------------------------------
    # Warmup & KV cache
    # ------------------------------------------------------------------

    def _warmup_model(self) -> None:
        torch.cuda.empty_cache()

        max_len = self.config.max_model_len
        max_tokens = self.config.max_num_batched_tokens
        num_seqs = min(max_tokens // max_len, self.config.max_num_seqs)

        seqs = [Sequence([0] * max_len) for _ in range(num_seqs)]
        self.run(seqs, is_prefill=True)

        torch.cuda.empty_cache()

    def _allocate_kv_cache(self) -> None:
        cfg = self.config
        hf = cfg.hf_config

        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        num_kv_heads = hf.num_key_value_heads // self.world_size
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)

        block_bytes = (
            2
            * hf.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf.torch_dtype.itemsize
        )

        cfg.num_kvcache_blocks = int(
            total * cfg.gpu_memory_utilization - used - peak + current
        ) // block_bytes

        assert cfg.num_kvcache_blocks > 0

        self.kv_cache = torch.empty(
            2,
            hf.num_hidden_layers,
            cfg.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )

        layer_id = 0
        for m in self.model.modules():
            if hasattr(m, "k_cache"):
                m.k_cache = self.kv_cache[0, layer_id]
                m.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, seqs: List[Sequence], is_prefill: bool) -> Optional[List[int]]:
        input_ids, positions = (
            self._prepare_prefill(seqs)
            if is_prefill
            else self._prepare_decode(seqs)
        )

        temps = self._prepare_sampling(seqs) if self.is_driver_rank else None
        logits = self._run_model(input_ids, positions, is_prefill)

        token_ids = (
            self.sampler(logits, temps).tolist()
            if self.is_driver_rank
            else None
        )

        reset_context()
        return token_ids

    # ------------------------------------------------------------------
    # CUDA Graph
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _capture_cuda_graphs(self) -> None:
        cfg = self.config
        hf = cfg.hf_config

        max_bs = min(cfg.max_num_seqs, 512)
        max_blocks = (cfg.max_model_len + self.block_size - 1) // self.block_size

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        input_ids = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        positions = torch.zeros_like(input_ids)
        outputs = torch.zeros(max_bs, hf.hidden_size, device="cuda")

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
