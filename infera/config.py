import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoConfig


@dataclass
class InferaConfig:
    """
    Global configuration for Infera inference engine.

    This config defines model properties, runtime constraints,
    scheduling limits, and memory management parameters.
    """

    # =========================
    # Model Configuration
    # =========================
    model_path: str
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    enforce_eager: bool = False

    # =========================
    # Scheduling Configuration
    # =========================
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512

    # =========================
    # Memory & KV Cache
    # =========================
    gpu_memory_utilization: float = 0.90
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1  # -1 means auto-compute

    # =========================
    # Derived / Internal
    # =========================
    hf_config: Optional[AutoConfig] = field(init=False)
    eos_token_id: int = field(init=False)

    def __post_init__(self) -> None:
        self._validate_model_path()
        self._load_hf_config()
        self._validate_parallelism()
        self._validate_lengths()
        self._validate_memory()
        self._setup_eos_token()

    # ---------------------------------------------------------------------
    # Validation & Initialization
    # ---------------------------------------------------------------------

    def _validate_model_path(self) -> None:
        if not os.path.isdir(self.model_path):
            raise ValueError(f"Invalid model_path: {self.model_path}")

    def _load_hf_config(self) -> None:
        self.hf_config = AutoConfig.from_pretrained(self.model_path)

        # Clamp model length to actual model capability
        if hasattr(self.hf_config, "max_position_embeddings"):
            self.max_model_len = min(
                self.max_model_len,
                self.hf_config.max_position_embeddings,
            )

    def _validate_parallelism(self) -> None:
        if not (1 <= self.tensor_parallel_size <= 8):
            raise ValueError(
                "tensor_parallel_size must be in range [1, 8]"
            )

    def _validate_lengths(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                "max_num_batched_tokens must be >= max_model_len"
            )

        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")

    def _validate_memory(self) -> None:
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError(
                "gpu_memory_utilization must be in (0, 1]"
            )

        if self.kvcache_block_size % 256 != 0:
            raise ValueError(
                "kvcache_block_size must be a multiple of 256"
            )

    def _setup_eos_token(self) -> None:
        # Prefer model-defined EOS
        self.eos_token_id = (
            self.hf_config.eos_token_id
            if self.hf_config.eos_token_id is not None
            else -1
        )
