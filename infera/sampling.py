from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingParams:
    """
    Sampling configuration for Infera decoding.

    This class defines how tokens are sampled during inference.
    It is intentionally decoupled from model and runtime configs.
    """

    # =========================
    # Core Sampling
    # =========================
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    # =========================
    # Advanced / Reserved
    # =========================
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    def __post_init__(self) -> None:
        self._validate_sampling_params()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_sampling_params(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if self.temperature <= 0.0:
            raise ValueError(
                "temperature must be > 0.0; "
                "greedy decoding should be handled explicitly"
            )

        if self.top_p is not None:
            if not (0.0 < self.top_p <= 1.0):
                raise ValueError("top_p must be in (0, 1]")

        if self.top_k is not None:
            if self.top_k <= 0:
                raise ValueError("top_k must be positive")
