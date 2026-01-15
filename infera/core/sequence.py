from enum import Enum, auto
from itertools import count
from typing import List

from infera.sampling import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    Runtime representation of a single inference sequence in Infera.

    A Sequence encapsulates:
      - Prompt tokens (immutable)
      - Sampling strategy (immutable)
      - Runtime decoding state (mutable)
    """

    _id_counter = count()

    def __init__(
        self,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        block_size: int,
    ):
        # -----------------------------
        # Identity & Status
        # -----------------------------
        self.seq_id: int = next(Sequence._id_counter)
        self.status: SequenceStatus = SequenceStatus.WAITING

        # -----------------------------
        # Immutable request data
        # -----------------------------
        self.prompt_token_ids: List[int] = list(prompt_token_ids)
        self.sampling_params: SamplingParams = sampling_params

        # -----------------------------
        # Runtime mutable state
        # -----------------------------
        self.token_ids: List[int] = list(prompt_token_ids)
        self.num_prompt_tokens: int = len(prompt_token_ids)
        self.num_tokens: int = self.num_prompt_tokens

        self.last_token_id: int = prompt_token_ids[-1]

        # KV cache bookkeeping
        self.block_size: int = block_size
        self.num_cached_tokens: int = 0
        self.block_table: List[int] = []

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_tokens

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self) -> List[int]:
        return self.token_ids[self.num_prompt_tokens :]

    # ------------------------------------------------------------------
    # KV Cache / Block helpers
    # ------------------------------------------------------------------

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def get_block_tokens(self, block_idx: int) -> List[int]:
        if not (0 <= block_idx < self.num_blocks):
            raise IndexError("block index out of range")

        start = block_idx * self.block_size
        end = start + self.block_size
        return self.token_ids[start:end]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token_id = token_id
        self.num_tokens += 1

        # Stop condition handled at sequence level
        if (
            not self.sampling_params.ignore_eos
            and token_id == self.sampling_params.eos_token_id
        ):
            self.status = SequenceStatus.FINISHED

        if self.num_completion_tokens >= self.sampling_params.max_tokens:
            self.status = SequenceStatus.FINISHED

    # ------------------------------------------------------------------
    # IPC / Runner view
    # ------------------------------------------------------------------

    def to_runner_state(self):
        """
        Minimal state needed by ModelRunner.

        Avoids shipping full token buffers when possible.
        """
        if self.num_completion_tokens == 0:
            return {
                "prompt": self.prompt_token_ids,
                "num_cached_tokens": self.num_cached_tokens,
                "block_table": self.block_table,
            }
        else:
            return {
                "last_token_id": self.last_token_id,
                "num_cached_tokens": self.num_cached_tokens,
                "block_table": self.block_table,
            }
