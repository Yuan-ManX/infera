from collections import deque
from typing import List, Tuple

from infera.config import InferaConfig
from infera.core.sequence import Sequence, SequenceStatus
from infera.core.block_manager import BlockManager


class Scheduler:
    """
    Scheduler for managing sequences during LLM inference.

    Responsibilities:
    - Prefill: allocate KV cache blocks for new sequences
    - Decode: append new tokens for running sequences
    - Preemption: temporarily suspend sequences if memory/blocks are insufficient
    """

    def __init__(self, config: InferaConfig):
        self.max_num_seqs: int = config.max_num_seqs
        self.max_num_batched_tokens: int = config.max_num_batched_tokens
        self.eos: int = config.eos

        self.block_manager: BlockManager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        """Check if all sequences have finished."""
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        """Add a new sequence to the scheduler."""
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], bool]:
        """
        Select sequences for the next engine step.

        Returns:
            scheduled_seqs: sequences to run in this step
            is_prefill: True if prefill phase, False if decode phase
        """
        scheduled_seqs: List[Sequence] = []

        # -----------------------------
        # 1️⃣ Prefill phase
        # -----------------------------
        num_seqs = 0
        num_batched_tokens = 0

        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            # Cannot fit batch tokens or insufficient KV cache blocks
            if (
                num_batched_tokens + len(seq) - seq.num_cached_tokens
                > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break

            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
            num_seqs += 1

        if scheduled_seqs:
            return scheduled_seqs, True

        # -----------------------------
        # 2️⃣ Decode phase
        # -----------------------------
        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            # Ensure enough free blocks to append; preempt if not
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # Safe to append
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                num_seqs += 1

        # Put scheduled sequences back into running queue (front)
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # -----------------------------
    # Preemption
    # -----------------------------
    def preempt(self, seq: Sequence) -> None:
        """
        Suspend a sequence temporarily and release its KV cache blocks.
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # -----------------------------
    # Postprocessing
    # -----------------------------
    def postprocess(self, seqs: List[Sequence], token_ids: List[int]) -> None:
        """
        Update sequences with newly generated tokens.

        Marks sequences as FINISHED if EOS is reached or max tokens exhausted.
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)

            if (
                (not seq.sampling_params.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens >= seq.sampling_params.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                # Release KV cache
                self.block_manager.deallocate(seq)
                # Safe removal from running
                if seq in self.running:
                    self.running.remove(seq)
