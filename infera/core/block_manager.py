from collections import deque
from typing import List, Dict, Deque, Set

import xxhash
import numpy as np

from infera.core.sequence import Sequence


class Block:
    """
    KV cache block for a fixed token span.
    """

    __slots__ = ("block_id", "ref_count", "hash", "token_ids")

    def __init__(self, block_id: int):
        self.block_id: int = block_id
        self.ref_count: int = 0
        self.hash: int = -1
        self.token_ids: List[int] = []

    def reset(self) -> None:
        """Reset block state when allocating."""
        self.ref_count = 1
        self.hash = -1
        self.token_ids.clear()

    def update(self, block_hash: int, token_ids: List[int]) -> None:
        """Set block content and hash."""
        self.hash = block_hash
        self.token_ids = token_ids


class BlockManager:
    """
    KV cache block manager with prefix-aware hashing.

    Design aligned with Infera:
    - Deterministic prefix reuse
    - Explicit cache hit/miss
    - Safe ref_count management
    - Decode-time append support
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size: int = block_size
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = {}

        self.free_block_ids: Deque[int] = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()

    # --------------------
    # Hashing
    # --------------------
    @staticmethod
    def compute_hash(token_ids: List[int], prefix_hash: int = -1) -> int:
        """
        Compute deterministic rolling hash for prefix caching.
        """
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little"))
        h.update(np.asarray(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    # --------------------
    # Allocation primitives
    # --------------------
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Cannot allocate a live block"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Cannot free a referenced block"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # --------------------
    # Public API
    # --------------------
    def can_allocate(self, seq: Sequence) -> bool:
        """Check if enough free blocks exist for sequence allocation."""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        """
        Allocate blocks for a sequence with prefix-aware cache reuse.
        """
        assert not seq.block_table, "Sequence already allocated"

        prefix_hash = -1
        cache_miss = False

        for block_idx in range(seq.num_blocks):
            token_ids = seq.block(block_idx)
            is_full_block = len(token_ids) == self.block_size
            block_hash = (
                self.compute_hash(token_ids, prefix_hash) if is_full_block else -1
            )

            block_id = self.hash_to_block_id.get(block_hash, -1)
            reuse = (
                block_id != -1
                and not cache_miss
                and self.blocks[block_id].token_ids == token_ids
            )

            if reuse:
                seq.num_cached_tokens += self.block_size
                block = self.blocks[block_id]
                if block_id in self.used_block_ids:
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            else:
                cache_miss = True
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)

            if block_hash != -1:
                block.update(block_hash, token_ids)
                self.hash_to_block_id[block_hash] = block_id
                prefix_hash = block_hash
            else:
                prefix_hash = -1

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence) -> None:
        """Deallocate all blocks used by a sequence."""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.block_table.clear()
        seq.num_cached_tokens = 0

    # --------------------
    # Decode-time append
    # --------------------
    def can_append(self, seq: Sequence) -> bool:
        """Check if appending a token may require a new block."""
        need_new_block = (len(seq) % self.block_size) == 1
        return not need_new_block or len(self.free_block_ids) > 0

    def may_append(self, seq: Sequence) -> None:
        """
        Handle KV cache updates during token-by-token decoding.

        Three cases:
        1. Start a new block
        2. Finish a block
        3. Middle of a block
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        offset = len(seq) % self.block_size

        if offset == 1:
            # Starting new block
            assert last_block.hash != -1, "Previous block must be sealed"
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif offset == 0:
            # Finish current block
            assert last_block.hash == -1, "Block already sealed"
            token_ids = seq.block(seq.num_blocks - 1)
            prefix_hash = (
                self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            )
            h = self.compute_hash(token_ids, prefix_hash)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # Middle of block
            assert last_block.hash == -1
