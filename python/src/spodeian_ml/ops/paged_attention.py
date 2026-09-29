"""
PagedAttention KV-Cache Manager for spodeian-ml.
Implements vLLM-style dynamic virtual memory block tables to eliminate VRAM fragmentation
and maximize batch throughput for continuous sequential alpha models.
"""

from typing import List, Dict
import math
import torch


class PagedKVCacheManager:
    """
    Manages physical KV memory as non-contiguous fixed-size blocks mapped through
    per-sequence virtual block tables.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Preallocated contiguous physical memory pool: (num_blocks, block_size, num_heads, head_dim)
        self.key_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
        self.val_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)

        # Free block indices
        self.free_blocks: List[int] = list(range(num_blocks))
        # Logical sequence block tables: seq_id -> list of physical block IDs
        self.block_tables: Dict[int, List[int]] = {}
        # Track sequence lengths: seq_id -> num_tokens
        self.seq_lengths: Dict[int, int] = {}

    def allocate_sequence(self, seq_id: int, initial_tokens: int = 1) -> bool:
        """Allocates physical blocks for a new sequence."""
        needed_blocks = max(1, math.ceil(initial_tokens / self.block_size))
        if len(self.free_blocks) < needed_blocks:
            return False

        assigned = []
        for _ in range(needed_blocks):
            assigned.append(self.free_blocks.pop(0))

        self.block_tables[seq_id] = assigned
        self.seq_lengths[seq_id] = initial_tokens
        return True

    def append_token(self, seq_id: int, key: torch.Tensor, value: torch.Tensor) -> bool:
        """
        Appends a key and value vector to the logical sequence cache.
        Automatically allocates new physical blocks when the current block fills up.
        """
        if seq_id not in self.block_tables:
            if not self.allocate_sequence(seq_id, 0):
                return False

        seq_len = self.seq_lengths[seq_id]
        block_idx_in_seq = seq_len // self.block_size
        offset_in_block = seq_len % self.block_size

        # Check if we need to allocate a new physical block
        if block_idx_in_seq >= len(self.block_tables[seq_id]):
            if not self.free_blocks:
                return False  # VRAM Cache Full
            new_block = self.free_blocks.pop(0)
            self.block_tables[seq_id].append(new_block)

        phys_block = self.block_tables[seq_id][block_idx_in_seq]

        # Write key and value into preallocated physical cache
        self.key_cache[phys_block, offset_in_block] = key.to(self.dtype)
        self.val_cache[phys_block, offset_in_block] = value.to(self.dtype)

        self.seq_lengths[seq_id] += 1
        return True

    def free_sequence(self, seq_id: int) -> None:
        """Frees all blocks allocated to a sequence back to the free pool."""
        if seq_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[seq_id])
            del self.block_tables[seq_id]
            del self.seq_lengths[seq_id]

    @property
    def memory_utilization(self) -> float:
        """Returns fraction of allocated physical blocks."""
        used = self.num_blocks - len(self.free_blocks)
        return used / max(1, self.num_blocks)
