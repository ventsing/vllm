# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Incremental KV-block migration planning (pure logic, no Ray/torch dep).

Model hot-switch or cross-node migration currently re-allocates the whole KV
cache, which is prohibitively expensive for large models / long contexts.
This module computes an *incremental* migration plan over KV blocks:

- transfer only blocks that are new or modified;
- reuse blocks whose content hash already exists at the destination
  (prefix-cache hit: common prompt prefixes need not be re-sent);
- map source physical block ids onto destination ids when the two pools have
  different sizes (heterogeneous migration);
- key prefix-cache reuse by KV cache group so that blocks from different
  groups (e.g. encoder vs. decoder, full vs. sliding-window) never collide.

The planner is deliberately metadata-only. The *source of truth* for which
blocks are live / dirty / prefix-cached is the vLLM KV cache manager living in
EngineCore; the executor only ships block *data* (see
``ExternalWorkerActor.export_kv_blocks`` / ``import_kv_blocks``). Keeping the
diff algorithm here makes it unit-testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class KVBlockRef:
    """Metadata for one KV block (in one KV cache group) in a migration.

    Attributes:
        block_id: Stable physical block id in the source block table.
        content_hash: Content hash of *this group's* KV for the block
            (prefix-cache key, hex). Equal hashes in the same group imply
            identical KV content regardless of block id; empty string means
            the block is not hashable (fall back to version comparison).
        version: Monotonic write version; detects modification when a block
            has no content hash.
        token_count: Number of tokens stored in the block (block_size).
        group_id: KV cache group this ref belongs to. Prefix-cache reuse is
            scoped to a group: blocks from different groups never match.
    """

    block_id: int
    content_hash: str = ""
    version: int = 0
    token_count: int = 16
    group_id: int = 0


@dataclass
class KVMigrationPlan:
    """Result of an incremental diff.

    Attributes:
        transfer: ``(block_id, group_id)`` refs whose data must be shipped
            (new, or modified without a reusable hash). Data plane ships these
            by unique ``block_id``.
        prefix_hits: ``(src_block_id, dst_block_id, group_id)`` tuples reused
            via prefix cache: no data transfer, only a block-table remap.
        unchanged: ``(block_id, group_id)`` already present at destination.
        block_mapping: ``src_block_id -> dst_block_id`` for transferred blocks
            (populated by :meth:`IncrementalKVPlanner.assign_targets`).
    """

    transfer: list[KVBlockRef] = field(default_factory=list)
    prefix_hits: list[tuple[int, int, int]] = field(default_factory=list)
    unchanged: list[tuple[int, int]] = field(default_factory=list)
    block_mapping: dict[int, int] = field(default_factory=dict)

    @property
    def transferred_count(self) -> int:
        return len({b.block_id for b in self.transfer})

    @property
    def reused_count(self) -> int:
        return len(self.prefix_hits) + len(self.unchanged)

    def summary(self) -> str:
        """One-line human-readable plan summary."""
        return (
            f"KV plan: transfer={self.transferred_count} "
            f"prefix_hits={len(self.prefix_hits)} "
            f"unchanged={len(self.unchanged)}"
        )


class IncrementalKVPlanner:
    """Compute a minimal, group-aware KV-block migration plan.

    Content hash is the primary signal: a source block whose hash exists at
    the destination (same group) is a prefix-cache hit and is reused without
    shipping data. For blocks without a hash, ``(block_id, version)`` equality
    decides unchanged vs. transfer.
    """

    def plan(
        self,
        src: Iterable[KVBlockRef],
        dst: Iterable[KVBlockRef],
    ) -> KVMigrationPlan:
        """Diff source blocks against destination blocks.

        Args:
            src: Source KV block metadata (live blocks to migrate).
            dst: Destination KV block metadata (already-resident blocks).

        Returns:
            A :class:`KVMigrationPlan` with the minimal transfer set.
        """
        dst_by_id = {(b.group_id, b.block_id): b for b in dst}
        dst_by_hash: dict[tuple[int, str], KVBlockRef] = {
            (b.group_id, b.content_hash): b
            for b in dst
            if b.content_hash
        }

        plan = KVMigrationPlan()
        for b in src:
            key = (b.group_id, b.block_id)
            # 1. Same id resident: unchanged if content matches (hash or
            #    version), else modified -> transfer.
            existing = dst_by_id.get(key)
            if existing is not None:
                if self._same_content(b, existing):
                    plan.unchanged.append((b.block_id, b.group_id))
                else:
                    plan.transfer.append(b)
                continue

            # 2. New block id: reuse via prefix cache when the destination
            #    already holds the same content (any id, same group).
            hash_key = (b.group_id, b.content_hash)
            if b.content_hash and hash_key in dst_by_hash:
                plan.prefix_hits.append(
                    (b.block_id, dst_by_hash[hash_key].block_id, b.group_id)
                )
                continue

            # 3. Otherwise ship the block data.
            plan.transfer.append(b)

        return plan

    def assign_targets(
        self,
        plan: KVMigrationPlan,
        total_blocks: int,
        dst_occupied: Iterable[int] = (),
    ) -> KVMigrationPlan:
        """Assign destination physical ids to transferred blocks.

        Heterogeneous migration: source and destination pools may differ in
        size, so a transferred block cannot always keep its source id. This
        fills ``plan.block_mapping`` (``src_block_id -> dst_block_id``) with
        free destination ids, in ascending order.

        Args:
            plan: Plan produced by :meth:`plan`.
            total_blocks: Destination pool size.
            dst_occupied: Destination block ids already in use (resident).

        Returns:
            ``plan`` with ``block_mapping`` populated.
        """
        allocator = BlockIdAllocator(total_blocks, set(dst_occupied))
        unique_src_ids = sorted({b.block_id for b in plan.transfer})
        plan.block_mapping = dict(
            zip(unique_src_ids, allocator.allocate(len(unique_src_ids)))
        )
        return plan

    @staticmethod
    def _same_content(a: KVBlockRef, b: KVBlockRef) -> bool:
        """Content equality within a group: hash wins, else version."""
        if a.content_hash and b.content_hash:
            return a.content_hash == b.content_hash
        return a.version == b.version


@dataclass
class BlockIdAllocator:
    """Assign free destination block ids (pure logic).

    Attributes:
        total_blocks: Destination pool size.
        occupied: Destination block ids already in use.
    """

    total_blocks: int
    occupied: set[int] = field(default_factory=set)

    def allocate(self, num: int) -> list[int]:
        """Return ``num`` free block ids, in ascending order.

        Raises:
            ValueError: If fewer than ``num`` blocks are free.
        """
        free = [i for i in range(self.total_blocks) if i not in self.occupied]
        if len(free) < num:
            raise ValueError(
                f"Not enough free destination blocks: need {num}, have "
                f"{len(free)}"
            )
        allocated = free[:num]
        self.occupied.update(allocated)
        return allocated