# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Incremental KV-block migration planning (pure logic, no Ray/torch dep).

Model hot-switch or cross-node migration currently re-allocates the whole KV
cache, which is prohibitively expensive for large models / long contexts.
This module computes an *incremental* migration plan over KV blocks:

- transfer only blocks that are new or modified;
- reuse blocks whose content hash already exists at the destination
  (prefix-cache hit: common prompt prefixes need not be re-sent).

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
    """Metadata for one KV block participating in a migration.

    Attributes:
        block_id: Stable block id in the source block table.
        content_hash: Content hash used by prefix caching. Equal hashes imply
            identical KV content regardless of block id; empty string means
            the block is not hashable (fall back to version comparison).
        version: Monotonic write version; detects modification when a block
            has no content hash.
        token_count: Number of tokens stored in the block (block_size).
    """

    block_id: int
    content_hash: str = ""
    version: int = 0
    token_count: int = 16


@dataclass
class KVMigrationPlan:
    """Result of an incremental diff.

    Attributes:
        transfer: Blocks whose data must be shipped (new, or modified without
            a reusable hash).
        prefix_hits: ``(src_block_id, dst_block_id)`` pairs reused via prefix
            cache: no data transfer, only a block-table remap is needed.
        unchanged: Source blocks already present at the destination.
    """

    transfer: list[KVBlockRef] = field(default_factory=list)
    prefix_hits: list[tuple[int, int]] = field(default_factory=list)
    unchanged: list[int] = field(default_factory=list)

    @property
    def transferred_count(self) -> int:
        return len(self.transfer)

    @property
    def reused_count(self) -> int:
        return len(self.prefix_hits) + len(self.unchanged)

    def summary(self) -> str:
        """One-line human-readable plan summary."""
        return (
            f"KV plan: transfer={len(self.transfer)} "
            f"prefix_hits={len(self.prefix_hits)} "
            f"unchanged={len(self.unchanged)}"
        )


class IncrementalKVPlanner:
    """Compute a minimal KV-block migration plan.

    Content hash is the primary signal: a source block whose hash exists at
    the destination is a prefix-cache hit and is reused without shipping data.
    For blocks without a hash, ``(block_id, version)`` equality decides
    unchanged vs. transfer.
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
        dst_by_id = {b.block_id: b for b in dst}
        dst_by_hash: dict[str, KVBlockRef] = {
            b.content_hash: b for b in dst if b.content_hash
        }

        plan = KVMigrationPlan()
        for b in src:
            # 1. Same id resident: unchanged if content matches (hash or
            #    version), else modified -> transfer.
            existing = dst_by_id.get(b.block_id)
            if existing is not None:
                if self._same_content(b, existing):
                    plan.unchanged.append(b.block_id)
                else:
                    plan.transfer.append(b)
                continue

            # 2. New block id: reuse via prefix cache when the destination
            #    already holds the same content (any id).
            if b.content_hash and b.content_hash in dst_by_hash:
                plan.prefix_hits.append(
                    (b.block_id, dst_by_hash[b.content_hash].block_id)
                )
                continue

            # 3. Otherwise ship the block data.
            plan.transfer.append(b)

        return plan

    @staticmethod
    def _same_content(a: KVBlockRef, b: KVBlockRef) -> bool:
        """Content equality: hash wins when present, else version equality."""
        if a.content_hash and b.content_hash:
            return a.content_hash == b.content_hash
        return a.version == b.version