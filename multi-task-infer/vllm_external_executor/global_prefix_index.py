# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Global prefix-cache index shared across actors.

The in-process vLLM prefix cache only serves a single block pool. A pooled
deployment needs a cluster-wide index that answers "which actor already holds
the KV for this prefix, so a new engine can reuse it instead of recomputing".

Correctness note: KV tensors are weight-dependent, so a prefix is reusable only
when the producer and consumer share the **same weight configuration**
(``weight_hash`` = base + adapter). "Cross-model" sharing is therefore valid
only across models with identical weights (e.g. two engines serving the same
checkpoint); the index keys on ``weight_hash`` to make that boundary explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrefixEntry:
    """One actor's resident copy of a cached prefix block.

    Attributes:
        content_hash: Deterministic prefix content hash (hex).
        group_id: KV cache group the block belongs to.
        weight_hash: Weight-configuration hash; only entries with the same
            weight_hash may share KV tensors.
        actor_id: Pool actor holding the block.
        node_id: Host node (fault domain) of that actor.
        block_id: Physical block id inside the actor's pool.
        refs: Reference count / hotness signal (reuse preference).
    """

    content_hash: str
    group_id: int
    weight_hash: str
    actor_id: str
    node_id: str
    block_id: int
    refs: int = 0


class GlobalPrefixIndex:
    """Cluster-wide ``(weight_hash, content_hash, group_id) -> entries`` map.

    Pure logic: it indexes metadata only. Producers register on
    ``import_prefix_cache``; the migration planner and the prefetch policy
    query it to find reuse sources. Entries are dropped per-actor on failure.
    """

    def __init__(self):
        self._index: dict[tuple[str, str, int], list[PrefixEntry]] = {}

    def register(self, entry: PrefixEntry) -> None:
        key = (entry.weight_hash, entry.content_hash, entry.group_id)
        self._index.setdefault(key, []).append(entry)

    def unregister(self, entry: PrefixEntry) -> int:
        key = (entry.weight_hash, entry.content_hash, entry.group_id)
        bucket = self._index.get(key)
        if not bucket:
            return 0
        before = len(bucket)
        bucket[:] = [e for e in bucket if e.actor_id != entry.actor_id]
        if not bucket:
            del self._index[key]
        return before - len(bucket)

    def lookup(
        self,
        content_hash: str,
        group_id: int,
        weight_hash: str,
        exclude_actors: tuple[str, ...] = (),
    ) -> list[PrefixEntry]:
        """Return reuse sources, most-referenced first, excluding some actors.

        ``exclude_actors`` keeps a migrating engine from "reusing" its own
        blocks (which would be a no-op self-reference).
        """
        bucket = self._index.get((weight_hash, content_hash, group_id), ())
        hits = [e for e in bucket if e.actor_id not in exclude_actors]
        return sorted(hits, key=lambda e: (-e.refs, e.actor_id))

    def drop_actor(self, actor_id: str) -> int:
        """Remove every entry owned by ``actor_id`` (actor failure/rebuild).

        Returns the number of entries removed.
        """
        removed = 0
        for key in list(self._index):
            bucket = self._index[key]
            kept = [e for e in bucket if e.actor_id != actor_id]
            removed += len(bucket) - len(kept)
            if kept:
                self._index[key] = kept
            else:
                del self._index[key]
        return removed

    def stats(self) -> dict:
        """Return per-weight-hash entry counts."""
        counts: dict[str, int] = {}
        for (weight_hash, _hash, _g), bucket in self._index.items():
            counts[weight_hash] = counts.get(weight_hash, 0) + len(bucket)
        return counts