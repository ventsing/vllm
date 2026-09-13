# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Weight-sharing ledger for base + adapter (LoRA) deployments.

When several tasks share one base checkpoint and differ only by a lightweight
adapter, an actor migration should load the base once and switch only the
adapter, cutting transfer cost by one to two orders of magnitude (an adapter is
typically <<1% of the base). This module tracks which actors hold which
``(base_hash, adapter_config)`` combination and estimates the migration cost
difference between a full reload and an adapter-only switch. Pure logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WeightShareLease:
    """State for one ``(base_hash, adapter_config)`` combination.

    Attributes:
        base_hash: Hash of the shared base weights.
        adapter_config: Adapter identifier (``None`` = base only). Sized in
            bytes by the caller for cost estimation.
        holders: Actor ids currently holding this exact combination.
        refs: Cumulative acquisition count (hotness).
    """

    base_hash: str
    adapter_config: str | None
    holders: set[str] = field(default_factory=set)
    refs: int = 0


class WeightShareLedger:
    """Tracks weight-combination residency across the actor pool.

    The ledger is pure bookkeeping. The executor consults
    :meth:`switch_cost` before a migration: if the target shares the source's
    base and only the adapter differs, the migration can reload just the
    adapter instead of the whole checkpoint.
    """

    def __init__(self):
        self._by_base: dict[str, dict[str | None, WeightShareLease]] = {}

    def register(
        self, actor_id: str, base_hash: str, adapter_config: str | None = None
    ) -> WeightShareLease:
        """Mark ``actor_id`` as holding the given combination."""
        base = self._by_base.setdefault(base_hash, {})
        lease = base.get(adapter_config)
        if lease is None:
            lease = WeightShareLease(base_hash, adapter_config)
            base[adapter_config] = lease
        lease.holders.add(actor_id)
        lease.refs += 1
        return lease

    def unregister(
        self, actor_id: str, base_hash: str, adapter_config: str | None = None
    ) -> int:
        """Drop ``actor_id`` from a combination; returns remaining holders."""
        base = self._by_base.get(base_hash, {})
        lease = base.get(adapter_config)
        if lease is None:
            return 0
        lease.holders.discard(actor_id)
        if not lease.holders:
            del base[adapter_config]
            if not base:
                del self._by_base[base_hash]
        return len(lease.holders)

    def holders(
        self, base_hash: str, adapter_config: str | None = None
    ) -> set[str]:
        """Actors currently holding a combination (empty if unknown)."""
        lease = self._by_base.get(base_hash, {}).get(adapter_config)
        return set(lease.holders) if lease else set()

    def cost_ratio(self, base_bytes: int, adapter_bytes: int) -> float:
        """Return ``adapter / base`` bytes, the migration-speed multiple.

        A ratio near 0 quantifies the one-to-two order-of-magnitude win of an
        adapter-only switch over reloading the shared base. Callers decide
        *whether* an adapter-only switch applies by comparing the source and
        target ``base_hash`` (matching hash = same base = cheap switch).
        """
        if base_bytes <= 0:
            raise ValueError("base_bytes must be positive")
        return adapter_bytes / base_bytes