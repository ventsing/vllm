# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Three-level storage tiering and swap-in/swap-out decisions.

The production storage base is a hierarchy, not a single backend:

- ``HBM``: Actor-local GPU memory (hot data).
- ``DRAM``: node-local host memory (warm data).
- ``REMOTE``: NVMe / distributed storage (cold data, cross-node).

This module is pure logic: it tracks entries, enforces per-tier capacity, and
emits :class:`EvictionDecision` records when data must move between tiers. The
record consumer (a real backend or executor) performs the actual tensor move;
the decisions themselves are deterministic and unit-testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StorageTier(str, Enum):
    """Storage tiers ordered from hot to cold."""

    HBM = "hbm"
    DRAM = "dram"
    REMOTE = "remote"

    def colder(self) -> "StorageTier | None":
        """Return the next colder tier, or ``None`` past REMOTE."""
        order = (StorageTier.HBM, StorageTier.DRAM, StorageTier.REMOTE)
        idx = order.index(self)
        return order[idx + 1] if idx + 1 < len(order) else None


@dataclass
class TierEntry:
    """Bookkeeping for one cached object across tiers.

    Attributes:
        key: Stable object key (e.g. block hash, checkpoint path).
        size_bytes: Footprint; drives capacity accounting.
        tier: Where the object currently resides.
        last_access: Monotonic timestamp of last touch (LRU ordering).
        access_count: Cumulative touches (hotness signal for promotion).
    """

    key: str
    size_bytes: int
    tier: StorageTier
    last_access: float
    access_count: int = 0


@dataclass
class EvictionDecision:
    """A tier move the caller must carry out.

    Attributes:
        key: Object to move.
        from_tier: Current tier.
        to_tier: Target tier (colder on eviction, warmer on promotion).
            ``None`` means drop entirely (evicted past the coldest tier).
        reason: Why the move was triggered (``"eviction"`` / ``"promotion"``).
    """

    key: str
    from_tier: StorageTier
    to_tier: StorageTier | None
    reason: str = "eviction"


@dataclass
class _TierState:
    capacity: int
    usage: int = 0
    entries: dict[str, TierEntry] = field(default_factory=dict)


class TieredCache:
    """Capacity accounting + tiering decisions for the three-tier hierarchy.

    Purely in-memory: it never moves tensors, it only decides what should move
    and when. Eviction is LRU within the over-capacity tier, demoting victims
    to the colder tier (or dropping them past REMOTE). Promotion moves an entry
    to a warmer tier, evicting from that tier if it overflows.
    """

    def __init__(self, capacities: dict[StorageTier, int]):
        """Args:
            capacities: Per-tier capacity in bytes. A missing tier means
                unbounded (``inf``) for that tier.
        """
        self._tiers: dict[StorageTier, _TierState] = {
            tier: _TierState(capacity=capacities.get(tier, float("inf")))
            for tier in StorageTier
        }
        self._by_key: dict[str, StorageTier] = {}

    # ------------------------------------------------------------- accounting
    def usage(self, tier: StorageTier) -> int:
        return self._tiers[tier].usage

    def location(self, key: str) -> StorageTier | None:
        return self._by_key.get(key)

    def resident(self) -> set[str]:
        return set(self._by_key)

    # ---------------------------------------------------------------- access
    def touch(self, key: str, now: float) -> None:
        """Mark an access, refreshing LRU position and hotness."""
        tier = self._by_key.get(key)
        if tier is None:
            return
        entry = self._tiers[tier].entries[key]
        entry.last_access = now
        entry.access_count += 1

    def put(
        self, key: str, size_bytes: int, tier: StorageTier, now: float
    ) -> list[EvictionDecision]:
        """Admit an object into ``tier``; may cascade evictions downward.

        Returns the eviction decisions to execute (``REMOTE`` => dropped,
        i.e. not cached) plus the admission itself is reflected immediately.
        """
        if key in self._by_key:
            self.touch(key, now)
            return []

        entry = TierEntry(key, size_bytes, tier, now)
        self._by_key[key] = tier
        self._tiers[tier].entries[key] = entry
        self._tiers[tier].usage += size_bytes

        return self._evict_if_needed(tier, now)

    def promote(
        self, key: str, to_tier: StorageTier, now: float
    ) -> list[EvictionDecision]:
        """Move an entry to a warmer tier, evicting the target if it fills."""
        return self._move(key, to_tier, now, reason="promotion")

    def demote(
        self, key: str, to_tier: StorageTier, now: float
    ) -> list[EvictionDecision]:
        """Move an entry to a colder tier, evicting the target if it fills."""
        return self._move(key, to_tier, now, reason="eviction")

    def _move(
        self, key: str, to_tier: StorageTier, now: float, reason: str
    ) -> list[EvictionDecision]:
        from_tier = self._by_key.get(key)
        if from_tier is None or from_tier == to_tier:
            return []
        entry = self._tiers[from_tier].entries.pop(key)
        entry.tier = to_tier
        entry.last_access = now
        self._tiers[from_tier].usage -= entry.size_bytes
        self._by_key[key] = to_tier
        self._tiers[to_tier].entries[key] = entry
        self._tiers[to_tier].usage += entry.size_bytes

        decisions = self._evict_if_needed(to_tier, now)
        decisions.append(
            EvictionDecision(key, from_tier, to_tier, reason=reason)
        )
        return decisions

    # --------------------------------------------------------------- eviction
    def _evict_if_needed(
        self, tier: StorageTier, now: float
    ) -> list[EvictionDecision]:
        decisions: list[EvictionDecision] = []
        state = self._tiers[tier]
        while state.usage > state.capacity and state.entries:
            victim = min(
                state.entries.values(),
                key=lambda e: (e.last_access, e.access_count),
            )
            self._remove(victim.key, tier)
            colder = tier.colder()
            if colder is None:
                decisions.append(
                    EvictionDecision(victim.key, tier, None, reason="eviction")
                )
                continue
            decisions.append(
                EvictionDecision(victim.key, tier, colder, reason="eviction")
            )
            self.put(victim.key, victim.size_bytes, colder, now)
        return decisions

    def remove(self, key: str) -> int:
        """Drop an entry wherever it resides; returns its size (0 if absent)."""
        tier = self._by_key.get(key)
        if tier is None:
            return 0
        entry = self._tiers[tier].entries.pop(key)
        self._tiers[tier].usage -= entry.size_bytes
        del self._by_key[key]
        return entry.size_bytes

    def _remove(self, key: str, tier: StorageTier) -> None:
        entry = self._tiers[tier].entries.pop(key)
        self._tiers[tier].usage -= entry.size_bytes
        del self._by_key[key]