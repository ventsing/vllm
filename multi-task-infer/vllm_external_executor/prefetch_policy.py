# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Access-heat tracking and asynchronous prefetch decisions.

Migration latency is hidden by loading the next hot state before it is
requested. This module decides *what* to prefetch and in *what order*; the
executor carries the loads out in the background while the current model keeps
serving. Pure logic: only timestamps and sizes flow in, a ranked plan flows
out.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class AccessHeatTracker:
    """Sliding-window access heat with exponential decay.

    A key touched recently and often scores higher than one touched long ago;
    decay folds recency into the score without unbounded growth.
    """

    def __init__(self, decay: float = 0.9):
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        self._decay = decay
        self._heat: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def record(self, key: str, now: float) -> None:
        """Fold one access into the heat score."""
        prev = self._heat.get(key, 0.0)
        dt = now - self._last.get(key, now)
        self._heat[key] = prev * (self._decay ** dt) + 1.0
        self._last[key] = now

    def score(self, key: str, now: float) -> float:
        """Return current heat (0.0 when unseen)."""
        if key not in self._heat:
            return 0.0
        dt = now - self._last.get(key, now)
        return self._heat[key] * (self._decay ** dt)


@dataclass
class PrefetchPlan:
    """Ranked prefetch decision.

    Attributes:
        keys: Keys to prefetch, hottest first, clipped to ``budget_bytes``.
        total_bytes: Bytes the plan will load.
        skipped: Candidates excluded (resident or over budget).
    """

    keys: list[str]
    total_bytes: int
    skipped: list[str] = field(default_factory=list)


class PrefetchPolicy:
    """Rank hot, non-resident candidates under a byte budget."""

    def __init__(self, budget_bytes: int):
        if budget_bytes < 0:
            raise ValueError("budget_bytes must be >= 0")
        self._budget = budget_bytes

    def plan(
        self,
        candidates: dict[str, int],
        heat: AccessHeatTracker,
        now: float,
        resident: set[str] | None = None,
    ) -> PrefetchPlan:
        """Select prefetch targets.

        Args:
            candidates: ``key -> size_bytes`` for everything known.
            heat: Heat tracker providing recency-weighted scores.
            now: Monotonic timestamp.
            resident: Keys already present (excluded from the plan).

        Returns:
            A :class:`PrefetchPlan` ordered by descending score.
        """
        resident = resident or set()
        ranked = sorted(
            candidates,
            key=lambda k: heat.score(k, now),
            reverse=True,
        )
        keys: list[str] = []
        skipped: list[str] = []
        total = 0
        for key in ranked:
            if key in resident:
                skipped.append(key)
                continue
            size = candidates[key]
            if total + size > self._budget:
                skipped.append(key)
                continue
            keys.append(key)
            total += size
        return PrefetchPlan(keys=keys, total_bytes=total, skipped=skipped)