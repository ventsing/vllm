# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""Elastic actor-pool autoscaling (decision layer, pure logic).

This module computes *whether* and *how many* actors the pool should add or
remove, given a load snapshot. It deliberately does not touch Ray: the pool
manager (``actor_pool_manager.py``) owns the actual create/kill side effects and
feeds this class a monotonic clock so it stays deterministic and offline-
testable.

The decision combines three signals (queue length, P99 latency, resource
utilization) with per-signal high/low watermarks, a configurable step size,
separate scale-up/scale-down cooldowns to damp oscillation, and optional time
windows that raise the floor/ceiling (or pin a target) for predictable periods
such as nightly batch windows.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum


class AutoscaleAction(Enum):
    """What the pool should do next."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NONE = "none"


@dataclass
class LoadMetrics:
    """One sampling of the pool's externally-visible load.

    Attributes:
        queue_length: Number of requests waiting for a worker lease.
        p99_latency_ms: Tail latency of served requests (ms).
        resource_utilization: Fractional device utilization ``0.0``-``1.0``
            (GPU/NPU busy ratio, averaged across the pool).
    """

    queue_length: int = 0
    p99_latency_ms: float = 0.0
    resource_utilization: float = 0.0


@dataclass
class TimeWindowScale:
    """Floor/ceiling (and optional target) that applies during an hour range.

    Hours are local, ``start_hour`` inclusive and ``end_hour`` exclusive
    (``24`` == midnight). A window that wraps midnight is expressed with
    ``start_hour > end_hour`` (e.g. ``22 -> 6`` covers 22:00-06:00).
    """

    start_hour: int
    end_hour: int
    min_actors: int
    max_actors: int
    target_actors: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.start_hour < 24 or not 0 <= self.end_hour <= 24:
            raise ValueError("time-window hours must be 0-23 (end may be 24)")
        if self.min_actors < 0 or self.max_actors < self.min_actors:
            raise ValueError("window actor bounds are inconsistent")
        if self.target_actors is not None and not (
            self.min_actors <= self.target_actors <= self.max_actors
        ):
            raise ValueError("window target_actors must lie within [min, max]")

    def contains(self, hour: int) -> bool:
        """Return whether ``hour`` (0-23) falls in this window."""
        if self.start_hour < self.end_hour:
            return self.start_hour <= hour < self.end_hour
        return hour >= self.start_hour or hour < self.end_hour


@dataclass
class AutoscalerConfig:
    """Watermarks, step, cooldowns, and time windows for the autoscaler.

    Scale-up fires when *any* signal crosses its high watermark (OR); scale-down
    fires only when *all* three signals are at or below their low watermark
    (AND) — conservative shrink avoids thrash on a single quiet metric.
    """

    min_actors: int = 1
    max_actors: int = 32
    scale_up_queue_length: int = 100
    scale_down_queue_length: int = 10
    scale_up_p99_ms: float = 200.0
    scale_down_p99_ms: float = 50.0
    scale_up_utilization: float = 0.8
    scale_down_utilization: float = 0.3
    scale_step: int = 2
    cooldown_scale_up_seconds: float = 60.0
    cooldown_scale_down_seconds: float = 300.0
    time_windows: list[TimeWindowScale] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.min_actors < 0 or self.max_actors < self.min_actors:
            raise ValueError("min_actors/max_actors are inconsistent")
        if self.scale_step <= 0:
            raise ValueError("scale_step must be positive")
        if self.cooldown_scale_up_seconds < 0 or self.cooldown_scale_down_seconds < 0:
            raise ValueError("cooldowns must be non-negative")
        if not (
            0.0
            <= self.scale_down_utilization
            <= self.scale_up_utilization
            <= 1.0
        ):
            raise ValueError(
                "utilization watermarks must be ordered within [0, 1]"
            )


@dataclass
class AutoscaleDecision:
    """The autoscaler's verdict for one sampling."""

    action: AutoscaleAction
    target_actors: int
    reason: str
    metrics: LoadMetrics | None = None
    time_window: TimeWindowScale | None = None


class Autoscaler:
    """Stateful decision engine behind :meth:`ActorPoolManager.maybe_autoscale`.

    Holds only the last scale-up/scale-down timestamps (cooldown bookkeeping);
    everything else comes in per-call. The caller owns the monotonic ``now``,
    so tests can advance time deterministically.
    """

    def __init__(self, config: AutoscalerConfig) -> None:
        self._config = config
        self._last_scale_up = float("-inf")
        self._last_scale_down = float("-inf")

    @property
    def config(self) -> AutoscalerConfig:
        return self._config

    def decide(
        self,
        metrics: LoadMetrics,
        current_actors: int,
        now: float,
    ) -> AutoscaleDecision:
        """Compute the next scale action for a load sample.

        Args:
            metrics: Load snapshot (queue, P99, utilization).
            current_actors: Current pool size (total, not idle).
            now: Monotonic seconds (``time.monotonic()``).

        Returns:
            The decision; ``target_actors`` is the new desired pool size (may
            equal ``current_actors`` for a no-op).
        """
        window = self._time_window(now)
        lo, hi = self.bounds(now)
        cfg = self._config

        if window is not None and window.target_actors is not None:
            return self._converge_to_target(
                metrics, current_actors, now, lo, hi, window
            )

        over = (
            metrics.queue_length >= cfg.scale_up_queue_length
            or metrics.p99_latency_ms >= cfg.scale_up_p99_ms
            or metrics.resource_utilization >= cfg.scale_up_utilization
        )
        under = (
            metrics.queue_length <= cfg.scale_down_queue_length
            and metrics.p99_latency_ms <= cfg.scale_down_p99_ms
            and metrics.resource_utilization <= cfg.scale_down_utilization
        )

        if over and current_actors < hi:
            if now - self._last_scale_up < cfg.cooldown_scale_up_seconds:
                return self._none(current_actors, "scale-up cooldown", metrics, window)
            target = min(hi, current_actors + cfg.scale_step)
            self._last_scale_up = now
            return AutoscaleDecision(
                AutoscaleAction.SCALE_UP, target,
                "load above high watermark", metrics, window,
            )

        if under and current_actors > lo:
            if now - self._last_scale_down < cfg.cooldown_scale_down_seconds:
                return self._none(
                    current_actors, "scale-down cooldown", metrics, window
                )
            target = max(lo, current_actors - cfg.scale_step)
            self._last_scale_down = now
            return AutoscaleDecision(
                AutoscaleAction.SCALE_DOWN, target,
                "load below low watermark", metrics, window,
            )

        return self._none(current_actors, "within target band", metrics, window)

    def bounds(self, now: float) -> tuple[int, int]:
        """Return the effective ``(min, max)`` for ``now`` (config ∩ window)."""
        cfg = self._config
        lo, hi = cfg.min_actors, cfg.max_actors
        window = self._time_window(now)
        if window is not None:
            lo = max(lo, window.min_actors)
            hi = min(hi, window.max_actors)
        return lo, hi

    def reset(self) -> None:
        """Clear cooldown state (e.g. after a manual pool resize)."""
        self._last_scale_up = float("-inf")
        self._last_scale_down = float("-inf")

    # ------------------------------------------------------------- internals
    def _time_window(self, now: float) -> TimeWindowScale | None:
        hour = datetime.datetime.fromtimestamp(now).hour
        for window in self._config.time_windows:
            if window.contains(hour):
                return window
        return None

    def _converge_to_target(self, metrics, current_actors, now, lo, hi, window):
        """Nudge toward a time-window ``target_actors`` one step at a time."""
        cfg = self._config
        target = max(lo, min(hi, window.target_actors))
        if current_actors < target:
            if now - self._last_scale_up < cfg.cooldown_scale_up_seconds:
                return self._none(
                    current_actors, "scale-up cooldown", metrics, window
                )
            new_target = min(target, current_actors + cfg.scale_step)
            self._last_scale_up = now
            return AutoscaleDecision(
                AutoscaleAction.SCALE_UP, new_target,
                "time-window target not yet met", metrics, window,
            )
        if current_actors > target:
            if now - self._last_scale_down < cfg.cooldown_scale_down_seconds:
                return self._none(
                    current_actors, "scale-down cooldown", metrics, window
                )
            new_target = max(target, current_actors - cfg.scale_step)
            self._last_scale_down = now
            return AutoscaleDecision(
                AutoscaleAction.SCALE_DOWN, new_target,
                "time-window target exceeded", metrics, window,
            )
        return self._none(current_actors, "at time-window target", metrics, window)

    @staticmethod
    def _none(current_actors, reason, metrics, window):
        return AutoscaleDecision(
            AutoscaleAction.NONE, current_actors, reason, metrics, window
        )
