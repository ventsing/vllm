# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""Offline tests for the elastic-autoscaling decision layer.

``autoscaling.py`` is pure logic (no torch/ray), so it is loadable via
``importlib`` in an environment without vLLM installed.
"""

import datetime
import importlib.util
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "vllm_external_executor"


def _load():
    spec = importlib.util.spec_from_file_location(
        "vllm_external_executor.autoscaling", PKG / "autoscaling.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


asc = _load()
LoadMetrics = asc.LoadMetrics
Autoscaler = asc.Autoscaler
AutoscalerConfig = asc.AutoscalerConfig
AutoscaleAction = asc.AutoscaleAction
TimeWindowScale = asc.TimeWindowScale


def ts(hour: int) -> float:
    """Local-timestamp seconds for ``hour`` on a fixed date (round-trips)."""
    return datetime.datetime(2024, 1, 15, hour, 0, 0).timestamp()


def cfg(**kwargs) -> AutoscalerConfig:
    defaults = dict(
        min_actors=1,
        max_actors=32,
        scale_up_queue_length=100,
        scale_down_queue_length=10,
        scale_up_p99_ms=200.0,
        scale_down_p99_ms=50.0,
        scale_up_utilization=0.8,
        scale_down_utilization=0.3,
        scale_step=2,
        cooldown_scale_up_seconds=60.0,
        cooldown_scale_down_seconds=300.0,
    )
    defaults.update(kwargs)
    return AutoscalerConfig(**defaults)


def idle_metrics() -> LoadMetrics:
    return LoadMetrics(
        queue_length=5, p99_latency_ms=20.0, resource_utilization=0.1
    )


def test_scale_up_on_queue_watermark():
    autoscaler = Autoscaler(cfg())
    decision = autoscaler.decide(
        LoadMetrics(queue_length=150), current_actors=4, now=ts(10)
    )
    assert decision.action == AutoscaleAction.SCALE_UP
    assert decision.target_actors == 6  # 4 + scale_step


def test_scale_up_on_utilization_only():
    autoscaler = Autoscaler(cfg())
    decision = autoscaler.decide(
        LoadMetrics(resource_utilization=0.9), current_actors=4, now=ts(10)
    )
    assert decision.action == AutoscaleAction.SCALE_UP


def test_scale_down_requires_all_signals_low():
    autoscaler = Autoscaler(cfg())
    # Queue alone is low; latency/utilization are mid-band: no shrink.
    decision = autoscaler.decide(
        LoadMetrics(
            queue_length=5, p99_latency_ms=100.0, resource_utilization=0.5
        ),
        current_actors=8, now=ts(10),
    )
    assert decision.action == AutoscaleAction.NONE
    # All three low: shrink by one step.
    decision = autoscaler.decide(idle_metrics(), current_actors=8, now=ts(10))
    assert decision.action == AutoscaleAction.SCALE_DOWN
    assert decision.target_actors == 6


def test_cooldown_blocks_repeated_scaling():
    autoscaler = Autoscaler(cfg())
    high = LoadMetrics(queue_length=150)
    first = autoscaler.decide(high, current_actors=4, now=ts(10))
    assert first.action == AutoscaleAction.SCALE_UP

    # Immediately re-sampled: cooldown suppresses another scale-up.
    second = autoscaler.decide(high, current_actors=6, now=ts(10))
    assert second.action == AutoscaleAction.NONE
    assert "cooldown" in second.reason

    # After the cooldown window elapses, scaling resumes.
    third = autoscaler.decide(high, current_actors=6, now=ts(10) + 61.0)
    assert third.action == AutoscaleAction.SCALE_UP


def test_time_window_target_converges_stepwise():
    window = TimeWindowScale(22, 6, min_actors=2, max_actors=16, target_actors=8)
    autoscaler = Autoscaler(cfg(time_windows=[window]))

    d1 = autoscaler.decide(idle_metrics(), current_actors=4, now=ts(22))
    assert d1.action == AutoscaleAction.SCALE_UP
    assert d1.target_actors == 6

    d2 = autoscaler.decide(idle_metrics(), current_actors=6, now=ts(22) + 61.0)
    assert d2.action == AutoscaleAction.SCALE_UP
    assert d2.target_actors == 8

    d3 = autoscaler.decide(idle_metrics(), current_actors=8, now=ts(22) + 122.0)
    assert d3.action == AutoscaleAction.NONE


def test_bounds_clamp_to_config_limits():
    autoscaler = Autoscaler(cfg(min_actors=1, max_actors=10))
    up = autoscaler.decide(
        LoadMetrics(queue_length=150), current_actors=9, now=ts(10)
    )
    assert up.target_actors == 10  # clamped, not 11

    down = autoscaler.decide(
        idle_metrics(), current_actors=2, now=ts(10) + 1000.0
    )
    assert down.target_actors == 1  # clamped, not 0


def test_wrapping_time_window_bounds():
    window = TimeWindowScale(22, 6, min_actors=5, max_actors=20)
    autoscaler = Autoscaler(cfg(time_windows=[window]))

    lo, hi = autoscaler.bounds(ts(23))
    assert (lo, hi) == (5, 20)  # inside the wrap-around window

    lo, hi = autoscaler.bounds(ts(12))
    assert (lo, hi) == (1, 32)  # outside: config defaults


def test_config_validation_rejects_inconsistent_bounds():
    try:
        AutoscalerConfig(min_actors=10, max_actors=5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for min > max")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
