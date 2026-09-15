# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""Unit tests for prefetch_policy (heat tracker + budgeted planner)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_EXEC_DIR = Path(__file__).resolve().parent.parent / "vllm_external_executor"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pp = _load_module(
    "vllm_external_executor.prefetch_policy",
    _EXEC_DIR / "prefetch_policy.py",
)
AccessHeatTracker = pp.AccessHeatTracker
PrefetchPolicy = pp.PrefetchPolicy


def test_unseen_key_scores_zero():
    heat = AccessHeatTracker()

    assert heat.score("k", now=1.0) == 0.0


def test_record_then_score_positive_and_decays():
    heat = AccessHeatTracker(decay=0.5)
    heat.record("k", now=0.0)

    assert heat.score("k", now=0.0) == 1.0
    # One time unit later with decay 0.5 -> 1.0 * (0.5 ** 1) = 0.5.
    assert heat.score("k", now=1.0) == pytest.approx(0.5)


def test_decay_out_of_bounds_rejected():
    with pytest.raises(ValueError):
        AccessHeatTracker(decay=1.5)
    with pytest.raises(ValueError):
        AccessHeatTracker(decay=-0.1)


def test_plan_ranks_hot_first_and_clips_budget():
    heat = AccessHeatTracker()
    heat.record("hot", now=0.0)
    heat.record("hot", now=1.0)
    heat.record("cold", now=0.0)

    plan = PrefetchPolicy(budget_bytes=10).plan(
        candidates={"hot": 6, "cold": 5},
        heat=heat,
        now=1.0,
    )

    # "hot" scores higher and fits; "cold" would exceed the remaining budget.
    assert plan.keys == ["hot"]
    assert plan.total_bytes == 6
    assert "cold" in plan.skipped


def test_plan_excludes_resident():
    heat = AccessHeatTracker()
    heat.record("a", now=0.0)

    plan = PrefetchPolicy(budget_bytes=100).plan(
        candidates={"a": 1, "b": 1},
        heat=heat,
        now=0.0,
        resident={"a"},
    )

    assert plan.keys == ["b"]
    assert "a" in plan.skipped


def test_zero_budget_skips_everything():
    heat = AccessHeatTracker()
    heat.record("a", now=0.0)

    plan = PrefetchPolicy(budget_bytes=0).plan(
        candidates={"a": 1},
        heat=heat,
        now=0.0,
    )

    assert plan.keys == []
    assert plan.total_bytes == 0
    assert plan.skipped == ["a"]


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        PrefetchPolicy(budget_bytes=-1)