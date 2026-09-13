# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Unit tests for storage tiering and prefetch decisions (pure logic).

Both modules keep no tensors and have no torch/ray import, so they load with
``importlib`` directly and run without a GPU.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent / "vllm_external_executor"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _PKG / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


st = _load("storage_tier_under_test", "storage_tier.py")
pp = _load("prefetch_policy_under_test", "prefetch_policy.py")

StorageTier = st.StorageTier
TieredCache = st.TieredCache
AccessHeatTracker = pp.AccessHeatTracker
PrefetchPolicy = pp.PrefetchPolicy


def test_tier_order_colder():
    HBM, DRAM, REMOTE = StorageTier.HBM, StorageTier.DRAM, StorageTier.REMOTE
    assert HBM.colder() is DRAM
    assert DRAM.colder() is REMOTE
    assert REMOTE.colder() is None


def test_put_within_capacity_no_eviction():
    cache = TieredCache({StorageTier.HBM: 100})
    assert cache.put("a", 60, StorageTier.HBM, 1.0) == []
    assert cache.location("a") is StorageTier.HBM


def test_put_evicts_lru_to_colder_tier():
    cache = TieredCache(
        {StorageTier.HBM: 100, StorageTier.DRAM: 100}
    )
    cache.put("a", 60, StorageTier.HBM, 1.0)
    decisions = cache.put("b", 60, StorageTier.HBM, 2.0)

    assert [(d.key, d.from_tier, d.to_tier) for d in decisions] == [
        ("a", StorageTier.HBM, StorageTier.DRAM)
    ]
    assert cache.location("a") is StorageTier.DRAM
    assert cache.usage(StorageTier.HBM) == 60


def test_evict_past_remote_drops_entry():
    cache = TieredCache({StorageTier.REMOTE: 10})
    decisions = cache.put("x", 20, StorageTier.REMOTE, 1.0)
    assert decisions[-1].to_tier is None
    assert cache.location("x") is None  # dropped, not resident anywhere


def test_touch_then_promote():
    cache = TieredCache(
        {StorageTier.HBM: 100, StorageTier.DRAM: 100}
    )
    cache.put("a", 60, StorageTier.HBM, 1.0)
    cache.put("b", 60, StorageTier.HBM, 2.0)  # "a" demoted to DRAM
    cache.touch("a", 3.0)  # "a" now hotter than "b"
    decisions = cache.promote("a", StorageTier.HBM, 4.0)

    assert cache.location("a") is StorageTier.HBM
    assert cache.location("b") is StorageTier.DRAM  # "b" pushed out
    assert decisions[-1].reason == "promotion"


def test_heat_tracker_decays_old_access():
    heat = AccessHeatTracker(decay=0.5)
    heat.record("k", 1.0)
    heat.record("k", 2.0)
    assert heat.score("k", 2.0) == pytest.approx(2.0)
    # One time-step of decay halves the score.
    assert heat.score("k", 3.0) == pytest.approx(1.0)


def test_prefetch_ranks_hot_and_respects_budget():
    heat = AccessHeatTracker(decay=0.9)
    heat.record("cold", 1.0)
    heat.record("hot", 9.0)
    heat.record("hot", 10.0)

    policy = PrefetchPolicy(40)
    plan = policy.plan({"hot": 40, "cold": 40}, heat, 11.0)

    assert plan.keys == ["hot"]  # budget clips the colder candidate
    assert plan.total_bytes == 40
    assert "cold" in plan.skipped


def test_prefetch_skips_resident_and_over_budget():
    heat = AccessHeatTracker(decay=0.9)
    heat.record("resident", 1.0)
    heat.record("hot", 9.0)

    plan = PrefetchPolicy(100).plan(
        {"hot": 40, "resident": 40}, heat, 10.0, resident={"resident"}
    )

    assert plan.keys == ["hot"]
    assert plan.skipped == ["resident"]