# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Unit tests for the global prefix index and weight-sharing ledger.

Both are pure-logic metadata structures (no tensors, no torch/ray import),
so they load with ``importlib`` directly and run without a GPU.
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


gpi = _load("global_prefix_index_under_test", "global_prefix_index.py")
wsh = _load("weight_sharing_under_test", "weight_sharing.py")

GlobalPrefixIndex = gpi.GlobalPrefixIndex
PrefixEntry = gpi.PrefixEntry
WeightShareLedger = wsh.WeightShareLedger


def _entry(content_hash, group_id, weight_hash, actor_id, block_id, refs=0):
    return PrefixEntry(
        content_hash=content_hash,
        group_id=group_id,
        weight_hash=weight_hash,
        actor_id=actor_id,
        node_id="n0",
        block_id=block_id,
        refs=refs,
    )


def test_lookup_only_reuses_same_weight_hash():
    index = GlobalPrefixIndex()
    index.register(_entry("h1", 0, "wA", "a0", 5, refs=2))
    index.register(_entry("h1", 0, "wB", "a1", 9, refs=9))

    # Same weight -> hit; different weight -> correctly excluded.
    assert [e.actor_id for e in index.lookup("h1", 0, "wA")] == ["a0"]
    assert [e.actor_id for e in index.lookup("h1", 0, "wB")] == ["a1"]


def test_lookup_orders_by_refs_and_excludes_actors():
    index = GlobalPrefixIndex()
    index.register(_entry("h1", 0, "wA", "a0", 5, refs=2))
    index.register(_entry("h1", 0, "wA", "a1", 9, refs=5))

    assert [e.actor_id for e in index.lookup("h1", 0, "wA")] == ["a1", "a0"]
    hits = index.lookup("h1", 0, "wA", exclude_actors=("a1",))
    assert [e.actor_id for e in hits] == ["a0"]


def test_drop_actor_removes_only_its_entries():
    index = GlobalPrefixIndex()
    index.register(_entry("h1", 0, "wA", "a0", 5))
    index.register(_entry("h1", 0, "wA", "a1", 9))

    assert index.drop_actor("a1") == 1
    assert [e.actor_id for e in index.lookup("h1", 0, "wA")] == ["a0"]
    assert index.stats() == {"wA": 1}


def test_weight_ledger_holders_and_unregister():
    ledger = WeightShareLedger()
    ledger.register("a0", "baseA", "lora1")
    ledger.register("a1", "baseA", "lora1")
    ledger.register("a2", "baseA", "lora2")

    assert ledger.holders("baseA", "lora1") == {"a0", "a1"}
    assert ledger.holders("baseA", "lora2") == {"a2"}
    assert ledger.unregister("a0", "baseA", "lora1") == 1
    assert ledger.holders("baseA", "lora1") == {"a1"}


def test_cost_ratio_quantifies_adapter_win():
    ledger = WeightShareLedger()
    # Adapter <<1% of the base -> ratio reflects the 1-2 order-of-magnitude
    # transfer saving of an adapter-only switch.
    assert ledger.cost_ratio(100_000_000, 500_000) < 0.01
    with pytest.raises(ValueError):
        ledger.cost_ratio(0, 1)