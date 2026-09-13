# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Unit tests for incremental KV-block migration planning.

kv_migration.py is dependency-free, so these run offline (no Ray/torch).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "vllm_external_executor"
    / "kv_migration.py"
)
_spec = importlib.util.spec_from_file_location("kv_migration_under_test", _MODULE_PATH)
kv_migration = importlib.util.module_from_spec(_spec)
sys.modules["kv_migration_under_test"] = kv_migration
_spec.loader.exec_module(kv_migration)

BlockIdAllocator = kv_migration.BlockIdAllocator
IncrementalKVPlanner = kv_migration.IncrementalKVPlanner
KVBlockRef = kv_migration.KVBlockRef
KVMigrationPlan = kv_migration.KVMigrationPlan


def _ref(block_id, hash_="", version=1, group_id=0):
    return KVBlockRef(block_id, hash_, version, group_id=group_id)


def test_new_block_transfers():
    src = [_ref(0, "h0")]
    dst = []
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == src
    assert plan.prefix_hits == []
    assert plan.unchanged == []


def test_unchanged_block_skipped():
    src = [_ref(0, "h0")]
    dst = [_ref(0, "h0")]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == []
    assert plan.unchanged == [(0, 0)]


def test_modified_block_transfers():
    src = [_ref(0, "h-new", version=2)]
    dst = [_ref(0, "h-old")]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == src


def test_prefix_cache_hit_reused_across_ids():
    """A source block whose hash exists at destination reuses, no transfer."""
    src = [_ref(5, "common-prefix")]
    dst = [_ref(0, "common-prefix")]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == []
    assert plan.prefix_hits == [(5, 0, 0)]


def test_hash_equality_reuses_even_when_version_differs():
    """Content hash is authoritative: same hash = reuse regardless of version."""
    src = [_ref(5, "same", version=9)]
    dst = [_ref(0, "same", version=1)]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == []
    assert plan.prefix_hits == [(5, 0, 0)]


def test_no_hash_falls_back_to_version():
    src = [_ref(0, "", version=2)]
    dst = [_ref(0, "", version=1)]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == src  # version differs -> transfer

    dst_same = [_ref(0, "", version=2)]
    plan2 = IncrementalKVPlanner().plan(src, dst_same)
    assert plan2.unchanged == [(0, 0)]  # version equal -> unchanged


def test_groups_are_isolated():
    """Same block id + hash in different groups must NOT collide."""
    src = [_ref(5, "shared", group_id=0), _ref(5, "shared", group_id=1)]
    dst = [_ref(9, "shared", group_id=0)]  # only group 0 has it
    plan = IncrementalKVPlanner().plan(src, dst)
    # group 0 -> prefix hit; group 1 -> transfer (no group-1 resident).
    assert plan.prefix_hits == [(5, 9, 0)]
    assert [b.group_id for b in plan.transfer] == [1]


def test_same_hash_different_group_is_not_reused():
    """A group-1 block whose hash exists only in group 0 must transfer."""
    src = [_ref(5, "shared", group_id=1)]
    dst = [_ref(0, "shared", group_id=0)]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == src
    assert plan.prefix_hits == []


def test_mixed_plan_counts():
    src = [
        _ref(0, "h0"),              # unchanged
        _ref(1, "h1"),              # new -> transfer
        _ref(2, "h2", version=2),   # modified
        _ref(3, "shared"),          # prefix hit
    ]
    dst = [
        _ref(0, "h0"),
        _ref(2, "old-h2"),
        _ref(9, "shared"),
    ]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.unchanged == [(0, 0)]
    assert {b.block_id for b in plan.transfer} == {1, 2}
    assert plan.prefix_hits == [(3, 9, 0)]
    assert plan.transferred_count == 2
    assert plan.reused_count == 2


def test_assign_targets_maps_blocks():
    plan = KVMigrationPlan(transfer=[_ref(7, "a"), _ref(3, "b"), _ref(7, "c")])
    # block 7 appears twice (two groups), so 2 unique source blocks.
    planner = IncrementalKVPlanner()
    planner.assign_targets(plan, total_blocks=10, dst_occupied=[0, 1, 5])
    assert plan.block_mapping == {3: 2, 7: 3}


def test_assign_targets_insufficient():
    plan = KVMigrationPlan(transfer=[_ref(0), _ref(1), _ref(2)])
    with pytest.raises(ValueError):
        IncrementalKVPlanner().assign_targets(
            plan, total_blocks=3, dst_occupied=[0]
        )


def test_allocator_sequential():
    alloc = BlockIdAllocator(5, {0, 3})
    assert alloc.allocate(2) == [1, 2]
    assert alloc.allocate(1) == [4]


def test_plan_summary_string():
    plan = KVMigrationPlan(
        transfer=[_ref(1), _ref(2)],
        prefix_hits=[(3, 9, 0)],
        unchanged=[(0, 0)],
    )
    assert "transfer=2" in plan.summary()
    assert "prefix_hits=1" in plan.summary()