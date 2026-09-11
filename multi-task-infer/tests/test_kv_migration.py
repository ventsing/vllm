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

IncrementalKVPlanner = kv_migration.IncrementalKVPlanner
KVBlockRef = kv_migration.KVBlockRef
KVMigrationPlan = kv_migration.KVMigrationPlan


def test_new_block_transfers():
    src = [KVBlockRef(block_id=0, content_hash="h0", version=1)]
    dst = []
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == src
    assert plan.prefix_hits == []
    assert plan.unchanged == []


def test_unchanged_block_skipped():
    src = [KVBlockRef(block_id=0, content_hash="h0", version=1)]
    dst = [KVBlockRef(block_id=0, content_hash="h0", version=1)]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == []
    assert plan.unchanged == [0]


def test_modified_block_transfers():
    src = [KVBlockRef(block_id=0, content_hash="h-new", version=2)]
    dst = [KVBlockRef(block_id=0, content_hash="h-old", version=1)]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == src


def test_prefix_cache_hit_reused_across_ids():
    """A source block whose hash exists at destination reuses, no transfer."""
    src = [KVBlockRef(block_id=5, content_hash="common-prefix", version=1)]
    dst = [KVBlockRef(block_id=0, content_hash="common-prefix", version=1)]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == []
    assert plan.prefix_hits == [(5, 0)]


def test_hash_equality_reuses_even_when_version_differs():
    """Content hash is authoritative: same hash = reuse regardless of version."""
    src = [KVBlockRef(block_id=5, content_hash="same", version=9)]
    dst = [KVBlockRef(block_id=0, content_hash="same", version=1)]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == []
    assert plan.prefix_hits == [(5, 0)]


def test_no_hash_falls_back_to_version():
    src = [KVBlockRef(block_id=0, content_hash="", version=2)]
    dst = [KVBlockRef(block_id=0, content_hash="", version=1)]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.transfer == src  # version differs -> transfer

    dst_same = [KVBlockRef(block_id=0, content_hash="", version=2)]
    plan2 = IncrementalKVPlanner().plan(src, dst_same)
    assert plan2.unchanged == [0]  # version equal -> unchanged


def test_mixed_plan_counts():
    src = [
        KVBlockRef(block_id=0, content_hash="h0", version=1),  # unchanged
        KVBlockRef(block_id=1, content_hash="h1", version=1),  # new -> transfer
        KVBlockRef(block_id=2, content_hash="h2", version=2),  # modified
        KVBlockRef(block_id=3, content_hash="shared", version=1),  # prefix hit
    ]
    dst = [
        KVBlockRef(block_id=0, content_hash="h0", version=1),
        KVBlockRef(block_id=2, content_hash="old-h2", version=1),
        KVBlockRef(block_id=9, content_hash="shared", version=1),
    ]
    plan = IncrementalKVPlanner().plan(src, dst)
    assert plan.unchanged == [0]
    assert {b.block_id for b in plan.transfer} == {1, 2}
    assert plan.prefix_hits == [(3, 9)]
    assert plan.transferred_count == 2
    assert plan.reused_count == 2


def test_plan_summary_string():
    plan = KVMigrationPlan(
        transfer=[KVBlockRef(1), KVBlockRef(2)],
        prefix_hits=[(3, 9)],
        unchanged=[0],
    )
    assert "transfer=2" in plan.summary()
    assert "prefix_hits=1" in plan.summary()