# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Unit tests for the migration orchestrator (decision-layer wiring).

The orchestrator depends only on pure-logic modules, so it loads with
``importlib`` once those modules are pre-registered in ``sys.modules`` under a
stub package (this mirrors how the plugin avoids the package ``__init__``'s
torch import in an offline test).
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent / "vllm_external_executor"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _PKG / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.modules.setdefault(
    "vllm_external_executor", types.ModuleType("vllm_external_executor")
)
migration = _load_module("vllm_external_executor.migration", "migration.py")
storage_tier = _load_module(
    "vllm_external_executor.storage_tier", "storage_tier.py"
)
prefetch = _load_module(
    "vllm_external_executor.prefetch_policy", "prefetch_policy.py"
)
prefix_mod = _load_module(
    "vllm_external_executor.global_prefix_index", "global_prefix_index.py"
)
weight_mod = _load_module(
    "vllm_external_executor.weight_sharing", "weight_sharing.py"
)
orch = _load_module(
    "vllm_external_executor.migration_orchestrator",
    "migration_orchestrator.py",
)

StorageTier = storage_tier.StorageTier
TieredCache = storage_tier.TieredCache
AccessHeatTracker = prefetch.AccessHeatTracker
PrefetchPolicy = prefetch.PrefetchPolicy
GlobalPrefixIndex = prefix_mod.GlobalPrefixIndex
PrefixEntry = prefix_mod.PrefixEntry
WeightShareLedger = weight_mod.WeightShareLedger
MigrationSpec = migration.MigrationSpec
MigrationOrchestrator = orch.MigrationOrchestrator


def _build(spec, heat=None, budget=100, start=None, await_=None):
    heat = heat or AccessHeatTracker(0.9)
    orchestrator = MigrationOrchestrator(
        tiered_cache=TieredCache(
            {StorageTier.HBM: 1000, StorageTier.DRAM: 1000,
             StorageTier.REMOTE: 1000}
        ),
        heat=heat,
        prefetch_policy=PrefetchPolicy(budget),
        prefix_index=GlobalPrefixIndex(),
        weight_ledger=WeightShareLedger(),
        start_prefetch=start,
        await_prefetch=await_,
    )
    return orchestrator


def _spec(target="target", payload=None):
    payload = payload or {
        "target_checkpoint": "target",
        "checkpoint_bytes": 100,
        "base_bytes": 50,
        "base_hash": "baseA",
        "adapter": "lora1",
        "prefix_entries": [
            PrefixEntry("h1", 0, "baseA", "a0", "n0", 5, refs=2)
        ],
    }
    return MigrationSpec(target=target, source_checkpoint="current",
                         payload=payload)


def test_async_prefetch_hook_and_tiering():
    """Prefetch starts at PREPARING and is awaited before the LOAD promote."""
    heat = AccessHeatTracker(0.9)
    heat.record("hot", 9.0)
    heat.record("hot", 10.0)
    calls = []
    orchestrator = _build(
        _spec(),
        heat=heat,
        budget=40,
        start=lambda keys: calls.append(("start", list(keys))),
        await_=lambda: calls.append("await"),
    )

    sm, script = orchestrator.migrate(
        _spec(), actor_id="a0",
        candidates={"hot": 40, "cold": 40},
        resident=set(),
    )

    # Prefetch: hot nominated and consumed in start-before-await order.
    assert calls == [("start", ["hot"]), "await"]
    assert script.prefetch_keys == ["hot"]
    assert script.prefetch_bytes == 40
    # Tiering: target staged to REMOTE, current demoted to DRAM, target
    # promoted to HBM.
    assert sorted(m.key for m in script.checkpoint_moves) == []
    assert [(m.key, m.to_tier) for m in script.unload_moves] == [
        ("current", StorageTier.DRAM)
    ]
    assert [(m.key, m.from_tier, m.to_tier) for m in script.load_moves] == [
        ("target", StorageTier.REMOTE, StorageTier.HBM)
    ]
    assert sm.phase.value == "completed"
    # Metadata restored.
    assert script.prefix_register[0].content_hash == "h1"
    assert script.weight_register == [("a0", "baseA", "lora1")]


def test_rollback_undoes_decision_bookkeeping():
    """Compensations remove the orchestrator's tier/prefix/ledger entries."""
    orchestrator = _build(_spec())
    sm, _ = orchestrator.migrate(
        _spec(), actor_id="a0",
        candidates={"hot": 40}, resident=set(),
    )

    tiered = orchestrator._tiered
    assert tiered.location("target") is StorageTier.HBM
    assert orchestrator._weights.holders("baseA", "lora1") == {"a0"}

    # The machine finished COMPLETED, but rollback() runs the registered
    # compensations regardless of phase; here we exercise those closures.
    assert sm.rollback().value == "rolled_back"

    assert tiered.location("target") is None
    assert tiered.location("current") is None
    assert orchestrator._weights.holders("baseA", "lora1") == set()
    assert orchestrator._prefix.stats() == {}


def test_prefetch_skips_resident_and_over_budget():
    """Resident keys and over-budget candidates are left out of the hook."""
    heat = AccessHeatTracker(0.9)
    heat.record("hot", 9.0)
    heat.record("already_there", 1.0)
    started = []
    orchestrator = _build(
        _spec(), heat=heat, budget=100,
        start=lambda keys: started.append(list(keys)),
    )

    orchestrator.migrate(
        _spec(), actor_id="a0",
        candidates={"hot": 40, "already_there": 40, "big": 200},
        resident={"already_there"},
    )

    assert started == [["hot"]]


def test_kv_migration_semantics_prefetch_and_index():
    """Executor glue: hot non-resident prefixes prefetched; transfer indexed.

    Mirrors ``ExternalExecutor._drive_migration_orchestrator``: candidates are
    source prefixes, resident are destination prefixes, and only the
    ``transfer`` blocks become prefix entries (remapped to dst ids).
    """
    heat = AccessHeatTracker(0.9)
    heat.record("hot:0", 1.0)
    heat.record("hot:0", 2.0)
    heat.record("cold:0", 1.0)  # colder than hot, present on source

    started = []
    orchestrator = _build(
        _spec(), heat=heat, budget=1,
        start=lambda keys: started.append(list(keys)),
    )

    spec = MigrationSpec(
        target="kv-migrate:wA",
        payload={
            "target_checkpoint": "kv:wA",
            "checkpoint_bytes": 2,
            "prefix_entries": [
                PrefixEntry("hot", 0, "wA", "dst0", "n1", 11, refs=1),
                PrefixEntry("cold", 0, "wA", "dst0", "n1", 13, refs=1),
            ],
        },
    )
    sm, script = orchestrator.migrate(
        spec, actor_id="dst0",
        candidates={"hot:0": 1, "cold:0": 1, "already:0": 1},
        resident={"already:0"},
    )

    # Budget=1 unit picks only the hottest non-resident prefix.
    assert started == [["hot:0"]]
    assert script.prefix_register[0].actor_id == "dst0"
    assert [e.content_hash for e in script.prefix_register] == ["hot", "cold"]
    assert sm.phase.value == "completed"


def test_phase_handlers_and_external_sm():
    """Phase handlers fire in order on an externally-owned state machine.

    Mirrors ``switch_model``: the executor pre-creates the machine inside
    ``MigrationIdempotencyRegistry``, passes it via ``sm=``, and injects
    per-phase execution callbacks (pause / checkpoint / switch / restore).
    """
    spec = MigrationSpec(
        target="m",
        source_checkpoint="cur",
        payload={
            "target_checkpoint": "tgt",
            "checkpoint_bytes": 0,
            "base_bytes": 0,
            "base_hash": "wA",
        },
    )
    external_sm = migration.MigrationStateMachine(spec)
    seen = []

    def phase_handler(phase):
        def handler(sm, script):
            seen.append((phase, sm is external_sm, script.migration_id))

        return handler

    handlers = {
        phase: phase_handler(phase)
        for phase in [
            migration.MigrationPhase.PREPARING,
            migration.MigrationPhase.GRACEFUL_PAUSE,
            migration.MigrationPhase.CHECKPOINT,
            migration.MigrationPhase.UNLOAD,
            migration.MigrationPhase.LOAD,
            migration.MigrationPhase.RESTORE,
        ]
    }

    orchestrator = _build(spec)
    sm, script = orchestrator.migrate(
        spec, actor_id="a0",
        candidates={}, resident=set(),
        phase_handlers=handlers, sm=external_sm,
    )

    assert sm is external_sm
    assert [p for p, _, _ in seen] == [
        migration.MigrationPhase.PREPARING,
        migration.MigrationPhase.GRACEFUL_PAUSE,
        migration.MigrationPhase.CHECKPOINT,
        migration.MigrationPhase.UNLOAD,
        migration.MigrationPhase.LOAD,
        migration.MigrationPhase.RESTORE,
    ]
    assert all(flag for _, flag, _ in seen)
    # switch_model payload has base_hash but no prefix entries: the ledger is
    # updated, the prefix index is not.
    assert script.weight_register == [("a0", "wA", None)]
    assert script.prefix_register == []
    assert external_sm.phase == migration.MigrationPhase.COMPLETED


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))