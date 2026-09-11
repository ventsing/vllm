# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Unit tests for the migration state machine + idempotency registry.

migration.py is dependency-free, so these tests run offline (no Ray/torch).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "vllm_external_executor"
    / "migration.py"
)
_spec = importlib.util.spec_from_file_location("migration_under_test", _MODULE_PATH)
migration = importlib.util.module_from_spec(_spec)
sys.modules["migration_under_test"] = migration
_spec.loader.exec_module(migration)

FlightBatchPolicy = migration.FlightBatchPolicy
InvalidTransitionError = migration.InvalidTransitionError
MigrationIdempotencyRegistry = migration.MigrationIdempotencyRegistry
MigrationOutcome = migration.MigrationOutcome
MigrationPhase = migration.MigrationPhase
MigrationSpec = migration.MigrationSpec
MigrationStateMachine = migration.MigrationStateMachine


def test_happy_path_ordered_phases():
    sm = MigrationStateMachine(MigrationSpec(target="m1"))
    order = [
        MigrationPhase.PREPARING,
        MigrationPhase.GRACEFUL_PAUSE,
        MigrationPhase.CHECKPOINT,
        MigrationPhase.UNLOAD,
        MigrationPhase.LOAD,
        MigrationPhase.RESTORE,
        MigrationPhase.COMPLETED,
    ]
    for phase in order:
        sm.transition(phase)
    assert sm.phase == MigrationPhase.COMPLETED
    assert sm.is_terminal()


def test_illegal_transition_raises():
    sm = MigrationStateMachine(MigrationSpec(target="m1"))
    sm.transition(MigrationPhase.PREPARING)
    with pytest.raises(InvalidTransitionError):
        # Cannot jump straight from PREPARING to LOAD.
        sm.transition(MigrationPhase.LOAD)


def test_skip_checkpoint_is_illegal():
    sm = MigrationStateMachine(MigrationSpec(target="m1"))
    sm.transition(MigrationPhase.PREPARING)
    sm.transition(MigrationPhase.GRACEFUL_PAUSE)
    with pytest.raises(InvalidTransitionError):
        sm.transition(MigrationPhase.UNLOAD)  # skipped CHECKPOINT


def test_fail_then_rollback_reverses_compensations():
    sm = MigrationStateMachine(MigrationSpec(target="m1"))
    order = []
    sm.transition(MigrationPhase.PREPARING)
    sm.transition(MigrationPhase.GRACEFUL_PAUSE)
    sm.transition(MigrationPhase.CHECKPOINT)
    sm.register_compensation(lambda: order.append("undo-checkpoint"))
    sm.transition(MigrationPhase.UNLOAD)
    sm.register_compensation(lambda: order.append("undo-unload"))
    sm.transition(MigrationPhase.LOAD)
    sm.register_compensation(lambda: order.append("undo-load"))

    sm.fail("load failed")
    outcome = sm.rollback()

    assert outcome == MigrationOutcome.ROLLED_BACK
    assert sm.phase == MigrationPhase.ROLLED_BACK
    assert order == ["undo-load", "undo-unload", "undo-checkpoint"]


def test_rollback_failure_sets_failed():
    sm = MigrationStateMachine(MigrationSpec(target="m1"))
    sm.transition(MigrationPhase.PREPARING)
    sm.transition(MigrationPhase.GRACEFUL_PAUSE)
    sm.transition(MigrationPhase.CHECKPOINT)

    def broken():
        raise RuntimeError("cannot undo")

    sm.register_compensation(broken)
    sm.fail("unload failed")
    outcome = sm.rollback()

    assert outcome == MigrationOutcome.FAILED
    assert sm.phase == MigrationPhase.FAILED


def test_fail_before_checkpoint_no_compensation():
    """Failing before any phase registers a compensation still rolls back."""
    sm = MigrationStateMachine(MigrationSpec(target="m1"))
    sm.transition(MigrationPhase.PREPARING)
    sm.fail("precondition unsatisfied")
    assert sm.rollback() == MigrationOutcome.ROLLED_BACK


def test_idempotency_runs_once():
    registry = MigrationIdempotencyRegistry()
    calls = []

    def orchestrate(sm):
        calls.append(1)
        sm.transition(MigrationPhase.PREPARING)
        sm.transition(MigrationPhase.GRACEFUL_PAUSE)
        sm.transition(MigrationPhase.CHECKPOINT)
        sm.transition(MigrationPhase.UNLOAD)
        sm.transition(MigrationPhase.LOAD)
        sm.transition(MigrationPhase.RESTORE)
        sm.transition(MigrationPhase.COMPLETED)

    spec = MigrationSpec(migration_id="same-id", target="m")
    first = registry.run(spec, orchestrate)
    second = registry.run(spec, orchestrate)

    assert first is second
    assert len(calls) == 1
    assert first.phase == MigrationPhase.COMPLETED


def test_idempotency_auto_rollback_on_exception():
    registry = MigrationIdempotencyRegistry()

    def orchestrate(sm):
        sm.transition(MigrationPhase.PREPARING)
        sm.transition(MigrationPhase.GRACEFUL_PAUSE)
        raise RuntimeError("scheduler pause failed")

    spec = MigrationSpec(migration_id="failing", target="m")
    result = registry.run(spec, orchestrate)
    assert result.outcome == MigrationOutcome.ROLLED_BACK


def test_flight_batch_policy_values():
    assert {p.value for p in FlightBatchPolicy} == {
        "drain", "pause_serialize", "preempt",
    }