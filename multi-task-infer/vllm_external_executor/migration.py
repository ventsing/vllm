# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Migration state machine and transaction primitives (pure logic, no Ray dep).

Models the lifecycle of a "migration" (model hot-switch on a running instance,
or cross-node actor rebuild) as an explicit state machine with:

- a fixed phase sequence with validated transitions;
- per-phase compensation handlers for **atomic rollback** (a failed migration
  unwinds to the pre-migration state instead of leaving dirty state);
- request-level **idempotency** keyed on ``migration_id``.

Keeping this free of Ray/vLLM/torch imports makes the state machine and the
idempotency registry unit-testable offline.
"""

from __future__ import annotations

import enum
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

# A rollback/compensation step: callable returning None, invoked in reverse
# registration order to undo a completed migration phase.
Compensation = Callable[[], None]


class MigrationPhase(str, enum.Enum):
    """Phases of the migration state machine.

    The ordered happy path is::

        IDLE -> PREPARING -> GRACEFUL_PAUSE -> CHECKPOINT -> UNLOAD
             -> LOAD -> RESTORE -> COMPLETED

    Any phase from GRACEFUL_PAUSE onwards may fail, which drives ROLLING_BACK,
    then either ROLLED_BACK (atomic rollback succeeded; equivalent to the
    pre-migration state) or FAILED (rollback itself failed).
    """

    IDLE = "idle"
    PREPARING = "preparing"
    GRACEFUL_PAUSE = "graceful_pause"
    CHECKPOINT = "checkpoint"
    UNLOAD = "unload"
    LOAD = "load"
    RESTORE = "restore"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class FlightBatchPolicy(str, enum.Enum):
    """How to handle in-flight batches during a migration.

    These policies require collaboration with the vLLM scheduler (owned by
    EngineCore, *not* the executor), so the executor surfaces them via
    ``begin_migration`` and the engine-core caller implements the actual
    scheduler coordination:

    - ``DRAIN``: stop scheduling new requests; let each in-flight batch run
      to completion, then migrate. Safest; no request is interrupted, at the
      cost of tail-latency before the switch.
    - ``PAUSE_SERIALIZE``: freeze the scheduler, serialize the pending
      sequence-group / step state, then resume deterministically after the
      switch. Lowest disruption to throughput; requires the caller to expose
      a checkpointable scheduler state.
    - ``PREEMPT``: interrupt in-flight batches immediately and rely on
      request-level idempotency/replay to retry the affected sequence groups.
      Fastest switch; risks duplicating partially-executed work unless the
      frontend is idempotent.
    """

    DRAIN = "drain"
    PAUSE_SERIALIZE = "pause_serialize"
    PREEMPT = "preempt"


class MigrationOutcome(str, enum.Enum):
    """Terminal outcome of a migration attempt."""

    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"  # rollback unsuccessful; manual intervention needed


# Allowed transitions. Key: current phase; value: reachable next phases.
_TRANSITIONS: dict[MigrationPhase, frozenset[MigrationPhase]] = {
    MigrationPhase.IDLE: frozenset({MigrationPhase.PREPARING}),
    MigrationPhase.PREPARING: frozenset({
        MigrationPhase.GRACEFUL_PAUSE, MigrationPhase.FAILED,
    }),
    MigrationPhase.GRACEFUL_PAUSE: frozenset({
        MigrationPhase.CHECKPOINT, MigrationPhase.FAILED,
    }),
    MigrationPhase.CHECKPOINT: frozenset({
        MigrationPhase.UNLOAD, MigrationPhase.FAILED,
    }),
    MigrationPhase.UNLOAD: frozenset({
        MigrationPhase.LOAD, MigrationPhase.FAILED,
    }),
    MigrationPhase.LOAD: frozenset({
        MigrationPhase.RESTORE, MigrationPhase.FAILED,
    }),
    MigrationPhase.RESTORE: frozenset({
        MigrationPhase.COMPLETED, MigrationPhase.FAILED,
    }),
    MigrationPhase.COMPLETED: frozenset(),
    MigrationPhase.FAILED: frozenset({MigrationPhase.ROLLING_BACK}),
    MigrationPhase.ROLLING_BACK: frozenset({
        MigrationPhase.ROLLED_BACK, MigrationPhase.FAILED,
    }),
    MigrationPhase.ROLLED_BACK: frozenset(),
}


class InvalidTransitionError(Exception):
    """Raised when the state machine is asked to take an illegal transition."""


@dataclass
class MigrationSpec:
    """Input describing one migration operation.

    Attributes:
        migration_id: Idempotency key. Reissuing the same id returns the same
            result instead of performing a second migration.
        target: Human-readable target descriptor (model name / node id).
        policy: In-flight batch handling policy.
        source_checkpoint: Checkpoint/path that can restore the pre-migration
            state during rollback (e.g. current model checkpoint). None when
            rollback uses the already-held references.
        payload: Opaque extras forwarded to the orchestration callback.
    """

    migration_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    target: str = ""
    policy: FlightBatchPolicy = FlightBatchPolicy.DRAIN
    source_checkpoint: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class MigrationStateMachine:
    """Validated migration state machine with compensation-based rollback.

    Each completed phase from ``CHECKPOINT`` onward may register a
    compensation handler. On ``fail()``, ``rollback()`` invokes the handlers in
    reverse registration order, attempting to return the system to the
    pre-migration state; the outcome is ROLLED_BACK if every handler succeeds,
    otherwise FAILED.
    """

    def __init__(self, spec: MigrationSpec):
        self.spec = spec
        self.phase = MigrationPhase.IDLE
        self.outcome: MigrationOutcome | None = None
        self.error: str | None = None
        self._compensations: list[Compensation] = []
        self._rollback_done = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------ transitions
    def transition(self, next_phase: MigrationPhase) -> None:
        """Advance to ``next_phase`` after validating the transition."""
        with self._lock:
            if next_phase not in _TRANSITIONS[self.phase]:
                raise InvalidTransitionError(
                    f"Illegal transition {self.phase.value} -> "
                    f"{next_phase.value}"
                )
            self.phase = next_phase

    def register_compensation(self, handler: Compensation) -> None:
        """Register a rollback step for the phase just completed.

        Handlers run in reverse registration order on rollback, so each phase
        should register exactly one handler that undoes its own side effect.
        """
        if not callable(handler):
            raise TypeError("compensation handler must be callable")
        with self._lock:
            self._compensations.append(handler)

    # ------------------------------------------------------------- failure
    def fail(self, error: str) -> None:
        """Mark the migration failed (keeps the current phase for forensics)."""
        with self._lock:
            self.error = error
            self.outcome = MigrationOutcome.FAILED
            self.phase = MigrationPhase.FAILED

    def rollback(self) -> MigrationOutcome:
        """Unwind compensations in reverse order.

        Idempotent: a second call returns the cached outcome without re-running
        the compensation handlers.

        Returns:
            ROLLED_BACK if the pre-migration state was restored, FAILED if a
            compensation handler raised.
        """
        with self._lock:
            if self._rollback_done:
                return self.outcome or MigrationOutcome.ROLLED_BACK
            self._rollback_done = True
            self.phase = MigrationPhase.ROLLING_BACK

        errors: list[str] = []
        for handler in reversed(self._compensations):
            try:
                handler()
            except Exception as e:  # noqa: BLE001 - record and continue
                errors.append(f"{e}")

        with self._lock:
            self._compensations.clear()
            if errors:
                self.outcome = MigrationOutcome.FAILED
                self.error = "rollback failed: " + "; ".join(errors)
                self.phase = MigrationPhase.FAILED
            else:
                self.outcome = MigrationOutcome.ROLLED_BACK
                self.error = None
                self.phase = MigrationPhase.ROLLED_BACK
        return self.outcome

    # ------------------------------------------------------------- queries
    def is_terminal(self) -> bool:
        return self.phase in (MigrationPhase.COMPLETED, MigrationPhase.ROLLED_BACK)

    def progress(self) -> str:
        """One-line status string for logs."""
        return (
            f"migration={self.spec.migration_id[:8]} phase={self.phase.value} "
            f"outcome={self.outcome.value if self.outcome else '-'}"
        )


class MigrationIdempotencyRegistry:
    """Request-level idempotency: one result per ``migration_id``.

    ``run`` executes ``execute_fn`` once per unique id (holding a per-id lock
    so concurrent callers block until the first completes) and thereafter
    returns the cached result. This makes a re-issued migration request (e.g.
    a retried RPC) a no-op returning the original outcome.
    """

    def __init__(self):
        self._results: dict[str, MigrationStateMachine] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, migration_id: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(migration_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[migration_id] = lock
            return lock

    def run(
        self,
        spec: MigrationSpec,
        execute_fn: Callable[[MigrationStateMachine], None],
    ) -> MigrationStateMachine:
        """Execute a migration idempotently.

        Args:
            spec: Migration specification carrying the idempotency key.
            execute_fn: Orchestration that drives ``sm`` through its phases,
                registers compensations, and calls ``fail`` / a final phase.

        Returns:
            The state machine, cached for ``spec.migration_id``.
        """
        identifier = spec.migration_id
        # Fast path: cached terminal result.
        existing = self._results.get(identifier)
        if existing is not None and existing.is_terminal():
            return existing

        lock = self._lock_for(identifier)
        with lock:
            existing = self._results.get(identifier)
            if existing is not None and existing.is_terminal():
                return existing

            sm = MigrationStateMachine(spec)
            try:
                execute_fn(sm)
            except Exception as e:  # noqa: BLE001 - orchestration may throw
                sm.error = str(e)
                sm.outcome = MigrationOutcome.FAILED
                sm.phase = MigrationPhase.FAILED

            # A machine left in FAILED (exception or explicit fail()) is
            # auto-rolled back unless the orchestration already rolled it back.
            if (
                sm.phase == MigrationPhase.FAILED
                and sm.outcome == MigrationOutcome.FAILED
            ):
                sm.rollback()

            self._results[identifier] = sm
            return sm

    def get(self, migration_id: str) -> MigrationStateMachine | None:
        """Return the cached result for ``migration_id``, if any."""
        return self._results.get(migration_id)