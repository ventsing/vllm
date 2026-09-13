# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Wire the decision-layer modules into the migration state machine.

The state machine (:mod:`migration`) validates *when* a migration advances;
the decision modules decide *what to move where*:

- ``PrefetchPolicy`` + ``AccessHeatTracker`` -> async prefetch hook: hot,
  non-resident state is nominated at PREPARING and awaited at LOAD, so its
  load overlaps the CHECKPOINT/UNLOAD work instead of blocking after it.
- ``TieredCache`` -> tiering-driven checkpoint movement: the target
  checkpoint is staged to REMOTE at CHECKPOINT, the current weights demoted
  at UNLOAD, and the target promoted to HBM at LOAD.
- ``GlobalPrefixIndex`` + ``WeightShareLedger`` -> metadata restore at
  RESTORE.

This module is pure logic: it emits a :class:`MigrationScript` of decisions.
An executor's data plane consumes the script to move tensors and registers its
own compensations for the physical moves; the orchestrator only registers
compensations for its own bookkeeping (tier entries, prefix index, ledger).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from vllm_external_executor.global_prefix_index import (
    GlobalPrefixIndex,
    PrefixEntry,
)
from vllm_external_executor.migration import (
    MigrationPhase,
    MigrationSpec,
    MigrationStateMachine,
)
from vllm_external_executor.prefetch_policy import (
    AccessHeatTracker,
    PrefetchPolicy,
)
from vllm_external_executor.storage_tier import StorageTier, TieredCache
from vllm_external_executor.weight_sharing import WeightShareLedger


@dataclass
class TierMove:
    """One tiering move the data plane must carry out.

    Attributes:
        key: Object to move (checkpoint path or weight key).
        from_tier: Current tier.
        to_tier: Target tier; ``None`` means drop (evicted past REMOTE).
    """

    key: str
    from_tier: StorageTier
    to_tier: StorageTier | None


@dataclass
class MigrationScript:
    """Per-phase decision script produced by the orchestrator."""

    migration_id: str
    prefetch_keys: list[str] = field(default_factory=list)
    prefetch_bytes: int = 0
    checkpoint_moves: list[TierMove] = field(default_factory=list)
    unload_moves: list[TierMove] = field(default_factory=list)
    load_moves: list[TierMove] = field(default_factory=list)
    prefix_register: list[PrefixEntry] = field(default_factory=list)
    weight_register: list[tuple[str, str, str | None]] = field(
        default_factory=list
    )


class MigrationOrchestrator:
    """Drive a migration while collecting decision-layer actions.

    ``payload`` conventions (``spec.payload``)::

        target_checkpoint: str      # key promoted to HBM at LOAD
        checkpoint_bytes: int       # target checkpoint size (REMOTE staging)
        base_bytes: int             # current base-weight size (UNLOAD demote)
        base_hash: str              # weight-sharing base hash
        adapter: str | None         # adapter config (None = base only)
        prefix_entries: [PrefixEntry]  # entries to re-index at RESTORE
    """

    def __init__(
        self,
        tiered_cache: TieredCache,
        heat: AccessHeatTracker,
        prefetch_policy: PrefetchPolicy,
        prefix_index: GlobalPrefixIndex,
        weight_ledger: WeightShareLedger,
        start_prefetch: Callable[[list], None] | None = None,
        await_prefetch: Callable[[], None] | None = None,
    ):
        self._tiered = tiered_cache
        self._heat = heat
        self._prefetch = prefetch_policy
        self._prefix = prefix_index
        self._weights = weight_ledger
        self._start_prefetch = start_prefetch or (lambda keys: None)
        self._await_prefetch = await_prefetch or (lambda: None)

    def migrate(
        self,
        spec: MigrationSpec,
        *,
        actor_id: str,
        candidates: dict[str, int],
        resident: set[str] | None = None,
        phase_handlers: dict[
            MigrationPhase, Callable[[MigrationStateMachine, MigrationScript], None]
        ] | None = None,
        sm: MigrationStateMachine | None = None,
    ) -> tuple[MigrationStateMachine, MigrationScript]:
        """Run the full happy path, returning the machine and its script.

        Args:
            spec: Migration input (idempotency key, target, payload).
            actor_id: Actor being migrated (prefix/weight ledger owner).
            candidates: ``key -> size_bytes`` for prefetch ranking.
            resident: Keys already resident (excluded from prefetch).
            phase_handlers: Optional per-phase execution callbacks. Each runs
                after the orchestrator has finished that phase's decisions (so
                the executor can, e.g., pause the scheduler at GRACEFUL_PAUSE,
                fan out worker switches at UNLOAD, or re-init KV at RESTORE).
                Signature: ``(state_machine, script) -> None``.
            sm: Optional externally-owned state machine to drive (e.g. one
                created by :class:`MigrationIdempotencyRegistry`), preserving
                the caller's idempotency/rollback wrapping. Defaults to a new
                machine.

        Returns:
            ``(state_machine, script)``; the machine ended in COMPLETED.
        """
        handlers = phase_handlers or {}
        now = time.monotonic()
        resident = resident or set()
        sm = sm or MigrationStateMachine(spec)
        script = MigrationScript(spec.migration_id)

        # The current weights start resident in HBM (the caller may already
        # have registered them; a no-op otherwise).
        current_key = spec.source_checkpoint or "current"
        if self._tiered.location(current_key) is None:
            self._tiered.put(
                current_key,
                int(spec.payload.get("base_bytes", 0)),
                StorageTier.HBM,
                now,
            )

        sm.on_enter(MigrationPhase.PREPARING, self._hook_prepare(
            script, candidates, resident, now
        ))
        sm.on_enter(MigrationPhase.LOAD, self._hook_await_load())

        sm.transition(MigrationPhase.PREPARING)
        self._run_handler(handlers, MigrationPhase.PREPARING, sm, script)
        sm.transition(MigrationPhase.GRACEFUL_PAUSE)
        self._run_handler(handlers, MigrationPhase.GRACEFUL_PAUSE, sm, script)

        sm.transition(MigrationPhase.CHECKPOINT)
        script.checkpoint_moves = self._stage_checkpoint(spec, now)
        self._register_tier_undo(sm, script.checkpoint_moves)
        self._run_handler(handlers, MigrationPhase.CHECKPOINT, sm, script)

        sm.transition(MigrationPhase.UNLOAD)
        script.unload_moves = self._demote_current(spec, now)
        self._register_tier_undo(sm, script.unload_moves)
        self._run_handler(handlers, MigrationPhase.UNLOAD, sm, script)

        sm.transition(MigrationPhase.LOAD)  # fires await_prefetch
        script.load_moves = self._promote_target(spec, now)
        self._register_tier_undo(sm, script.load_moves)
        self._run_handler(handlers, MigrationPhase.LOAD, sm, script)

        sm.transition(MigrationPhase.RESTORE)
        self._restore_metadata(spec, actor_id, script, sm)
        self._run_handler(handlers, MigrationPhase.RESTORE, sm, script)

        sm.transition(MigrationPhase.COMPLETED)
        return sm, script

    @staticmethod
    def _run_handler(handlers, phase, sm, script) -> None:
        handler = handlers.get(phase)
        if handler is not None:
            handler(sm, script)

    # ---------------------------------------------------------------- hooks
    def _hook_prepare(self, script, candidates, resident, now):
        def handler() -> None:
            plan = self._prefetch.plan(
                candidates, self._heat, now, resident
            )
            script.prefetch_keys = list(plan.keys)
            script.prefetch_bytes = plan.total_bytes
            self._start_prefetch(plan.keys)

        return handler

    def _hook_await_load(self):
        def handler() -> None:
            self._await_prefetch()

        return handler

    # ------------------------------------------------------------ decisions
    def _stage_checkpoint(self, spec, now) -> list[TierMove]:
        key = spec.payload.get("target_checkpoint", spec.target)
        size = int(spec.payload.get("checkpoint_bytes", 0))
        decisions = self._tiered.put(key, size, StorageTier.REMOTE, now)
        return [TierMove(d.key, d.from_tier, d.to_tier) for d in decisions]

    def _demote_current(self, spec, now) -> list[TierMove]:
        # The old weights drop to node-local DRAM (warm) so rollback can
        # restore them quickly, rather than being pushed to REMOTE alongside
        # the freshly staged target checkpoint.
        key = spec.source_checkpoint or "current"
        decisions = self._tiered.demote(key, StorageTier.DRAM, now)
        return [TierMove(d.key, d.from_tier, d.to_tier) for d in decisions]

    def _promote_target(self, spec, now) -> list[TierMove]:
        key = spec.payload.get("target_checkpoint", spec.target)
        decisions = self._tiered.promote(key, StorageTier.HBM, now)
        return [TierMove(d.key, d.from_tier, d.to_tier) for d in decisions]

    def _restore_metadata(self, spec, actor_id, script, sm) -> None:
        entries = list(spec.payload.get("prefix_entries", ()))
        base_hash = spec.payload.get("base_hash")
        adapter = spec.payload.get("adapter")

        for entry in entries:
            self._prefix.register(entry)
            script.prefix_register.append(entry)
        if base_hash:
            self._weights.register(actor_id, base_hash, adapter)
            script.weight_register.append((actor_id, base_hash, adapter))

        def undo() -> None:
            for entry in entries:
                self._prefix.unregister(entry)
            if base_hash:
                self._weights.unregister(actor_id, base_hash, adapter)

        sm.register_compensation(undo)

    # ------------------------------------------------------------- rollback
    def _register_tier_undo(self, sm, moves: list[TierMove]) -> None:
        """Undo tier bookkeeping: drop whatever these moves admitted/promoted.

        The physical tensor moves are the data plane's concern; this only
        removes the entries the orchestrator added to the tier cache.
        """

        def undo() -> None:
            for move in reversed(moves):
                # Only entries this orchestrator *created* are removed; moves
                # caused by cascade eviction are not re-admitted here (the data
                # plane replays those from its own compensation list).
                if move.to_tier is not None:
                    self._tiered.remove(move.key)

        sm.register_compensation(undo)