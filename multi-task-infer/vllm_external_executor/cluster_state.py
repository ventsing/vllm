# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Cluster state model and global scheduler (pure logic, no Ray dependency).

This module defines the domain model for a multi-node actor pool:

- ``NodeInfo``: cross-node registration record (IP, hostname, fault domain,
  GPU capacity, liveness heartbeat).
- ``ActorRegistration``: per-actor record (node, device, state, lease,
  liveness heartbeat).
- ``GlobalScheduler``: fault-domain-aware selection of actors for a lease,
  used by :class:`ActorPoolManager.acquire`.

Keeping these algorithms free of Ray imports makes them unit-testable
without a running cluster and keeps the pool manager thin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

# Actor states are imported lazily in helpers to avoid a hard dependency on
# the actor module (which itself imports torch). The scheduler compares
# against the string values declared here, which mirror ActorState.
IDLE_STATE = "idle"

# Detached registry actor identity. The namespace must be shared by every
# process (owner and attached clients); the default/anonymous namespace is
# per-process, so an owner-created detached actor would be invisible to a
# worker that looks it up by name alone.
REGISTRY_ACTOR_NAME = "external-executor-node-registry"
REGISTRY_ACTOR_NAMESPACE = "external-executor"


@dataclass
class NodeInfo:
    """Cross-node registration record maintained by the central registry.

    One instance per physical Ray node participating in the pool.

    Attributes:
        node_id: Ray runtime node id (hex string).
        ip: Node IP address (from ``ray.util.get_node_ip_address``).
        hostname: Reserved for reporting; may be empty on some platforms.
        fault_domain: First-class failure-domain label. Defaults to the node
            id (each node is its own failure domain); may be set to a rack /
            zone / region label so the scheduler can spread a lease across
            independent failure domains.
        total_gpus: Number of GPUs/NPUs this node contributes to the pool.
        free_gpus: Idle GPUs, updated on acquire/release.
        registered_at: ``time.time()`` at registration.
        last_heartbeat: ``time.time()`` of the last node-level heartbeat.
    """

    node_id: str
    ip: str
    fault_domain: str
    total_gpus: int
    free_gpus: int
    hostname: str = ""
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)


@dataclass
class ActorRegistration:
    """Per-actor record maintained by the central registry.

    Attributes:
        actor_id: Stable id for the actor (``{node_id}-gpu{device_id}`` or a
            user-supplied id). Ray handles are not stable across rebuilds, so
            the registry keys on this id.
        node_id: Ray node id hosting the actor.
        device_id: Physical device index bound by the actor.
        fault_domain: Inherited from the hosting node.
        state: Current lifecycle state (see ``ActorState``).
        lease_id: Id of the vLLM instance currently leasing this actor, or
            ``None`` when idle.
        lease_generation: Monotonic counter bumped on every grant. A stale
            release (older generation) must not disturb the current lease.
        last_heartbeat: ``time.time()`` of the last successful heartbeat.
        registered_at: ``time.time()`` at registration.
    """

    actor_id: str
    node_id: str
    device_id: int
    fault_domain: str
    state: str = IDLE_STATE
    lease_id: str | None = None
    lease_generation: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)


class GlobalScheduler:
    """Fault-domain-aware actor selection for a lease.

    Pure logic: consumes registration snapshots, returns selected actor ids.
    The pool manager owns side effects (marking ``LEASED``, recording the
    lease id).

    Selection priorities, in order:

    1. Hard ``fault_domain_constraint`` (per-domain minimum/maximum counts)
       when provided.
    2. Spread: avoid concentrating a lease in one failure domain, so a single
       node/rack failure degrades the running instance at most partially.
    3. Load balance: prefer domains with more free actors.
    4. Driver-node preference (optional tie-break).
    """

    @staticmethod
    def group_idle_by_domain(
        actors: Iterable[ActorRegistration],
    ) -> dict[str, list[ActorRegistration]]:
        """Group idle actors by fault domain, preserving registration order."""
        groups: dict[str, list[ActorRegistration]] = {}
        for actor in actors:
            if actor.state != IDLE_STATE:
                continue
            groups.setdefault(actor.fault_domain, []).append(actor)
        return groups

    @classmethod
    def select_actors(
        cls,
        actors: Iterable[ActorRegistration],
        nodes: Iterable[NodeInfo],
        world_size: int,
        fault_domain_constraint: dict[str, int] | None = None,
        prefer_driver_node: str | None = None,
        require_contiguous_devices: bool = False,
    ) -> list[str]:
        """Select ``world_size`` idle actor ids for a lease.

        Args:
            actors: Snapshot of actor registrations to choose from.
            nodes: Snapshot of node registrations (for liveness filtering and
                driver-node preference).
            world_size: Number of actors to select.
            fault_domain_constraint: Optional hard per-domain counts. Only
                these domains are considered, and each domain contributes at
                most its mandated count. Not supported together with
                ``require_contiguous_devices``.
            prefer_driver_node: Optional node id; its domain acts as the first
                tie-break when two domains have equivalent free counts.
            require_contiguous_devices: When True, the whole lease must come
                from one node as a *contiguous* run of device ids (e.g. TP=2 ->
                {0,1}/{2,3}, TP=4 -> {0-3}/{4-7}), required by accelerators
                whose inter-device links (e.g. Ascend HCCS) only span adjacent
                devices.

        Returns:
            List of selected actor ids.

        Raises:
            ValueError: If ``world_size`` <= 0, a constraint references an
                unknown / dead domain, or a constraint is combined with
                ``require_contiguous_devices``.
            RuntimeError: If fewer than ``world_size`` idle actors satisfy the
                constraints.
        """
        if world_size <= 0:
            raise ValueError("world_size must be positive")

        alive_nodes = {n.node_id for n in nodes}

        if require_contiguous_devices:
            if fault_domain_constraint is not None:
                raise ValueError(
                    "require_contiguous_devices does not support "
                    "fault_domain_constraint"
                )
            selected = cls._select_contiguous(
                actors,
                alive_nodes,
                world_size,
                prefer_driver_node,
            )
        elif fault_domain_constraint is not None:
            selected = cls._select_with_constraint(
                actors,
                alive_nodes,
                world_size,
                fault_domain_constraint,
            )
        else:
            selected = cls._select_spread(
                actors,
                alive_nodes,
                world_size,
                prefer_driver_node,
            )

        if len(selected) < world_size:
            raise RuntimeError(
                f"Not enough idle actors: {len(selected)} < {world_size} "
                f"(constraint={fault_domain_constraint})"
            )
        return selected

    @classmethod
    def _select_with_constraint(
        cls,
        actors: Iterable[ActorRegistration],
        alive_nodes: set[str],
        world_size: int,
        constraint: dict[str, int],
    ) -> list[str]:
        groups = cls.group_idle_by_domain(actors)
        if not constraint:
            return []

        total = 0
        per_domain: dict[str, int] = {}
        for domain, count in constraint.items():
            if count < 0:
                raise ValueError(f"Negative constraint for domain {domain}")
            per_domain[domain] = count
            total += count

        if total < world_size:
            raise RuntimeError(
                f"Constraint sum ({total}) < world_size ({world_size})"
            )

        selected: list[str] = []
        # Driver-node domain first for deterministic output.
        for domain, count in per_domain.items():
            candidates = [
                a for a in groups.get(domain, [])
                if a.node_id in alive_nodes
            ]
            if len(candidates) < count:
                raise RuntimeError(
                    f"Not enough idle actors in domain {domain}: "
                    f"{len(candidates)} < {count}"
                )
            selected.extend(a.actor_id for a in candidates[:count])

        return selected[:world_size]

    @classmethod
    def _select_spread(
        cls,
        actors: Iterable[ActorRegistration],
        alive_nodes: set[str],
        world_size: int,
        prefer_driver_node: str | None,
    ) -> list[str]:
        groups = cls.group_idle_by_domain(actors)
        # Drop actors on dead nodes.
        groups = {
            domain: [a for a in group if a.node_id in alive_nodes]
            for domain, group in groups.items()
            if group
        }
        groups = {d: g for d, g in groups.items() if g}

        if not groups:
            return []

        def domain_key(domain: str) -> tuple[int, int]:
            """(driver-node-priority, -free_count) for ordering."""
            driver_priority = 0
            if prefer_driver_node is not None:
                # A domain is "driver" if any of its actors live on the
                # driver node. Prefer it by giving it the smallest key.
                if any(a.node_id == prefer_driver_node for a in groups[domain]):
                    driver_priority = 0
                else:
                    driver_priority = 1
            return (driver_priority, -len(groups[domain]))

        domains = sorted(groups, key=domain_key)

        # Greedy spread: repeatedly take from the domain with the fewest
        # selected actors so far that still has free actors, breaking ties by
        # the ordering above (driver first, then most-free).
        selected: list[str] = []
        taken: dict[str, int] = {d: 0 for d in domains}
        while len(selected) < world_size:
            best: str | None = None
            for domain in domains:
                if taken[domain] >= len(groups[domain]):
                    continue
                if best is None:
                    best = domain
                    continue
                # Prefer the domain with fewer selections so far.
                if taken[domain] < taken[best]:
                    best = domain
            if best is None:
                break
            selected.append(groups[best][taken[best]].actor_id)
            taken[best] += 1

        return selected

    @classmethod
    def _select_contiguous(
        cls,
        actors: Iterable[ActorRegistration],
        alive_nodes: set[str],
        world_size: int,
        prefer_driver_node: str | None,
    ) -> list[str]:
        """Select a contiguous run of ``world_size`` devices on one node.

        Tensor parallelism within one node requires adjacent devices because
        the high-bandwidth inter-device fabric (e.g. Ascend HCCS) links only
        neighbouring devices. Idle actors are grouped per node, sorted by
        device id, split into ascending runs, and the run with the lowest
        driver-priority then lowest starting device id is chosen.
        """
        by_node: dict[str, list[ActorRegistration]] = {}
        for actor in actors:
            if actor.state != IDLE_STATE or actor.node_id not in alive_nodes:
                continue
            by_node.setdefault(actor.node_id, []).append(actor)

        best: list[ActorRegistration] | None = None
        best_key: tuple[int, int] | None = None
        for node_id, node_actors in by_node.items():
            for run in cls._contiguous_runs(node_actors):
                if len(run) < world_size:
                    continue
                candidate = run[:world_size]
                driver_priority = 0 if node_id == prefer_driver_node else 1
                key = (driver_priority, candidate[0].device_id)
                if best_key is None or key < best_key:
                    best_key = key
                    best = candidate
        return [a.actor_id for a in best] if best else []

    @staticmethod
    def _contiguous_runs(
        actors: Iterable[ActorRegistration],
    ) -> list[list[ActorRegistration]]:
        """Split actors into runs of consecutive device ids."""
        runs: list[list[ActorRegistration]] = []
        run: list[ActorRegistration] = []
        prev: int | None = None
        for actor in sorted(actors, key=lambda a: a.device_id):
            if prev is not None and actor.device_id != prev + 1:
                runs.append(run)
                run = []
            run.append(actor)
            prev = actor.device_id
        if run:
            runs.append(run)
        return runs


def actor_resource_kwargs(device_key: str) -> dict:
    """Reserve one platform accelerator for a Ray worker actor."""
    if not device_key:
        raise ValueError("The current platform does not define a Ray device resource")
    if device_key == "GPU":
        return {"num_cpus": 0, "num_gpus": 1}
    return {"num_cpus": 0, "num_gpus": 0, "resources": {device_key: 1}}
