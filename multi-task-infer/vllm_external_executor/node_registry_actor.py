# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the VLLM project

"""
NodeRegistryActor - centralized cross-node registry, heartbeat and view.

This Ray actor is the single source of truth for the cluster state of the
actor pool:

- Registration of nodes and their actors (IP, hostname, fault domain, GPUs).
- Liveness heartbeats for actors and nodes.
- A unified resource view across all nodes (global snapshot).
- Dead-actor / dead-node detection based on heartbeat staleness.

The actor is intentionally stateless with respect to scheduling *policy*:
selection logic lives in :class:`GlobalScheduler` (``cluster_state.py``) and is
invoked by :class:`ActorPoolManager` with the snapshots returned here.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from vllm_external_executor.cluster_state import ActorRegistration, NodeInfo

logger = logging.getLogger(__name__)


class NodeRegistryActor:
    """Central cluster registry.

    Methods are serialized by Ray's actor model, so no external locking is
    required; a re-entrant lock is still held as a defensive measure for any
    re-entrancy (e.g. a callback invoking the registry again).
    """

    def __init__(self):
        self._nodes: dict[str, NodeInfo] = {}
        self._actors: dict[str, ActorRegistration] = {}
        self._node_actors: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ nodes
    def register_node(
        self,
        node_id: str,
        ip: str,
        fault_domain: str | None = None,
        total_gpus: int = 0,
        hostname: str = "",
    ) -> None:
        """Register or refresh a node record."""
        with self._lock:
            fault_domain = fault_domain or node_id
            if node_id in self._nodes:
                info = self._nodes[node_id]
                info.ip = ip
                info.hostname = hostname
                info.fault_domain = fault_domain
                info.total_gpus = total_gpus
                info.last_heartbeat = time.time()
            else:
                info = NodeInfo(
                    node_id=node_id,
                    ip=ip,
                    hostname=hostname,
                    fault_domain=fault_domain,
                    total_gpus=total_gpus,
                    free_gpus=total_gpus,
                )
                self._nodes[node_id] = info
                self._node_actors.setdefault(node_id, set())
            logger.info(
                "Registered node %s (ip=%s domain=%s gpus=%d)",
                node_id, ip, fault_domain, total_gpus,
            )

    def node_heartbeat(self, node_id: str) -> None:
        """Refresh a node's liveness timestamp."""
        with self._lock:
            info = self._nodes.get(node_id)
            if info is not None:
                info.last_heartbeat = time.time()

    def unregister_node(self, node_id: str) -> list[str]:
        """Remove a node and return its actor ids (for caller-side rebuild)."""
        with self._lock:
            self._nodes.pop(node_id, None)
            actor_ids = sorted(self._node_actors.pop(node_id, ()))
            for actor_id in actor_ids:
                self._actors.pop(actor_id, None)
            return actor_ids

    # ----------------------------------------------------------------- actors
    def register_actor(
        self,
        actor_id: str,
        node_id: str,
        device_id: int,
        fault_domain: str | None = None,
    ) -> None:
        """Register or refresh an actor record."""
        with self._lock:
            node = self._nodes.get(node_id)
            fault_domain = fault_domain or (node.fault_domain if node else node_id)
            self._actors[actor_id] = ActorRegistration(
                actor_id=actor_id,
                node_id=node_id,
                device_id=device_id,
                fault_domain=fault_domain,
            )
            self._node_actors.setdefault(node_id, set()).add(actor_id)
            logger.info(
                "Registered actor %s (node=%s device=%d domain=%s)",
                actor_id, node_id, device_id, fault_domain,
            )

    def heartbeat(self, actor_id: str, state: str | None = None) -> bool:
        """Record an actor heartbeat; returns False if the actor is unknown."""
        with self._lock:
            reg = self._actors.get(actor_id)
            if reg is None:
                return False
            reg.last_heartbeat = time.time()
            if state is not None:
                reg.state = state
            return True

    def set_actor_state(
        self,
        actor_id: str,
        state: str,
        lease_id: str | None = None,
    ) -> None:
        """Update lifecycle state and optional lease id."""
        with self._lock:
            reg = self._actors.get(actor_id)
            if reg is None:
                logger.warning("set_actor_state on unknown actor %s", actor_id)
                return
            reg.state = state
            if lease_id is not None:
                reg.lease_id = lease_id
            if state == "idle":
                reg.lease_id = None
            self._sync_free_gpus(reg.node_id)

    def unregister_actor(self, actor_id: str) -> None:
        """Remove an actor record (e.g. after rebuild)."""
        with self._lock:
            reg = self._actors.pop(actor_id, None)
            if reg is not None:
                actors = self._node_actors.get(reg.node_id)
                if actors is not None:
                    actors.discard(actor_id)
                self._sync_free_gpus(reg.node_id)

    # ------------------------------------------------------------------ view
    def list_nodes(self) -> list[NodeInfo]:
        with self._lock:
            return list(self._nodes.values())

    def list_actors(self) -> list[ActorRegistration]:
        with self._lock:
            return list(self._actors.values())

    def get_global_view(self) -> dict[str, Any]:
        """Unified cross-node resource snapshot.

        Returns a JSON-serializable dict aggregating per-node free/total GPU
        counts and per-domain actor distribution, plus per-actor state.
        """
        with self._lock:
            nodes = []
            for node in self._nodes.values():
                actor_states = {}
                for actor_id in self._node_actors.get(node.node_id, ()):
                    reg = self._actors.get(actor_id)
                    if reg is not None:
                        actor_states[reg.state] = actor_states.get(reg.state, 0) + 1
                nodes.append({
                    "node_id": node.node_id,
                    "ip": node.ip,
                    "hostname": node.hostname,
                    "fault_domain": node.fault_domain,
                    "total_gpus": node.total_gpus,
                    "free_gpus": node.free_gpus,
                    "actor_states": actor_states,
                    "last_heartbeat": node.last_heartbeat,
                })
            domains: dict[str, dict[str, int]] = {}
            for reg in self._actors.values():
                d = domains.setdefault(reg.fault_domain, {"idle": 0, "total": 0})
                d["total"] += 1
                if reg.state == "idle":
                    d["idle"] += 1
            return {
                "nodes": nodes,
                "fault_domains": domains,
                "num_nodes": len(self._nodes),
                "num_actors": len(self._actors),
            }

    # ---------------------------------------------------------------- liveness
    def detect_dead_actors(self, timeout: float) -> list[str]:
        """Return actor ids whose heartbeat is older than ``timeout`` seconds."""
        now = time.time()
        with self._lock:
            dead = [
                actor_id
                for actor_id, reg in self._actors.items()
                if now - reg.last_heartbeat > timeout
            ]
        return dead

    def detect_dead_nodes(self, timeout: float) -> list[str]:
        """Return node ids whose heartbeat is older than ``timeout`` seconds."""
        now = time.time()
        with self._lock:
            dead = [
                node_id
                for node_id, info in self._nodes.items()
                if now - info.last_heartbeat > timeout
            ]
        return dead

    def mark_node_failed(self, node_id: str) -> list[str]:
        """Mark a node as failed; returns its actor ids for rebuild.

        The node and its actors are removed from the registry. Callers use the
        returned actor ids to rebuild equivalent actors on healthy nodes.
        """
        with self._lock:
            actor_ids = sorted(self._node_actors.pop(node_id, ()))
            for actor_id in actor_ids:
                self._actors.pop(actor_id, None)
            self._nodes.pop(node_id, None)
            logger.warning("Marked node %s failed (%d actors)", node_id, len(actor_ids))
            return actor_ids

    def mark_actor_failed(self, actor_id: str) -> None:
        """Remove a single failed actor from the registry."""
        with self._lock:
            reg = self._actors.pop(actor_id, None)
            if reg is not None:
                actors = self._node_actors.get(reg.node_id)
                if actors is not None:
                    actors.discard(actor_id)
                self._sync_free_gpus(reg.node_id)

    def _sync_free_gpus(self, node_id: str) -> None:
        """Recompute a node's free_gpus from its idle actor count."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        idle = 0
        for actor_id in self._node_actors.get(node_id, ()):
            reg = self._actors.get(actor_id)
            if reg is not None and reg.state == "idle":
                idle += 1
        node.free_gpus = idle


def create_registry_actor() -> "Any":
    """Create a detached Ray NodeRegistryActor.

    Detached actors survive the driver process, which is important for the
    multi-tenant pool: the registry must outlive any single client.
    """
    import ray

    return (
        ray.remote(NodeRegistryActor)
        .options(name="external-executor-node-registry", lifetime="detached")
        .remote()
    )
