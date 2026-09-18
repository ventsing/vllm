# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
ActorPoolManager - Manages a pool of pre-started Ray Actors.

This module provides the ActorPoolManager class which handles:
- Pre-starting Ray Actors bound to GPU/NPU devices
- Managing actor lifecycle (IDLE -> LEASED -> RUNNING -> RELEASED)
- Cross-node registration + heartbeat + unified resource view (via
  :class:`NodeRegistryActor`)
- Fault-domain-aware global scheduling (via :class:`GlobalScheduler`)
- Per-node failure isolation (dead-node detection and actor rebuild)
- Acquiring and releasing actors for vLLM instances
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import TYPE_CHECKING

from vllm.platforms import current_platform

from vllm_external_executor.cluster_state import actor_resource_kwargs
from vllm_external_executor.external_worker_actor import ActorState, ExternalWorkerActor

if TYPE_CHECKING:
    import ray

logger = logging.getLogger(__name__)

# Ray RPC timeout: prevents a dead node from hanging pool operations forever.
RPC_TIMEOUT = 30.0
DEFAULT_HEARTBEAT_INTERVAL = 5.0
DEFAULT_HEARTBEAT_TIMEOUT = 30.0
# Node-level liveness is more forgiving than actor-level: a single actor may
# flap, but a node is only declared dead when its *node* heartbeat stalls.
DEFAULT_NODE_HEARTBEAT_TIMEOUT = 60.0
MAX_HEARTBEAT_FAILURES = 3
# Name of the detached registry actor; a second process attaches with
# ray.get_actor(REGISTRY_ACTOR_NAME) to share a pre-started pool.
REGISTRY_ACTOR_NAME = "external-executor-node-registry"


class ActorPoolManager:
    """
    Manages a pool of pre-started Ray Actors across one or more nodes.

    The pool pre-starts actors that bind to GPU/NPU devices and pre-import
    common libraries. These actors can then be acquired by ExternalExecutor
    to serve as workers in vLLM instances.

    Distributed capabilities:

    - Cross-node registry: every actor and node is registered with a central
      :class:`NodeRegistryActor` holding IP, fault domain, GPU capacity and
      liveness heartbeat, exposed through :meth:`get_global_view`.
    - Global scheduling: :meth:`acquire` spreads a lease across independent
      fault domains (nodes by default) instead of grabbing the first N idle
      actors, improving resilience to a single-node failure.
    - Node-failure isolation: a background heartbeat thread marks unresponsive
      actors failed; :meth:`recover_node` removes a dead node's actors and
      rebuilds them on healthy nodes.

    Example:
        pool = ActorPoolManager()
        pool.pre_start(num_actors=8, devices_per_node=[0, 1, 2, 3, 4, 5, 6, 7])

        actors = pool.acquire(tp_size=4, pp_size=2)

        # ... use actors with ExternalExecutor ...

        pool.release(actors)
    """

    def __init__(self):
        """Initialize the ActorPoolManager."""
        self.actors: list = []  # list[ray.actor.ActorHandle]
        self.states: dict[int, ActorState] = {}
        self.node_mapping: dict[str, list[int]] = {}  # node_id -> pool indices
        self.placement_group = None
        self.cache_manager = None  # CacheManagerActor handle
        self.registry = None  # NodeRegistryActor handle
        self.fault_domain: str | None = None
        self._initialized = False

        # Stable identity maps: actor_id <-> handle <-> pool index.
        self.actor_ids: dict[int, str] = {}          # pool index -> actor_id
        self._actor_id_to_idx: dict[str, int] = {}   # actor_id -> pool index
        # Registration snapshot cached for rebuilds (survives registry edits).
        self._actor_regs: dict[str, dict] = {}       # actor_id -> {node/device}

        # Heartbeat machinery.
        self._hb_thread: threading.Thread | None = None
        self._hb_stop = threading.Event()
        self._hb_failures: dict[int, int] = {}       # pool index -> failures
        self.heartbeat_interval = DEFAULT_HEARTBEAT_INTERVAL
        self.heartbeat_timeout = DEFAULT_HEARTBEAT_TIMEOUT
        self.node_heartbeat_timeout = DEFAULT_NODE_HEARTBEAT_TIMEOUT

        # Active leases: lease_id -> [actor_id] (for future accounting).
        self._leases: dict[str, list[str]] = {}

        # Idempotency registry for actor migrations (migrate_actor).
        from vllm_external_executor.migration import MigrationIdempotencyRegistry
        self._migration_registry = MigrationIdempotencyRegistry()

        # Elastic autoscaling (decision layer + injected execution callbacks).
        self.autoscaler = None  # vllm_external_executor.autoscaling.Autoscaler
        self._scale_up_fn = None
        self._scale_down_fn = None

    # ==================================================================== start
    def pre_start(
        self,
        num_actors: int,
        devices_per_node: list[int],
        placement_group=None,
        warmup_distributed: bool = True,
        shared_cache_dir: str | None = None,
        enable_cache_compression: bool = True,
        registry=None,
        fault_domain: str | None = None,
        strategy: str = "pack",
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT,
        node_heartbeat_timeout: float = DEFAULT_NODE_HEARTBEAT_TIMEOUT,
    ) -> None:
        """
        Pre-start actors and bind them to GPU/NPU devices.

        Args:
            num_actors: Number of actors to pre-start.
            devices_per_node: List of GPU/NPU device IDs, one per actor. For a
                multi-node pool, repeat local device indices per node (e.g.
                ``[0,1,2,3, 0,1,2,3]`` for two 4-GPU nodes) and use
                ``strategy="spread"``.
            placement_group: Optional Ray placement group.
            warmup_distributed: Whether to warm up NCCL/HCCl at init time.
            shared_cache_dir: NFS directory for large compilation caches.
            enable_cache_compression: Whether to compress large caches.
            registry: Optional existing NodeRegistryActor handle (multi-tenant
                pools share one registry). One is created when omitted.
            fault_domain: Optional failure-domain label applied to all nodes.
                Defaults to the node id, so each node is its own fault domain.
            strategy: Placement group strategy ("pack" or "spread").
            heartbeat_interval: Seconds between heartbeat rounds.
            heartbeat_timeout: Seconds before an actor is considered dead.
            node_heartbeat_timeout: Seconds before a whole node is considered
                dead (auto-triggers :meth:`recover_node`).
        """
        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        if num_actors != len(devices_per_node):
            raise ValueError(
                f"num_actors ({num_actors}) must equal len(devices_per_node) "
                f"({len(devices_per_node)})"
            )

        if self._initialized:
            logger.warning("ActorPoolManager already initialized, shutting down first")
            self.shutdown()

        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.node_heartbeat_timeout = node_heartbeat_timeout
        self.fault_domain = fault_domain

        logger.info(
            "Pre-starting %d actors on devices %s (strategy=%s)",
            num_actors, devices_per_node, strategy,
        )

        device_key = current_platform.ray_device_key
        resource_kwargs = actor_resource_kwargs(device_key)

        # 1. Cache manager actor.
        from vllm_external_executor.cache_manager_actor import (
            create_cache_manager_actor,
        )
        self.cache_manager = create_cache_manager_actor(
            shared_cache_dir=shared_cache_dir,
            enable_compression=enable_cache_compression,
        )

        # 2. Registry actor (shared across tenants when injected).
        if registry is None:
            from vllm_external_executor.node_registry_actor import (
                create_registry_actor,
            )
            registry = create_registry_actor()
        self.registry = registry

        # 3. Placement group.
        if placement_group is None:
            self.placement_group = ray.util.placement_group(
                bundles=[{device_key: 1}] * num_actors + [{"CPU": 1}],
                strategy="PACK" if strategy == "pack" else "SPREAD",
            )
            ray.get(self.placement_group.ready(), timeout=RPC_TIMEOUT)
        else:
            self.placement_group = placement_group

        # 4. Runtime env + actor creation.
        runtime_env = RuntimeEnv(env_vars={
            "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "1",
        })

        for i, device_id in enumerate(devices_per_node):
            actor = (
                ray.remote(ExternalWorkerActor)
                .options(
                    **resource_kwargs,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.placement_group,
                        placement_group_bundle_index=i,
                    ),
                    runtime_env=runtime_env,
                )
                .remote(
                    device_id=device_id,
                    warmup_distributed=warmup_distributed,
                )
            )
            self.actors.append(actor)
            self.states[i] = ActorState.IDLE

        # 5. Wait for readiness and collect node placement.
        logger.info("Waiting for actors to be ready...")
        ray.get(
            [actor.wait_for_ready.remote() for actor in self.actors],
            timeout=RPC_TIMEOUT,
        )
        infos = ray.get(
            [actor.get_info.remote() for actor in self.actors],
            timeout=RPC_TIMEOUT,
        )

        # 6. Build node map + register nodes with IP/hostname.
        node_ip = self._collect_node_addresses()
        self.node_mapping = {}
        for i, info in enumerate(infos):
            node_id = info["node_id"]
            self.node_mapping.setdefault(node_id, []).append(i)

        for node_id in self.node_mapping:
            total_gpus = len(self.node_mapping[node_id])
            ray.get(
                self.registry.register_node.remote(
                    node_id=node_id,
                    ip=node_ip.get(node_id, ""),
                    fault_domain=self.fault_domain,
                    total_gpus=total_gpus,
                    hostname=node_ip.get(node_id + ":host", ""),
                ),
                timeout=RPC_TIMEOUT,
            )

        # 7. Backfill stable identity + register actors.
        per_device_count: dict[str, int] = {}
        for i, info in enumerate(infos):
            node_id = info["node_id"]
            device_id = info["device_id"]
            domain = self.fault_domain or node_id
            # De-duplicate actor ids when the same (node, device) recurs.
            key = f"{node_id}:{device_id}"
            per_device_count[key] = per_device_count.get(key, 0) + 1
            actor_id = f"{node_id}-g{device_id}"
            if per_device_count[key] > 1:
                actor_id += f"-{per_device_count[key]}"

            ray.get(
                self.actors[i].configure_pool_identity.remote(
                    actor_id, domain
                ),
                timeout=RPC_TIMEOUT,
            )
            ray.get(
                self.registry.register_actor.remote(
                    actor_id=actor_id,
                    node_id=node_id,
                    device_id=device_id,
                    fault_domain=domain,
                    handle=self.actors[i],
                ),
                timeout=RPC_TIMEOUT,
            )
            self.actor_ids[i] = actor_id
            self._actor_id_to_idx[actor_id] = i
            self._actor_regs[actor_id] = {
                "node_id": node_id,
                "device_id": device_id,
                "fault_domain": domain,
                "warmup_distributed": warmup_distributed,
            }

        self._initialized = True
        self._start_heartbeat()
        logger.info(
            "Actor pool ready: %d actors on %d nodes",
            num_actors, len(self.node_mapping),
        )

    # ================================================================== attach
    def attach(
        self,
        registry=None,
        registry_name: str = REGISTRY_ACTOR_NAME,
    ) -> None:
        """Attach to a pre-started pool owned by another process.

        Rebuilds the local actor index from the central registry (which stores
        both the registrations and their Ray handles) so this process can call
        :meth:`acquire` / :meth:`release` against the same actors without
        re-running :meth:`pre_start`. Liveness heartbeats stay with the owning
        process; an attached client only schedules work and releases it.

        Args:
            registry: Optional registry handle; resolved by name when omitted.
            registry_name: Name of the detached registry actor.

        Raises:
            RuntimeError: If the registry holds no actors (the owner has not
                finished ``pre_start`` yet, or already shut down).
        """
        import ray

        if registry is None:
            registry = ray.get_actor(registry_name)
        self.registry = registry

        regs = ray.get(registry.list_actors.remote(), timeout=RPC_TIMEOUT)
        handles = ray.get(
            registry.get_actor_handles.remote(), timeout=RPC_TIMEOUT
        )
        if not regs or not handles:
            raise RuntimeError(
                "Registry has no actors to attach to; run pre_start first"
            )

        # Deterministic local indices: sort by actor_id so every attached
        # process lands on the same (index -> actor_id) map as the owner.
        regs = sorted(regs, key=lambda r: r.actor_id)
        self.actors = []
        self.actor_ids = {}
        self._actor_id_to_idx = {}
        self._actor_regs = {}
        self.states = {}
        self.node_mapping = {}
        for i, reg in enumerate(regs):
            handle = handles.get(reg.actor_id)
            if handle is None:
                logger.warning(
                    "Actor %s has no stored handle; skipped", reg.actor_id
                )
                continue
            self.actors.append(handle)
            self.actor_ids[i] = reg.actor_id
            self._actor_id_to_idx[reg.actor_id] = i
            self._actor_regs[reg.actor_id] = {
                "node_id": reg.node_id,
                "device_id": reg.device_id,
                "fault_domain": reg.fault_domain,
            }
            try:
                self.states[i] = ActorState(reg.state)
            except ValueError:
                self.states[i] = ActorState.IDLE
            self.node_mapping.setdefault(reg.node_id, []).append(i)

        self._initialized = True
        logger.info(
            "Attached to pool: %d actors across %d nodes",
            len(self.actors), len(self.node_mapping),
        )

    def _collect_node_addresses(self) -> dict[str, str]:
        """Map node id -> (ip, hostname) using the Ray cluster view."""
        import ray
        address: dict[str, str] = {}
        try:
            for node in ray.nodes():
                nid = node.get("NodeID")
                if nid is None:
                    continue
                address[nid] = node.get("NodeManagerAddress", "")
                host = node.get("NodeName", "")
                address[nid + ":host"] = host
        except Exception as e:  # pragma: no cover - best effort
            logger.warning("Failed to collect node addresses: %s", e)
        return address

    # ================================================================== acquire
    def acquire(
        self,
        tp_size: int,
        pp_size: int,
        fault_domain_constraint: dict[str, int] | None = None,
        prefer_driver_node: bool = True,
        node_constraint: dict[str, int] | None = None,
    ) -> list:
        """
        Acquire idle actors for a vLLM instance, spread across fault domains.

        Args:
            tp_size: Tensor Parallel size.
            pp_size: Pipeline Parallel size.
            fault_domain_constraint: Optional per-fault-domain actor counts
                (e.g. ``{"node-a": 2, "node-b": 2}``). Only these domains are
                considered.
            prefer_driver_node: Prefer the driver node when counts tie.
            node_constraint: Deprecated alias of ``fault_domain_constraint``
                (default fault domain is the node id, so the two coincide).

        Returns:
            List of acquired actor handles.

        Raises:
            RuntimeError: If not enough idle actors satisfy the constraints.
        """
        import ray

        if not self._initialized:
            raise RuntimeError("Pool not initialized; call pre_start() first")

        if fault_domain_constraint is None and node_constraint is not None:
            fault_domain_constraint = node_constraint

        world_size = tp_size * pp_size
        prefer = (
            ray.get_runtime_context().get_node_id() if prefer_driver_node else None
        )
        lease_id = str(uuid.uuid4())

        # Atomic select + grant in the registry: concurrent acquirers cannot
        # grab the same actor and a shortfall leaves no partial lease.
        selected_ids = ray.get(
            self.registry.try_acquire.remote(
                world_size,
                lease_id,
                fault_domain_constraint=fault_domain_constraint,
                prefer_driver_node=prefer,
            ),
            timeout=RPC_TIMEOUT,
        )
        if len(selected_ids) < world_size:
            raise RuntimeError(
                f"Not enough idle actors: {len(selected_ids)} < {world_size} "
                f"(constraint={fault_domain_constraint})"
            )

        selected: list = []
        for actor_id in selected_ids:
            idx = self._actor_id_to_idx[actor_id]
            selected.append(self.actors[idx])
            self.states[idx] = ActorState.LEASED

        self._leases[lease_id] = selected_ids
        logger.info(
            "Acquired %d actors for TP=%d, PP=%d (lease=%s)",
            len(selected), tp_size, pp_size, lease_id[:8],
        )
        return selected

    # ================================================================== release
    def release(self, actors: list, lease_id: str | None = None) -> None:
        """
        Release actors back to the pool, gated on lease identity.

        Each actor is reset (worker resources torn down) *first*; only actors
        that reset cleanly are returned to idle in the registry via
        :meth:`release_actors`, which also verifies the lease still belongs to
        this caller. A delayed/duplicated release from an older task therefore
        cannot disturb a newer task holding the same actor.

        Args:
            actors: List of actor handles to release.
            lease_id: Optional lease id. When omitted, it is recovered from the
                pool's lease table (the lease owning these actors).
        """
        import ray

        actor_ids = self._actor_ids_of(actors)
        if not actor_ids:
            return
        if lease_id is None:
            lease_id = self._lease_id_for(actor_ids)
        if lease_id is None:
            # Idempotent repeat release: nothing to do.
            logger.warning("Release with no matching lease; skipping")
            return

        reset_ok: list[str] = []
        for actor in actors:
            try:
                idx = self.actors.index(actor)
            except ValueError:
                logger.warning("Actor not found in pool, skipping")
                continue

            actor_id = self.actor_ids.get(idx)
            self.states[idx] = ActorState.RELEASED
            try:
                ray.get(actor.reset.remote(), timeout=RPC_TIMEOUT)
            except Exception as e:
                # Do NOT return a failed actor to idle: isolate it.
                logger.error("Failed to reset actor %d: %s", idx, e)
                self.states[idx] = ActorState.FAILED
                if actor_id:
                    ray.get(
                        self.registry.mark_actor_failed.remote(actor_id),
                        timeout=RPC_TIMEOUT,
                    )
                continue
            self.states[idx] = ActorState.IDLE
            if actor_id:
                reset_ok.append(actor_id)

        # Only reset-clean actors are released; the registry re-validates the
        # lease_identity so a stale release cannot free a re-leased actor.
        if reset_ok:
            ray.get(
                self.registry.release_actors.remote(reset_ok, lease_id),
                timeout=RPC_TIMEOUT,
            )
        self._leases.pop(lease_id, None)
        logger.info("Released %d actors back to pool", len(reset_ok))

    def _actor_ids_of(self, actors: list) -> list[str]:
        """Map actor handles to their stable ids (skip unknown handles)."""
        ids: list[str] = []
        for actor in actors:
            try:
                idx = self.actors.index(actor)
            except ValueError:
                continue
            actor_id = self.actor_ids.get(idx)
            if actor_id:
                ids.append(actor_id)
        return ids

    def _lease_id_for(self, actor_ids: list[str]) -> str | None:
        """Return the lease id owning any of ``actor_ids``, or None."""
        wanted = set(actor_ids)
        for lease_id, ids in self._leases.items():
            if wanted & set(ids):
                return lease_id
        return None

    # =============================================================== autoscale
    def set_autoscaler(
        self,
        config,
        scale_up_fn=None,
        scale_down_fn=None,
    ) -> None:
        """
        Enable elastic autoscaling with the given policy config.

        Args:
            config: :class:`AutoscalerConfig` (watermarks, step, cooldowns,
                time windows).
            scale_up_fn: Optional ``fn(count)`` invoked on SCALE_UP; defaults
                to :meth:`_scale_up_actors` (Ray create + register).
            scale_down_fn: Optional ``fn(count)`` invoked on SCALE_DOWN;
                defaults to :meth:`_scale_down_actors` (Ray kill + unregister).
        """
        from vllm_external_executor.autoscaling import Autoscaler
        self.autoscaler = Autoscaler(config)
        self._scale_up_fn = scale_up_fn
        self._scale_down_fn = scale_down_fn

    def maybe_autoscale(self, metrics, now=None):
        """
        Sample load and nudge the pool toward the autoscaler's target size.

        Pure decision (in :class:`Autoscaler`) + execution side effects. When
        no autoscaler is configured, this is a no-op returning ``None``.

        Args:
            metrics: :class:`LoadMetrics` snapshot (queue, P99, utilization).
            now: Optional monotonic seconds (defaults to ``time.monotonic``).

        Returns:
            The :class:`AutoscaleDecision`, or ``None`` if disabled.
        """
        import time

        from vllm_external_executor.autoscaling import AutoscaleAction

        if self.autoscaler is None:
            return None
        now = now if now is not None else time.monotonic()
        current = len(self.actors)
        decision = self.autoscaler.decide(metrics, current, now)

        up = self._scale_up_fn or self._scale_up_actors
        down = self._scale_down_fn or self._scale_down_actors
        if decision.action is AutoscaleAction.SCALE_UP:
            up(decision.target_actors - current)
        elif decision.action is AutoscaleAction.SCALE_DOWN:
            down(current - decision.target_actors)

        logger.info(
            "Autoscale %s -> target=%d (reason=%s)",
            decision.action.value, decision.target_actors, decision.reason,
        )
        return decision

    def _scale_up_actors(self, count) -> None:
        """Create and register ``count`` additional idle actors (Ray side).

        New actors reuse the pool's recorded device layout (round-robin) and
        are scheduled without a placement-group bundle index, since the group
        was sized at ``pre_start``; extending the bundle set is a real-cluster
        concern left to the operator (or an injected ``scale_up_fn``).
        """
        import ray

        if count <= 0:
            return
        devices = [reg["device_id"] for reg in self._actor_regs.values()]
        if not devices:
            logger.warning("Cannot scale up: no device layout (pre_start first)")
            return

        for i in range(count):
            device_id = devices[i % len(devices)]
            actor = (
                ray.remote(ExternalWorkerActor)
                .options(**actor_resource_kwargs(current_platform.ray_device_key))
                .remote(device_id=device_id, warmup_distributed=True)
            )
            ray.get(actor.wait_for_ready.remote(), timeout=RPC_TIMEOUT)
            info = ray.get(actor.get_info.remote(), timeout=RPC_TIMEOUT)
            node_id = info["node_id"]
            domain = self.fault_domain or node_id
            actor_id = f"{node_id}-g{device_id}-{uuid.uuid4().hex[:8]}"

            ray.get(
                actor.configure_pool_identity.remote(actor_id, domain),
                timeout=RPC_TIMEOUT,
            )
            ray.get(
                self.registry.register_actor.remote(
                    actor_id=actor_id,
                    node_id=node_id,
                    device_id=device_id,
                    fault_domain=domain,
                    handle=actor,
                ),
                timeout=RPC_TIMEOUT,
            )

            idx = len(self.actors)
            self.actors.append(actor)
            self.states[idx] = ActorState.IDLE
            self.actor_ids[idx] = actor_id
            self._actor_id_to_idx[actor_id] = idx
            self._actor_regs[actor_id] = {
                "node_id": node_id,
                "device_id": device_id,
                "fault_domain": domain,
                "warmup_distributed": True,
            }
            self.node_mapping.setdefault(node_id, []).append(idx)

        logger.info("Scaled up: created %d actors", count)

    def _scale_down_actors(self, count) -> None:
        """Kill ``count`` surplus idle actors from the tail (Ray side).

        Only IDLE actors are eligible; leased/failed actors are left alone.
        Removing from the tail keeps the remaining pool indices stable (no
        shift), so ``node_mapping``/``actor_ids`` stay consistent.
        """
        import ray

        destroyed = 0
        for idx in range(len(self.actors) - 1, -1, -1):
            if destroyed >= count:
                break
            if self.states.get(idx) is not ActorState.IDLE:
                continue
            handle = self.actors[idx]
            actor_id = self.actor_ids.get(idx)
            try:
                ray.kill(handle)
            except Exception:
                logger.warning("Failed to kill actor %d", idx)
                continue
            if actor_id:
                try:
                    ray.get(
                        self.registry.unregister_actor.remote(actor_id),
                        timeout=RPC_TIMEOUT,
                    )
                except Exception:
                    pass
                reg = self._actor_regs.pop(actor_id, None)
                if reg:
                    self.node_mapping.get(reg["node_id"], []).remove(idx)
                self._actor_id_to_idx.pop(actor_id, None)
                self.actor_ids.pop(idx, None)
            self.actors.pop(idx)
            self.states.pop(idx, None)
            destroyed += 1

        logger.info("Scaled down: destroyed %d idle actors", destroyed)

    # ========================================================== health / fault
    def get_global_view(self) -> dict:
        """Return the unified cross-node resource snapshot from the registry."""
        import ray
        if self.registry is None:
            return {"nodes": [], "fault_domains": {}, "num_nodes": 0, "num_actors": 0}
        return ray.get(self.registry.get_global_view.remote(), timeout=RPC_TIMEOUT)

    def check_health(self) -> dict:
        """Detect dead actors and dead nodes via stale heartbeats."""
        import ray
        if self.registry is None:
            return {"dead_actors": [], "dead_nodes": []}
        dead_actors = ray.get(
            self.registry.detect_dead_actors.remote(self.heartbeat_timeout),
            timeout=RPC_TIMEOUT,
        )
        dead_nodes = ray.get(
            self.registry.detect_dead_nodes.remote(self.heartbeat_timeout),
            timeout=RPC_TIMEOUT,
        )
        return {"dead_actors": dead_actors, "dead_nodes": dead_nodes}

    def recover_node(self, node_id: str) -> list:
        """
        Isolate a failed node and rebuild its actors on healthy nodes.

        Removes the node and its actors from the registry (and local state),
        then rebuilds each lost actor on a healthy node. This is the
        node-failure path: a single dead node only loses its own actors, the
        rest of the pool keeps serving.

        Args:
            node_id: Ray node id of the failed node.

        Returns:
            List of (old_actor_id, new_actor_handle) tuples.
        """
        import ray

        if self.registry is None:
            raise RuntimeError("Pool not initialized")

        failed_ids = ray.get(
            self.registry.mark_node_failed.remote(node_id), timeout=RPC_TIMEOUT
        )
        logger.warning(
            "Recovering node %s: rebuilding %d actors", node_id, len(failed_ids)
        )

        rebuilt = []
        for actor_id in failed_ids:
            handle = self.rebuild_actor(actor_id)
            if handle is not None:
                rebuilt.append((actor_id, handle))

        self.node_mapping.pop(node_id, None)
        return rebuilt

    def rebuild_actor(
        self,
        actor_id: str,
        target_node_id: str | None = None,
    ):
        """
        Rebuild a lost actor on a healthy node (cross-node "migration").

        GPU actors cannot be live-migrated, so failure/负载-driven migration is
        implemented as rebuild: kill the local handle if any, create an
        equivalent actor on a healthy node, and re-register it. The old actor
        id is retired and a new id is minted for the replacement.

        Args:
            actor_id: The retired actor id.
            target_node_id: Optional preferred node for the replacement.

        Returns:
            New actor handle, or None if the actor's registration info is gone.
        """
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        reg = self._actor_regs.get(actor_id)
        if reg is None:
            logger.warning("No registration cached for actor %s", actor_id)
            return None

        old_idx = self._actor_id_to_idx.pop(actor_id, None)
        if old_idx is not None:
            old_handle = self.actors[old_idx]
            try:
                ray.kill(old_handle)
            except Exception:
                pass
            self.states[old_idx] = ActorState.FAILED

        try:
            options = actor_resource_kwargs(current_platform.ray_device_key)
            if target_node_id:
                options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                    node_id=target_node_id, soft=False
                )
            new_actor = ray.remote(ExternalWorkerActor).options(**options).remote(
                device_id=reg["device_id"],
                warmup_distributed=bool(reg.get("warmup_distributed", True)),
            )
            ray.get(new_actor.wait_for_ready.remote(), timeout=RPC_TIMEOUT)
            info = ray.get(new_actor.get_info.remote(), timeout=RPC_TIMEOUT)

            new_node_id = info["node_id"]
            new_id = f"{new_node_id}-g{reg['device_id']}-{uuid.uuid4().hex[:8]}"
            domain = reg["fault_domain"]

            ray.get(
                new_actor.configure_pool_identity.remote(new_id, domain),
                timeout=RPC_TIMEOUT,
            )
            ray.get(
                self.registry.register_actor.remote(
                    actor_id=new_id,
                    node_id=new_node_id,
                    device_id=reg["device_id"],
                    fault_domain=domain,
                    handle=new_actor,
                ),
                timeout=RPC_TIMEOUT,
            )

            # Slot the replacement into the local pool.
            if old_idx is not None and old_idx < len(self.actors):
                self.actors[old_idx] = new_actor
                self.states[old_idx] = ActorState.IDLE
                idx = old_idx
            else:
                idx = len(self.actors)
                self.actors.append(new_actor)
                self.states[idx] = ActorState.IDLE

            self.actor_ids[idx] = new_id
            self._actor_id_to_idx[new_id] = idx
            self._actor_regs[new_id] = {
                "node_id": new_node_id,
                "device_id": reg["device_id"],
                "fault_domain": domain,
                "warmup_distributed": reg.get("warmup_distributed", True),
            }
            self.node_mapping.setdefault(new_node_id, []).append(idx)
            self._actor_regs.pop(actor_id, None)
            logger.info(
                "Rebuilt actor %s as %s on node %s",
                actor_id, new_id, new_node_id,
            )
            return new_actor
        except Exception as e:
            logger.error("Failed to rebuild actor %s: %s", actor_id, e)
            return None

    def migrate_actor(
        self,
        actor_id: str,
        target_node_id: str | None = None,
        migration_id: str | None = None,
    ):
        """
        Migrate a healthy actor across nodes atomically (state machine).

        Unlike :meth:`rebuild_actor` (which kills first, for dead-actor
        recovery), this is the *proactive* migration path and follows
        "build-before-swap": a replacement actor is created and registered
        first, then the local pool is atomically switched to it, and only then
        is the old actor killed. A failure before the swap leaves the original
        actor untouched, so the migration rolls back cleanly.

        Phases: PREPARING -> GRACEFUL_PAUSE -> CHECKPOINT -> UNLOAD (build)
        -> LOAD (swap) -> RESTORE (kill old) -> COMPLETED. Idempotent on
        ``migration_id``.
        """
        import ray
        import uuid
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from vllm_external_executor.migration import (
            FlightBatchPolicy,
            MigrationPhase,
            MigrationSpec,
        )

        spec = MigrationSpec(
            migration_id=migration_id or uuid.uuid4().hex,
            target=f"actor:{actor_id}",
            policy=FlightBatchPolicy.DRAIN,
        )

        def orchestrate(sm):
            sm.transition(MigrationPhase.PREPARING)
            reg = self._actor_regs.get(actor_id)
            if reg is None:
                raise ValueError(f"Unknown actor id {actor_id}")
            old_idx = self._actor_id_to_idx.get(actor_id)

            sm.transition(MigrationPhase.GRACEFUL_PAUSE)
            if old_idx is not None and self.states.get(old_idx) in (
                ActorState.LEASED, ActorState.RUNNING,
            ):
                raise RuntimeError(
                    f"Actor {actor_id} is leased/running; release it before "
                    f"migrating"
                )

            sm.transition(MigrationPhase.CHECKPOINT)
            old_handle = self.actors[old_idx] if old_idx is not None else None

            # Build the replacement on the target node (build-before-swap).
            sm.transition(MigrationPhase.UNLOAD)
            options = actor_resource_kwargs(current_platform.ray_device_key)
            if target_node_id:
                options["scheduling_strategy"] = (
                    NodeAffinitySchedulingStrategy(
                        node_id=target_node_id, soft=False
                    )
                )
            new_actor = ray.remote(ExternalWorkerActor).options(**options).remote(
                device_id=reg["device_id"],
                warmup_distributed=bool(reg.get("warmup_distributed", True)),
            )
            ray.get(new_actor.wait_for_ready.remote(), timeout=RPC_TIMEOUT)
            info = ray.get(new_actor.get_info.remote(), timeout=RPC_TIMEOUT)
            new_node_id = info["node_id"]
            new_id = f"{new_node_id}-g{reg['device_id']}-{uuid.uuid4().hex[:8]}"
            domain = reg["fault_domain"]

            # Swap into the pool (LOAD): update local maps + registry.
            sm.transition(MigrationPhase.LOAD)
            ray.get(
                new_actor.configure_pool_identity.remote(new_id, domain),
                timeout=RPC_TIMEOUT,
            )
            ray.get(
                self.registry.register_actor.remote(
                    actor_id=new_id,
                    node_id=new_node_id,
                    device_id=reg["device_id"],
                    fault_domain=domain,
                    handle=new_actor,
                ),
                timeout=RPC_TIMEOUT,
            )
            if old_idx is not None:
                self.actors[old_idx] = new_actor
                self.states[old_idx] = ActorState.IDLE
                idx = old_idx
            else:
                idx = len(self.actors)
                self.actors.append(new_actor)
                self.states[idx] = ActorState.IDLE
            self.actor_ids[idx] = new_id
            self._actor_id_to_idx.pop(actor_id, None)
            self._actor_id_to_idx[new_id] = idx
            self._actor_regs[new_id] = dict(reg)
            self._actor_regs[new_id]["node_id"] = new_node_id
            self._actor_regs.pop(actor_id, None)
            self.node_mapping.setdefault(new_node_id, []).append(idx)
            if old_idx is not None:
                old_node = reg.get("node_id")
                node_actors = self.node_mapping.get(old_node, [])
                if old_idx in node_actors:
                    node_actors.remove(old_idx)

            # RESTORE: only now retire the old actor (swap is committed).
            sm.transition(MigrationPhase.RESTORE)
            if old_handle is not None:
                try:
                    ray.get(
                        self.registry.unregister_actor.remote(actor_id),
                        timeout=RPC_TIMEOUT,
                    )
                except Exception:
                    pass
                try:
                    ray.kill(old_handle)
                except Exception:
                    pass

            sm.transition(MigrationPhase.COMPLETED)
            logger.info(
                "Migrated actor %s -> %s on node %s (%s)",
                actor_id, new_id, new_node_id, sm.progress(),
            )

        return self._migration_registry.run(spec, orchestrate)

    # ================================================================ heartbeat
    def _start_heartbeat(self) -> None:
        """Start the background heartbeat thread."""
        if self._hb_thread is not None and self._hb_thread.is_alive():
            return
        self._hb_stop.clear()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, name="actor-pool-heartbeat", daemon=True
        )
        self._hb_thread.start()

    def _heartbeat_loop(self) -> None:
        """Periodically probe actors/nodes and register liveness."""
        import ray

        while not self._hb_stop.is_set():
            try:
                self._heartbeat_round(ray)
            except Exception as e:  # pragma: no cover - keep thread alive
                logger.warning("Heartbeat round failed: %s", e)
            self._hb_stop.wait(self.heartbeat_interval)

    def _heartbeat_round(self, ray) -> None:
        # Ray actor and registry RPCs can queue behind model load, inference,
        # shutdown, or reset. Probing during those phases turns normal work
        # into heartbeat timeouts and can race actor reset. The next idle
        # round refreshes node and actor liveness together.
        if any(state is not ActorState.IDLE for state in self.states.values()):
            return
        if self.registry is not None:
            self._heartbeat_nodes_and_recover(ray)
        self._heartbeat_actors(ray)

    def _heartbeat_nodes_and_recover(self, ray) -> None:
        """Refresh node heartbeats and auto-recover stalled nodes."""
        for node_id in self.node_mapping:
            ray.get(
                self.registry.node_heartbeat.remote(node_id),
                timeout=self.heartbeat_timeout,
            )
        dead_nodes = ray.get(
            self.registry.detect_dead_nodes.remote(self.node_heartbeat_timeout),
            timeout=self.heartbeat_timeout,
        )
        for node_id in dead_nodes:
            logger.warning(
                "Node %s heartbeat stale (>%ss); auto-recovering its actors",
                node_id, self.node_heartbeat_timeout,
            )
            self.recover_node(node_id)

    def _registry_idle_actor_ids(self, ray) -> set[str] | None:
        """Return ids the registry reports idle, or None if unavailable."""
        if self.registry is None:
            return None
        try:
            regs = ray.get(
                self.registry.list_actors.remote(),
                timeout=self.heartbeat_timeout,
            )
        except Exception:
            return None
        return {reg.actor_id for reg in regs if reg.state == "idle"}

    def _heartbeat_actors(self, ray) -> None:
        # Another process may lease actors without touching this process's
        # local states, so probe only actors the registry reports idle
        # (authoritative), falling back to local state when unavailable.
        idle_ids = self._registry_idle_actor_ids(ray)
        for idx, actor in enumerate(self.actors):
            # Ray actors execute methods serially by default. A leased actor
            # may spend tens of seconds loading a model or tearing down an
            # engine, so its heartbeat RPC would sit behind that operation
            # and be reported as a false failure. Node heartbeats continue to
            # cover process/node liveness while the actor is in use.
            if self.states.get(idx) is not ActorState.IDLE:
                self._hb_failures[idx] = 0
                continue
            actor_id = self.actor_ids.get(idx)
            if idle_ids is not None and actor_id not in idle_ids:
                self._hb_failures[idx] = 0
                continue
            try:
                ray.get(actor.heartbeat.remote(), timeout=self.heartbeat_timeout)
                self._hb_failures[idx] = 0
                # Liveness only: the registry heartbeat no longer mutates
                # lease/state, so a busy actor keeps its leased status.
                if actor_id and self.registry is not None:
                    ray.get(
                        self.registry.heartbeat.remote(actor_id),
                        timeout=self.heartbeat_timeout,
                    )
            except Exception as e:
                fails = self._hb_failures.get(idx, 0) + 1
                self._hb_failures[idx] = fails
                logger.warning(
                    "Heartbeat failed for actor %d (%s) attempt %d: %s",
                    idx, actor_id, fails, e,
                )
                if fails >= MAX_HEARTBEAT_FAILURES and actor_id:
                    self.states[idx] = ActorState.FAILED
                    if self.registry is not None:
                        try:
                            ray.get(
                                self.registry.mark_actor_failed.remote(actor_id),
                                timeout=self.heartbeat_timeout,
                            )
                        except Exception:
                            pass
                    # Schedule a replacement on another node.
                    self.rebuild_actor(actor_id)

    # ================================================================== queries
    def get_idle_count(self) -> int:
        """Get the number of idle actors."""
        return sum(1 for state in self.states.values() if state == ActorState.IDLE)

    def get_actor_states(self) -> dict[int, ActorState]:
        """Get the state of all actors."""
        return self.states.copy()

    def get_node_mapping(self) -> dict[str, list[int]]:
        """Get the node to actor index mapping."""
        return self.node_mapping.copy()

    def get_placement_group(self):
        """Get the placement group used by this pool."""
        return self.placement_group

    def get_registry(self):
        """Get the central NodeRegistryActor handle."""
        return self.registry

    # ================================================================= shutdown
    def shutdown(self) -> None:
        """Shutdown the pool and release all resources."""
        import ray

        # Stop heartbeat first so it does not race the teardown.
        self._hb_stop.set()
        if self._hb_thread is not None and self._hb_thread.is_alive():
            self._hb_thread.join(timeout=self.heartbeat_interval + 1)

        if not self._initialized:
            self.actors = []
            self.states = {}
            self.node_mapping = {}
            self._initialized = False
            return

        logger.info("Shutting down actor pool...")

        for idx, actor in enumerate(self.actors):
            try:
                ray.get(actor.reset.remote(), timeout=RPC_TIMEOUT)
            except Exception as e:
                logger.warning("Failed to reset actor %d: %s", idx, e)

        for idx, actor in enumerate(self.actors):
            actor_id = self.actor_ids.get(idx)
            if actor_id and self.registry is not None:
                try:
                    ray.get(
                        self.registry.unregister_actor.remote(actor_id),
                        timeout=RPC_TIMEOUT,
                    )
                except Exception:
                    pass
            try:
                ray.kill(actor)
            except Exception:
                pass

        if self.placement_group is not None:
            try:
                ray.util.remove_placement_group(self.placement_group)
            except Exception:
                pass

        # Clear local state (keep registry alive for other tenants).
        self.actors = []
        self.states = {}
        self.node_mapping = {}
        self.actor_ids = {}
        self._actor_id_to_idx = {}
        self._actor_regs = {}
        self._hb_failures = {}
        self._leases = {}
        self.placement_group = None
        self._initialized = False

        logger.info("Actor pool shutdown complete")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass