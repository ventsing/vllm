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

from vllm_external_executor.cluster_state import GlobalScheduler
from vllm_external_executor.external_worker_actor import ActorState, ExternalWorkerActor

if TYPE_CHECKING:
    import ray

logger = logging.getLogger(__name__)

# Ray RPC timeout: prevents a dead node from hanging pool operations forever.
RPC_TIMEOUT = 30.0
DEFAULT_HEARTBEAT_INTERVAL = 5.0
DEFAULT_HEARTBEAT_TIMEOUT = 30.0
MAX_HEARTBEAT_FAILURES = 3


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

        # Active leases: lease_id -> [actor_id] (for future accounting).
        self._leases: dict[str, list[str]] = {}

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
            heartbeat_timeout: Seconds before an actor/node is considered dead.
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
        self.fault_domain = fault_domain

        logger.info(
            "Pre-starting %d actors on devices %s (strategy=%s)",
            num_actors, devices_per_node, strategy,
        )

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
                bundles=[{"GPU": 1}] * num_actors + [{"CPU": 1}],
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
                    num_gpus=1,
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

        regs = ray.get(
            self.registry.list_actors.remote(), timeout=RPC_TIMEOUT
        )
        nodes = ray.get(
            self.registry.list_nodes.remote(), timeout=RPC_TIMEOUT
        )
        selected_ids = GlobalScheduler.select_actors(
            actors=regs,
            nodes=nodes,
            world_size=world_size,
            fault_domain_constraint=fault_domain_constraint,
            prefer_driver_node=prefer,
        )

        lease_id = str(uuid.uuid4())
        selected: list = []
        for actor_id in selected_ids:
            idx = self._actor_id_to_idx[actor_id]
            selected.append(self.actors[idx])
            self.states[idx] = ActorState.LEASED
            ray.get(
                self.registry.set_actor_state.remote(
                    actor_id, ActorState.LEASED.value, lease_id
                ),
                timeout=RPC_TIMEOUT,
            )

        self._leases[lease_id] = selected_ids
        logger.info(
            "Acquired %d actors for TP=%d, PP=%d (lease=%s)",
            len(selected), tp_size, pp_size, lease_id[:8],
        )
        return selected

    # ================================================================== release
    def release(self, actors: list) -> None:
        """
        Release actors back to the pool.

        Args:
            actors: List of actor handles to release.
        """
        import ray

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
                ray.get(
                    self.registry.set_actor_state.remote(
                        actor_id, ActorState.IDLE.value
                    ),
                    timeout=RPC_TIMEOUT,
                )

        logger.info("Released %d actors back to pool", len(actors))

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
            options = {"num_gpus": 1}
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
        """Periodically probe each actor and register liveness."""
        import ray

        while not self._hb_stop.is_set():
            try:
                self._heartbeat_round(ray)
            except Exception as e:  # pragma: no cover - keep thread alive
                logger.warning("Heartbeat round failed: %s", e)
            self._hb_stop.wait(self.heartbeat_interval)

    def _heartbeat_round(self, ray) -> None:
        for idx, actor in enumerate(self.actors):
            actor_id = self.actor_ids.get(idx)
            try:
                info = ray.get(actor.heartbeat.remote(), timeout=self.heartbeat_timeout)
                self._hb_failures[idx] = 0
                if actor_id and self.registry is not None:
                    ray.get(
                        self.registry.heartbeat.remote(
                            actor_id, info.get("state")
                        ),
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