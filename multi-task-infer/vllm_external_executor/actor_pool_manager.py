# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
ActorPoolManager - Manages a pool of pre-started Ray Actors.

This module provides the ActorPoolManager class which handles:
- Pre-starting Ray Actors bound to GPU/NPU devices
- Managing actor lifecycle (IDLE -> LEASED -> RUNNING -> RELEASED)
- Acquiring and releasing actors for vLLM instances
"""

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from vllm_external_executor.external_worker_actor import ActorState, ExternalWorkerActor

if TYPE_CHECKING:
    import ray

logger = logging.getLogger(__name__)


class ActorPoolManager:
    """
    Manages a pool of pre-started Ray Actors.
    
    The pool pre-starts actors that bind to GPU/NPU devices and pre-import
    common libraries. These actors can then be acquired by ExternalExecutor
    to serve as workers in vLLM instances.
    
    Example:
        pool = ActorPoolManager()
        pool.pre_start(num_actors=8, devices_per_node=[0, 1, 2, 3, 4, 5, 6, 7])
        
        # Acquire actors for a vLLM instance
        actors = pool.acquire(tp_size=4, pp_size=2)
        
        # ... use actors with ExternalExecutor ...
        
        # Release actors back to pool
        pool.release(actors)
    """
    
    def __init__(self):
        """Initialize the ActorPoolManager."""
        self.actors: list = []  # list[ray.actor.ActorHandle]
        self.states: dict[int, ActorState] = {}
        self.node_mapping: dict[str, list[int]] = {}  # node_id -> actor_indices
        self.placement_group = None
        self.cache_manager = None  # CacheManagerActor handle
        self._initialized = False
    
    def pre_start(
        self,
        num_actors: int,
        devices_per_node: list[int],
        placement_group=None,
        warmup_distributed: bool = True,
        shared_cache_dir: str | None = None,
        enable_cache_compression: bool = True,
    ) -> None:
        """
        Pre-start actors and bind them to GPU/NPU devices.
        
        Args:
            num_actors: Number of actors to pre-start
            devices_per_node: List of GPU/NPU device IDs
            placement_group: Optional Ray placement group for resource constraints
            warmup_distributed: Whether to warm up NCCL/HCCl at init time
            shared_cache_dir: NFS directory for large compilation caches
            enable_cache_compression: Whether to compress large caches
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
        
        logger.info(f"Pre-starting {num_actors} actors on devices {devices_per_node}")
        
        # 1. Create CacheManagerActor for compilation cache sharing
        from vllm_external_executor.cache_manager_actor import (
            create_cache_manager_actor,
        )
        self.cache_manager = create_cache_manager_actor(
            shared_cache_dir=shared_cache_dir,
            enable_compression=enable_cache_compression,
        )
        logger.info(
            f"Created CacheManagerActor: shared_dir={shared_cache_dir}, "
            f"compression={enable_cache_compression}"
        )
        
        # 2. Create placement group if not provided
        if placement_group is None:
            self.placement_group = ray.util.placement_group(
                bundles=[{"GPU": 1}] * num_actors + [{"CPU": 1}],
                strategy="PACK",
            )
            ray.get(self.placement_group.ready())
            logger.info("Created placement group")
        else:
            self.placement_group = placement_group
        
        # 2. Build runtime_env
        runtime_env = RuntimeEnv(env_vars={
            "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "1",
        })
        
        # 3. Create actors
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
        
        # 4. Wait for all actors to be ready
        logger.info("Waiting for actors to be ready...")
        ray.get([actor.wait_for_ready.remote() for actor in self.actors])
        
        # 5. Build node_mapping
        infos = ray.get([actor.get_info.remote() for actor in self.actors])
        for i, info in enumerate(infos):
            node_id = info["node_id"]
            if node_id not in self.node_mapping:
                self.node_mapping[node_id] = []
            self.node_mapping[node_id].append(i)
        
        self._initialized = True
        logger.info(
            f"Actor pool ready: {num_actors} actors on "
            f"{len(self.node_mapping)} nodes"
        )
    
    def acquire(
        self,
        tp_size: int,
        pp_size: int,
        node_constraint: dict[str, int] | None = None,
    ) -> list:
        """
        Acquire idle actors for a vLLM instance.
        
        Args:
            tp_size: Tensor Parallel size
            pp_size: Pipeline Parallel size
            node_constraint: Optional node constraints
                (e.g., {"node_0": 2, "node_1": 2})
        
        Returns:
            List of acquired actor handles
        
        Raises:
            RuntimeError: If not enough idle actors are available
        """
        world_size = tp_size * pp_size
        
        if node_constraint is not None:
            # Acquire with node constraints
            selected = self._acquire_with_constraints(
                world_size, node_constraint
            )
        else:
            # Simple selection: pick first world_size idle actors
            selected = []
            for idx, state in self.states.items():
                if state == ActorState.IDLE and len(selected) < world_size:
                    selected.append(self.actors[idx])
                    self.states[idx] = ActorState.LEASED
        
        if len(selected) < world_size:
            raise RuntimeError(
                f"Not enough idle actors: {len(selected)} < {world_size}"
            )
        
        logger.info(f"Acquired {len(selected)} actors for TP={tp_size}, PP={pp_size}")
        return selected
    
    def _acquire_with_constraints(
        self,
        world_size: int,
        node_constraint: dict[str, int],
    ) -> list:
        """
        Acquire actors with node constraints.
        
        Args:
            world_size: Total number of actors needed
            node_constraint: Dict mapping node_id to required actor count
        
        Returns:
            List of acquired actor handles
        """
        selected = []
        
        for node_id, required_count in node_constraint.items():
            if node_id not in self.node_mapping:
                raise ValueError(f"Unknown node: {node_id}")
            
            # Find idle actors on this node
            node_actors = []
            for idx in self.node_mapping[node_id]:
                if self.states[idx] == ActorState.IDLE:
                    node_actors.append(idx)
            
            if len(node_actors) < required_count:
                raise RuntimeError(
                    f"Not enough idle actors on node {node_id}: "
                    f"{len(node_actors)} < {required_count}"
                )
            
            # Acquire required actors from this node
            for idx in node_actors[:required_count]:
                selected.append(self.actors[idx])
                self.states[idx] = ActorState.LEASED
        
        return selected
    
    def release(self, actors: list) -> None:
        """
        Release actors back to the pool.
        
        Args:
            actors: List of actor handles to release
        """
        import ray
        
        for actor in actors:
            # Find the corresponding index
            try:
                idx = self.actors.index(actor)
            except ValueError:
                logger.warning(f"Actor not found in pool, skipping")
                continue
            
            # Mark as RELEASED (transitional state before reset completes)
            self.states[idx] = ActorState.RELEASED
            
            # Reset the actor
            try:
                ray.get(actor.reset.remote())
            except Exception as e:
                logger.error(f"Failed to reset actor {idx}: {e}")
                self.states[idx] = ActorState.FAILED
                continue
            
            # Update state
            self.states[idx] = ActorState.IDLE
        
        logger.info(f"Released {len(actors)} actors back to pool")
    
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
    
    def shutdown(self) -> None:
        """Shutdown the pool and release all resources."""
        import ray
        
        if not self._initialized:
            return
        
        logger.info("Shutting down actor pool...")
        
        # Reset all actors
        for idx, actor in enumerate(self.actors):
            try:
                ray.get(actor.reset.remote())
            except Exception as e:
                logger.warning(f"Failed to reset actor {idx}: {e}")
        
        # Kill all actors
        for actor in self.actors:
            try:
                ray.kill(actor)
            except Exception:
                pass
        
        # Remove placement group
        if self.placement_group is not None:
            try:
                ray.util.remove_placement_group(self.placement_group)
            except Exception:
                pass
        
        # Clear state
        self.actors = []
        self.states = {}
        self.node_mapping = {}
        self.placement_group = None
        self._initialized = False
        
        logger.info("Actor pool shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass
