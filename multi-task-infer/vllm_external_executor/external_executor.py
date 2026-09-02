# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
ExternalExecutor - Executor that uses pre-started Ray Actors.

This executor inherits from RayExecutorV2 and reuses its MessageQueue-based
communication mechanism. Instead of creating new Ray Actors, it acquires
pre-started actors from an ActorPoolManager.
"""

import logging
import weakref
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_distributed_init_method
from vllm.v1.executor.ray_executor_v2 import RayExecutorV2, RayWorkerHandle

if TYPE_CHECKING:
    import ray

logger = logging.getLogger(__name__)


class ExternalExecutor(RayExecutorV2):
    """
    Executor that uses pre-started Ray Actors from a pool.
    
    This executor inherits from RayExecutorV2 to reuse its MessageQueue-based
    communication mechanism. The key difference is that instead of creating
    new Ray Actors, it uses pre-started actors provided via the external_actors
    parameter.
    
    Benefits:
    - Faster initialization (actors are pre-warmed)
    - Actor reuse across multiple vLLM instances
    - Support for model hot-switching via weight_transfer
    
    Example:
        from vllm_external_executor import ActorPoolManager, ExternalExecutor
        
        pool = ActorPoolManager()
        pool.pre_start(num_actors=8, devices_per_node=[0, 1, 2, 3, 4, 5, 6, 7])
        
        actors = pool.acquire(tp_size=4, pp_size=2)
        
        llm = AsyncLLM(
            vllm_config=config,
            executor_class=ExternalExecutor,
            log_stats=True,
            external_actors=actors,
        )
    """
    
    uses_ray: bool = True
    supports_pp: bool = True
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        external_actors: list | None = None,
        cache_manager: Any | None = None,
    ):
        """
        Initialize the ExternalExecutor.
        
        Args:
            vllm_config: vLLM configuration
            external_actors: Pre-started Ray Actor handles. If None, falls back
                           to standard RayExecutorV2 behavior.
            cache_manager: CacheManagerActor handle for cache sharing.
        """
        self.external_actors = external_actors
        self._cache_manager = cache_manager
        
        if external_actors is not None:
            world_size = vllm_config.parallel_config.world_size
            if len(external_actors) != world_size:
                raise ValueError(
                    f"external_actors count ({len(external_actors)}) must equal "
                    f"world_size ({world_size})"
                )
        
        super().__init__(vllm_config)
    
    def _init_executor(self) -> None:
        """
        Initialize the executor.
        
        If external_actors is provided, uses those actors instead of creating
        new ones. Otherwise, falls back to standard RayExecutorV2 behavior.
        """
        if self.external_actors is None:
            # Fall back to standard RayExecutorV2 behavior
            logger.info("No external_actors provided, using standard RayExecutorV2")
            super()._init_executor()
            return
        
        logger.info(
            f"Initializing ExternalExecutor with {len(self.external_actors)} "
            f"pre-started actors"
        )
        
        # Initialize state
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback = None
        self.shutting_down = False
        
        import threading
        self.shutdown_lock = threading.Lock()
        
        # Get parallel config
        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )
        
        # Step 1: Create RayWorkerHandle from external actors
        self.ray_worker_handles: list[RayWorkerHandle] = []
        
        for i, actor in enumerate(self.external_actors):
            import ray
            info = ray.get(actor.get_info.remote())
            
            handle = RayWorkerHandle(
                actor=actor,
                rank=i,
                local_rank=-1,  # Set later after GPU ID discovery
                node_id=info["node_id"],
            )
            self.ray_worker_handles.append(handle)
        
        # Step 2: Group workers by node
        self._group_workers_by_node()
        
        # Step 3: Create MessageQueue (same as RayExecutorV2)
        import ray
        driver_node = ray.get_runtime_context().get_node_id()
        
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        n_local = sum(1 for h in self.ray_worker_handles if h.node_id == driver_node)
        
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size,
            n_local,
            max_chunk_bytes=max_chunk_bytes,
            connect_ip=ray.util.get_node_ip_address(),
        )
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        
        # Step 4: Initialize workers
        # Get distributed init method from first worker
        distributed_init_method = ray.get(
            self.ray_worker_handles[0].actor.create_dist_init_method.remote()
        )
        
        # Discover physical GPU IDs
        worker_node_and_physical_gpu_ids = ray.get([
            h.actor.get_node_and_physical_gpu_ids.remote()
            for h in self.ray_worker_handles
        ])
        
        node_workers: dict[str, list[int]] = defaultdict(list)
        node_physical_gpu_ids: dict[str, list[int]] = defaultdict(list)
        
        for i, (node_id, physical_gpu_ids) in enumerate(
            worker_node_and_physical_gpu_ids
        ):
            node_workers[node_id].append(i)
            node_physical_gpu_ids[node_id].extend(physical_gpu_ids)
        
        for node_id in node_physical_gpu_ids:
            node_physical_gpu_ids[node_id] = sorted(node_physical_gpu_ids[node_id])
        
        # Initialize each worker
        init_worker_refs = []
        for i, (node_id, _) in enumerate(worker_node_and_physical_gpu_ids):
            local_rank = node_workers[node_id].index(i)
            assigned_physical_gpu_ids = sorted(node_physical_gpu_ids[node_id])
            
            self.ray_worker_handles[i].local_rank = local_rank
            
            is_driver_worker = self._is_driver_worker(
                self.ray_worker_handles[i].rank
            )
            is_driver_node = node_id == driver_node
            
            init_worker_refs.append(
                self.ray_worker_handles[i].actor.initialize_worker.remote(
                    vllm_config=self.vllm_config,
                    rank=self.ray_worker_handles[i].rank,
                    local_rank=local_rank,
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    is_driver_worker=is_driver_worker,
                    is_driver_node=is_driver_node,
                )
            )
        
        # Set assigned_physical_gpu_ids on config for consistency
        if len(node_physical_gpu_ids) == 1:
            node_id_0 = worker_node_and_physical_gpu_ids[0][0]
            self.vllm_config.parallel_config.assigned_physical_gpu_ids = sorted(
                node_physical_gpu_ids[node_id_0]
            )
        
        ray.get(init_worker_refs)
        
        # Step 5: Collect response MQ handles
        init_results = ray.get([
            h.actor.wait_for_init.remote() for h in self.ray_worker_handles
        ])
        
        self.response_mqs: list[MessageQueue] = []
        for i, result in enumerate(init_results):
            if result["status"] != "READY":
                raise RuntimeError(f"Worker {i} failed to initialize: {result}")
            self.response_mqs.append(
                MessageQueue.create_from_handle(result["handle"], 0)
            )
        
        # Step 6: Start run() before wait_until_ready()
        for handle in self.ray_worker_handles:
            handle.run()
        
        # Step 7: wait_until_ready() barrier
        self.rpc_broadcast_mq.wait_until_ready()
        for response_mq in self.response_mqs:
            response_mq.wait_until_ready()
        
        from collections import deque
        from concurrent.futures import Future
        self.futures_queue = deque()
        
        self._post_init_executor()
        
        # Step 8: Start worker monitor
        self.start_worker_monitor()
        
        self.output_rank = self._get_output_rank()
        
        # Step 9: Handle compilation optimization and cache management
        self._handle_compilation_optimization()
        
        logger.info("ExternalExecutor initialization complete")
    
    def _handle_compilation_optimization(self):
        """
        Handle compilation optimization with lazy-loading pattern.
        
        Flow:
        1. Check local cache → hit: skip compilation
        2. Pull from CacheManagerActor → hit: extract to local, skip compilation
        3. Fallback: compile via RPC, then push to CacheManagerActor
        
        This ensures workers only compile when necessary, and compiled
        caches are shared across the cluster.
        """
        import os
        
        # Get cache manager reference
        cache_manager = getattr(self, '_cache_manager', None)
        if cache_manager is None:
            # No cache manager, just compile
            logger.info("No cache manager, proceeding with compilation")
            self.collective_rpc("compile_or_warm_up_model")
            return
        
        # Compute cache hash
        cache_hash = self._compute_cache_hash()
        
        # Step 1: Check local cache
        from vllm_external_executor.cache_manager_actor import local_cache_exists
        if local_cache_exists(cache_hash):
            logger.info(f"Local cache hit: {cache_hash}, skipping compilation")
            return
        
        # Step 2: Pull from CacheManagerActor
        logger.info(f"Local cache miss: {cache_hash}, pulling from manager")
        try:
            import ray
            cache_data = ray.get(cache_manager.pull.remote(cache_hash))
            if cache_data is not None:
                from vllm_external_executor.cache_manager_actor import (
                    extract_cache_to_local,
                )
                if extract_cache_to_local(cache_hash, cache_data):
                    logger.info(
                        f"Pulled cache from manager: {cache_hash} "
                        f"({len(cache_data) / 1024 / 1024:.1f} MB)"
                    )
                    return
        except Exception as e:
            logger.warning(f"Failed to pull from cache manager: {e}")
        
        # Step 3: Fallback - compile and push
        logger.info(f"Cache not found, compiling: {cache_hash}")
        
        # Try to acquire compile lock to avoid duplicate work
        worker_id = f"executor-{id(self)}"
        try:
            import ray
            lock_result = ray.get(
                cache_manager.try_acquire_compile_lock.remote(
                    cache_hash, worker_id
                )
            )
            
            if lock_result["status"] == "done":
                # Another worker already compiled, pull again
                logger.info(
                    "Cache compiled by another worker, pulling again"
                )
                cache_data = ray.get(cache_manager.pull.remote(cache_hash))
                if cache_data is not None:
                    from vllm_external_executor.cache_manager_actor import (
                        extract_cache_to_local,
                    )
                    extract_cache_to_local(cache_hash, cache_data)
                return
            
            if lock_result["status"] == "wait":
                # Another worker is compiling, wait and pull
                logger.info(
                    f"Waiting for {lock_result['holder']} to finish "
                    f"compilation"
                )
                # TODO: Implement proper waiting with timeout
                import time
                time.sleep(5)
                cache_data = ray.get(cache_manager.pull.remote(cache_hash))
                if cache_data is not None:
                    from vllm_external_executor.cache_manager_actor import (
                        extract_cache_to_local,
                    )
                    extract_cache_to_local(cache_hash, cache_data)
                return
            
            # Lock acquired, proceed with compilation
            logger.info(f"Compile lock acquired, compiling: {cache_hash}")
        
        except Exception as e:
            logger.warning(f"Failed to acquire compile lock: {e}")
        
        try:
            # Compile
            self.collective_rpc("compile_or_warm_up_model")
            
            # Package and push
            from vllm_external_executor.cache_manager_actor import (
                package_local_cache,
            )
            cache_data = package_local_cache(cache_hash)
            if cache_data is not None:
                import ray
                ray.get(
                    cache_manager.push.remote(
                        cache_hash, cache_data, worker_id
                    )
                )
                logger.info(
                    f"Pushed compiled cache to manager: {cache_hash} "
                    f"({len(cache_data) / 1024 / 1024:.1f} MB)"
                )
        finally:
            # Release lock
            try:
                import ray
                ray.get(
                    cache_manager.release_compile_lock.remote(
                        cache_hash, worker_id
                    )
                )
            except Exception:
                pass
    
    def _compute_cache_hash(self) -> str:
        """Compute cache hash from vLLM config."""
        import hashlib
        import json
        
        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config
        compilation_config = self.vllm_config.compilation_config
        
        factors = {
            "model_arch": (
                model_config.architectures[0]
                if hasattr(model_config, "architectures")
                and model_config.architectures
                else "unknown"
            ),
            "model_revision": (
                model_config.revision
                if hasattr(model_config, "revision")
                else "unknown"
            ),
            "tp_size": parallel_config.tensor_parallel_size,
            "pp_size": parallel_config.pipeline_parallel_size,
            "dp_size": parallel_config.data_parallel_size,
            "batch_sizes": (
                compilation_config.cudagraph_capture_sizes
                if hasattr(compilation_config, "cudagraph_capture_sizes")
                else []
            ),
            "cudagraph_mode": (
                compilation_config.cudagraph_mode.value
                if hasattr(compilation_config, "cudagraph_mode")
                and compilation_config.cudagraph_mode
                else 0
            ),
        }
        
        hash_content = json.dumps(factors, sort_keys=True)
        return hashlib.sha256(hash_content.encode()).hexdigest()[:10]
    
    def _load_model_via_weight_transfer(self) -> None:
        """
        Load model via weight_transfer mechanism.
        
        This method is called when weight_transfer is configured.
        It triggers each worker to load the model via weight_transfer.
        """
        weight_transfer_config = self.vllm_config.weight_transfer_config
        if weight_transfer_config is None:
            raise RuntimeError("weight_transfer_config is not set")
        
        init_info = weight_transfer_config.init_info
        
        self.collective_rpc(
            "load_model_via_weight_transfer",
            kwargs={"weight_transfer_init_info": init_info}
        )
    
    def release_actors(self) -> None:
        """
        Release actors back to the pool.
        
        This method should be called when the vLLM instance is no longer needed.
        It resets all actors and returns them to the IDLE state.
        """
        import ray
        
        logger.info("Releasing actors back to pool...")
        
        for handle in self.ray_worker_handles:
            try:
                ray.get(handle.actor.reset.remote())
            except Exception as e:
                logger.warning(f"Failed to reset actor {handle.rank}: {e}")
        
        self.ray_worker_handles = []
        logger.info("Actors released")
    
    def switch_model(
        self,
        new_vllm_config: VllmConfig,
        checkpoint_path: str | None = None,
        storage_backend: str = "nfs",
        storage_config: dict | None = None,
        weight_transfer_init_info: dict | None = None,
        reinitialize_cache: bool = True,
    ) -> None:
        """
        Switch to a new model (hot-switching).
        
        Releases the current model on every worker, rebuilds them with the
        new vllm_config, and loads the new weights from the chosen source.
        Optionally re-profiles and re-allocates KV cache, then re-runs the
        compilation optimization (cache-aware lazy loading).
        
        Note: The caller is responsible for ensuring no in-flight inference
        requests when calling this method (stop scheduling new requests and
        wait for running requests to finish). The executor cannot safely
        drain the engine's scheduler from here.
        
        Args:
            new_vllm_config: New vLLM configuration. The parallel layout
                (world_size) must match the current one - actors are bound
                to fixed devices, so TP/PP changes require a new acquire().
            checkpoint_path: Checkpoint path for storage loading (optional).
            storage_backend: Storage backend name ("nfs" or "mooncake").
            storage_config: Backend-specific configuration.
            weight_transfer_init_info: Weight transfer init info (optional).
            reinitialize_cache: Whether to re-profile and re-allocate KV
                cache after the model switch.
        """
        import ray
        
        parallel_config = new_vllm_config.parallel_config
        if parallel_config.world_size != self.world_size:
            raise ValueError(
                f"switch_model requires world_size to stay constant: "
                f"current={self.world_size}, new={parallel_config.world_size}. "
                f"Change TP/PP by releasing actors and acquiring a new set."
            )
        
        logger.info("Switching model (hot-switching)...")
        
        # 1. Update executor-side config reference
        self.vllm_config = new_vllm_config
        
        # 2. Switch model on every worker (parallel ray calls)
        switch_refs = [
            handle.actor.switch_model.remote(
                vllm_config=new_vllm_config,
                checkpoint_path=checkpoint_path,
                storage_backend=storage_backend,
                storage_config=storage_config,
                weight_transfer_init_info=weight_transfer_init_info,
            )
            for handle in self.ray_worker_handles
        ]
        ray.get(switch_refs)
        logger.info("All workers switched to the new model")
        
        # 3. Re-initialize KV cache (profile available memory and allocate)
        if reinitialize_cache:
            self._reinitialize_kv_cache()
        
        # 4. Compilation optimization (cache-aware lazy loading)
        self._handle_compilation_optimization()
        
        logger.info("Model switch complete")
    
    def _reinitialize_kv_cache(self) -> None:
        """
        Re-profile available GPU memory and re-allocate KV cache.
        
        Mirrors the engine-core initialization path (EngineCore.
        _initialize_kv_caches) so that a switched model gets a consistent
        KV cache. Best-effort: falls back with a warning when profiling
        fails (e.g. attention backends are not registered in this process).
        """
        try:
            from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
            from vllm.v1.core.single_type_kv_cache_manager import (
                register_all_kvcache_specs,
            )
            from vllm.v1.attention.backends.utils import (
                resolve_kv_cache_layout,
            )
            
            # 1. Register KV cache specs for the new model
            register_all_kvcache_specs(self.vllm_config)
            
            # 2. Collect KV cache specs and resolve the cache layout
            kv_cache_specs = self.get_kv_cache_specs()
            supported_layouts = self.get_supported_kv_cache_layouts()
            layout = resolve_kv_cache_layout(
                self.vllm_config, supported_layouts,
                [s for specs in kv_cache_specs for s in specs.values()],
            )
            self.set_kv_cache_layout(layout.name)
            
            # 3. Profile available GPU memory
            available_gpu_memory = self.determine_available_memory()
            
            # 4. Compute KV cache configs and allocate on workers
            kv_cache_configs = get_kv_cache_configs(
                self.vllm_config, kv_cache_specs, available_gpu_memory
            )
            self.initialize_from_config(kv_cache_configs)
        except Exception as e:
            logger.warning(
                f"Failed to re-initialize KV cache after switch: {e}. "
                f"Call executor.initialize_from_config() manually with "
                f"the kv_cache_configs for the new model."
            )
