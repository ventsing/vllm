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

# Timeout for the ray.get calls inside a migration, so a dead worker does not
# hang switch_model/rollback indefinitely.
_MIGRATION_RPC_TIMEOUT = 60.0


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
        prefix_index: Any | None = None,
        weight_ledger: Any | None = None,
        heat_tracker: Any | None = None,
        tiered_cache: Any | None = None,
        prefetch_policy: Any | None = None,
    ):
        """
        Initialize the ExternalExecutor.
        
        Args:
            vllm_config: vLLM configuration
            external_actors: Pre-started Ray Actor handles. If None, falls back
                           to standard RayExecutorV2 behavior.
            cache_manager: CacheManagerActor handle for cache sharing.
            prefix_index: Shared :class:`GlobalPrefixIndex` (cluster-wide prefix
                reuse). Defaults to a per-executor instance.
            weight_ledger: Shared :class:`WeightShareLedger` (base+adapter
                residency). Defaults to a per-executor instance.
            heat_tracker: Shared :class:`AccessHeatTracker` for prefetch
                ranking. Defaults to a per-executor instance.
            tiered_cache: Shared :class:`TieredCache` for checkpoint-tier
                bookkeeping. Defaults to an unbounded per-executor instance.
            prefetch_policy: :class:`PrefetchPolicy` for async prefetch
                decisions. Defaults to a per-executor instance.
        """
        self.external_actors = external_actors
        self._cache_manager = cache_manager
        # Idempotency registry for model migrations (switch_model).
        from vllm_external_executor.migration import MigrationIdempotencyRegistry
        self._migration_registry = MigrationIdempotencyRegistry()
        # Engine-core scheduler gate (injected via set_scheduler_gate).
        self._migration_pause_fn = None
        self._migration_resume_fn = None
        # EngineCore scheduler reference, bound via bind_scheduler (the
        # EngineCore hook): source of truth for KV-block metadata.
        self._scheduler = None

        # Decision-layer wiring (storage tiering / prefetch / prefix reuse).
        self._init_decision_layer(
            prefix_index=prefix_index,
            weight_ledger=weight_ledger,
            heat_tracker=heat_tracker,
            tiered_cache=tiered_cache,
            prefetch_policy=prefetch_policy,
        )

        if external_actors is not None:
            world_size = vllm_config.parallel_config.world_size
            if len(external_actors) != world_size:
                raise ValueError(
                    f"external_actors count ({len(external_actors)}) must equal "
                    f"world_size ({world_size})"
                )

        # MVP envelope: reject out-of-scope configuration and shared
        # decision-layer handles before they reach untested paths (P2).
        self._validate_mvp_scope(vllm_config)
        if cache_manager is not None:
            raise NotImplementedError("MVP does not enable cache_manager (P2)")
        optional_decision = (
            prefix_index,
            weight_ledger,
            heat_tracker,
            tiered_cache,
            prefetch_policy,
        )
        if any(x is not None for x in optional_decision):
            raise NotImplementedError(
                "MVP does not enable shared decision-layer handles "
                "(prefix_index/weight_ledger/heat_tracker/tiered_cache/"
                "prefetch_policy) (P2)"
            )

        super().__init__(vllm_config)

    def _validate_mvp_scope(self, vllm_config: VllmConfig) -> None:
        """Reject configurations outside the single-node sequential-reuse MVP.

        Delegates to :func:`mvp_entry.validate_mvp_config` (single source of
        truth for the MVP envelope) so the executor-level guard cannot drift
        from the entry-point guard.

        Raises:
            ValueError: For TP/PP, LoRA, KV-connector, or elastic-EP settings
                that the MVP does not implement yet (P2).
        """
        from vllm_external_executor.mvp_entry import validate_mvp_config

        pc = vllm_config.parallel_config
        lora = vllm_config.lora_config
        ktc = vllm_config.kv_transfer_config
        has_lora = lora is not None and getattr(lora, "max_loras", 0) > 0
        has_kv_connector = (
            ktc is not None and getattr(ktc, "kv_connector", None) is not None
        )
        validate_mvp_config(
            tp_size=pc.tensor_parallel_size,
            pp_size=pc.pipeline_parallel_size,
            enable_lora=has_lora,
            kv_transfer_config=ktc if has_kv_connector else None,
            elastic_ep=getattr(pc, "enable_elastic_ep", False),
        )

    def _init_decision_layer(
        self,
        prefix_index=None,
        weight_ledger=None,
        heat_tracker=None,
        tiered_cache=None,
        prefetch_policy=None,
    ) -> None:
        """Create (or adopt) the decision-layer objects and the orchestrator."""
        from vllm_external_executor.global_prefix_index import GlobalPrefixIndex
        from vllm_external_executor.migration_orchestrator import (
            MigrationOrchestrator,
        )
        from vllm_external_executor.prefetch_policy import (
            AccessHeatTracker,
            PrefetchPolicy,
        )
        from vllm_external_executor.storage_tier import StorageTier, TieredCache
        from vllm_external_executor.weight_sharing import WeightShareLedger

        self._prefix_index = prefix_index or GlobalPrefixIndex()
        self._weight_ledger = weight_ledger or WeightShareLedger()
        self._heat = heat_tracker or AccessHeatTracker()
        self._tiered_cache = tiered_cache or TieredCache(
            {
                StorageTier.HBM: float("inf"),
                StorageTier.DRAM: float("inf"),
                StorageTier.REMOTE: float("inf"),
            }
        )
        self._prefetch_policy = prefetch_policy or PrefetchPolicy(
            budget_bytes=256 * 1024 * 1024
        )
        self._orchestrator = MigrationOrchestrator(
            tiered_cache=self._tiered_cache,
            heat=self._heat,
            prefetch_policy=self._prefetch_policy,
            prefix_index=self._prefix_index,
            weight_ledger=self._weight_ledger,
            start_prefetch=self._start_prefetch,
            await_prefetch=self._await_prefetch,
        )

    # ----------------------------------------------------- prefetch execution
    def _start_prefetch(self, keys) -> None:
        """Begin background prefetch of ``keys`` (override for async I/O).

        Default is a no-op log: the decision layer names *what* to prefetch;
        a production executor launches a thread/ray task to load those keys
        while the current model keeps serving, then signals completion for
        :meth:`_await_prefetch`.
        """
        if keys:
            logger.info("Prefetch nominated (%d keys, not started)", len(keys))

    def _await_prefetch(self) -> None:
        """Block until in-flight prefetch completes (override for async I/O).

        Called by the orchestrator at LOAD, after CHECKPOINT/UNLOAD, so the
        prefetched state overlaps unrelated migration work.
        """
        return

    # ------------------------------------------------------- migration helpers
    def _weight_hash(self, config: "VllmConfig | None" = None) -> str:
        """Stable weight-configuration hash for prefix/weight reuse groups."""
        config = config or self.vllm_config
        model = getattr(config.model_config, "model", "unknown")
        tp = getattr(config.parallel_config, "tensor_parallel_size", 1)
        pp = getattr(
            config.parallel_config, "pipeline_parallel_size", 1
        )
        return f"{model}@tp{tp}pp{pp}"

    def _actor_id(self) -> str:
        """A stable identity for this executor's actor group (first node)."""
        if self.ray_worker_handles:
            return getattr(self.ray_worker_handles[0], "node_id", None) or "exec-0"
        return "exec-0"
    
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
        
        # Step 2: Create MessageQueue (same as RayExecutorV2). Worker
        # grouping by node happens inline in Step 4 (node_workers below) so
        # local_rank reflects per-node ordering.
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
        # Get distributed init method from the first worker. Pass world_size so
        # the worker does not need a vllm_config of its own yet (see 2.3).
        distributed_init_method = ray.get(
            self.ray_worker_handles[0].actor.create_dist_init_method.remote(
                self.world_size
            )
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
        
        tp_size = self.parallel_config.tensor_parallel_size
        
        # Build the full per-rank kwargs list (same shape as RayExecutorV2) so
        # every actor can pick its own entry by global rank.
        all_kwargs = []
        for rank, (node_id, _) in enumerate(worker_node_and_physical_gpu_ids):
            local_rank = node_workers[node_id].index(rank)
            assigned_physical_gpu_ids = sorted(node_physical_gpu_ids[node_id])
            self.ray_worker_handles[rank].local_rank = local_rank
            all_kwargs.append({
                "vllm_config": self.vllm_config,
                "assigned_physical_gpu_ids": assigned_physical_gpu_ids,
                "local_rank": local_rank,
                "rank": rank,
                "distributed_init_method": distributed_init_method,
                "is_driver_worker": rank % tp_size == 0,
            })
        
        # Each actor independently initializes, passing the full all_kwargs
        # with its own rank so rpc_rank and all_kwargs length stay aligned.
        init_worker_refs = []
        for rank, (node_id, _) in enumerate(worker_node_and_physical_gpu_ids):
            init_worker_refs.append(
                self.ray_worker_handles[rank].actor.initialize_worker.remote(
                    vllm_config=self.vllm_config,
                    rank=rank,
                    all_kwargs=all_kwargs,
                    input_shm_handle=scheduler_output_handle,
                    is_driver_node=(node_id == driver_node),
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
        
        logger.info("ExternalExecutor initialization complete")
    
    def _handle_compilation_optimization(self):
        """
        Handle compilation optimization with lazy-loading pattern (P2).

        Not called by the MVP path: with no CacheManagerActor the executor
        defers each model's compilation to vLLM's standard
        ``EngineCore._initialize_kv_caches`` -> ``compile_or_warm_up_model``
        flow, which runs *after* KV caches are sized (an early
        ``compile_or_warm_up_model`` here would capture graphs against an
        uninitialized cache). When P2 re-enables cache sharing, restore the
        lazy flow:
        1. Check local cache → hit: skip compilation
        2. Pull from CacheManagerActor → hit: extract to local, skip compilation
        3. Fallback: compile via RPC, then push to CacheManagerActor
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

    def start_worker_monitor(self, inline=False) -> None:
        """Monitor pre-started actor liveness via heartbeat polling.

        Overrides :meth:`RayExecutorV2.start_worker_monitor`: our
        ``ExternalWorkerActor.run`` returns immediately (its busy loop runs on
        a daemon thread), so the inherited ``run_ref`` sentinel would complete
        at once and be misread as a worker death. Poll each actor's
        ``heartbeat`` instead; a failed poll marks the executor failed and
        shuts it down.
        """
        import ray
        import threading
        import time

        if not self.ray_worker_handles:
            raise RuntimeError("Ray workers have not started successfully.")

        self_ref = weakref.ref(self)
        handles = list(self.ray_worker_handles)

        def _should_stop() -> bool:
            executor = self_ref()
            return not executor or executor.shutting_down

        def monitor_workers() -> None:
            while not _should_stop() and ray.is_initialized():
                try:
                    ray.get([h.actor.heartbeat.remote() for h in handles])
                except Exception:
                    if _should_stop():
                        return
                    executor = self_ref()
                    if not executor:
                        return
                    executor.is_failed = True
                    logger.error(
                        "ExternalWorkerActor heartbeat failed, shutting down "
                        "executor."
                    )
                    executor.shutdown()
                    if executor.failure_callback is not None:
                        callback = executor.failure_callback
                        executor.failure_callback = None
                        callback()
                    return
                time.sleep(5.0)

        thread = threading.Thread(
            target=monitor_workers, daemon=True, name="ExternalWorkerMonitor"
        )
        thread.start()
        self._monitor_thread = thread

    def shutdown(self) -> None:
        """Tear down the executor without killing pool-owned actors.

        ``RayExecutorV2.shutdown`` ``ray.kill()``s the workers it created;
        this executor only *leases* pre-started actors from an
        ``ActorPoolManager``, so shutdown instead resets each actor (freeing
        worker/model/KV resources and stopping its busy loop) and closes the
        message queues, leaving the actors alive in the pool for reuse.
        Idempotent.
        """
        import ray

        lock = getattr(self, "shutdown_lock", None)
        if lock is None:
            return
        with lock:
            if getattr(self, "shutting_down", False):
                return
            self.shutting_down = True

        self._join_monitor_thread()

        for handle in getattr(self, "ray_worker_handles", []):
            try:
                ray.get(handle.actor.reset.remote())
            except Exception:
                logger.exception("Failed to reset actor rank=%d", handle.rank)

        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None

        for mq in getattr(self, "response_mqs", []):
            mq.shutdown()
        self.response_mqs = []
        self.ray_worker_handles = []

    def switch_model(
        self,
        new_vllm_config: VllmConfig,
        checkpoint_path: str | None = None,
        storage_backend: str = "nfs",
        storage_config: dict | None = None,
        weight_transfer_init_info: dict | None = None,
        reinitialize_cache: bool = True,
        migration_id: str | None = None,
        flight_batch_policy: str = "drain",
    ):
        """
        Switch to a new model (hot-switching) as an atomic migration.

        Drives a validated state machine
        (PREPARING -> GRACEFUL_PAUSE -> CHECKPOINT -> UNLOAD -> LOAD ->
        RESTORE -> COMPLETED) with compensation-based rollback: if any worker
        fails to switch, the already-switched workers are rolled back to the
        previous model and the executor config reference is restored, leaving
        the instance in its pre-migration state (request-level idempotency via
        ``migration_id``).

        In-flight batches are governed by ``flight_batch_policy``. The vLLM
        scheduler lives in EngineCore, so the executor surfaces
        ``set_scheduler_gate`` to let the caller inject the pause/resume
        coordination; the executor cannot drain the scheduler itself.

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
            migration_id: Idempotency key. Reissuing the same id returns the
                cached result instead of re-running the migration.
            flight_batch_policy: One of "drain" / "pause_serialize" /
                "preempt" (see FlightBatchPolicy).

        Returns:
            MigrationStateMachine describing the terminal outcome.

        Raises:
            NotImplementedError: In-place hot-switching is a P2 feature and is
                intentionally disabled in the MVP; use sequential reuse
                (release actors, then acquire + ``initialize_worker`` for the
                next model) instead.
        """
        raise NotImplementedError(
            "switch_model (in-place hot-switch) is disabled in the MVP (P2); "
            "use sequential reuse: release actors then acquire and initialize "
            "the next model"
        )
        import ray  # pragma: no cover - unreachable until P2 enables hot-switch
        import uuid

        from vllm_external_executor.migration import (
            FlightBatchPolicy,
            MigrationPhase,
            MigrationSpec,
        )

        new_weight_hash = self._weight_hash(new_vllm_config)
        spec = MigrationSpec(
            migration_id=migration_id or uuid.uuid4().hex,
            target=getattr(new_vllm_config.model_config, "model", "unknown"),
            policy=FlightBatchPolicy(flight_batch_policy),
            payload={
                "target_checkpoint": checkpoint_path or f"ckpt:{new_weight_hash}",
                "checkpoint_bytes": 0,  # advisory; filled from real tensors
                "base_bytes": 0,        # advisory; filled from real tensors
                "base_hash": new_weight_hash,
                "adapter": None,
            },
        )

        def orchestrate(sm):
            # PREPARING precondition check (world_size stability).
            parallel_config = new_vllm_config.parallel_config
            if parallel_config.world_size != self.world_size:
                raise ValueError(
                    f"switch_model requires world_size to stay constant: "
                    f"current={self.world_size}, "
                    f"new={parallel_config.world_size}. "
                    f"Change TP/PP via release+acquire."
                )

            snapshots_box: list = []

            def on_graceful_pause(_sm, _script):
                self._pause_for_migration(_sm)
                # Resume on success (RESTORE) and rollback (compensation runs
                # last), so the scheduler is never left paused.
                _sm.register_compensation(
                    lambda: self._resume_after_migration(_sm)
                )

            def on_checkpoint(_sm, _script):
                old_config = self.vllm_config
                _sm.register_compensation(
                    lambda: setattr(self, "vllm_config", old_config)
                )
                snapshots_box.extend(
                    ray.get(
                        [
                            h.actor.switch_model_snapshot.remote()
                            for h in self.ray_worker_handles
                        ],
                        timeout=_MIGRATION_RPC_TIMEOUT,
                    )
                )

            def on_unload(_sm, _script):
                self._switch_all_workers(
                    new_vllm_config,
                    checkpoint_path,
                    storage_backend,
                    storage_config,
                    weight_transfer_init_info,
                    list(snapshots_box),
                    _sm,
                )

            def on_restore(_sm, _script):
                if reinitialize_cache:
                    self._reinitialize_kv_cache()
                self._handle_compilation_optimization()
                self._resume_after_migration(_sm)

            _sm, script = self._orchestrator.migrate(
                spec,
                actor_id=self._actor_id(),
                candidates={},
                resident=set(),
                phase_handlers={
                    MigrationPhase.GRACEFUL_PAUSE: on_graceful_pause,
                    MigrationPhase.CHECKPOINT: on_checkpoint,
                    MigrationPhase.UNLOAD: on_unload,
                    MigrationPhase.RESTORE: on_restore,
                },
                sm=sm,
            )
            logger.info(
                "Model switch complete (%s; script prefetch=%d, "
                "tier_moves=%d, weight_register=%d)",
                _sm.progress(),
                len(script.prefetch_keys),
                len(script.checkpoint_moves)
                + len(script.unload_moves)
                + len(script.load_moves),
                len(script.weight_register),
            )

        return self._migration_registry.run(spec, orchestrate)

    def _switch_all_workers(
        self,
        new_vllm_config: VllmConfig,
        checkpoint_path: str | None,
        storage_backend: str,
        storage_config: dict | None,
        weight_transfer_init_info: dict | None,
        snapshots: list[dict],
        sm,
    ) -> None:
        """Fan out worker switches and roll back successes on partial failure.

        All workers are launched in parallel; results are collected one by one
        so a partial failure can be compensated precisely. On failure, the
        already-switched workers are rolled back to their snapshots and the
        error is re-raised to drive the state machine's rollback.
        """
        import ray

        refs = [
            h.actor.switch_model.remote(
                vllm_config=new_vllm_config,
                checkpoint_path=checkpoint_path,
                storage_backend=storage_backend,
                storage_config=storage_config,
                weight_transfer_init_info=weight_transfer_init_info,
            )
            for h in self.ray_worker_handles
        ]

        switched: list[tuple] = []  # (handle, snapshot)
        errors: list[str] = []
        for i, ref in enumerate(refs):
            try:
                ray.get(ref, timeout=_MIGRATION_RPC_TIMEOUT)
                switched.append((self.ray_worker_handles[i], snapshots[i]))
            except Exception as e:  # noqa: BLE001 - collect all failures
                errors.append(f"worker {i}: {e}")

        if errors:
            def rollback_workers():
                for handle, snap in switched:
                    try:
                        ray.get(
                            handle.actor.switch_model_rollback.remote(snap),
                            timeout=_MIGRATION_RPC_TIMEOUT,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.error("Worker rollback failed: %s", e)

            sm.register_compensation(rollback_workers)
            raise RuntimeError("; ".join(errors))

        logger.info("All workers switched to the new model")

    def _pause_for_migration(self, sm) -> None:
        """Invoke the engine-core scheduler gate for the migration pause.

        The vLLM scheduler runs in EngineCore, not in this executor, so this
        is a coordination point: if the caller registered a pause callback via
        :meth:`set_scheduler_gate`, it is invoked with the flight-batch policy.
        """
        if self._migration_pause_fn is not None:
            self._migration_pause_fn(sm.spec.policy)
        else:
            logger.info(
                "No scheduler gate registered; assuming caller paused "
                "scheduling (policy=%s)", sm.spec.policy.value,
            )

    def _resume_after_migration(self, sm) -> None:
        """Invoke the engine-core scheduler gate to resume scheduling."""
        if self._migration_resume_fn is not None:
            self._migration_resume_fn()

    def set_scheduler_gate(self, pause_fn, resume_fn) -> None:
        """Register engine-core callbacks for scheduler pause/resume.

        Args:
            pause_fn: ``callable(FlightBatchPolicy) -> None`` invoked at
                GRACEFUL_PAUSE to stop scheduling / drain or freeze batches.
            resume_fn: ``callable() -> None`` invoked after RESTORE.
        """
        self._migration_pause_fn = pause_fn
        self._migration_resume_fn = resume_fn

    def bind_scheduler(self, scheduler) -> None:
        """Receive the EngineCore scheduler reference (core.py hook).

        The scheduler owns ``kv_cache_manager.block_pool``, the source of truth
        for KV-block metadata (block ids, prefix-cache hashes, ref counts).
        This enables :meth:`snapshot_kv_blocks` and
        :meth:`import_prefix_cache` without reaching into private structures
        from outside the engine loop.
        """
        self._scheduler = scheduler

    def snapshot_kv_blocks(self) -> list:
        """Export live KV blocks as group-aware ``KVBlockRef`` metadata.

        Walks the scheduler's block pool and returns one ref per
        ``(block_id, kv_cache_group)`` that is either referenced by a request
        (``ref_cnt > 0``) or resident in the prefix cache. A block may appear
        once per group (multi-group models: encoder/decoder, full/sliding
        window). The content hash is the group's *pure* content hash (hex,
        group id stripped), stable across processes thanks to vLLM's
        deterministic prefix hashing.

        Returns:
            ``KVBlockRef`` list suitable for :class:`IncrementalKVPlanner`.
        """
        from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id

        from vllm_external_executor.kv_migration import KVBlockRef

        if self._scheduler is None:
            raise RuntimeError(
                "scheduler not bound; this executor requires the EngineCore "
                "bind_scheduler hook"
            )
        block_pool = self._scheduler.kv_cache_manager.block_pool
        refs: list[KVBlockRef] = []
        for block in block_pool.blocks:
            hashes: set = set()
            if block.block_hash is not None:
                hashes.add(block.block_hash)
            hashes.update(
                block_pool.cached_block_hashes_by_block.get(block.block_id, ())
            )
            if hashes:
                for block_hash in sorted(hashes):
                    refs.append(
                        KVBlockRef(
                            block_id=block.block_id,
                            content_hash=get_block_hash(block_hash).hex(),
                            version=block.ref_cnt,
                            group_id=get_group_id(block_hash),
                        )
                    )
            elif block.ref_cnt > 0:
                # Live but unhashed (mid-fill / caching off): still migrate,
                # but no group/prefix signal; group 0 is a placeholder.
                refs.append(
                    KVBlockRef(
                        block_id=block.block_id,
                        version=block.ref_cnt,
                    )
                )
        return refs

    def import_prefix_cache(
        self,
        block_group_hashes: list[tuple[int, int, str]],
    ) -> int:
        """Restore the prefix-cache index after an incremental KV import.

        After the data plane copies KV tensors into the destination workers,
        the scheduler-side prefix cache is still stale. This re-registers the
        imported blocks' hashes (one per group) so subsequent requests hit
        them.

        Args:
            block_group_hashes: ``(dst_block_id, group_id, content_hash_hex)``
                triples produced by :meth:`snapshot_kv_blocks` / the planner.

        Returns:
            Number of hashes registered.
        """
        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            make_block_hash_with_group_id,
        )

        if self._scheduler is None:
            raise RuntimeError(
                "scheduler not bound; this executor requires the EngineCore "
                "bind_scheduler hook"
            )
        by_block: dict[int, list] = {}
        for block_id, group_id, content_hash in block_group_hashes:
            if not content_hash:
                continue
            by_block.setdefault(block_id, []).append(
                make_block_hash_with_group_id(
                    BlockHash(bytes.fromhex(content_hash)), group_id
                )
            )
        self._scheduler.kv_cache_manager.block_pool.import_block_hashes(by_block)
        return len(by_block)

    def import_request_blocks(
        self,
        request_id: str,
        block_ids_by_group: list[list[int]],
        block_id_to_hashes: dict | None = None,
    ) -> None:
        """Attach pre-filled KV blocks to an admitted request's block table.

        Final scheduler-side step of cross-engine KV migration: after the data
        plane copied the block tensors, this occupies them (ref-count + free
        queue) and points the request's per-group block table at them, without
        an ``allocate_slots`` lookup. The request must already be admitted to
        the target scheduler.

        Args:
            request_id: Target request id (already added via ``add_request``).
            block_ids_by_group: Destination physical ids per KV cache group.
                Length must equal the number of KV cache groups.
            block_id_to_hashes: Optional ``block_id -> [hash...]`` to restore
                into the prefix cache (as produced by the diff/planner).
        """
        if self._scheduler is None:
            raise RuntimeError(
                "scheduler not bound; this executor requires the EngineCore "
                "bind_scheduler hook"
            )
        self._scheduler.kv_cache_manager.import_request_blocks(
            request_id,
            block_ids_by_group,
            block_id_to_hashes=block_id_to_hashes,
        )

    def snapshot_requests(self) -> list[dict]:
        """Export resident requests' state for cross-engine migration.

        Returns one dict per resident request: serialized admission inputs
        (``EngineCoreRequest``), the full token sequence, the delivered output
        tokens, the KV compute position, and the per-group block ids covering
        the computed tokens (the blocks that carry migrated KV).
        """
        if self._scheduler is None:
            raise RuntimeError(
                "scheduler not bound; this executor requires the EngineCore "
                "bind_scheduler hook"
            )
        snapshots: list[dict] = []
        for request in self._scheduler.requests.values():
            block_ids = (
                self._scheduler.kv_cache_manager.get_block_ids_for_computed_tokens(
                    request.request_id, request.num_computed_tokens
                )
            )
            snapshots.append(
                {
                    "engine_core_request": request.to_engine_core_request(),
                    "all_token_ids": list(request.all_token_ids),
                    "output_token_ids": list(request.output_token_ids),
                    "num_computed_tokens": request.num_computed_tokens,
                    "block_ids_by_group": [list(ids) for ids in block_ids],
                }
            )
        return snapshots

    def restore_requests(
        self,
        snapshots: list[dict],
        block_mapping: dict[int, int] | None = None,
    ) -> None:
        """Rebuild migrated requests and mount their KV blocks on this executor.

        For each snapshot: reconstruct the ``Request`` from its admission
        inputs, restore the running state (full token sequence + compute
        position), admit it to the scheduler, then attach the migrated KV
        blocks via :meth:`import_request_blocks` (mapping source ids through
        ``block_mapping`` for heterogeneous pools).

        Args:
            snapshots: List produced by :meth:`snapshot_requests`.
            block_mapping: Optional ``src_block_id -> dst_block_id`` remap.
        """
        from vllm.v1.request import Request

        if self._scheduler is None:
            raise RuntimeError(
                "scheduler not bound; this executor requires the EngineCore "
                "bind_scheduler hook"
            )
        mapping = block_mapping or {}
        for snap in snapshots:
            request = Request.from_engine_core_request(
                snap["engine_core_request"], block_hasher=None
            )
            request.restore_running_state(
                snap["all_token_ids"],
                snap["output_token_ids"],
                snap["num_computed_tokens"],
            )
            self._scheduler.add_request(request)
            block_ids_by_group = [
                [mapping.get(bid, bid) for bid in group]
                for group in snap["block_ids_by_group"]
            ]
            if any(block_ids_by_group):
                self.import_request_blocks(
                    request.request_id, block_ids_by_group
                )

    def migrate_kv_cache_incremental(
        self,
        src_executor: "ExternalExecutor",
        src_blocks: list,
        dst_blocks: list,
        block_mapping: dict[int, int] | None = None,
        transport: str | None = None,
    ):
        """
        Ship only the KV blocks that actually changed to this executor.

        Data-plane half of incremental KV migration. The diff (which blocks are
        new/modified/prefix-reused) is computed by :class:`IncrementalKVPlanner`
        from EngineCore's KV cache manager block tables; this method moves the
        ``transfer`` block tensors rank-for-rank from ``src_executor`` workers
        to this executor's workers, leaving prefix-cache hits and unchanged
        blocks untouched.

        Args:
            src_executor: Source executor whose workers hold the live KV cache.
            src_blocks: Source ``KVBlockRef`` metadata (live blocks).
            dst_blocks: Destination ``KVBlockRef`` metadata (resident blocks).
            block_mapping: Optional ``src_block_id -> dst_block_id`` map for
                heterogeneous pools. When ``None``, ids map 1:1 (same-layout).
            transport: KV transport backend name
                (``ray_object_store`` or ``mooncake_rdma``). Defaults to
                Ray Object Store (TCP relay).

        Returns:
            The computed :class:`KVMigrationPlan` (transfer / prefix_hits /
            unchanged), so EngineCore can remap the destination block table.
        """
        from vllm_external_executor.kv_migration import IncrementalKVPlanner
        from vllm_external_executor.kv_transport import KVTransportFactory

        if len(src_executor.ray_worker_handles) != len(self.ray_worker_handles):
            raise ValueError(
                "src/dst executor world_size mismatch: "
                f"{len(src_executor.ray_worker_handles)} vs "
                f"{len(self.ray_worker_handles)}"
            )

        plan = IncrementalKVPlanner().plan(src_blocks, dst_blocks)
        logger.info("Incremental KV migration: %s", plan.summary())

        if plan.transfer:
            src_ids = sorted({b.block_id for b in plan.transfer})
            selected = KVTransportFactory.create(
                transport or "ray_object_store"
            )
            shipped = selected.ship(
                src_executor,
                self,
                src_ids,
                block_mapping=block_mapping,
                timeout=_MIGRATION_RPC_TIMEOUT,
            )
            logger.info(
                "Transferred %d KV blocks via %s (rank-for-rank)",
                shipped, selected.name,
            )

        return plan

    def migrate_kv_cache_to(
        self,
        target_executor: "ExternalExecutor",
        total_blocks: int | None = None,
        dst_occupied: list[int] | None = None,
        transport: str | None = None,
        prefetch: bool = True,
    ):
        """
        End-to-end incremental KV migration from this executor to another.

        Flows through snapshot -> diff -> (optional) target-id assignment ->
        data-plane copy -> prefix-cache restore. When ``total_blocks`` is
        provided, source block ids are remapped onto free destination ids
        (heterogeneous pools); otherwise ids map 1:1 (same-layout).

        When ``prefetch`` is set, the migration is also driven through the
        decision layer (:class:`MigrationOrchestrator`): hot, non-resident
        prefixes are nominated at PREPARING (firing ``_start_prefetch``) and
        the transferred prefixes are registered on the shared
        :class:`GlobalPrefixIndex` so later engines can reuse them.

        Args:
            target_executor: Destination executor (empty or partially-populated
                KV cache).
            total_blocks: Destination pool size. Provide to enable
                heterogeneous block-id remapping.
            dst_occupied: Destination block ids already in use (when remapping).
            transport: KV transport backend name passed through to the data
                plane (``ray_object_store`` or ``mooncake_rdma``).
            prefetch: Drive the prefetch hook + prefix-index registration.

        Returns:
            The computed :class:`KVMigrationPlan`.
        """
        from vllm_external_executor.kv_migration import IncrementalKVPlanner

        src_blocks = self.snapshot_kv_blocks()
        dst_blocks = target_executor.snapshot_kv_blocks()
        plan = IncrementalKVPlanner().plan(src_blocks, dst_blocks)

        if total_blocks is not None:
            IncrementalKVPlanner().assign_targets(
                plan,
                total_blocks=total_blocks,
                dst_occupied=dst_occupied or (),
            )

        if prefetch:
            self._drive_migration_orchestrator(
                plan, target_executor, src_blocks, dst_blocks
            )

        target_executor.migrate_kv_cache_incremental(
            self,
            src_blocks,
            dst_blocks,
            block_mapping=plan.block_mapping or None,
            transport=transport,
        )

        if plan.transfer:
            # Register transferred blocks' hashes on the target, using the
            # remapped destination ids when present.
            block_group_hashes = []
            for b in plan.transfer:
                if not b.content_hash:
                    continue
                dst_id = plan.block_mapping.get(b.block_id, b.block_id)
                block_group_hashes.append(
                    (dst_id, b.group_id, b.content_hash)
                )
            if block_group_hashes:
                imported = target_executor.import_prefix_cache(
                    block_group_hashes
                )
                logger.info(
                    "Restored prefix-cache index for %d blocks", imported
                )

        return plan

    def _drive_migration_orchestrator(
        self,
        plan,
        target_executor: "ExternalExecutor",
        src_blocks: list,
        dst_blocks: list,
    ):
        """Drive the decision layer for a KV migration (prefetch + index).

        Nominates hot, non-resident prefixes for async prefetch via the
        orchestrator's PREPARING hook and registers the transferred prefixes
        on the shared :class:`GlobalPrefixIndex` under the destination actor.
        The tiering bookkeeping uses a synthetic ``kv:{weight_hash}`` object
        and is advisory only: actual KV tensors move via the transport, not
        through :class:`TieredCache`.
        """
        import time

        from vllm_external_executor.global_prefix_index import PrefixEntry
        from vllm_external_executor.migration import MigrationSpec

        now = time.monotonic()
        weight_hash = self._weight_hash()
        candidates: dict[str, int] = {}
        resident: set[str] = set()

        for block in src_blocks:
            if block.content_hash:
                key = f"{block.content_hash}:{block.group_id}"
                candidates[key] = candidates.get(key, 0) + 1
                self._heat.record(key, now)
        for block in dst_blocks:
            if block.content_hash:
                resident.add(f"{block.content_hash}:{block.group_id}")

        prefix_entries = []
        for block in plan.transfer:
            if not block.content_hash:
                continue
            dst_id = plan.block_mapping.get(block.block_id, block.block_id)
            prefix_entries.append(
                PrefixEntry(
                    content_hash=block.content_hash,
                    group_id=block.group_id,
                    weight_hash=weight_hash,
                    actor_id=target_executor._actor_id(),
                    node_id=target_executor._actor_id(),
                    block_id=dst_id,
                    refs=1,
                )
            )

        spec = MigrationSpec(
            target=f"kv-migrate:{weight_hash}",
            payload={
                "target_checkpoint": f"kv:{weight_hash}",
                "checkpoint_bytes": len(plan.transfer),
                "prefix_entries": prefix_entries,
            },
        )
        _sm, script = self._orchestrator.migrate(
            spec,
            actor_id=target_executor._actor_id(),
            candidates=candidates,
            resident=resident,
        )
        logger.info(
            "KV migration decision script: prefetch=%d keys (%d units), "
            "prefix=%d entries registered",
            len(script.prefetch_keys), script.prefetch_bytes,
            len(script.prefix_register),
        )
        return script

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
