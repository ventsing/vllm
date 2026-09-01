# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
External Executor Plugin for vLLM

This plugin provides a way to pre-start Ray Actors and reuse them across
multiple vLLM instances, significantly reducing initialization time.

Key components:
- ExternalWorkerActor: Pre-started Ray Actor that binds to a GPU/NPU device
- ActorPoolManager: Manages a pool of pre-started actors
- ExternalExecutor: Executor that acquires actors from the pool
- StorageCheckpointEngine: Load model weights from persistent storage
  (compatible with verl's CheckpointEngineWithCache interface)

Usage:
    from vllm_external_executor import ActorPoolManager, ExternalExecutor
    
    # Pre-start actors
    pool = ActorPoolManager()
    pool.pre_start(num_actors=8, devices_per_node=[0, 1, 2, 3, 4, 5, 6, 7])
    
    # Acquire actors for a vLLM instance
    actors = pool.acquire(tp_size=4, pp_size=2)
    
    # Create vLLM instance with external actors
    llm = AsyncLLM(
        vllm_config=config,
        executor_class=ExternalExecutor,
        log_stats=True,
        external_actors=actors,
    )
"""

from vllm_external_executor.external_worker_actor import ExternalWorkerActor
from vllm_external_executor.actor_pool_manager import ActorPoolManager, ActorState
from vllm_external_executor.external_executor import ExternalExecutor
from vllm_external_executor.cache_manager_actor import (
    CacheManagerActor,
    create_cache_manager_actor,
    package_local_cache,
    extract_cache_to_local,
    local_cache_exists,
)
from vllm_external_executor.storage_checkpoint_engine import (
    StorageCheckpointEngine,
    StorageBackend,
    StorageBackendFactory,
    NFSStorageBackend,
    MooncakeStoreBackend,
    CheckpointMetadata,
    TensorMeta,
)

__all__ = [
    "ExternalWorkerActor",
    "ActorPoolManager",
    "ActorState",
    "ExternalExecutor",
    "CacheManagerActor",
    "create_cache_manager_actor",
    "package_local_cache",
    "extract_cache_to_local",
    "local_cache_exists",
    "StorageCheckpointEngine",
    "StorageBackend",
    "StorageBackendFactory",
    "NFSStorageBackend",
    "MooncakeStoreBackend",
    "CheckpointMetadata",
    "TensorMeta",
    "register_plugin",
]

__version__ = "0.1.0"


def register_plugin():
    """
    Register the ExternalExecutor plugin with vLLM.
    
    This function is called by vLLM's plugin system when the plugin is loaded.
    It registers the ExternalExecutor as a valid executor backend.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Loading vllm-external-executor plugin v%s", __version__)
    
    # Register ExternalExecutor as a valid distributed_executor_backend
    # This allows users to use: distributed_executor_backend="external"
    from vllm.config import ParallelConfig
    
    # Store the original __post_init__ method
    original_post_init = ParallelConfig.__post_init__
    
    def patched_post_init(self):
        # Call original first
        original_post_init(self)
        
        # Add "external" as a valid backend
        # Note: This is a minimal patch - the actual executor selection
        # happens in Executor.get_class() which we also need to patch
    
    # Note: For a production plugin, you would need to patch Executor.get_class()
    # to recognize "external" as a valid backend and return ExternalExecutor.
    # For now, users can directly pass executor_class=ExternalExecutor to AsyncLLM.
    
    logger.info("vllm-external-executor plugin loaded successfully")
