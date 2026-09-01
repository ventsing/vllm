# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
ExternalWorkerActor - Pre-started Ray Actor for vLLM workers.

This actor binds to a GPU/NPU device at creation time and pre-imports
common libraries to speed up subsequent vLLM instance initialization.
"""

import os
from enum import Enum
from typing import Any

import torch


class ActorState(str, Enum):
    """Actor lifecycle states."""
    IDLE = "idle"                    # 空闲，可被租用
    LEASED = "leased"                # 已租用，等待初始化
    INIT_DEVICE = "init_device"      # 设备已初始化（NCCL/HCCl 已初始化）
    INIT_MODEL = "init_model"        # 模型已加载
    RUNNING = "running"              # 正在运行推理
    RELEASED = "released"            # 已释放，等待重置
    FAILED = "failed"                # 失败，需要重建


class ExternalWorkerActor:
    """
    Pre-started Ray Actor that binds to a GPU/NPU device.
    
    This actor performs minimal initialization at creation time:
    1. Binds to the specified GPU/NPU device
    2. Imports common libraries (torch, vllm, etc.)
    3. Optionally warms up NCCL/HCCl for faster subsequent initialization
    
    The actor can then be acquired by an ExternalExecutor to serve as a
    worker in a vLLM instance.
    
    Note: This class is designed to be used with @ray.remote decorator.
    The actual Ray actor creation is handled by ActorPoolManager.
    """
    
    def __init__(
        self,
        device_id: int,
        warmup_distributed: bool = True,
    ):
        """
        Initialize the ExternalWorkerActor.
        
        Args:
            device_id: The GPU/NPU device ID to bind to
            warmup_distributed: Whether to warm up NCCL/HCCl at init time
        """
        # 1. Bind to device
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        
        # 2. Import common libraries (warm up imports)
        self._import_common_libraries()
        
        # 3. Optionally warm up distributed
        if warmup_distributed:
            self._warmup_distributed()
        
        # 4. Initialize state
        self.worker = None
        self.vllm_config = None
        self.state = ActorState.IDLE
        
        # For distributed initialization
        self._dist_init_store = None
        
        # Message queues (set during initialize_worker)
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
    
    def _import_common_libraries(self):
        """Import common libraries to speed up subsequent initialization."""
        import torch.distributed as dist
        import vllm
        from vllm.v1.worker.gpu_worker import Worker
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm.v1.worker.worker_base import WorkerWrapperBase
        from vllm.distributed.device_communicators.shm_broadcast import (
            MessageQueue, Handle
        )
    
    def _warmup_distributed(self):
        """
        Warm up NCCL/HCCl by creating and destroying a temporary ProcessGroup.
        
        This speeds up subsequent distributed initialization because the
        NCCL/HCCl library is already loaded and initialized.
        """
        import torch.distributed as dist
        
        try:
            # Create a temporary TCPStore
            store = dist.TCPStore(
                host_name="localhost",
                port=0,
                world_size=1,
                is_master=True,
                wait_for_workers=False,
            )
            
            # Initialize process group
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://localhost:{store.port}",
                world_size=1,
                rank=0,
            )
            
            # Immediately destroy
            dist.destroy_process_group()
            
            # Cleanup
            del store
            
        except Exception as e:
            # Warmup is optional, don't fail if it doesn't work
            import logging
            logging.warning(f"NCCL warmup failed: {e}")
    
    def wait_for_ready(self) -> dict:
        """
        Wait for actor to be ready and return basic info.
        
        Returns:
            Dictionary with device_id, node_id, physical_gpu_ids, state
        """
        import ray
        
        return {
            "device_id": self.device_id,
            "node_id": ray.get_runtime_context().get_node_id(),
            "physical_gpu_ids": [self.device_id],
            "state": self.state.value,
        }
    
    def get_info(self) -> dict:
        """
        Get actor information for pool management.
        
        Returns:
            Dictionary with device_id, node_id, physical_gpu_ids, state
        """
        import ray
        
        return {
            "device_id": self.device_id,
            "node_id": ray.get_runtime_context().get_node_id(),
            "physical_gpu_ids": [self.device_id],
            "state": self.state.value,
        }
    
    def get_node_and_physical_gpu_ids(self) -> tuple:
        """
        Return (node_id, physical_gpu_ids) for this actor.
        
        This method is compatible with RayExecutorV2's interface.
        
        Returns:
            Tuple of (node_id, physical_gpu_ids)
        """
        import ray
        from vllm.platforms import current_platform
        
        node_id = ray.get_runtime_context().get_node_id()
        device_key = current_platform.ray_device_key
        
        if not device_key:
            raise RuntimeError(
                f"current platform {current_platform.device_name} does not support ray."
            )
        
        physical_gpu_ids = ray.get_runtime_context().get_accelerator_ids()[device_key]
        
        return node_id, [
            current_platform.device_control_id_to_physical_device_id(str(x))
            for x in physical_gpu_ids
        ]
    
    def initialize_worker(
        self,
        vllm_config,
        rank: int,
        local_rank: int,
        distributed_init_method: str,
        input_shm_handle,
        is_driver_worker: bool,
        is_driver_node: bool = False,
    ) -> None:
        """
        Initialize the worker (called by ExternalExecutor).
        
        Creates WorkerWrapperBase but does not perform device initialization.
        
        Args:
            vllm_config: vLLM configuration
            rank: Global rank of this worker
            local_rank: Local rank within the node
            distributed_init_method: Distributed initialization method
            input_shm_handle: Shared memory handle for broadcast MessageQueue
            is_driver_worker: Whether this is the driver worker
            is_driver_node: Whether this actor is on the driver node
        """
        from vllm.v1.worker.worker_base import WorkerWrapperBase
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
        
        # Store config
        self.vllm_config = vllm_config
        self._is_driver_node = is_driver_node
        
        # Create WorkerWrapperBase
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        wrapper.init_worker(all_kwargs=[{
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
        }])
        
        self.worker = wrapper
        
        # Initialize message queues
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank
        )
        
        # Use ray's internal IP for cross-node communication
        import ray
        n_local = 1 if is_driver_node else 0
        self.worker_response_mq = MessageQueue(
            n_reader=1,
            n_local_reader=n_local,
            connect_ip=ray.util.get_node_ip_address(),
        )
        
        self.state = ActorState.LEASED
    
    def create_dist_init_method(self) -> str:
        """
        Create a distributed initialization method.
        
        Returns:
            Distributed initialization method string
        """
        import ray
        from torch.distributed import TCPStore
        from vllm.utils.network_utils import get_distributed_init_method
        
        host = ray.util.get_node_ip_address()
        store = TCPStore(
            host_name=host,
            port=0,
            world_size=self.vllm_config.parallel_config.world_size,
            is_master=True,
            wait_for_workers=False,
            multi_tenant=True,
        )
        self._dist_init_store = store
        
        return get_distributed_init_method(host, store.port)
    
    def init_device(self) -> None:
        """
        Initialize device (distributed environment).
        
        Calls Worker.init_device() which:
        - Sets GPU device
        - Initializes NCCL/HCCl
        - Takes memory snapshot
        - Creates ModelRunner (empty shell)
        """
        self.worker.init_device()
        self.state = ActorState.INIT_DEVICE
    
    def load_model(self, load_dummy_weights: bool = False) -> None:
        """
        Load model weights.
        
        Args:
            load_dummy_weights: If True, load dummy weights instead of real weights
        """
        self.worker.load_model(load_dummy_weights=load_dummy_weights)
        self.state = ActorState.INIT_MODEL
    
    def load_model_via_weight_transfer(
        self,
        weight_transfer_init_info: dict,
    ) -> None:
        """
        Load model via weight_transfer mechanism.
        
        Flow:
        1. Create model structure (with dummy weights)
        2. Initialize weight_transfer engine
        3. Pull real weights via weight_transfer
        
        Args:
            weight_transfer_init_info: Initialization info for weight_transfer engine
        """
        # 1. Load dummy model (create model structure)
        self.worker.load_model(load_dummy_weights=True)
        
        # 2. Initialize weight_transfer engine
        model = self.worker.get_model()
        device = self.worker.device
        weight_transfer_config = self.vllm_config.weight_transfer_config
        
        from vllm.distributed.weight_transfer import WeightTransferEngineFactory
        engine = WeightTransferEngineFactory.create_engine(
            weight_transfer_config, self.vllm_config, device, model
        )
        self.worker.weight_transfer_engine = engine
        
        # 3. Initialize transfer mechanism
        init_info = engine.parse_init_info(weight_transfer_init_info)
        engine.init_transfer_engine(init_info)
        
        # 4. Wait for and receive weights (triggered by external trainer)
        engine.start_weight_update()
        # update_weights will be called externally
        engine.finish_weight_update()
        
        self.state = ActorState.INIT_MODEL
    
    def load_model_from_storage(
        self,
        checkpoint_path: str,
        storage_backend: str = "nfs",
        storage_config: dict | None = None,
    ) -> None:
        """
        Load model weights from persistent storage.
        
        This method uses StorageCheckpointEngine (compatible with verl's
        CheckpointEngineWithCache interface) to load weights from storage
        backends like NFS or Mooncake Store.
        
        Flow:
        1. Create model structure (with dummy weights)
        2. Create StorageCheckpointEngine
        3. Load weights from storage and update model parameters
        
        Args:
            checkpoint_path: Logical path for the checkpoint in storage
            storage_backend: Storage backend name ("nfs" or "mooncake")
            storage_config: Backend-specific configuration
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 1. Load dummy model (create model structure)
        logger.info(
            f"Loading model from storage: checkpoint={checkpoint_path}, "
            f"backend={storage_backend}"
        )
        self.worker.load_model(load_dummy_weights=True)
        
        # 2. Create StorageCheckpointEngine
        from vllm_external_executor.storage_checkpoint_engine import (
            StorageCheckpointEngine,
        )
        
        engine = StorageCheckpointEngine(
            backend=storage_backend,
            config=storage_config or {},
            device=str(self.worker.device),
        )
        engine.set_checkpoint(checkpoint_path)
        
        # 3. Load weights from storage and update model
        model = self.worker.get_model()
        loaded_count = 0
        
        for name, tensor in engine.get_weights():
            try:
                param = model.get_parameter(name)
                param.data.copy_(tensor)
                loaded_count += 1
            except AttributeError:
                logger.warning(f"Parameter {name} not found in model, skipping")
        
        logger.info(
            f"Loaded {loaded_count} parameters from storage checkpoint"
        )
        
        self.state = ActorState.INIT_MODEL
    
    def initialize_kv_cache(self, kv_cache_config) -> None:
        """
        Initialize KV Cache.
        
        Args:
            kv_cache_config: KV Cache configuration
        """
        self.worker.initialize_kv_cache(kv_cache_config)
    
    def determine_available_memory(self) -> int:
        """
        Determine available memory for KV Cache.
        
        Returns:
            Available memory in bytes
        """
        return self.worker.determine_available_memory()
    
    def compile_or_warm_up_model(self) -> None:
        """Compile/warm up model (CUDA Graph, kernel warmup, etc.)."""
        self.worker.compile_or_warm_up_model()
        self.state = ActorState.RUNNING
    
    def wait_for_init(self) -> dict:
        """
        Wait for initialization to complete and return status.
        
        Returns:
            Dictionary with status and response MQ handle
        """
        return {
            "status": "READY",
            "handle": self.worker_response_mq.export_handle(),
        }
    
    def run(self) -> None:
        """Start the worker busy loop."""
        try:
            assert self.rpc_broadcast_mq is not None
            self.rpc_broadcast_mq.wait_until_ready()
            assert self.worker_response_mq is not None
            self.worker_response_mq.wait_until_ready()
            
            self.worker_busy_loop()
        except Exception as e:
            import logging
            logging.exception(f"ExternalWorkerActor failed: {e}")
            raise
        finally:
            self.shutdown()
    
    def worker_busy_loop(self) -> None:
        """Worker main loop (same as WorkerProc.worker_busy_loop)."""
        while True:
            rpc_request = self.rpc_broadcast_mq.dequeue(indefinite=True)
            self._execute_worker_rpc(rpc_request)
    
    def _execute_worker_rpc(self, rpc_request) -> None:
        """Execute an RPC request."""
        method, args, kwargs, output_rank = rpc_request
        
        try:
            result = getattr(self.worker, method)(*args, **kwargs)
            
            # Send response
            if output_rank is not None:
                self.worker_response_mq.enqueue((True, result))
            else:
                self.worker_response_mq.enqueue((True, None))
                
        except Exception as e:
            self.worker_response_mq.enqueue((False, str(e)))
    
    def reset(self) -> None:
        """
        Reset actor state.
        
        Releases model, KV Cache, distributed environment, but keeps device binding.
        Actor returns to IDLE state and can be re-acquired.
        """
        if self.worker is not None:
            try:
                self.worker.shutdown()
            except Exception:
                pass
            self.worker = None
        
        self.vllm_config = None
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        self._dist_init_store = None
        self.state = ActorState.IDLE
    
    def shutdown(self) -> None:
        """Shutdown the actor completely."""
        self.reset()
