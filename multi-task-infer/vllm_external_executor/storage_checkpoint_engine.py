# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Storage-based Checkpoint Engine for ExternalExecutor.

This module provides a checkpoint engine that loads model weights from
persistent storage (Mooncake Store, NFS) instead of real-time transfer
from a trainer. This is compatible with verl's CheckpointEngineWithCache
interface.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│              StorageCheckpointEngine                             │
│  - Extends CheckpointEngineWithCache interface                   │
│  - Supports backends: Mooncake Store / NFS                       │
│  - get_weights(): yields named tensors from storage              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              ExternalExecutor Worker                             │
│  - Calls engine.get_weights() to get named tensors               │
│  - Loads weights into model via model.load_weights()             │
└─────────────────────────────────────────────────────────────────┘
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generator

import torch


@dataclass
class TensorMeta:
    """Metadata for a stored tensor."""
    name: str
    shape: torch.Size
    dtype: torch.dtype
    storage_key: str
    offset: int = 0
    size: int = 0


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint stored in storage."""
    model_name: str
    global_steps: int
    tensors: dict[str, TensorMeta]
    total_size: int


class StorageBackend(ABC):
    """Abstract storage backend interface.
    
    This interface defines the contract for storage backends that can
    store and retrieve model weights.
    """
    
    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the storage backend.
        
        Args:
            config: Backend-specific configuration
        """
        raise NotImplementedError
    
    @abstractmethod
    def save_checkpoint(
        self,
        checkpoint_path: str,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a checkpoint to storage.
        
        Args:
            checkpoint_path: Logical path for the checkpoint
            weights: Generator of (name, tensor) pairs
            metadata: Optional metadata to store with checkpoint
            
        Returns:
            Storage key for the saved checkpoint
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_metadata(self, checkpoint_path: str) -> CheckpointMetadata:
        """Load checkpoint metadata.
        
        Args:
            checkpoint_path: Logical path for the checkpoint
            
        Returns:
            Checkpoint metadata
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_tensor(self, tensor_meta: TensorMeta) -> torch.Tensor:
        """Load a single tensor from storage.
        
        Args:
            tensor_meta: Tensor metadata
            
        Returns:
            Loaded tensor
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_weights(
        self, checkpoint_path: str
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get all weights from a checkpoint.
        
        Args:
            checkpoint_path: Logical path for the checkpoint
            
        Yields:
            (name, tensor) pairs
        """
        raise NotImplementedError
    
    @abstractmethod
    def exists(self, checkpoint_path: str) -> bool:
        """Check if a checkpoint exists in storage.
        
        Args:
            checkpoint_path: Logical path for the checkpoint
            
        Returns:
            True if checkpoint exists
        """
        raise NotImplementedError
    
    @abstractmethod
    def delete(self, checkpoint_path: str) -> None:
        """Delete a checkpoint from storage.
        
        Args:
            checkpoint_path: Logical path for the checkpoint
        """
        raise NotImplementedError


class NFSStorageBackend(StorageBackend):
    """NFS-based storage backend.
    
    Stores checkpoints as safetensors files on NFS.
    """
    
    def __init__(self):
        self.base_path: str = ""
        self._initialized = False
    
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize NFS backend.
        
        Args:
            config: Configuration with 'base_path' key
        """
        self.base_path = config.get("base_path", "/tmp/vllm_checkpoints")
        import os
        os.makedirs(self.base_path, exist_ok=True)
        self._initialized = True
    
    def _get_checkpoint_dir(self, checkpoint_path: str) -> str:
        """Get the directory for a checkpoint."""
        import os
        return os.path.join(self.base_path, checkpoint_path)
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save checkpoint as safetensors on NFS."""
        from safetensors.torch import save_file
        
        checkpoint_dir = self._get_checkpoint_dir(checkpoint_path)
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Collect all tensors
        tensors = {}
        tensor_metas = {}
        total_size = 0
        
        for name, tensor in weights:
            tensors[name] = tensor
            tensor_metas[name] = TensorMeta(
                name=name,
                shape=tensor.shape,
                dtype=tensor.dtype,
                storage_key=f"{checkpoint_path}/model.safetensors",
                size=tensor.nbytes,
            )
            total_size += tensor.nbytes
        
        # Save to safetensors
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        save_file(tensors, model_path)
        
        # Save metadata
        import json
        meta = CheckpointMetadata(
            model_name=metadata.get("model_name", "unknown") if metadata else "unknown",
            global_steps=metadata.get("global_steps", 0) if metadata else 0,
            tensors=tensor_metas,
            total_size=total_size,
        )
        meta_path = os.path.join(checkpoint_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({
                "model_name": meta.model_name,
                "global_steps": meta.global_steps,
                "total_size": meta.total_size,
                "tensors": {
                    name: {
                        "name": t.name,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "storage_key": t.storage_key,
                        "size": t.size,
                    }
                    for name, t in meta.tensors.items()
                },
            }, f)
        
        return checkpoint_path
    
    def load_metadata(self, checkpoint_path: str) -> CheckpointMetadata:
        """Load checkpoint metadata from NFS."""
        import json
        import os
        
        meta_path = os.path.join(
            self._get_checkpoint_dir(checkpoint_path), "metadata.json"
        )
        with open(meta_path, "r") as f:
            data = json.load(f)
        
        tensors = {}
        for name, t in data["tensors"].items():
            tensors[name] = TensorMeta(
                name=t["name"],
                shape=torch.Size(t["shape"]),
                dtype=getattr(torch, t["dtype"].replace("torch.", "")),
                storage_key=t["storage_key"],
                size=t["size"],
            )
        
        return CheckpointMetadata(
            model_name=data["model_name"],
            global_steps=data["global_steps"],
            tensors=tensors,
            total_size=data["total_size"],
        )
    
    def load_tensor(self, tensor_meta: TensorMeta) -> torch.Tensor:
        """Load a single tensor from NFS."""
        from safetensors import safe_open
        
        model_path = os.path.join(
            self.base_path, tensor_meta.storage_key
        )
        with safe_open(model_path, framework="pt", device="cpu") as f:
            return f.get_tensor(tensor_meta.name)
    
    def get_weights(
        self, checkpoint_path: str
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get all weights from NFS checkpoint."""
        from safetensors import safe_open
        import os
        
        model_path = os.path.join(
            self._get_checkpoint_dir(checkpoint_path), "model.safetensors"
        )
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)
    
    def exists(self, checkpoint_path: str) -> bool:
        """Check if checkpoint exists on NFS."""
        import os
        checkpoint_dir = self._get_checkpoint_dir(checkpoint_path)
        return os.path.exists(
            os.path.join(checkpoint_dir, "model.safetensors")
        )
    
    def delete(self, checkpoint_path: str) -> None:
        """Delete checkpoint from NFS."""
        import shutil
        checkpoint_dir = self._get_checkpoint_dir(checkpoint_path)
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)


class MooncakeStoreBackend(StorageBackend):
    """Mooncake Store-based storage backend.

    Uses MooncakeDistributedStore for high-performance RDMA-based
    distributed weight storage. Supports both embedded mode (each node
    contributes memory) and standalone-store mode (external store process).

    Key naming convention:
        metadata:  ckpt:{checkpoint_path}:metadata
        tensor:    ckpt:{checkpoint_path}:tensor:{tensor_name}
        index:     ckpt:{checkpoint_path}:index

    Config keys:
        metadata_server: Mooncake metadata server URL
            (e.g. "http://127.0.0.1:8080/metadata")
        master_server_address: Mooncake master server address
            (e.g. "127.0.0.1:50051")
        protocol: Transfer protocol ("rdma" or "tcp")
        device_name: RDMA device name (e.g. "mlx5_0")
        global_segment_size: Per-node memory pool size in bytes
        local_buffer_size: Local transfer buffer size in bytes
        tenant_id: Mooncake tenant ID (optional)
        preferred_segment: Preferred storage segment (optional)
    """

    # Default pool sizes
    DEFAULT_GLOBAL_SEGMENT = 512 * 1024 * 1024  # 512 MB
    DEFAULT_LOCAL_BUFFER = 64 * 1024 * 1024  # 64 MB

    def __init__(self):
        self.store = None
        self.replicate_config = None
        self._config: dict[str, Any] = {}
        self._initialized = False

    def _make_key(self, checkpoint_path: str, suffix: str) -> str:
        """Build a store key for a checkpoint component."""
        return f"ckpt:{checkpoint_path}:{suffix}"

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize MooncakeDistributedStore.

        Args:
            config: Mooncake connection configuration.
        """
        try:
            from mooncake.store import (
                MooncakeDistributedStore,
                ReplicateConfig,
            )
        except ImportError:
            try:
                from mooncake import (
                    MooncakeDistributedStore,
                    ReplicateConfig,
                )
            except ImportError as e:
                raise ImportError(
                    "Mooncake is not installed. "
                    "Please install mooncake-transfer-engine or build from "
                    "source: https://github.com/kvcache-ai/Mooncake"
                ) from e

        import socket
        self._config = config

        metadata_server = config.get(
            "metadata_server",
            "http://127.0.0.1:8080/metadata",
        )
        master = config.get("master_server_address", "127.0.0.1:50051")
        protocol = config.get("protocol", "tcp")
        device_name = config.get("device_name", "")
        segment_size = config.get(
            "global_segment_size", self.DEFAULT_GLOBAL_SEGMENT
        )
        local_buffer = config.get(
            "local_buffer_size", self.DEFAULT_LOCAL_BUFFER
        )
        tenant_id = config.get("tenant_id")

        hostname = config.get(
            "hostname", socket.gethostname()
        )

        self.store = MooncakeDistributedStore()
        setup_kwargs: dict[str, str] = {}
        if tenant_id:
            setup_kwargs["tenant_id"] = tenant_id

        ret = self.store.setup(
            hostname,
            metadata_server,
            segment_size,
            local_buffer,
            protocol,
            device_name,
            master,
            **setup_kwargs,
        )
        if ret != 0:
            raise RuntimeError(
                f"MooncakeDistributedStore setup failed: {ret}"
            )

        self.replicate_config = ReplicateConfig()
        self.replicate_config.with_soft_pin = True
        preferred = config.get("preferred_segment")
        if preferred:
            self.replicate_config.preferred_segment = preferred

        self._initialized = True

    def save_checkpoint(
        self,
        checkpoint_path: str,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save checkpoint to Mooncake Store.

        Stores each tensor individually as a keyed entry, plus a metadata
        index entry describing the full checkpoint layout.

        Args:
            checkpoint_path: Logical checkpoint path.
            weights: Generator of (name, tensor) pairs.
            metadata: Optional metadata (model_name, global_steps, etc.).

        Returns:
            The checkpoint_path on success.
        """
        import json
        import time
        import logging
        logger = logging.getLogger(__name__)

        if not self._initialized:
            raise RuntimeError("MooncakeStoreBackend not initialized")

        start = time.monotonic()
        tensor_index: dict[str, dict] = {}
        total_bytes = 0
        count = 0

        for name, tensor in weights:
            tensor_key = self._make_key(
                checkpoint_path, f"tensor:{name}"
            )
            # Serialize tensor to bytes: [dtype_str, shape..., data]
            raw = self._serialize_tensor(tensor)

            ret = self.store.put(
                tensor_key, raw, self.replicate_config
            )
            if isinstance(ret, bool):
                ok = ret
            else:
                ok = ret is None or ret == 0
            if not ok:
                raise RuntimeError(
                    f"Mooncake put failed for {tensor_key}: {ret}"
                )

            tensor_index[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "nbytes": tensor.nbytes,
            }
            total_bytes += tensor.nbytes
            count += 1

        # Store metadata + index
        meta_payload = json.dumps({
            "model_name": (metadata or {}).get(
                "model_name", "unknown"
            ),
            "global_steps": (metadata or {}).get(
                "global_steps", 0
            ),
            "total_size": total_bytes,
            "num_tensors": count,
            "tensors": tensor_index,
        }).encode("utf-8")

        meta_key = self._make_key(checkpoint_path, "metadata")
        ret = self.store.put(
            meta_key, meta_payload, self.replicate_config
        )
        if isinstance(ret, bool):
            ok = ret
        else:
            ok = ret is None or ret == 0
        if not ok:
            raise RuntimeError(
                f"Mooncake metadata put failed: {ret}"
            )

        elapsed = time.monotonic() - start
        bw = total_bytes / elapsed / (1024 ** 3) if elapsed > 0 else 0
        logger.info(
            f"Mooncake save checkpoint {checkpoint_path}: "
            f"{count} tensors, {total_bytes / 1024 ** 2:.1f} MB, "
            f"{elapsed:.2f}s, {bw:.2f} GB/s"
        )
        return checkpoint_path

    def load_metadata(self, checkpoint_path: str) -> CheckpointMetadata:
        """Load checkpoint metadata from Mooncake Store."""
        import json

        if not self._initialized:
            raise RuntimeError("MooncakeStoreBackend not initialized")

        meta_key = self._make_key(checkpoint_path, "metadata")
        raw = self.store.get(meta_key)
        if not raw:
            raise FileNotFoundError(
                f"Checkpoint metadata not found: {checkpoint_path}"
            )

        data = json.loads(
            raw if isinstance(raw, str) else raw.decode("utf-8")
        )

        tensors = {}
        for name, t in data["tensors"].items():
            tensors[name] = TensorMeta(
                name=name,
                shape=torch.Size(t["shape"]),
                dtype=getattr(
                    torch, t["dtype"].replace("torch.", "")
                ),
                storage_key=self._make_key(
                    checkpoint_path, f"tensor:{name}"
                ),
                size=t["nbytes"],
            )

        return CheckpointMetadata(
            model_name=data["model_name"],
            global_steps=data["global_steps"],
            tensors=tensors,
            total_size=data["total_size"],
        )

    def load_tensor(self, tensor_meta: TensorMeta) -> torch.Tensor:
        """Load a single tensor from Mooncake Store."""
        if not self._initialized:
            raise RuntimeError("MooncakeStoreBackend not initialized")

        raw = self.store.get(tensor_meta.storage_key)
        if not raw:
            raise FileNotFoundError(
                f"Tensor not found: {tensor_meta.storage_key}"
            )
        return self._deserialize_tensor(raw, tensor_meta)

    def get_weights(
        self, checkpoint_path: str
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get all weights from Mooncake Store.

        Yields named tensors streamed from the distributed store.
        """
        if not self._initialized:
            raise RuntimeError("MooncakeStoreBackend not initialized")

        meta = self.load_metadata(checkpoint_path)

        for name, tensor_meta in meta.tensors.items():
            raw = self.store.get(tensor_meta.storage_key)
            if not raw:
                raise FileNotFoundError(
                    f"Tensor not found: {tensor_meta.storage_key}"
                )
            tensor = self._deserialize_tensor(raw, tensor_meta)
            yield name, tensor

    def exists(self, checkpoint_path: str) -> bool:
        """Check if checkpoint exists in Mooncake Store."""
        if not self._initialized:
            return False
        meta_key = self._make_key(checkpoint_path, "metadata")
        raw = self.store.get(meta_key)
        return raw is not None and len(raw) > 0

    def delete(self, checkpoint_path: str) -> None:
        """Delete checkpoint from Mooncake Store.

        Note: MooncakeDistributedStore does not have an explicit delete
        API. Data will be garbage collected when the store is under
        memory pressure. This method is a no-op.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "Mooncake Store does not support explicit delete. "
            "Checkpoint %s will be garbage collected.",
            checkpoint_path,
        )

    # ---- tensor serialization helpers ----

    @staticmethod
    def _serialize_tensor(tensor: torch.Tensor) -> bytes:
        """Serialize a tensor to bytes.

        Format: [4B dtype_len][dtype_str][8*ndim B shape][raw data]
        """
        import struct

        dtype_str = str(tensor.dtype).encode("utf-8")
        shape_bytes = struct.pack(
            f"{len(tensor.shape)}Q", *tensor.shape
        )
        header = struct.pack("I", len(dtype_str)) + dtype_str
        header += struct.pack("I", len(tensor.shape))
        header += shape_bytes
        data = tensor.contiguous().cpu().numpy().tobytes()
        return header + data

    @staticmethod
    def _deserialize_tensor(
        raw: bytes, meta: TensorMeta
    ) -> torch.Tensor:
        """Deserialize bytes to a tensor using metadata."""
        import struct

        buf = raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)
        offset = 0

        # dtype
        (dtype_len,) = struct.unpack_from("I", buf, offset)
        offset += 4
        dtype_str = buf[offset:offset + dtype_len].decode("utf-8")
        offset += dtype_len

        # shape
        (ndim,) = struct.unpack_from("I", buf, offset)
        offset += 4
        shape = struct.unpack_from(f"{ndim}Q", buf, offset)
        offset += 8 * ndim

        # data
        dtype = getattr(torch, dtype_str.replace("torch.", ""))
        numel = 1
        for s in shape:
            numel *= s
        nbytes = numel * torch.tensor(
            [], dtype=dtype
        ).element_size()
        data = buf[offset:offset + nbytes]

        tensor = torch.frombuffer(
            bytearray(data), dtype=dtype
        ).view(*shape)
        return tensor.clone()


class StorageBackendFactory:
    """Factory for creating storage backends."""
    
    _registry: dict[str, type[StorageBackend]] = {
        "nfs": NFSStorageBackend,
        "mooncake": MooncakeStoreBackend,
    }
    
    @classmethod
    def register(cls, name: str, backend_cls: type[StorageBackend]) -> None:
        """Register a storage backend."""
        cls._registry[name] = backend_cls
    
    @classmethod
    def create(cls, name: str, config: dict[str, Any]) -> StorageBackend:
        """Create a storage backend instance.
        
        Args:
            name: Backend name
            config: Backend configuration
            
        Returns:
            Initialized storage backend
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown storage backend: {name}. "
                f"Available: {list(cls._registry.keys())}"
            )
        backend = cls._registry[name]()
        backend.initialize(config)
        return backend


class StorageCheckpointEngine:
    """Checkpoint engine that loads weights from persistent storage.
    
    This engine is compatible with verl's CheckpointEngineWithCache interface.
    It provides a `get_weights()` method that yields named tensors from storage,
    which can be used to update model weights in ExternalExecutor.
    
    Supported backends:
        - "nfs": NFS/shared filesystem with safetensors format
        - "mooncake": MooncakeDistributedStore with RDMA transfer
    
    Usage:
        # Create engine
        engine = StorageCheckpointEngine(
            backend="nfs",
            config={"base_path": "/shared/checkpoints"},
        )
        
        # Load weights from storage
        for name, tensor in engine.get_weights("model_v1/step_100"):
            model.get_parameter(name).data.copy_(tensor)
    
    Architecture:
        Storage (NFS/Mooncake) → StorageCheckpointEngine → ExternalExecutor
    """
    
    # Wire format compatible with verl
    wire_format = "named_tensors"
    
    def __init__(
        self,
        backend: str = "nfs",
        config: dict[str, Any] | None = None,
        bucket_size: int = 256 * 1024 * 1024,  # 256 MB
        device: str = "cuda",
    ):
        """Initialize storage checkpoint engine.
        
        Args:
            backend: Storage backend name ("nfs" or "mooncake")
            config: Backend-specific configuration
            bucket_size: Bucket size for weight transfer
            device: Target device for loaded tensors
        """
        self.backend_name = backend
        self.config = config or {}
        self.bucket_size = bucket_size
        self.device = device
        
        # Create storage backend
        self.storage = StorageBackendFactory.create(backend, self.config)
        
        self._current_checkpoint: str | None = None
    
    def prepare(self) -> dict[str, Any]:
        """Prepare checkpoint engine (no-op for storage backend).
        
        Returns:
            Empty metadata dict
        """
        return {}
    
    @classmethod
    def build_topology(
        cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Build topology (no-op for storage backend).
        
        Storage backend doesn't need communication topology.
        """
        return {}, {}
    
    def init_process_group(self, **kwargs) -> None:
        """Init process group (no-op for storage backend).
        
        Storage backend doesn't need process group.
        """
        pass
    
    def finalize(self) -> None:
        """Finalize checkpoint engine (no-op for storage backend)."""
        pass
    
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ) -> None:
        """Send weights to storage.
        
        Args:
            weights: Generator of (name, tensor) pairs
            global_steps: Optional global step count
        """
        if self._current_checkpoint is None:
            raise ValueError("No checkpoint path set. Call set_checkpoint() first.")
        
        metadata = {"global_steps": global_steps} if global_steps else None
        self.storage.save_checkpoint(self._current_checkpoint, weights, metadata)
    
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive weights from storage.
        
        Args:
            global_steps: Optional global step count
            
        Yields:
            (name, tensor) pairs
        """
        if self._current_checkpoint is None:
            raise ValueError("No checkpoint path set. Call set_checkpoint() first.")
        
        # Note: We call the sync get_weights() and yield each item
        # This maintains compatibility with verl's async interface
        for name, tensor in self.get_weights():
            yield name, tensor
    
    def get_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get weights from storage.
        
        This is the main method for loading weights from persistent storage.
        It yields named tensors that can be used to update model weights.
        
        Yields:
            (name, tensor) pairs
        """
        if self._current_checkpoint is None:
            raise ValueError("No checkpoint path set. Call set_checkpoint() first.")
        
        for name, tensor in self.storage.get_weights(self._current_checkpoint):
            # Move tensor to target device
            tensor = tensor.to(self.device)
            yield name, tensor
    
    def set_checkpoint(self, checkpoint_path: str) -> None:
        """Set the current checkpoint path.
        
        Args:
            checkpoint_path: Logical path for the checkpoint
        """
        self._current_checkpoint = checkpoint_path
    
    def exists(self, checkpoint_path: str) -> bool:
        """Check if a checkpoint exists in storage.
        
        Args:
            checkpoint_path: Logical path for the checkpoint
            
        Returns:
            True if checkpoint exists
        """
        return self.storage.exists(checkpoint_path)
    
    def load_metadata(self, checkpoint_path: str) -> CheckpointMetadata:
        """Load checkpoint metadata.
        
        Args:
            checkpoint_path: Logical path for the checkpoint
            
        Returns:
            Checkpoint metadata
        """
        return self.storage.load_metadata(checkpoint_path)
    
    def delete(self, checkpoint_path: str) -> None:
        """Delete a checkpoint from storage.
        
        Args:
            checkpoint_path: Logical path for the checkpoint
        """
        self.storage.delete(checkpoint_path)


# Register with verl's CheckpointEngineRegistry if available
try:
    from verl.checkpoint_engine.base import CheckpointEngineRegistry
    CheckpointEngineRegistry.register("storage")(StorageCheckpointEngine)
except ImportError:
    # verl is not installed, skip registration
    pass
