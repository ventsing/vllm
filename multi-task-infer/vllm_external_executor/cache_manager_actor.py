# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
CacheManagerActor - A Ray Actor that manages compilation cache sharing.

This actor serves as the central cache manager for the cluster:
- Holds Ray Object Store references for small caches
- Manages NFS files for large caches
- Provides pull/push API for workers
- Coordinates compilation locks to avoid duplicate work

Workers follow a lazy-loading pattern:
1. Check local cache → hit: use directly
2. Pull from CacheManagerActor → hit: write local, use
3. Fallback: compile locally, then push to CacheManagerActor
"""

import gzip
import hashlib
import io
import json
import logging
import os
import tarfile
import tempfile
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class CacheManagerActor:
    """
    A Ray Actor that manages compilation cache sharing across the cluster.
    
    This actor is the single source of truth for compilation caches.
    Workers pull caches from it when needed, and push compiled caches
    back after compilation.
    
    It uses a hybrid storage strategy:
    - Small caches (< threshold): Ray Object Store (fast, zero-copy)
    - Large caches (>= threshold): NFS with compression
    
    It also provides compilation locking to prevent multiple workers
    from compiling the same cache simultaneously.
    """
    
    RAY_OBJECT_THRESHOLD = 50 * 1024 * 1024  # 50 MB
    COMPRESSION_LEVEL = 6
    
    def __init__(
        self,
        shared_cache_dir: str | None = None,
        enable_compression: bool = True,
    ):
        """
        Initialize the cache manager actor.
        
        Args:
            shared_cache_dir: NFS directory for large caches (optional)
            enable_compression: Whether to compress large caches
        """
        self.shared_cache_dir = shared_cache_dir
        self.enable_compression = enable_compression
        
        # Ray Object Store registry: hash_key -> ObjectRef
        self._object_registry: dict[str, Any] = {}
        
        # Compilation lock registry: hash_key -> lock info
        # Prevents multiple workers from compiling the same cache
        self._compile_locks: dict[str, dict] = {}
        
        # Metadata: hash_key -> {size, created_at, source}
        self._metadata: dict[str, dict] = {}
        
        logger.info(
            f"CacheManagerActor initialized: "
            f"shared_dir={shared_cache_dir}, "
            f"compression={enable_compression}"
        )
    
    def pull(self, hash_key: str) -> bytes | None:
        """
        Pull cache from the manager.
        
        Called by workers when local cache miss.
        Tries Ray Object Store first, then NFS.
        
        Args:
            hash_key: Cache hash key
        
        Returns:
            Cache data (tar.gz bytes), or None if not found
        """
        # Try Ray Object Store first
        if hash_key in self._object_registry:
            try:
                import ray
                data = ray.get(self._object_registry[hash_key])
                logger.info(
                    f"Cache hit in Ray Object Store: {hash_key} "
                    f"({len(data) / 1024 / 1024:.1f} MB)"
                )
                return data
            except Exception as e:
                logger.warning(
                    f"Failed to get from Ray Object Store: {e}"
                )
        
        # Try NFS
        if self.shared_cache_dir:
            data = self._read_from_nfs(hash_key)
            if data is not None:
                logger.info(
                    f"Cache hit in NFS: {hash_key} "
                    f"({len(data) / 1024 / 1024:.1f} MB)"
                )
                return data
        
        logger.info(f"Cache miss: {hash_key}")
        return None
    
    def push(
        self,
        hash_key: str,
        cache_data: bytes,
        source: str = "unknown",
    ) -> bool:
        """
        Push cache to the manager.
        
        Called by workers after compilation.
        Automatically selects storage based on size.
        
        Args:
            hash_key: Cache hash key
            cache_data: Cache data (tar.gz bytes)
            source: Source identifier (e.g., "worker-0", "node-1")
        
        Returns:
            True if push succeeded
        """
        cache_size = len(cache_data)
        
        # Store metadata
        self._metadata[hash_key] = {
            "size": cache_size,
            "created_at": time.time(),
            "source": source,
        }
        
        # Select storage
        if cache_size < self.RAY_OBJECT_THRESHOLD:
            return self._push_to_ray(hash_key, cache_data)
        else:
            return self._push_to_nfs(hash_key, cache_data)
    
    def try_acquire_compile_lock(
        self,
        hash_key: str,
        worker_id: str,
    ) -> dict:
        """
        Try to acquire a compilation lock.
        
        This prevents multiple workers from compiling the same cache.
        
        Args:
            hash_key: Cache hash key
            worker_id: Worker identifier
        
        Returns:
            Dict with status:
            - {"status": "acquired"} - lock acquired, proceed to compile
            - {"status": "wait", "holder": worker_id} - another worker
              is compiling, wait and pull later
            - {"status": "done"} - cache already exists, just pull
        """
        # Check if cache already exists
        if self._cache_exists(hash_key):
            return {"status": "done"}
        
        # Check if another worker is compiling
        if hash_key in self._compile_locks:
            lock_info = self._compile_locks[hash_key]
            # Check if the lock is stale (> 10 minutes)
            if time.time() - lock_info["acquired_at"] < 600:
                return {
                    "status": "wait",
                    "holder": lock_info["worker_id"],
                }
            else:
                logger.warning(
                    f"Stale compile lock for {hash_key}, "
                    f"held by {lock_info['worker_id']}"
                )
        
        # Acquire lock
        self._compile_locks[hash_key] = {
            "worker_id": worker_id,
            "acquired_at": time.time(),
        }
        
        logger.info(
            f"Compile lock acquired for {hash_key} by {worker_id}"
        )
        return {"status": "acquired"}
    
    def release_compile_lock(self, hash_key: str, worker_id: str):
        """
        Release a compilation lock.
        
        Called after compilation is done (success or failure).
        
        Args:
            hash_key: Cache hash key
            worker_id: Worker identifier
        """
        if hash_key in self._compile_locks:
            lock_info = self._compile_locks[hash_key]
            if lock_info["worker_id"] == worker_id:
                del self._compile_locks[hash_key]
                logger.info(
                    f"Compile lock released for {hash_key} "
                    f"by {worker_id}"
                )
    
    def get_cache_hash(
        self,
        model_config: dict,
        parallel_config: dict,
        compilation_config: dict,
    ) -> str:
        """
        Compute cache hash key from configuration dicts.
        
        This is a remote-callable version that accepts dicts
        instead of vLLM config objects.
        
        Args:
            model_config: Model config as dict
            parallel_config: Parallel config as dict
            compilation_config: Compilation config as dict
        
        Returns:
            10-character hex string hash key
        """
        factors = {
            "model_arch": model_config.get("architectures", ["unknown"])[0],
            "model_revision": model_config.get("revision", "unknown"),
            "tp_size": parallel_config.get("tensor_parallel_size", 1),
            "pp_size": parallel_config.get("pipeline_parallel_size", 1),
            "dp_size": parallel_config.get("data_parallel_size", 1),
            "batch_sizes": compilation_config.get(
                "cudagraph_capture_sizes", []
            ),
            "cudagraph_mode": compilation_config.get("cudagraph_mode", 0),
        }
        
        hash_content = json.dumps(factors, sort_keys=True)
        return hashlib.sha256(hash_content.encode()).hexdigest()[:10]
    
    def get_stats(self) -> dict:
        """Get cache manager statistics."""
        return {
            "ray_objects": len(self._object_registry),
            "nfs_caches": self._count_nfs_caches(),
            "compile_locks": len(self._compile_locks),
            "metadata": self._metadata,
        }
    
    def list_caches(self) -> list[str]:
        """List all available cache hash keys."""
        keys = set(self._object_registry.keys())
        if self.shared_cache_dir and os.path.exists(self.shared_cache_dir):
            for f in os.listdir(self.shared_cache_dir):
                if f.endswith(".bin"):
                    keys.add(f[:-4])
        return sorted(keys)
    
    # ========== Internal methods ==========
    
    def _cache_exists(self, hash_key: str) -> bool:
        """Check if cache exists in any storage."""
        if hash_key in self._object_registry:
            return True
        if self.shared_cache_dir:
            path = os.path.join(self.shared_cache_dir, f"{hash_key}.bin")
            if os.path.exists(path):
                return True
        return False
    
    def _push_to_ray(self, hash_key: str, data: bytes) -> bool:
        """Push cache to Ray Object Store."""
        try:
            import ray
            object_ref = ray.put(data)
            self._object_registry[hash_key] = object_ref
            logger.info(
                f"Pushed to Ray Object Store: {hash_key} "
                f"({len(data) / 1024 / 1024:.1f} MB)"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to push to Ray: {e}")
            return False
    
    def _push_to_nfs(self, hash_key: str, data: bytes) -> bool:
        """Push cache to NFS with compression."""
        if not self.shared_cache_dir:
            logger.warning(
                f"Cache {hash_key} too large for Ray Object Store "
                f"({len(data) / 1024 / 1024:.1f} MB), "
                f"but no NFS configured"
            )
            return False
        
        try:
            # Compress
            if self.enable_compression:
                data = gzip.compress(data, compresslevel=self.COMPRESSION_LEVEL)
            
            # Atomic write
            target = os.path.join(
                self.shared_cache_dir, f"{hash_key}.bin"
            )
            os.makedirs(self.shared_cache_dir, exist_ok=True)
            
            fd, temp_path = tempfile.mkstemp(
                dir=self.shared_cache_dir, suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                os.replace(temp_path, target)
            except Exception:
                os.unlink(temp_path)
                raise
            
            logger.info(
                f"Pushed to NFS: {hash_key} "
                f"({len(data) / 1024 / 1024:.1f} MB)"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to push to NFS: {e}")
            return False
    
    def _read_from_nfs(self, hash_key: str) -> bytes | None:
        """Read cache from NFS with decompression."""
        if not self.shared_cache_dir:
            return None
        
        path = os.path.join(self.shared_cache_dir, f"{hash_key}.bin")
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, "rb") as f:
                data = f.read()
            
            # Try decompress
            try:
                data = gzip.decompress(data)
            except (gzip.BadGzipFile, OSError):
                pass  # Not compressed
            
            return data
        except Exception as e:
            logger.error(f"Failed to read from NFS: {e}")
            return None
    
    def _count_nfs_caches(self) -> int:
        """Count caches in NFS."""
        if not self.shared_cache_dir:
            return 0
        if not os.path.exists(self.shared_cache_dir):
            return 0
        return len([
            f for f in os.listdir(self.shared_cache_dir)
            if f.endswith(".bin")
        ])


def create_cache_manager_actor(
    shared_cache_dir: str | None = None,
    enable_compression: bool = True,
) -> Any:
    """
    Create a CacheManagerActor as a Ray Actor.
    
    Args:
        shared_cache_dir: NFS directory for large caches
        enable_compression: Whether to compress large caches
    
    Returns:
        Ray Actor handle
    """
    import ray
    
    actor = ray.remote(CacheManagerActor).options(
        num_cpus=0.5,
        memory=500 * 1024 * 1024,  # 500 MB for Object Store refs
    ).remote(
        shared_cache_dir=shared_cache_dir,
        enable_compression=enable_compression,
    )
    
    return actor


def package_local_cache(hash_key: str) -> bytes | None:
    """
    Package local cache directory into tar.gz bytes.
    
    Args:
        hash_key: Cache hash key
    
    Returns:
        tar.gz bytes, or None if not found
    """
    local_dir = os.path.expanduser(
        f"~/.cache/vllm/torch_compile_cache/{hash_key}"
    )
    
    if not os.path.exists(local_dir):
        return None
    
    try:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            tar.add(local_dir, arcname=hash_key)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Failed to package local cache: {e}")
        return None


def extract_cache_to_local(hash_key: str, data: bytes) -> bool:
    """
    Extract cache data to local cache directory.
    
    Args:
        hash_key: Cache hash key
        data: tar.gz bytes
    
    Returns:
        True if extraction succeeded
    """
    local_dir = os.path.expanduser("~/.cache/vllm/torch_compile_cache")
    target_dir = os.path.join(local_dir, hash_key)
    
    if os.path.exists(target_dir):
        return True
    
    try:
        os.makedirs(local_dir, exist_ok=True)
        buffer = io.BytesIO(data)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(path=local_dir)
        return True
    except Exception as e:
        logger.error(f"Failed to extract cache: {e}")
        return False


def local_cache_exists(hash_key: str) -> bool:
    """Check if cache exists in local directory."""
    local_dir = os.path.expanduser(
        f"~/.cache/vllm/torch_compile_cache/{hash_key}"
    )
    return os.path.exists(local_dir)
