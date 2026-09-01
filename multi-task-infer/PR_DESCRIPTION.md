# ExternalExecutor Plugin: Actor Pooling + Storage Backend

## Summary

This PR introduces **ExternalExecutor**, a vLLM plugin that enables actor pooling, compilation cache sharing, and storage-based weight loading for faster multi-task inference.

## Key Features

### 1. Actor Pooling
- Pre-start Ray Actors bound to GPU/NPU devices
- Reuse actors across multiple vLLM instances
- Eliminate initialization overhead (device setup, library imports, NCCL/HCCl warmup)

### 2. Compilation Cache Sharing
- **CacheManagerActor**: Independent Ray Actor managing torch.compile caches
- **Lazy-loading pattern**: Check local → Pull from manager → Fallback compile
- **Compile lock**: Prevents duplicate compilation across nodes
- **Hybrid storage**: Ray Object Store (<50MB) + NFS compression (>=50MB)

### 3. Storage Checkpoint Engine
- Load model weights from persistent storage (NFS/Mooncake Store)
- Compatible with verl's `CheckpointEngineWithCache` interface
- Can register to verl's `CheckpointEngineRegistry`

## Storage Backends

| Backend | Status | Performance | Use Case |
|---------|--------|-------------|----------|
| **NFS** | ✅ Implemented | Depends on network | Shared filesystem, simple setup |
| **Mooncake Store** | ✅ Implemented | ~9 GB/s (InfiniBand) | Large-scale clusters, RDMA |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Storage (NFS/Mooncake Store)                        │
│  - Model checkpoints (safetensors)                               │
│  - Compilation caches                                            │
└─────────────────────────────────────────────────────────────────┘
         │                              │
         ↓                              ↓
┌─────────────────────┐    ┌─────────────────────────────────────┐
│  StorageCheckpoint  │    │        CacheManagerActor            │
│     Engine          │    │  - Compilation cache sharing         │
│  - Load model       │    │  - pull/push API                     │
│    weights          │    │  - Compile lock coordination         │
└─────────────────────┘    └─────────────────────────────────────┘
         │                              │
         ↓                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              ExternalExecutor Workers                            │
│  - Pre-started Ray Actors bound to GPU/NPU devices              │
│  - Load model via StorageCheckpointEngine                       │
│  - Share compilation cache via CacheManagerActor                │
└─────────────────────────────────────────────────────────────────┘
```

## Performance

| Scenario | Without Sharing | With Sharing |
|----------|----------------|--------------|
| First node (compile) | 30-180s | 30-180s |
| Second node | 30-180s | 5-10s (cache pull) |
| Subsequent | 5-10s (local) | 5-10s (local) |

## Usage Example

```python
from vllm_external_executor import ActorPoolManager, ExternalExecutor

# 1. Create ActorPoolManager (auto-creates CacheManagerActor)
pool = ActorPoolManager()
pool.pre_start(
    num_actors=8,
    devices_per_node=[0, 1, 2, 3, 4, 5, 6, 7],
    warmup_distributed=True,
    shared_cache_dir="/shared/vllm_compile_cache",
)

# 2. Acquire actors and create vLLM instance
actors = pool.acquire(tp_size=4, pp_size=2)

llm = LLM(
    model="facebook/opt-125m",
    executor_class=ExternalExecutor,
    external_actors=actors,
    cache_manager=pool.cache_manager,
)
```

### Storage Backend Usage

```python
from vllm_external_executor import StorageCheckpointEngine

# NFS backend
engine = StorageCheckpointEngine(
    backend="nfs",
    config={"base_path": "/shared/checkpoints"},
    device="cuda",
)
engine.set_checkpoint("opt-125m/step_1000")

for name, tensor in engine.get_weights():
    model.get_parameter(name).data.copy_(tensor)

# Mooncake Store backend
engine = StorageCheckpointEngine(
    backend="mooncake",
    config={
        "metadata_server": "http://127.0.0.1:8080/metadata",
        "master_server_address": "127.0.0.1:50051",
        "protocol": "rdma",
    },
    device="cuda",
)
```

## vLLM Core Changes

Minimal parameter threading only (4 files):
- `vllm/v1/engine/async_llm.py`: Add `external_actors` param
- `vllm/v1/engine/core_client.py`: Thread parameter
- `vllm/v1/engine/utils.py`: Thread parameter
- `vllm/v1/engine/core.py`: Pass to executor

No changes to vLLM's core logic.

## Files

```
multi-task-infer/
├── README.md                                    # Main documentation
├── design.md                                    # 4+1 view design document
├── STORAGE_CHECKPOINT_ENGINE_DESIGN.md          # Storage backend design
├── STARTUP_DEPENDENCIES.md                      # Startup dependencies
├── pyproject.toml                               # Plugin package config
├── vllm_external_executor/                      # Plugin code
│   ├── __init__.py                              # Module entry + plugin registration
│   ├── external_worker_actor.py                 # Pre-started Ray Actor
│   ├── actor_pool_manager.py                    # Actor pool manager
│   ├── external_executor.py                     # ExternalExecutor implementation
│   ├── cache_manager_actor.py                   # CacheManagerActor implementation
│   └── storage_checkpoint_engine.py             # Storage backend checkpoint engine
├── examples/
│   ├── basic_usage.py                           # Usage examples
│   └── mooncake_config.json                     # Mooncake config template
├── tests/
│   └── test_storage_checkpoint_engine.py        # Test cases
└── verify_dependencies.sh                       # Dependency verification script
```

## Testing

```bash
# NFS backend test (no external dependencies)
python tests/test_storage_checkpoint_engine.py nfs

# Mooncake mock test (no services needed)
python tests/test_storage_checkpoint_engine.py mooncake_mock

# Mooncake integration test (requires services)
MOONCAKE_CONFIG_PATH=examples/mooncake_config.json \
    python tests/test_storage_checkpoint_engine.py mooncake

# Verify all dependencies
bash verify_dependencies.sh
```

## Dependencies

### Core (required)
- `vllm`
- `ray[default]>=2.9`
- `safetensors>=0.4.0`

### Mooncake Store (optional)
- `mooncake-transfer-engine`
- `mooncake_master` service running
- RDMA network (recommended) or TCP fallback

## Compatibility

- **verl**: `StorageCheckpointEngine` is compatible with verl's `CheckpointEngineWithCache` interface
- **Hardware**: Supports both NVIDIA GPU (CUDA) and Ascend NPU
- **Python**: 3.9+

## Future Work

- [ ] Mooncake batch API (`batch_put_from_multi_buffers`) for better throughput
- [ ] Delta/incremental weight updates
- [ ] Cache eviction policies
- [ ] CacheManagerActor high availability

## Checklist

- [x] Code follows vLLM style guidelines (88 char line limit, Google-style docstrings)
- [x] Tests included (NFS, Mooncake mock, Mooncake integration)
- [x] Documentation complete (README, design docs, startup guide)
- [x] Minimal vLLM core changes (parameter threading only)
- [x] verl-compatible interface
- [x] Supports both GPU and NPU
