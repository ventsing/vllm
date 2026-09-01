# ExternalExecutor 多任务推理插件

vLLM 的 ExternalExecutor 插件，支持 Actor 池化、模型热切换、编译缓存共享和存储后端权重加载。

## 核心特性

- **Actor 池化**：预启动一组 Ray Actor，绑定设备并预热公共库
- **动态重组**：支持动态调整 TP/PP 大小，重新分配 Actor
- **模型热切换**：通过 weight_transfer 或 storage backend 动态加载新模型权重
- **数据通路复用**：复用 RayExecutorV2 的 MessageQueue 通信机制
- **编译缓存共享**：通过 CacheManagerActor 实现跨节点缓存共享
- **存储后端加载**：通过 StorageCheckpointEngine 从 NFS/S3/Mooncake 加载模型权重

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│              Storage (NFS/S3/Mooncake Store)                     │
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

## 存储后端权重加载

### StorageCheckpointEngine

`StorageCheckpointEngine` 与 verl 的 `CheckpointEngineWithCache` 接口兼容，支持从持久化存储加载模型权重。

**支持的存储后端**：

| 后端 | 状态 | 说明 |
|------|------|------|
| **NFS** | ✅ 已实现 | 使用 safetensors 格式，适合共享文件系统 |
| **Mooncake Store** | ✅ 已实现 | 使用 MooncakeDistributedStore，RDMA 高性能 |

### 使用方式

```python
from vllm_external_executor import StorageCheckpointEngine

# 1. 创建 StorageCheckpointEngine
engine = StorageCheckpointEngine(
    backend="nfs",
    config={"base_path": "/shared/checkpoints"},
    device="cuda",
)

# 2. 设置 checkpoint 路径
engine.set_checkpoint("opt-125m/step_1000")

# 3. 从存储加载权重
for name, tensor in engine.get_weights():
    model.get_parameter(name).data.copy_(tensor)
```

### 在 ExternalWorkerActor 中使用

```python
# 在 Worker 中从存储加载模型
ray.get(actor.load_model_from_storage.remote(
    checkpoint_path="opt-125m/step_1000",
    storage_backend="nfs",
    storage_config={"base_path": "/shared/checkpoints"},
))
```

### 保存 checkpoint 到存储

```python
import asyncio
from vllm_external_executor import StorageCheckpointEngine

engine = StorageCheckpointEngine(
    backend="nfs",
    config={"base_path": "/shared/checkpoints"},
)
engine.set_checkpoint("my-model/step_100")

# 保存权重
def weight_generator():
    yield "model.layers.0.weight", torch.randn(1024, 1024)
    yield "model.layers.0.bias", torch.randn(1024)

asyncio.run(engine.send_weights(weight_generator(), global_steps=100))
```

## 编译缓存共享

### CacheManagerActor

CacheManagerActor 是独立的 Ray Actor，负责编译缓存的共享和管理。

**混合存储策略**：
- 小缓存（< 50 MB）：Ray Object Store（快，零拷贝同节点）
- 大缓存（>= 50 MB）：压缩(gzip) + 共享存储(NFS)

### 懒加载模式

```
Worker 需要编译缓存
  → ① 检查本地 (~/.cache/vllm/torch_compile_cache/)
    → 命中：直接使用
    → 未命中：② 向 CacheManagerActor 拉取
      → 命中：解压到本地，使用
      → 未命中：③ 尝试获取编译锁
        → 获得锁：编译，推送，释放锁
        → 其他 worker 正在编译：等待后拉取
        → 缓存已存在：直接拉取
```

### 使用方式

```python
from vllm_external_executor import ActorPoolManager, ExternalExecutor

# 创建 ActorPoolManager（自动创建 CacheManagerActor）
pool = ActorPoolManager()
pool.pre_start(
    num_actors=8,
    devices_per_node=[0, 1, 2, 3, 4, 5, 6, 7],
    warmup_distributed=True,
    shared_cache_dir="/shared/vllm_compile_cache",  # 可选
    enable_cache_compression=True,
)

# 获取 Actor 并创建 vLLM 实例
actors = pool.acquire(tp_size=4, pp_size=2)

llm = LLM(
    model="facebook/opt-125m",
    executor_class=ExternalExecutor,
    external_actors=actors,
    cache_manager=pool.cache_manager,  # 传递 CacheManagerActor
)
```

## 性能对比

| 场景 | 无共享存储 | 有共享存储 |
|------|-----------|-----------|
| 首次启动（节点 1） | 30-180s（编译） | 30-180s（编译） |
| 首次启动（节点 2） | 30-180s（编译） | 5-10s（拉取缓存） |
| 后续启动 | 5-10s（本地缓存） | 5-10s（本地缓存） |

## 安装

```bash
cd vllm/multi-task-infer
pip install -e .
```

## 环境变量

```bash
# NFS 共享缓存目录（可选）
export VLLM_SHARED_CACHE_DIR=/shared/vllm_compile_cache

# 是否启用压缩（默认启用）
export VLLM_CACHE_COMPRESSION=1

# Mooncake Store 配置（可选）
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

Mooncake 配置文件格式：
```json
{
    "metadata_server": "http://127.0.0.1:8080/metadata",
    "master_server_address": "127.0.0.1:50051",
    "protocol": "rdma",
    "device_name": "mlx5_0",
    "global_segment_size": 536870912,
    "local_buffer_size": 67108864
}
```

## 文件结构

```
multi-task-infer/
├── pyproject.toml                          # 插件包配置
├── design.md                               # 详细设计文档（4+1 视图）
├── CACHE_SIZE_AND_RAY_DESIGN.md            # 缓存大小分析
├── CACHE_MANAGER_ACTOR_DESIGN.md           # CacheManagerActor 设计
├── README.md                               # 本文件
├── vllm_external_executor/                 # 插件代码
│   ├── __init__.py                         # 模块入口 + plugin 注册
│   ├── external_worker_actor.py            # 预启动的 Ray Actor
│   ├── actor_pool_manager.py               # Actor 池管理器
│   ├── external_executor.py                # ExternalExecutor 实现
│   ├── cache_manager_actor.py              # CacheManagerActor 实现
│   ├── storage_checkpoint_engine.py        # 存储后端 checkpoint engine
│   └── compilation_cache.py                # 编译缓存工具（旧）
└── examples/                               # 使用示例
    └── basic_usage.py                      # 基础使用示例
```

## 与 verl 的兼容性

`StorageCheckpointEngine` 实现了与 verl `CheckpointEngineWithCache` 兼容的接口：

| verl 接口 | StorageCheckpointEngine | 说明 |
|-----------|------------------------|------|
| `prepare()` | ✅ | 准备引擎（no-op） |
| `build_topology()` | ✅ | 构建拓扑（no-op） |
| `init_process_group()` | ✅ | 初始化进程组（no-op） |
| `finalize()` | ✅ | 完成引擎（no-op） |
| `send_weights()` | ✅ | 发送权重到存储 |
| `receive_weights()` | ✅ | 从存储接收权重 |
| `get_weights()` | ✅ | 从存储获取权重 |

这意味着 `StorageCheckpointEngine` 可以直接注册到 verl 的 `CheckpointEngineRegistry`：

```python
from verl.checkpoint_engine.base import CheckpointEngineRegistry
CheckpointEngineRegistry.register("storage")(StorageCheckpointEngine)
```

## 开发计划

- [x] ExternalWorkerActor 实现
- [x] ActorPoolManager 实现
- [x] ExternalExecutor 实现
- [x] Plugin 注册机制
- [x] CacheManagerActor 实现（独立 Ray Actor）
- [x] 编译缓存懒加载模式
- [x] 编译锁机制（避免重复编译）
- [x] StorageCheckpointEngine 实现（NFS 后端）
- [x] Mooncake Store 后端实现
- [ ] vLLM 核心代码修改（最小侵入）
- [ ] 单元测试
- [ ] 模型热切换支持

