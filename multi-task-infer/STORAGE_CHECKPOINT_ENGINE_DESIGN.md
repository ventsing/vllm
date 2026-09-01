# StorageCheckpointEngine 设计文档

## 概述

`StorageCheckpointEngine` 是一个与 verl `CheckpointEngineWithCache` 接口兼容的存储后端 checkpoint engine，用于从持久化存储（NFS/Mooncake Store）加载模型权重。

## 设计目标

1. **与 verl 接口兼容**：实现相同的 API，可以注册到 verl 的 `CheckpointEngineRegistry`
2. **支持多种存储后端**：NFS、Mooncake Store
3. **流式加载**：通过 generator 逐个 yield 权重 tensor，避免内存溢出
4. **与 ExternalExecutor 集成**：在 Worker 中直接从存储加载模型

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│              StorageCheckpointEngine                             │
│  - 继承 verl CheckpointEngineWithCache 接口                      │
│  - 支持后端：NFS / Mooncake Store                                │
│  - get_weights(): yield named tensors from storage              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              StorageBackend (抽象接口)                           │
│  - initialize(config)                                           │
│  - save_checkpoint(path, weights, metadata)                     │
│  - load_metadata(path)                                          │
│  - load_tensor(tensor_meta)                                     │
│  - get_weights(path)                                            │
│  - exists(path)                                                 │
│  - delete(path)                                                 │
└─────────────────────────────────────────────────────────────────┘
         │                    │
         ↓                    ↓
┌──────────────┐    ┌──────────────────────────────────┐
│ NFS Backend  │    │      Mooncake Store Backend       │
│              │    │                                    │
│ safetensors  │    │  MooncakeDistributedStore          │
│              │    │  - store.put(key, data, config)    │
│              │    │  - store.get(key)                  │
│              │    │  - RDMA / TCP 传输                 │
└──────────────┘    └──────────────────────────────────┘
```

## 与 verl 的接口对比

| verl 接口 | StorageCheckpointEngine | 说明 |
|-----------|------------------------|------|
| `prepare()` | ✅ 实现（no-op） | 存储后端不需要准备 |
| `build_topology()` | ✅ 实现（no-op） | 存储后端不需要拓扑 |
| `init_process_group()` | ✅ 实现（no-op） | 存储后端不需要进程组 |
| `finalize()` | ✅ 实现（no-op） | 存储后端不需要清理 |
| `send_weights()` | ✅ 实现 | 发送权重到存储 |
| `receive_weights()` | ✅ 实现 | 从存储接收权重 |
| `get_weights()` | ✅ 实现 | 从存储获取权重 |

## 存储后端

### NFS Backend（已实现）

使用 safetensors 格式存储模型权重。

**存储结构**：
```
/shared/checkpoints/
└── opt-125m/
    └── step_1000/
        ├── model.safetensors    # 模型权重
        └── metadata.json        # 元数据
```

**优点**：
- 简单可靠，适合共享文件系统
- safetensors 格式高效，支持零拷贝加载
- 易于调试和检查

**缺点**：
- 需要 NFS 挂载
- 跨节点性能依赖网络带宽

### Mooncake Store Backend（已实现）

使用 MooncakeDistributedStore 的分布式 KV 存储，支持 RDMA 高性能传输。

**Key 命名约定**：
```
ckpt:{checkpoint_path}:metadata          # 元数据（JSON）
ckpt:{checkpoint_path}:tensor:{name}     # 单个权重 tensor
```

**配置参数**：
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

**存储模式**：
- **embedded**：每个节点贡献 `global_segment_size` 内存作为存储池
- **standalone-store**：外部 `mooncake_client` 进程管理存储池和 SSD 层

**优点**：
- 高性能 RDMA 传输（~9 GB/s on InfiniBand）
- 支持 GPU 直接传输（GPUDirect）
- 适合大规模集群
- 支持 Ascend NPU（ascend_direct transport）

**缺点**：
- 需要部署 Mooncake metadata server 和 master server
- 不支持显式删除（依赖 GC）

**Tensor 序列化格式**：
```
[4B dtype_len][dtype_str][4B ndim][8*ndim B shape][raw data]
```

**性能参考**（来自 verl benchmark）：
| 硬件 | 后端 | 耗时 | 带宽 |
|------|------|------|------|
| 4*8 H100, ConnectX-7 | NCCL | ~7s | 8.25 GB/s |
| 4*8 H100, ConnectX-7 | NIXL | ~7s | 8.25 GB/s |
| 2*16 Ascend 910C | HCCL | ~11s | 5.3 GB/s |
| 2*16 Ascend 910C | kimi_ckpt_engine | 7+3.5s | 16.5 GB/s |
| 2*8 H100, ConnectX-7 | mooncake | 5.93s | 9.44 GB/s |

## 使用示例

### 基本使用

```python
from vllm_external_executor import StorageCheckpointEngine

# 创建 engine
engine = StorageCheckpointEngine(
    backend="nfs",
    config={"base_path": "/shared/checkpoints"},
    device="cuda",
)

# 设置 checkpoint 路径
engine.set_checkpoint("opt-125m/step_1000")

# 从存储加载权重
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

### 使用 Mooncake Store

```python
engine = StorageCheckpointEngine(
    backend="mooncake",
    config={
        "metadata_server": "http://127.0.0.1:8080/metadata",
        "master_server_address": "127.0.0.1:50051",
        "protocol": "rdma",
        "device_name": "mlx5_0",
        "global_segment_size": 512 * 1024 * 1024,
        "local_buffer_size": 64 * 1024 * 1024,
    },
    device="cuda",
)
engine.set_checkpoint("opt-125m/step_1000")

# 从 Mooncake Store 加载权重
for name, tensor in engine.get_weights():
    model.get_parameter(name).data.copy_(tensor)
```

### 保存 checkpoint

```python
import asyncio
from vllm_external_executor import StorageCheckpointEngine

engine = StorageCheckpointEngine(
    backend="mooncake",
    config={
        "metadata_server": "http://127.0.0.1:8080/metadata",
        "master_server_address": "127.0.0.1:50051",
        "protocol": "rdma",
    },
)
engine.set_checkpoint("my-model/step_100")

# 保存权重
def weight_generator():
    yield "model.layers.0.weight", torch.randn(1024, 1024)
    yield "model.layers.0.bias", torch.randn(1024)

asyncio.run(engine.send_weights(weight_generator(), global_steps=100))
```

### 注册到 verl

```python
from verl.checkpoint_engine.base import CheckpointEngineRegistry
from vllm_external_executor import StorageCheckpointEngine

# 注册到 verl
CheckpointEngineRegistry.register("storage")(StorageCheckpointEngine)

# 在 verl 配置中使用
config = {
    "checkpoint_engine": {
        "backend": "storage",
        "engine_kwargs": {
            "storage": {
                "backend": "mooncake",
                "config": {
                    "metadata_server": "http://127.0.0.1:8080/metadata",
                    "master_server_address": "127.0.0.1:50051",
                    "protocol": "rdma",
                }
            }
        }
    }
}
```

## 性能优化

### 流式加载

`get_weights()` 使用 generator 逐个 yield 权重 tensor，避免一次性加载整个模型到内存：

```python
def get_weights(self):
    for name, tensor in self.storage.get_weights(self.checkpoint_path):
        tensor = tensor.to(self.device)
        yield name, tensor
```

### 设备预分配

在加载前预分配目标设备的内存：

```python
# 在 Worker 中
model = self.worker.get_model()
for name, tensor in engine.get_weights():
    param = model.get_parameter(name)
    param.data.copy_(tensor)  # 直接复制到预分配的 GPU 内存
```

### Mooncake RDMA 优化

Mooncake Store 支持：
- **GPUDirect RDMA**：GPU 内存直接传输，避免 CPU 拷贝
- **软绑定**（`with_soft_pin`）：灵活的数据放置策略
- **preferred_segment**：指定数据存储的目标节点

## 未来工作

1. **增量更新**：支持只加载变化的权重（delta update）
2. **压缩传输**：支持 gzip/zstd 压缩，减少网络传输
3. **缓存层**：在本地 SSD 缓存热点 checkpoint，减少远程访问
4. **batch_put_from_multi_buffers**：使用 Mooncake 批量 API 提升吞吐
