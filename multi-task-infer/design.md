# ExternalExecutor 架构设计文档

---

# 0. 文档概述

## 0.1 背景

vLLM 的 Executor 体系（UniProcExecutor、MultiprocExecutor、RayExecutorV2）采用**一次性初始化**模式：创建 Worker 进程/Actor → 初始化分布式环境 → 加载模型权重 → 初始化 KV Cache → 编译 CUDA Graph。整个初始化流程耗时 68-495s，其中模型加载（30-300s）和算子编译（25-170s）是主要瓶颈。

**关键认知**：CUDA Graph 不能直接缓存（涉及运行时状态：内存地址、Stream 资源等），但通过配置优化（减少 batch size、切换 PIECEWISE 模式）和确保 torch.compile 缓存启用，可以显著减少 Graph Capture 时间（从 5-10s 降低到 1-3s）。

在多模型推理、弹性伸缩、模型热切换等场景下，需要频繁创建和销毁 vLLM 实例。如果每次都能复用已预热的 Worker，可以大幅降低初始化开销。

## 0.2 设计目标

| 编号 | 目标 | 说明 |
|------|------|------|
| G1 | Actor 池化 | 预启动一组 Ray Actor，绑定设备并预热公共库，按需分配给 vLLM 实例 |
| G2 | 动态重组 | 支持动态调整 TP/PP 大小，重新分配 Actor 到新的并行组 |
| G3 | 模型热切换 | 通过 storage 存储加载 / weight_transfer 机制动态加载新模型权重，不重启进程 |
| G4 | 数据通路复用 | 复用 RayExecutorV2 的 MessageQueue 通信机制，保持性能一致 |
| G5 | 最小侵入 | 尽量以插件形式实现，减少对 vLLM 核心代码的修改 |
| G6 | 编译缓存共享 | 通过 CacheManagerActor（独立 Ray Actor）跨节点共享 torch.compile 缓存，避免重复编译 |
| G7 | 存储权重加载 | 通过 StorageCheckpointEngine 从持久化存储（NFS/Mooncake Store）加载模型权重 |

## 0.3 术语定义

| 术语 | 定义 |
|------|------|
| **Actor Pool** | 预启动的 Ray Actor 集合，每个 Actor 绑定一个 NPU/GPU 设备 |
| **ExternalExecutor** | 从 Actor Pool 获取 Worker 的 Executor 实现 |
| **ExternalWorkerActor** | 预启动的 Ray Actor，实现了 Worker 接口 |
| **Worker Lease** | Executor 从 Pool 中"租用"一组 Actor 的过程 |
| **Worker Release** | Executor 将 Actor 归还 Pool 的过程 |
| **weight_transfer** | vLLM 已有的权重热更新机制，支持 NCCL/IPC/sharded_rdt 后端 |
| **CacheManagerActor** | 独立的 Ray Actor，作为编译缓存的中央管理器，提供 pull/push API 和编译锁协调 |
| **编译缓存懒加载** | Worker 编译前按需查找缓存的模式：本地缓存 → CacheManagerActor → fallback 自编译 |
| **编译锁** | CacheManagerActor 提供的互斥机制，防止多个 Worker 重复编译同一缓存 |
| **混合存储** | 缓存按大小分流：<50MB 走 Ray Object Store，≥50MB 走 NFS（gzip 压缩） |
| **StorageCheckpointEngine** | 与 verl CheckpointEngineWithCache 接口兼容的存储后端，从 NFS/Mooncake Store 加载权重 |
| **StorageBackend** | 存储后端的抽象接口，具体实现：NFSStorageBackend、MooncakeStoreBackend |

---

# 1. 逻辑视图（Logical View）

> 关注系统的功能分解，描述系统提供给用户的服务和关键抽象。

## 1.1 核心类图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Executor (抽象基类)                                 │
│  vllm/v1/executor/abstract.py                                                   │
│  ─────────────────────────────                                                   │
│  + collective_rpc(method, args, kwargs) → list[Any]                            │
│  + execute_model(scheduler_output) → ModelRunnerOutput                          │
│  + sample_tokens(grammar_output) → ModelRunnerOutput                            │
│  + initialize_kv_cache(kv_cache_config)                                         │
│  + check_health()                                                               │
│  + shutdown()                                                                   │
└────────────────────────────────┬────────────────────────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ↓                  ↓                  ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐
│ MultiprocExecutor│  │RayExecutorV2     │  │ ExternalExecutor (新增)   │
│                  │  │                  │  │ ───────────────────────── │
│ + _init_executor │  │ + _init_executor │  │ + __init__(vllm_config,  │
│ + collective_rpc │  │ + _build_runtime │  │     external_actors)     │
│ + execute_model  │  │ + _init_executor │  │ + _init_executor()       │
│ + sample_tokens  │  │ + start_monitor  │  │ + _load_model_via_       │
│                  │  │ + shutdown       │  │   weight_transfer()      │
│                  │  │                  │  │ + release_actors()       │
└──────────────────┘  └──────────────────┘  └──────────────────────────┘
        ↑                                            │
        │                                            │ 继承
        └────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ActorPoolManager (新增)                                  │
│  ─────────────────────────────────────                                           │
│  - actors: list[ActorHandle]                                                     │
│  - states: dict[int, ActorState]                                                 │
│  - node_mapping: dict[str, list[int]]                                            │
│  ─────────────────────────────────────                                           │
│  + pre_start(num_actors, devices_per_node, placement_group)                      │
│  + acquire(tp_size, pp_size, node_constraint) → list[ActorHandle]               │
│  + release(actors: list[ActorHandle])                                            │
│  + get_idle_count() → int                                                        │
│  + get_actor_states() → dict[int, ActorState]                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                      ExternalWorkerActor (新增, @ray.remote)                      │
│  ─────────────────────────────────────────────                                   │
│  - device_id: int                                                                │
│  - worker: WorkerWrapperBase | None                                              │
│  - vllm_config: VllmConfig | None                                                │
│  - state: str                                                                    │
│  ─────────────────────────────────────────────                                   │
│  + __init__(device_id)              # 预启动：绑定设备 + import 公共库            │
│  + get_info() → dict                # 获取 Actor 信息                            │
│  + initialize_worker(...)           # 创建 WorkerWrapperBase                     │
│  + init_device()                    # 初始化分布式环境                            │
│  + load_model_via_weight_transfer(init_info)  # 通过 weight_transfer 加载模型    │
│  + load_model_from_storage(path, backend)     # 从存储加载模型 (G7)               │
│  + initialize_kv_cache(config)      # 初始化 KV Cache                           │
│  + compile_or_warm_up_model()       # 编译/预热模型                              │
│  + run() → worker_busy_loop()       # 启动推理循环                               │
│  + reset()                          # 释放资源，回到 IDLE 状态                    │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CacheManagerActor (新增, @ray.remote)                      │
│  ─────────────────────────────────────────────                                   │
│  - _object_registry: dict[str, ObjectRef]   # 小缓存 → Ray Object Store          │
│  - _compile_locks: dict[str, dict]          # 编译锁表                            │
│  - _metadata: dict[str, dict]               # 缓存元数据                          │
│  - shared_cache_dir: str | None             # 大缓存 → NFS                        │
│  ─────────────────────────────────────────────                                   │
│  + pull(hash_key) → bytes | None            # 拉取缓存 (Object Store → NFS)       │
│  + push(hash_key, data, source) → bool      # 推送缓存 (按大小自动分流)           │
│  + try_acquire_compile_lock(hash, worker_id) → dict  # 获取编译锁                 │
│  + release_compile_lock(hash, worker_id)              # 释放编译锁                │
│  + get_cache_hash(model_cfg, parallel_cfg, comp_cfg)   # 计算缓存哈希 (远端)      │
│  + get_stats() / list_caches()             # 监控                                │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                      StorageCheckpointEngine (新增)                               │
│  ─────────────────────────────────────────────                                   │
│  (兼容 verl CheckpointEngineWithCache 接口)                                      │
│  - storage: StorageBackend                    # 具体后端                          │
│  - bucket_size: int                                                              │
│  - device: str                                                                   │
│  ─────────────────────────────────────────────                                   │
│  + prepare() → dict                          # no-op (存储不需要通信拓扑)          │
│  + build_topology(...)                       # no-op                              │
│  + init_process_group(**kwargs)              # no-op                              │
│  + finalize()                                # no-op                              │
│  + send_weights(gen, global_steps)           # 保存权重到存储 (G7)                │
│  + receive_weights(global_steps)             # 从存储接收权重                      │
│  + get_weights() → gen[(name, tensor)]       # 从存储流式获取权重                  │
│  + set_checkpoint(path)                      # 设置 checkpoint 路径               │
│  + exists(path) / load_metadata(path) / delete(path)                             │
│  ─────────────────────────────────────────────                                   │
│  StorageBackend (抽象)                                                           │
│  ├── NFSStorageBackend        # safetensors + metadata.json                      │
│  └── MooncakeStoreBackend     # MooncakeDistributedStore (RDMA)                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 1.2 功能模块划分

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ExternalExecutor 模块                               │
│                                                                                  │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────────┐ │
│  │  Actor Pool 管理    │  │  Executor 核心      │  │  Weight Transfer 集成     │ │
│  │  ────────────────  │  │  ────────────────  │  │  ────────────────────────  │ │
│  │                    │  │                    │  │                            │ │
│  │  ActorPoolManager  │  │  ExternalExecutor  │  │  WeightTransferEngine     │ │
│  │  - pre_start()     │  │  - _init_executor  │  │  - init_transfer_engine   │ │
│  │  - acquire()       │  │  - collective_rpc  │  │  - start_weight_update    │ │
│  │  - release()       │  │  - execute_model   │  │  - update_weights         │ │
│  │  - monitor()       │  │  - sample_tokens   │  │  - finish_weight_update   │ │
│  └────────────────────┘  └────────────────────┘  └────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────────┐ │
│  │ 缓存管理 (G6)       │  │ 存储加载 (G7)       │  │  Worker 适配层             │ │
│  │  ────────────────  │  │  ────────────────  │  │  ────────────────────────  │ │
│  │                    │  │                    │  │                            │ │
│  │  CacheManagerActor │  │  StorageCheckpoint │  │  ExternalWorkerActor      │ │
│  │  - pull/push       │  │    Engine          │  │  - init_device()          │ │
│  │  - 编译锁          │  │  - NFS Backend     │  │  - load_model()           │ │
│  │  - 懒加载协调      │  │  - Mooncake Backend│  │  - load_model_from_       │ │
│  │                    │  │  - get_weights     │  │    storage()              │ │
│  │                    │  │  - send_weights    │  │  - run()                  │ │
│  └────────────────────┘  └────────────────────┘  └────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────────┐ │
│  │ 分布式控制面        │  │ 全局调度器          │  │  节点故障隔离              │ │
│  │  ────────────────  │  │  ────────────────  │  │  ────────────────────────  │ │
│  │                    │  │                    │  │                            │ │
│  │  NodeRegistryActor │  │  GlobalScheduler   │  │  - 心跳线程                │ │
│  │  - register_node   │  │  - select_actors   │  │  - detect_dead_*           │ │
│  │  - register_actor  │  │  - 故障域 spread   │  │  - recover_node            │ │
│  │  - heartbeat       │  │  - 硬约束/负载均衡 │  │  - rebuild_actor           │ │
│  │  - get_global_view │  │  - driver 就近     │  │  - RPC_TIMEOUT             │ │
│  └────────────────────┘  └────────────────────┘  └────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────┐                                                         │
│  │  通信层 (复用)      │                                                         │
│  │  ────────────────  │                                                         │
│  │  MessageQueue      │                                                         │
│  │  - rpc_broadcast   │                                                         │
│  │  - response_mqs    │                                                         │
│  │  - SHM (同节点)     │                                                         │
│  │  - TCP (跨节点)     │                                                         │
│  └────────────────────┘                                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 1.3 关键接口定义

### 1.3.1 ActorPoolManager 接口

```python
class ActorPoolManager:
    """管理预启动的 Actor 池"""
    
    def pre_start(
        self,
        num_actors: int,
        devices_per_node: list[int],
        placement_group: PlacementGroup | None = None,
        registry: NodeRegistryActor | None = None,   # 共享全局注册表
        fault_domain: str | None = None,             # 故障域标签(默认=node_id)
        strategy: str = "pack",                       # "pack" | "spread"
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ) -> None:
        """预启动 Actor 池 + 注册节点/Actor + 启动心跳线程"""
    
    def acquire(
        self,
        tp_size: int,
        pp_size: int,
        fault_domain_constraint: dict[str, int] | None = None,
        prefer_driver_node: bool = True,
        node_constraint: dict[str, int] | None = None,  # 已废弃别名
    ) -> list[ray.actor.ActorHandle]:
        """按故障域 spread 全局调度，获取空闲 Actor"""
    
    def release(self, actors: list[ray.actor.ActorHandle]) -> None:
        """释放 Actor 回池（ray.get 带 RPC_TIMEOUT）"""
    
    def get_idle_count(self) -> int:
        """获取空闲 Actor 数量"""
    
    def get_actor_states(self) -> dict[int, ActorState]:
        """获取所有 Actor 的状态"""
    
    def get_global_view(self) -> dict:
        """跨节点统一资源视图（每节点空闲 GPU + 故障域分布 + 心跳年龄）"""
    
    def check_health(self) -> dict:
        """检测死节点/死 Actor（心跳超时）"""
    
    def recover_node(self, node_id: str) -> list:
        """节点故障隔离：移除该节点 Actor 并在健康节点重建"""
    
    def rebuild_actor(self, actor_id: str, target_node_id: str | None = None):
        """跨节点迁移：在健康节点等价重建（GPU Actor 无法 live-migrate）"""
```

### 1.3.2 ExternalExecutor 接口

```python
class ExternalExecutor(RayExecutorV2):
    """从预启动的 Ray Actor 池获取 Worker 的执行器"""
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        external_actors: list[ray.actor.ActorHandle],
    ):
        """
        Args:
            vllm_config: vLLM 配置
            external_actors: 预启动的 Actor 列表（数量必须等于 world_size）
        """
    
    def release_actors(self) -> None:
        """释放 Actor 回池"""
```

### 1.3.3 CacheManagerActor 接口

```python
class CacheManagerActor:
    """编译缓存中央管理器（独立 Ray Actor）"""
    
    RAY_OBJECT_THRESHOLD = 50 * 1024 * 1024  # 小缓存阈值: 50 MB
    
    def pull(self, hash_key: str) -> bytes | None:
        """拉取缓存。先查 Ray Object Store，再查 NFS。
        返回 tar.gz 字节数据，未命中返回 None。"""
    
    def push(self, hash_key: str, cache_data: bytes, source: str) -> bool:
        """推送缓存。≤阈值存入 Ray Object Store，>阈值压缩后写 NFS。"""
    
    def try_acquire_compile_lock(self, hash_key: str, worker_id: str) -> dict:
        """尝试获取编译锁。
        返回状态: {"status": "acquired" | "wait" | "done", "holder": str}"""
    
    def release_compile_lock(self, hash_key: str, worker_id: str) -> None:
        """释放编译锁（仅锁持有者）。"""
    
    def get_cache_hash(
        self, model_config: dict, parallel_config: dict, compilation_config: dict
    ) -> str:
        """远端计算缓存哈希（与 ExternalExecutor._compute_cache_hash 因子一致）。"""
```

### 1.3.4 StorageCheckpointEngine / StorageBackend 接口

```python
class StorageCheckpointEngine:
    """存储后端 checkpoint engine，兼容 verl CheckpointEngineWithCache。"""
    
    wire_format = "named_tensors"  # (name, tensor) 流
        
    def prepare(self) -> dict[str, Any]: ...
    def build_topology(cls, *args, **kwargs) -> tuple[dict, dict]: ...
    def init_process_group(self, **kwargs) -> None: ...
    def finalize(self) -> None: ...
    
    async def send_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ) -> None:
        """保存权重到存储。"""
    
    async def receive_weights(
        self, global_steps: int | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """从存储接收权重。"""
    
    def get_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """从存储流式获取权重（主路径，用于模型加载）。"""
    
    def set_checkpoint(self, checkpoint_path: str) -> None: ...
    def exists(self, checkpoint_path: str) -> bool: ...
    def load_metadata(self, checkpoint_path: str) -> CheckpointMetadata: ...
    def delete(self, checkpoint_path: str) -> None: ...


class StorageBackend(ABC):
    """存储后端抽象接口。"""
    
    def initialize(self, config: dict[str, Any]) -> None: ...
    def save_checkpoint(
        self, path: str, weights: Generator, metadata: dict | None
    ) -> str: ...
    def load_metadata(self, path: str) -> CheckpointMetadata: ...
    def load_tensor(self, tensor_meta: TensorMeta) -> torch.Tensor: ...
    def get_weights(self, path: str) -> Generator: ...
    def exists(self, path: str) -> bool: ...
    def delete(self, path: str) -> None: ...


class NFSStorageBackend(StorageBackend):
    """NFS 后端: safetensors + metadata.json，无额外服务依赖。"""


class MooncakeStoreBackend(StorageBackend):
    """Mooncake 后端: MooncakeDistributedStore (RDMA ~9 GB/s on IB)。
    Key 约定: ckpt:{path}:metadata / ckpt:{path}:tensor:{name}"""
```

### 1.3.5 NodeRegistryActor / GlobalScheduler 接口

```python
class NodeRegistryActor:
    """中心化跨节点注册表 (Ray detached actor, 多租户共享)。
    方法由 Ray actor 模型串行化, 无并发写。"""

    def register_node(self, node_id, ip, fault_domain=None,
                      total_gpus=0, hostname="") -> None: ...
    def register_actor(self, actor_id, node_id, device_id,
                       fault_domain=None) -> None: ...
    def node_heartbeat(self, node_id) -> None: ...
    def heartbeat(self, actor_id, state=None) -> bool: ...
    def set_actor_state(self, actor_id, state, lease_id=None) -> None: ...
    def unregister_actor(self, actor_id) -> None: ...
    def unregister_node(self, node_id) -> list[str]: ...

    def list_nodes(self) -> list[NodeInfo]: ...
    def list_actors(self) -> list[ActorRegistration]: ...
    def get_global_view(self) -> dict: ...
        """统一视图: 每节点 free/total GPU + 每故障域分布 + 心跳年龄"""

    def detect_dead_actors(self, timeout) -> list[str]: ...
    def detect_dead_nodes(self, timeout) -> list[str]: ...
    def mark_node_failed(self, node_id) -> list[str]: ...
        """节点故障: 移除节点及其 Actor, 返回 actor ids 供重建"""
    def mark_actor_failed(self, actor_id) -> None: ...


@dataclass
class NodeInfo:
    node_id: str; ip: str; fault_domain: str
    total_gpus: int; free_gpus: int
    hostname: str = ""; last_heartbeat: float = ...


@dataclass
class ActorRegistration:
    actor_id: str; node_id: str; device_id: int
    fault_domain: str; state: str; lease_id: str | None
    last_heartbeat: float = ...


class GlobalScheduler:
    """故障域感知调度 (纯逻辑, 无 Ray 依赖, 可单测)。"""

    def select_actors(self, actors, nodes, world_size,
                      fault_domain_constraint=None,
                      prefer_driver_node=None) -> list[str]: ...
        """优先级: 硬约束 > 故障域 spread > 负载均衡 > driver 就近"""
```

### 1.3.6 迁移状态机 / 事务 / 幂等

```python
class MigrationPhase(str, Enum):
    IDLE, PREPARING, GRACEFUL_PAUSE, CHECKPOINT, UNLOAD, LOAD,
    RESTORE, COMPLETED, FAILED, ROLLING_BACK, ROLLED_BACK


class FlightBatchPolicy(str, Enum):
    """飞行中 batch 处理策略（需与 EngineCore 调度器协同）。"""
    DRAIN = "drain"                       # 完成当前步再迁移（最安全，默认）
    PAUSE_SERIALIZE = "pause_serialize"   # 暂停 + 序列化调度状态
    PREEMPT = "preempt"                   # 抢占式中断 + 请求级重放


class MigrationStateMachine:
    """校验状态转换 + 补偿式回滚（纯逻辑）。"""
    def transition(self, next_phase) -> None: ...      # 非法跳转抛异常
    def register_compensation(self, handler) -> None:  # 每步成功后注册补偿
    def fail(self, error) -> None: ...
    def rollback(self) -> MigrationOutcome: ...        # 逆序执行补偿(幂等)


class MigrationIdempotencyRegistry:
    """请求级幂等: 同一 migration_id 只执行一次，重发返回缓存结果。"""
    def run(self, spec, execute_fn) -> MigrationStateMachine: ...
    def get(self, migration_id) -> MigrationStateMachine | None: ...
```

---

# 2. 进程视图（Process View）

> 关注运行时并发、同步、通信，描述进程/线程结构和交互。

## 2.1 进程拓扑

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  进程拓扑                                                                        │
│                                                                                  │
│  ┌──────────────┐    ZMQ     ┌──────────────┐    MessageQueue    ┌────────────┐ │
│  │  API Server  │ ←────────→ │  EngineCore  │ ←────────────────→ │ Worker 0   │ │
│  │  (前端进程)   │            │  (调度进程)   │    (SHM/TCP)       │ (Actor)    │ │
│  └──────────────┘            └──────────────┘                    └────────────┘ │
│        │                                    │                                   │
│        │ TensorIPC                          │ MessageQueue                      │
│        │ (多模态张量)                        │                                   │
│        │                                    ↓                                   │
│        │                              ┌────────────┐                            │
│        │                              │ Worker 1   │                            │
│        │                              │ (Actor)    │                            │
│        │                              └────────────┘                            │
│        │                                    │                                   │
│        │                                    ↓ NCCL/HCCl (数据面)                │
│        │                              ┌────────────┐                            │
│        │                              │ Worker N   │                            │
│        │                              │ (Actor)    │                            │
│        │                              └────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

CacheManagerActor 作为**独立的 Ray Actor** 常驻集群，与 EngineCore/Worker 并列：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         缓存管理进程拓扑                                          │
│                                                                                  │
│                    ┌──────────────────────────────┐                             │
│                    │  CacheManagerActor           │                             │
│                    │  (独立 Ray Actor, 持久常驻)   │                             │
│                    │  ────────────────────────    │                             │
│                    │  - _object_registry          │  ←→ Ray Object Store        │
│                    │  - _compile_locks            │                              │
│                    │  - _metadata                 │                              │
│                    │  - shared_cache_dir          │  ←→ NFS (大缓存, 压缩)       │
│                    └──────────────────────────────┘                             │
│                              ▲              ▲                                    │
│                    pull/push │              │ 编译锁                              │
│                    (Ray 调用) │              │ (Ray 调用)                         │
│                              │              │                                    │
│        ┌─────────────────────┴─────┐  ┌─────┴─────────────────────┐              │
│        │  EngineCore (调度进程)     │  │  ExternalWorkerActor     │              │
│        │  ExternalExecutor         │  │  (每 GPU 一个)            │              │
│        │  - 懒加载协调              │  │  - 本地缓存检查           │              │
│        │  - 计算 cache_hash        │  │  - compile_or_warm_up     │              │
│        └───────────────────────────┘  └───────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 2.2 Actor 状态机

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Actor 状态机                                        │
│                                                                                  │
│   ┌──────────┐    acquire     ┌──────────┐    init_device    ┌────────────────┐ │
│   │  IDLE    │ ─────────────→ │ LEASED   │ ────────────────→ │ INIT_DEVICE    │ │
│   │ (空闲)   │                │ (已租用)  │                   │ (设备已初始化)  │ │
│   └──────────┘ ←───────────── └──────────┘ ←──────────────── └────────────────┘ │
│        ↑            release          ↑              init_worker                  │
│        │                             │                                           │
│        │         ┌──────────┐        │              load_model                   │
│        └──────── │ RELEASED │ ───────┘ ←─────────────────────────────────────────│
│                  │ (已释放)  │                      ↓                             │
│                  └──────────┘              ┌────────────────┐                     │
│                                            │ INIT_MODEL     │                     │
│                                            │ (模型已加载)    │                     │
│                                            └────────────────┘                     │
│                                                   ↓                               │
│                                            ┌────────────────┐                     │
│                                            │ RUNNING        │                     │
│                                            │ (运行中)        │                     │
│                                            └────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 2.3 通信时序图

### 2.3.1 预启动阶段

```
ActorPoolManager          Ray Cluster          ExternalWorkerActor
       │                      │                       │
       │  placement_group()   │                       │
       │─────────────────────→│                       │
       │                      │                       │
       │  ray.remote(Actor)   │                       │
       │  .options(num_gpus=1)│                       │
       │  .remote(device_id)  │                       │
       │─────────────────────→│                       │
       │                      │   create actor        │
       │                      │──────────────────────→│
       │                      │                       │
       │                      │                       │ 1. 绑定 GPU 设备
       │                      │                       │ 2. Import 公共库
       │                      │                       │ 3. 预热 NCCL/HCCl
       │                      │                       │ 4. 状态 = IDLE
       │                      │                       │
       │  wait_for_ready()    │                       │
       │─────────────────────────────────────────────→│
       │                      │                       │
       │  {device_id, node_id,│                       │
       │   physical_gpu_ids}  │                       │
       │←─────────────────────────────────────────────│
       │                      │                       │
```

### 2.3.2 Acquire 阶段（构建 vLLM 实例）

```
AsyncLLM    EngineCore    ExternalExecutor    ActorPoolManager    ExternalWorkerActor
   │             │               │                  │                    │
   │ acquire()   │               │                  │                    │
   │────────────────────────────────────────────────→│                    │
   │             │               │                  │                    │
   │             │               │  actors = [a0..a7]│                   │
   │←────────────────────────────────────────────────│                    │
   │             │               │                  │                    │
   │ AsyncLLM(   │               │                  │                    │
   │  executor_  │               │                  │                    │
   │  class=     │               │                  │                    │
   │  External   │               │                  │                    │
   │  Executor,  │               │                  │                    │
   │  actors=..) │               │                  │                    │
   │────────────→│               │                  │                    │
   │             │               │                  │                    │
   │             │ EngineCore(   │                  │                    │
   │             │  executor_    │                  │                    │
   │             │  class,       │                  │                    │
   │             │  actors)      │                  │                    │
   │             │──────────────→│                  │                    │
   │             │               │                  │                    │
   │             │               │ _init_executor() │                    │
   │             │               │──────────────────│                    │
   │             │               │                  │                    │
   │             │               │ 1. 创建 MessageQueue                  │
   │             │               │ 2. 获取 Actor 信息                    │
   │             │               │    get_info() ───────────────────────→│
   │             │               │    ← {node_id, gpu_ids} ─────────────│
   │             │               │                                       │
   │             │               │ 3. initialize_worker() ──────────────→│
   │             │               │    (vllm_config, rank, local_rank,    │
   │             │               │     distributed_init_method,          │
   │             │               │     input_shm_handle)                 │
   │             │               │                                       │
   │             │               │ 4. init_device() ────────────────────→│
   │             │               │    (NCCL/HCCl 初始化, 已预热, 快)     │
   │             │               │                                       │
   │             │               │ 5. load_model_via_weight_transfer() ─→│
   │             │               │    (创建模型结构 + weight_transfer)    │
   │             │               │                                       │
   │             │               │ 6. initialize_kv_cache() ────────────→│
   │             │               │                                       │
   │             │               │ 7. compile_or_warm_up_model() ───────→│
   │             │               │                                       │
   │             │               │ 8. run() → worker_busy_loop() ───────→│
   │             │               │                                       │
   │             │               │ 9. wait_until_ready() ───────────────→│
   │             │               │                                       │
   │             │ ←─────────────│                                       │
   │             │               │                  │                    │
   │ ←───────────│               │                  │                    │
   │             │               │                  │                    │
```

### 2.3.3 推理运行阶段

```
EngineCore         ExternalExecutor         ExternalWorkerActor (Worker 0..N)
    │                     │                          │
    │ schedule()          │                          │
    │ → SchedulerOutput   │                          │
    │                     │                          │
    │ execute_model()     │                          │
    │────────────────────→│                          │
    │                     │ rpc_broadcast_mq         │
    │                     │ .enqueue(SchedulerOutput)│
    │                     │─────────────────────────→│ (SHM 同节点 / TCP 跨节点)
    │                     │                          │
    │                     │                          │ Worker.execute_model()
    │                     │                          │ → ModelRunner.execute_model()
    │                     │                          │ → NCCL AllReduce (TP)
    │                     │                          │ → NCCL P2P (PP)
    │                     │                          │
    │                     │ response_mq[i].dequeue() │
    │                     │←─────────────────────────│
    │                     │                          │
    │ sample_tokens()     │                          │
    │────────────────────→│                          │
    │                     │ rpc_broadcast_mq         │
    │                     │ .enqueue(GrammarOutput)  │
    │                     │─────────────────────────→│
    │                     │                          │
    │                     │                          │ Worker.sample_tokens()
    │                     │                          │ → ModelRunner.sample_tokens()
    │                     │                          │
    │                     │ response_mq[i].dequeue() │
    │                     │←─────────────────────────│
    │                     │                          │
    │ ←───────────────────│                          │
    │ ModelRunnerOutput   │                          │
    │                     │                          │
```

### 2.3.4 Release 阶段

```
ActorPoolManager    ExternalExecutor    ExternalWorkerActor
       │                   │                    │
       │ release(actors)   │                    │
       │──────────────────→│                    │
       │                   │                    │
       │                   │ actor.reset()      │
       │                   │───────────────────→│
       │                   │                    │
       │                   │                    │ 1. worker.shutdown()
       │                   │                    │ 2. worker = None
       │                   │                    │ 3. vllm_config = None
       │                   │                    │ 4. state = IDLE
       │                   │                    │
       │                   │ ←──────────────────│
       │                   │                    │
       │ states[idx]=IDLE  │                    │
       │←──────────────────│                    │
       │                   │                    │
```

### 2.3.5 编译缓存懒加载阶段（G6）

```
Worker(Executor)            本地 ~/.cache/vllm           CacheManagerActor          NFS / Object Store
      │                            │                          │                          │
      │ _handle_compilation_       │                          │                          │
      │   _optimization()          │                          │                          │
      │───────────────────────────→│                          │                          │
      │                            │                          │                          │
      │ ① 检查本地缓存             │                          │                          │
      │  local_cache_exists(hash)  │                          │                          │
      │───────────────────────────→│                          │                          │
      │      ← 命中: 跳过编译 ─────│                          │                          │
      │   (未命中)                  │                          │                          │
      │                            │                          │                          │
      │ ② pull(hash_key)           │                          │                          │
      │───────────────────────────────────────────────────────→│                          │
      │                            │                          │ 查 Object Store / NFS    │
      │                            │                          │─────────────────────────→│
      │                            │                          │← 未命中 (None) ──────────│
      │      ← 未命中 ────────────────────────────────────────│                          │
      │                            │                          │                          │
      │ ③ try_acquire_compile_lock │                          │                          │
      │───────────────────────────────────────────────────────→│                          │
      │      ← {"status":"acquired"} ─────────────────────────│                          │
      │                            │                          │                          │
      │ ④ collective_rpc("compile_or_warm_up_model")          │                          │
      │───────────────────────────→ Worker 编译                │                          │
      │                            │                          │                          │
      │ ⑤ package_local_cache →    │                          │                          │
      │    push(hash, data)        │                          │                          │
      │───────────────────────────────────────────────────────→│ 按大小分流存储           │
      │                            │                          │─────────────────────────→│
      │ ⑥ release_compile_lock     │                          │                          │
      │───────────────────────────────────────────────────────→│                          │
      │                            │                          │                          │
      │ (其他 Worker: status="wait" → sleep(5) → 再 pull)      │                          │
      │ (其他 Worker: status="done" → 直接 pull)               │                          │
```

### 2.3.6 存储权重加载阶段（G7）

```
ExternalWorkerActor    StorageCheckpointEngine    StorageBackend (NFS/Mooncake)
      │                          │                          │
      │ load_model_from_storage( │                          │
      │   checkpoint, backend)   │                          │
      │─────────────────────────→│                          │
      │                          │ set_checkpoint(path)     │
      │─────────────────────────→│                          │
      │                          │                          │
      │ get_weights()            │                          │
      │─────────────────────────→│                          │
      │                          │ load_metadata(path)      │
      │─────────────────────────→│                          │
      │← metadata (tensor 列表) ─│                          │
      │                          │                          │
      │                          │ for each tensor:         │
      │                          │   get(storage_key)       │
      │─────────────────────────→│                          │
      │← (name, tensor) 流 ─────│                          │
      │ param.data.copy_(tensor) │                          │
      │                          │                          │
```

## 2.4 并发模型

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              并发模型                                            │
│                                                                                  │
│  API Server 进程 (主进程)                                                        │
│  ├── asyncio event loop                                                         │
│  │   ├── InputProcessor (请求预处理)                                             │
│  │   ├── OutputProcessor (输出后处理)                                            │
│  │   └── EngineCoreClient (ZMQ 通信)                                            │
│  └── TensorIpcSender (多模态张量发送)                                            │
│                                                                                  │
│  EngineCore 进程                                                                 │
│  ├── busy_loop (主线程)                                                          │
│  │   ├── Scheduler.schedule()                                                    │
│  │   ├── Executor.execute_model()                                                │
│  │   └── Executor.sample_tokens()                                                │
│  ├── input_thread (ZMQ 接收)                                                     │
│  ├── output_thread (ZMQ 发送)                                                    │
│  └── monitor_thread (Worker 健康监控)                                            │
│                                                                                  │
│  ExternalWorkerActor (Ray Actor, 每个 GPU 一个)                                  │
│  ├── worker_busy_loop (主循环)                                                   │
│  │   ├── dequeue SchedulerOutput (从 MessageQueue)                               │
│  │   ├── Worker.execute_model()                                                  │
│  │   ├── Worker.sample_tokens()                                                  │
│  │   └── enqueue ModelRunnerOutput (到 response_mq)                              │
│  └── NCCL/HCCl 通信线程 (数据面, 由 NCCL 管理)                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

CacheManagerActor 是**串行 Actor**（Ray Actor 默认单线程执行方法），天然串行化编译锁的申请与释放，保证锁的原子性：

```
CacheManagerActor (单线程 Actor):
├── 方法调用串行执行 (pull / push / try_acquire_compile_lock / ...)
├── _compile_locks 无需加锁 (Actor 天然互斥)
├── 线程模型: 简单，无后台线程
└── 可扩展: 如需并行 pull/push，可在 Actor 内引入线程池
```

---

# 3. 开发视图（Development View）

> 关注代码组织、模块划分、依赖关系。

## 3.1 模块划分

```
vllm/                                        # vLLM 核心 (仅 4 处参数传递)
├── v1/
│   └── engine/
│       ├── async_llm.py             # ⭐ 修改: 添加 external_actors 参数
│       ├── core.py                  # ⭐ 修改: 传递 external_actors
│       ├── core_client.py           # ⭐ 修改: 传递 external_actors
│       └── utils.py                 # ⭐ 修改: 传递 external_actors
│
multi-task-infer/                            # ⭐ 新增: 独立插件包 (G5 最小侵入)
├── pyproject.toml                     # 包配置 + vllm.general_plugins 入口点
├── vllm_external_executor/            # 插件代码
│   ├── __init__.py                    # 模块入口 + register_plugin()
│   ├── external_executor.py           # ⭐ ExternalExecutor (继承 RayExecutorV2)
│   ├── actor_pool_manager.py          # ⭐ ActorPoolManager
│   ├── external_worker_actor.py       # ⭐ ExternalWorkerActor + ActorState
│   ├── cache_manager_actor.py         # ⭐ CacheManagerActor (G6)
│   ├── cluster_state.py               # ⭐ NodeInfo / ActorRegistration / GlobalScheduler
│   ├── node_registry_actor.py         # ⭐ NodeRegistryActor (注册/心跳/统一视图/死检测)
│   ├── migration.py                   # ⭐ 迁移状态机 + 事务 + 幂等 + FlightBatchPolicy
│   └── storage_checkpoint_engine.py   # ⭐ StorageCheckpointEngine + StorageBackend (G7)
├── examples/
│   ├── basic_usage.py                 # 使用示例
│   ├── verify_multi_task_sharing.py   # 多任务共享验证脚本
│   └── mooncake_config.json           # Mooncake 配置模板
├── tests/
│   ├── test_global_scheduler.py       # 测试: 故障域感知调度
│   ├── test_migration.py              # 测试: 迁移状态机 + 事务 + 幂等
│   └── test_storage_checkpoint_engine.py   # 测试: nfs / mooncake_mock / mooncake
├── README.md                          # 使用文档
├── STORAGE_CHECKPOINT_ENGINE_DESIGN.md
├── STARTUP_DEPENDENCIES.md            # 启动依赖清单
└── verify_dependencies.sh             # 依赖验证脚本

说明:
- vLLM 核心零侵入 (G5): ExternalExecutor 不放入 vllm/v1/executor/，
  而是作为独立插件包通过 executor_class 参数注入
- 插件通过 pyproject.toml 的 entry-points.vllm.general_plugins 注册
- 代码行宽 ≤ 88 字符, Google 风格 docstring
```

## 3.2 依赖关系

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              依赖关系图                                          │
│                                                                                  │
│  ┌─────────────────────┐                                                        │
│  │  async_llm.py       │                                                        │
│  │  (AsyncLLM)         │                                                        │
│  └─────────┬───────────┘                                                        │
│            │ uses                                                                │
│            ↓                                                                     │
│  ┌─────────────────────┐     ┌─────────────────────┐                            │
│  │  core_client.py     │     │  external_executor  │                            │
│  │  (AsyncMPClient)    │     │  .py                │                            │
│  └─────────┬───────────┘     │  (ExternalExecutor) │                            │
│            │ uses            └─────────┬───────────┘                            │
│            ↓                           │ inherits                                │
│  ┌─────────────────────┐     ┌─────────┴───────────┐                            │
│  │  core.py            │     │  ray_executor_v2.py │                            │
│  │  (EngineCore)       │     │  (RayExecutorV2)    │                            │
│  └─────────┬───────────┘     └─────────┬───────────┘                            │
│            │ uses                       │ inherits                                │
│            ↓                           ↓                                         │
│  ┌─────────────────────┐     ┌─────────────────────┐                            │
│  │  utils.py           │     │  multiproc_executor │                            │
│  │  (CoreEngineProc    │     │  .py                │                            │
│  │   Manager)          │     │  (MultiprocExecutor)│                            │
│  └─────────────────────┘     └─────────────────────┘                            │
│                                                                                  │
│  ┌─────────────────────┐     ┌─────────────────────┐                            │
│  │  actor_pool_manager │     │  external_worker    │                            │
│  │  .py                │────→│  _actor.py          │                            │
│  │  (ActorPoolManager) │     │  (ExternalWorker    │                            │
│  └─────────────────────┘     │   Actor)            │                            │
│                              └─────────────────────┘                            │
│                                                                                  │
│  ┌─────────────────────┐     ┌─────────────────────┐                            │
│  │  external_executor  │────→│  cache_manager      │                            │
│  │  .py                │     │  _actor.py          │                            │
│  │  (ExternalExecutor) │     │  (CacheManagerActor)│                            │
│  └─────────────────────┘     └─────────┬───────────┘                            │
│                                        │ uses                                    │
│                                        ↓                                         │
│                              ┌─────────────────────┐                            │
│                              │  Ray Object Store    │                            │
│                              │  + NFS (大缓存)      │                            │
│                              └─────────────────────┘                            │
│                                                                                  │
│  ┌─────────────────────┐     ┌─────────────────────┐                            │
│  │  external_worker    │────→│  storage_checkpoint │                            │
│  │  _actor.py          │     │  _engine.py         │                            │
│  │  (ExternalWorker    │     │  (StorageCheckpoint │                            │
│  │   Actor)            │     │   Engine)           │                            │
│  └─────────────────────┘     └─────────┬───────────┘                            │
│                                        │ 使用                                     │
│                                        ↓                                         │
│                              ┌─────────────────────┐                            │
│                              │  StorageBackend     │                            │
│                              │  ├─ NFSStorageBackend                             │
│                              │  └─ MooncakeStoreBackend                          │
│                              └─────────────────────┘                            │
│                                                                                  │
│  外部依赖:                                                                       │
│  ├── ray (Actor 管理)                                                            │
│  ├── torch.distributed (NCCL/HCCl)                                              │
│  ├── vllm.distributed.weight_transfer (权重热更新)                               │
│  ├── vllm.distributed.device_communicators.shm_broadcast (MessageQueue)         │
│  ├── safetensors (NFS 权重序列化)                                                │
│  ├── mooncake-transfer-engine (可选, Mooncake Store 后端)                        │
│  └── verl.checkpoint_engine (可选, 注册到 verl CheckpointEngineRegistry)        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 3.3 需要修改的 vLLM 代码

### 3.3.1 AsyncLLM 构造函数

```python
# vllm/v1/engine/async_llm.py
class AsyncLLM(EngineClient):
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        ...
        external_actors: list | None = None,  # ⭐ 新增参数
    ) -> None:
        ...
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            ...
            external_actors=external_actors,  # ⭐ 传递
        )
```

### 3.3.2 EngineCoreClient.make_async_mp_client

```python
# vllm/v1/engine/core_client.py
@staticmethod
def make_async_mp_client(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    ...
    external_actors: list | None = None,  # ⭐ 新增参数
) -> "AsyncMPClient":
    ...
    return AsyncMPClient(
        vllm_config, executor_class, log_stats, ...,
        external_actors=external_actors,  # ⭐ 传递
    )
```

### 3.3.3 launch_core_engines

```python
# vllm/v1/engine/utils.py
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
    external_actors: list | None = None,  # ⭐ 新增参数
) -> Iterator[CoreEngineLaunch]:
    ...
    local_engine_manager = CoreEngineProcManager(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=log_stats,
        ...
        external_actors=external_actors,  # ⭐ 传递
    )
```

### 3.3.4 CoreEngineProcManager

```python
# vllm/v1/engine/utils.py
class CoreEngineProcManager:
    def __init__(
        self,
        ...
        external_actors: list | None = None,  # ⭐ 新增参数
    ):
        ...
        common_kwargs = {
            "vllm_config": vllm_config,
            "executor_class": executor_class,
            ...
            "external_actors": external_actors,  # ⭐ 传递
        }
```

### 3.3.5 EngineCoreProc

```python
# vllm/v1/engine/core.py
class EngineCoreProc(EngineCore):
    def __init__(
        self,
        ...
        external_actors: list | None = None,  # ⭐ 新增参数
    ):
        ...
        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            executor_fail_callback,
            internal_dp_balancing,
            external_actors=external_actors,  # ⭐ 传递
        )
```

### 3.3.6 EngineCore

```python
# vllm/v1/engine/core.py
class EngineCore:
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        ...
        external_actors: list | None = None,  # ⭐ 新增参数
    ):
        ...
        # ⭐ 创建 Executor，传入 external_actors
        if external_actors is not None:
            self.model_executor = executor_class(
                vllm_config, external_actors=external_actors
            )
        else:
            self.model_executor = executor_class(vllm_config)
```

## 3.4 新增文件清单

| 文件 | 说明 | 实际行数 |
|------|------|---------|
| `multi-task-infer/pyproject.toml` | 包配置 + vllm.general_plugins 入口点 | ~50 |
| `vllm_external_executor/__init__.py` | 模块入口 + register_plugin() | ~123 |
| `vllm_external_executor/external_executor.py` | ExternalExecutor 实现（G1/G4 + 迁移状态机） | ~718 |
| `vllm_external_executor/actor_pool_manager.py` | ActorPoolManager 实现（G1/G2 + 分布式 + 迁移） | ~849 |
| `vllm_external_executor/external_worker_actor.py` | ExternalWorkerActor 实现 | ~693 |
| `vllm_external_executor/cluster_state.py` | NodeInfo / ActorRegistration / GlobalScheduler（纯逻辑） | ~272 |
| `vllm_external_executor/node_registry_actor.py` | NodeRegistryActor：注册/心跳/统一视图/死检测 | ~279 |
| `vllm_external_executor/migration.py` | 迁移状态机 + 原子事务 + 幂等 + FlightBatchPolicy | ~313 |
| `vllm_external_executor/cache_manager_actor.py` | CacheManagerActor 实现（G6） | ~474 |
| `vllm_external_executor/storage_checkpoint_engine.py` | StorageCheckpointEngine + 后端（G7） | ~884 |
| `examples/basic_usage.py` | 使用示例 | ~223 |
| `examples/verify_multi_task_sharing.py` | 多任务共享验证脚本 | ~369 |
| `examples/mooncake_config.json` | Mooncake 配置模板 | ~10 |
| `tests/test_global_scheduler.py` | 测试：GlobalScheduler 故障域调度 | ~137 |
| `tests/test_migration.py` | 测试：迁移状态机 + 事务 + 幂等 | ~151 |
| `tests/test_storage_checkpoint_engine.py` | 测试：nfs / mooncake_mock / mooncake | ~483 |
| `verify_dependencies.sh` | 依赖验证脚本 | ~120 |

vLLM 核心修改（最小侵入，G5）：

| 文件 | 修改内容 |
|------|---------|
| `vllm/v1/engine/async_llm.py` | +5 行：添加 external_actors 参数并传递 |
| `vllm/v1/engine/core.py` | +11 行：添加 external_actors 参数并传给 executor |
| `vllm/v1/engine/core_client.py` | +9 行：透传 external_actors |
| `vllm/v1/engine/utils.py` | +4 行：透传 external_actors |

---

# 4. 物理视图（Physical View）

> 关注硬件拓扑、部署架构、网络通信。

## 4.1 部署拓扑

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Ray Cluster                                         │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Node 0 (Head Node)                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │  API Server  │  │ EngineCore   │  │ Actor 0      │  │ Actor 1      │  │   │
│  │  │  (进程)      │  │ (进程)       │  │ (GPU 0)      │  │ (GPU 1)      │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │         │                │                 │                 │             │   │
│  │         │ ZMQ            │ MessageQueue    │ SHM             │ SHM         │   │
│  │         │                │                 │                 │             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ Actor 2      │  │ Actor 3      │  │ Actor 4      │  │ Actor 5      │  │   │
│  │  │ (GPU 2)      │  │ (GPU 3)      │  │ (GPU 4)      │  │ (GPU 5)      │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │         │ SHM              │ SHM             │ SHM             │ SHM         │   │
│  └─────────┼──────────────────┼─────────────────┼─────────────────┼───────────┘   │
│            │                  │                 │                 │                │
│            │ NCCL/HCCl (数据面, 跨 GPU 通信)     │                 │                │
│            └──────────────────┴─────────────────┴─────────────────┘                │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Node 1 (Worker Node)                                                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ Actor 6      │  │ Actor 7      │  │ Actor 8      │  │ Actor 9      │  │   │
│  │  │ (GPU 0)      │  │ (GPU 1)      │  │ (GPU 2)      │  │ (GPU 3)      │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │         │                │                 │                 │             │   │
│  │         │ NCCL/HCCl      │ NCCL/HCCl       │ NCCL/HCCl       │ NCCL/HCCl   │   │
│  └─────────┼──────────────────┼─────────────────┼─────────────────┼───────────┘   │
│            │                  │                 │                 │                │
│            └──────────────────┴─────────────────┴─────────────────┘                │
│                                      │                                            │
│                              NCCL/HCCl (跨节点)                                    │
│                              (TCP/RDMA)                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 4.2 通信路径

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              通信路径                                            │
│                                                                                  │
│  控制面 (Executor ↔ Worker):                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  同节点: 共享内存 (SHM) + ZMQ IPC (unix domain socket)                   │   │
│  │  ──────────────────────────────────────────────────────────────────────  │   │
│  │  - ShmRingBuffer: 传递序列化数据 (SchedulerOutput, ModelRunnerOutput)    │   │
│  │  - ZMQ IPC: 大数据通知                                                   │   │
│  │  - SpinCondition: 低延迟唤醒                                             │   │
│  │  - 延迟: < 1μs                                                           │   │
│  │                                                                          │   │
│  │  跨节点: TCP (ZMQ)                                                       │   │
│  │  ──────────────────────────────────────────────────────────────────────  │   │
│  │  - ZMQ TCP socket: 传递所有数据                                          │   │
│  │  - 延迟: ~10-100μs (取决于网络)                                          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  数据面 (Worker ↔ Worker):                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  同节点: NVLink / PCIe (NCCL)                                            │   │
│  │  ──────────────────────────────────────────────────────────────────────  │   │
│  │  - AllReduce: TP 层的激活值聚合                                           │   │
│  │  - AllGather: EP 层的 expert 输出聚合                                     │   │
│  │  - P2P Send/Recv: PP 层的 IntermediateTensors                           │   │
│  │  - 带宽: ~300 GB/s (NVLink) / ~30 GB/s (PCIe)                           │   │
│  │                                                                          │   │
│  │  跨节点: RDMA / TCP (NCCL)                                               │   │
│  │  ──────────────────────────────────────────────────────────────────────  │   │
│  │  - AllReduce: 跨节点 TP                                                  │   │
│  │  - 带宽: ~100 Gbps (RDMA) / ~10 Gbps (TCP)                              │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  权重传输面 (Trainer ↔ Worker):                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  NCCL: 通过 NCCL broadcast 传输权重                                      │   │
│  │  IPC: 通过 CUDA IPC 共享内存传输权重                                     │   │
│  │  sharded_rdt: 通过 Ray tensor transport 传输权重                         │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 4.3 资源分配

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              资源分配                                            │
│                                                                                  │
│  Actor Pool 预启动时:                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  Placement Group:                                                        │   │
│  │  bundles = [                                                             │   │
│  │      {"GPU": 1},  # Actor 0 → GPU 0                                     │   │
│  │      {"GPU": 1},  # Actor 1 → GPU 1                                     │   │
│  │      {"GPU": 1},  # Actor 2 → GPU 2                                     │   │
│  │      ...                                                                 │   │
│  │      {"CPU": 1},  # 控制节点                                              │   │
│  │  ]                                                                       │   │
│  │  strategy = "PACK"  # 尽量放在同一节点                                    │   │
│  │                                                                          │   │
│  │  每个 Actor 占用:                                                        │   │
│  │  - 1 GPU (已绑定)                                                        │   │
│  │  - ~2 GB GPU 内存 (公共库 + 预热)                                        │   │
│  │  - ~500 MB CPU 内存 (Python 进程)                                        │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  vLLM 实例运行时:                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  每个 Worker (Actor) 占用:                                               │   │
│  │  - 模型权重: ~1-40 GB (取决于模型大小和 TP)                               │   │
│  │  - KV Cache: ~1-20 GB (取决于 KV Cache 大小)                             │   │
│  │  - CUDA Graph: ~1-5 GB                                                   │   │
│  │  - 激活值: ~1-10 GB                                                      │   │
│  │  - MessageQueue: ~24 MB (共享内存)                                       │   │
│  │                                                                          │   │
│  │  EngineCore 进程占用:                                                    │   │
│  │  - Scheduler: ~100 MB                                                    │   │
│  │  - MessageQueue: ~24 MB                                                  │   │
│  │  - ZMQ sockets: ~10 MB                                                   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3.1 编译缓存混合存储架构（G6）

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          编译缓存存储拓扑                                         │
│                                                                                  │
│  torch.compile 缓存 (本地 ~/.cache/vllm/torch_compile_cache/{hash}/)             │
│  ├── 1B-3B 模型:   50-200 MB                                                    │
│  ├── 7B-13B 模型:  200 MB - 1 GB                                                 │
│  └── 30B-70B 模型: 1-5 GB                                                        │
│                                                                                  │
│  混合分流策略 (ROT = 50 MB):                                                     │
│  ┌────────────────────────────────────────────┐   ┌───────────────────────────┐ │
│  │  小缓存 (< 50 MB)                          │   │  大缓存 (≥ 50 MB)          │ │
│  │  ─────────────────────────────            │   │  ────────────────────────  │ │
│  │  Ray Object Store (默认 30% 节点内存)      │   │  NFS + gzip 压缩            │ │
│  │  - 同节点零拷贝                             │   │  - tar.gz 打包              │ │
│  │  - 无需额外部署                             │   │  - 跨节点共享               │ │
│  │  - 适合 1B-3B 模型                          │   │  - 适合 7B+ 模型            │ │
│  │  限制: ~100 MB/对象阈值, 超限 spill          │   │  限制: 依赖 NFS 带宽        │ │
│  └────────────────────────────────────────────┘   └───────────────────────────┘ │
│                                                                                  │
│  缓存流程:                                                                       │
│  ① 本地缓存命中 → 直接使用 (最快)                                                │
│  ② CacheManagerActor.pull() → 解压到本地 → 使用                                  │
│  ③ fallback: 获取编译锁 → 编译 → 打包 → push → 释放锁                            │
│                                                                                  │
│  编译锁: 防止多个节点同时编译同一缓存                                             │
│  status: acquired (获锁编译) / wait (他人编译中, 等待) / done (缓存已就绪)       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3.2 存储权重加载资源（G7）

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          存储权重加载拓扑                                         │
│                                                                                  │
│  StorageBackend 对比:                                                            │
│  ┌───────────────────────────┐   ┌───────────────────────────────┐             │
│  │  NFSStorageBackend        │   │  MooncakeStoreBackend         │             │
│  │  ─────────────────────    │   │  ──────────────────────────   │             │
│  │  格式: safetensors        │   │  格式: MooncakeDistributedStore│             │
│  │  部署: NFS 挂载即可       │   │  部署: mooncake_master +         │             │
│  │  性能: 依赖网络带宽       │   │        (可选) mooncake_client   │             │
│  │         (~1-3 GB/s)       │   │  性能: RDMA ~9 GB/s (IB)       │             │
│  │                           │   │        GPUDirect 支持          │             │
│  │  适用: 小集群/无 RDMA      │   │  适用: 大规模集群/热切换频繁    │             │
│  └───────────────────────────┘   └───────────────────────────────┘             │
│                                                                                  │
│  Mooncake Key 约定:                                                              │
│    ckpt:{checkpoint_path}:metadata      → 元数据 JSON                            │
│    ckpt:{checkpoint_path}:tensor:{name} → 权重 tensor 字节                        │
│                                                                                  │
│  内存占用: 加载时按 tensor 流式处理, 峰值 = 单 tensor 大小                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 4.4 故障域

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              故障域                                              │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │  故障类型                    影响范围              恢复策略               │   │
│  │  ──────────────────────────────────────────────────────────────────────  │   │
│  │                                                                          │   │
│  │  Actor 崩溃                  单个 Worker 不可用    ActorPoolManager       │   │
│  │                              EngineCore 检测到     检测到后标记 FAILED     │   │
│  │                              Worker 死亡          触发 release + 重建     │   │
│  │                                                                          │   │
│  │  EngineCore 崩溃             整个 vLLM 实例不可用  AsyncLLM 检测到        │   │
│  │                              需要 release actors   触发 release actors    │   │
│  │                              回 Pool              可重新 acquire 创建     │   │
│  │                                                                          │   │
│  │  NCCL/HCCl 故障              TP/PP 通信失败       Worker 检测到通信超时   │   │
│  │                              需要重新初始化        触发 reset + 重新初始化 │   │
│  │                                                                          │   │
│  │  weight_transfer 失败        模型加载失败          回退到磁盘加载          │   │
│  │                              需要重新加载模型      或标记 Actor FAILED     │   │
│  │                                                                          │   │
│  │  CacheManagerActor 崩溃     编译缓存共享不可用     Worker 降级为直接编译   │   │
│  │                             缓存拉取失败          编译锁失效, 仅损失      │   │
│  │                                                   去重能力, 不影响推理    │   │
│  │                                                                          │   │
│  │  Mooncake/NFS 存储故障      权重加载失败          回退到磁盘加载 /         │   │
│  │                             编译缓存读写失败      跳过缓存直接编译         │   │
│  │                                                                          │   │
│  │  节点故障                    该节点所有 Actor 不可用  ActorPoolManager     │   │
│  │                              (心跳超时)              心跳线程检测到死节点     │   │
│  │                                                    recover_node() 重建      │   │
│  │                                                    (见 §4.5)               │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 4.5 分布式控制面（节点注册 / 心跳 / 全局调度 / 故障隔离）

> 三项分布式能力在 `node_registry_actor.py` + `cluster_state.py` +
> `actor_pool_manager.py` 落地，与 §4.4 故障域对应。

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            分布式控制面                                         │
│                                                                                  │
│  ┌────────────────────┐        ┌─────────────────────────────────────────────┐  │
│  │  NodeRegistryActor  │◄───── │  ActorPoolManager (每租户一个, 共享全局视图) │  │
│  │  (Ray 单例, detached)│       │  - pre_start: 注册节点 + Actor               │  │
│  │                    │       │  - acquire: 委托 GlobalScheduler              │  │
│  │  _nodes:           │       │  - 心跳线程: 周期探测 Actor → registry.heartbeat│
│  │    node_id → NodeInfo │     │  - recover_node: 节点故障重建                 │  │
│  │  _actors:          │       └─────────────────────────────────────────────┘  │
│  │    actor_id → ActorRegistration │                                            │
│  └────────────────────┘                                                        │
│           ▲                                                                    │
│           │ heartbeat(state, timestamp) / register / set_actor_state           │
│  ┌────────┴────────────────────────────────────────────────────────────┐       │
│  │  三个能力 → 三类实现对标                                             │       │
│  │  ① 跨节点注册/心跳/统一视图                                          │       │
│  │     - NodeInfo: node_id/ip/hostname/fault_domain/total_gpus/free_gpus │      │
│  │     - ActorRegistration: actor_id/node/device/state/lease/heartbeat  │       │
│  │     - get_global_view(): 每节点空闲 GPU + 每故障域分布 + 心跳年龄      │       │
│  │     - detect_dead_actors/nodes(timeout): 失联检测                     │       │
│  │  ② 全局调度器 (GlobalScheduler, 纯逻辑可单测)                          │      │
│  │     - select_actors(): 按故障域 spread → 负载均衡 → driver 就近        │       │
│  │     - fault_domain_constraint: {故障域: 数量} 硬约束                  │       │
│  │     - 跨节点迁移 = rebuild_actor(): GPU Actor 无法 live-migrate,      │       │
│  │       故障/负载触发的等价重建                                          │       │
│  │  ③ 节点级故障域隔离                                                   │       │
│  │     - fault_domain 默认 = node_id (每节点独立故障域)                  │       │
│  │     - acquire 默认 spread, 单节点故障只损失其自身 Actor               │       │
│  │     - 心跳超时 → mark_node_failed → recover_node 在健康节点重建        │       │
│  │     - 所有 ray.get 加 RPC_TIMEOUT, 节点失联不挂起                     │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

# 5. 场景视图（Scenarios / Use Case View）

> 用关键用例串联其他四个视图，展示系统如何工作。

## 5.1 场景 1：预启动 Actor Pool

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  场景: 预启动 Actor Pool                                                       │
│  参与者: 运维人员, ActorPoolManager, Ray Cluster, ExternalWorkerActor          │
│                                                                                │
│  前置条件: Ray Cluster 已启动，GPU 资源可用                                    │
│                                                                                │
│  主流程:                                                                       │
│  1. 运维人员调用 ActorPoolManager.pre_start(num_actors=8, devices_per_node=...)│
│  2. ActorPoolManager 创建 Placement Group                                      │
│  3. ActorPoolManager 创建 8 个 ExternalWorkerActor                             │
│  4. 每个 Actor:                                                                │
│     a. 绑定 GPU 设备                                                           │
│     b. Import 公共库 (torch, vllm, Worker, ModelRunner)                        │
│     c. 预热 NCCL/HCCl (创建临时 ProcessGroup 并销毁)                           │
│     d. 状态设为 IDLE                                                           │
│  5. ActorPoolManager 等待所有 Actor 就绪                                       │
│  6. ActorPoolManager 构建 node_mapping (node_id → actor_indices)               │
│                                                                                │
│  后置条件: 8 个 Actor 处于 IDLE 状态，可被租用                                 │
│                                                                                │
│  耗时: ~10-20s                                                                 │
│                                                                                │
│  涉及的视图:                                                                   │
│  - 逻辑视图: ActorPoolManager.pre_start()                                      │
│  - 进程视图: Actor 创建、预初始化                                              │
│  - 开发视图: actor_pool_manager.py, external_worker_actor.py                   │
│  - 物理视图: Placement Group、GPU 绑定                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 5.2 场景 2：从 Pool 构建 vLLM 实例

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  场景: 从 Pool 构建 vLLM 实例                                                   │
│  参与者: 用户, AsyncLLM, EngineCore, ExternalExecutor, ActorPoolManager, Actors  │
│                                                                                  │
│  前置条件: ActorPoolManager 已预启动，有足够空闲 Actor                            │
│                                                                                  │
│  主流程:                                                                         │
│  1. 用户调用 pool.acquire(tp_size=4, pp_size=2) 获取 8 个 Actor                 │
│  2. 用户创建 AsyncLLM(executor_class=ExternalExecutor, external_actors=actors)   │
│  3. AsyncLLM → AsyncMPClient → launch_core_engines → EngineCore                 │
│  4. EngineCore 创建 ExternalExecutor(vllm_config, external_actors=actors)        │
│  5. ExternalExecutor._init_executor():                                           │
│     a. 从 actors 创建 RayWorkerHandle (不创建新 Actor)                           │
│     b. 创建 MessageQueue (广播 + 响应)                                           │
│     c. 调用 Actor.initialize_worker() (创建 WorkerWrapperBase)                   │
│     d. 调用 Actor.init_device() (NCCL/HCCl 初始化, 已预热)                       │
│     e. 调用 Actor.load_model_via_weight_transfer() (加载模型)                    │
│     f. 调用 Actor.initialize_kv_cache() (分配 KV Cache)                          │
│     g. 调用 Actor.compile_or_warm_up_model() (Graph Capture, 应用编译优化)       │
│     h. 调用 Actor.run() (启动 busy loop)                                         │
│     i. 等待所有 Worker 就绪                                                      │
│  6. vLLM 实例就绪，可以处理推理请求                                              │
│                                                                                  │
│  后置条件: vLLM 实例就绪，8 个 Actor 处于 RUNNING 状态                            │
│                                                                                  │
│  耗时: ~7-36s (相比标准流程 68-495s 减少 84-93%)                                │
│                                                                                  │
│  涉及的视图:                                                                     │
│  - 逻辑视图: ExternalExecutor._init_executor()                                   │
│  - 进程视图: Actor 初始化、模型加载、KV Cache 分配                                │
│  - 开发视图: vllm_external_executor/external_executor.py                  │
│  - 物理视图: MessageQueue (SHM/TCP)、NCCL/HCCl、weight_transfer                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 5.3 场景 3：模型热切换

> ✅ **实现状态：已实现（迁移状态机 + 原子事务 + 幂等）**。
> `ExternalExecutor.switch_model()` 用 `MigrationStateMachine` 驱动
> PREPARING → GRACEFUL_PAUSE → CHECKPOINT → UNLOAD → LOAD → RESTORE →
> COMPLETED；`migration_id` 保证请求级幂等；任一 worker 切换失败触发
> 补偿回滚（已切换 worker 通过 `switch_model_rollback` 重载旧模型、
> executor 配置引用恢复），原子回到迁移前状态。飞行中 batch 由
> `FlightBatchPolicy`（drain / pause_serialize / preempt）+ `set_scheduler_gate`
> 与 EngineCore 调度器协同。

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  场景: 模型热切换（迁移状态机 + 原子事务）                                        │
│  参与者: 用户, ExternalExecutor, EngineCore(调度器), Actors, StorageBackend       │
│                                                                                  │
│  前置条件: vLLM 实例正在运行模型 A，需要切换到模型 B                              │
│            (world_size 不变; 飞行中 batch 按策略处理)                             │
│                                                                                  │
│  状态机主流程:                                                                   │
│  ① PREPARING     校验 world_size / migration_id 幂等去重                        │
│  ② GRACEFUL_PAUSE 调用 set_scheduler_gate 注入的 pause 回调                     │
│                   (DRAIN=排空 / PAUSE_SERIALIZE=冻结序列化 / PREEMPT=抢占中断)   │
│  ③ CHECKPOINT    保存旧 vllm_config + 各 worker switch_model_snapshot()         │
│                   注册补偿: 恢复 config + 重载旧模型                             │
│  ④ UNLOAD + LOAD 并行切换所有 worker; 逐个收集结果                              │
│                   部分失败 → 对已切换 worker switch_model_rollback(snapshot)    │
│  ⑤ RESTORE       _reinitialize_kv_cache() + 编译缓存懒加载 + resume 回调         │
│  ⑥ COMPLETED     终端态; 或 FAILED → ROLLING_BACK → ROLLED_BACK(原子回滚)      │
│                                                                                  │
│  事务性:                                                                         │
│  - 迁移失败自动回滚到迁移前状态（请求级幂等，重复 migration_id 直接返回结果）     │
│  - FlightBatchPolicy:                                                            │
│    · DRAIN           完成当前步再迁移（最安全，默认）                             │
│    · PAUSE_SERIALIZE 暂停并序列化调度状态，迁移后确定性恢复                       │
│    · PREEMPT         抢占式中断 + 请求级重放（需前端幂等）                        │
│                                                                                  │
│  后置条件: vLLM 实例已切换到模型 B（或原子回滚到模型 A），可处理推理请求          │
│                                                                                  │
│  耗时: ~10-60s（回滚另需重载旧模型 ~同量级）                                     │
│                                                                                  │
│  涉及的视图:                                                                     │
│  - 逻辑视图: MigrationStateMachine, ExternalExecutor.switch_model(),             │
│             ExternalWorkerActor.switch_model_snapshot/rollback()                 │
│  - 进程视图: 调度器暂停/恢复、模型卸载、权重加载、KV Cache 重新分配、补偿回滚     │
│  - 开发视图: vllm_external_executor/migration.py, external_executor.py,          │
│             external_worker_actor.py, storage_checkpoint_engine.py               │
│  - 物理视图: StorageBackend (NFS/Mooncake) / weight_transfer 通信                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 5.3.1 场景 3b：跨节点 Actor 迁移（原子 build-before-swap）

> `ActorPoolManager.migrate_actor()` 实现主动迁移：先建后换（build 新 Actor
> → 注册 → 原子切换本地映射 → 最后才 kill 旧 Actor），失败时旧 Actor 完好，
> 回滚无损。区别于 `rebuild_actor()`（先 kill 后建，用于死节点故障恢复）。

```
  ① PREPARING    校验 actor_id 存在、非 LEASED/RUNNING
  ② GRACEFUL_PAUSE 心跳确认旧 actor 健康（DRAIN 语义）
  ③ CHECKPOINT   记录旧 handle/registration
  ④ UNLOAD       在目标节点 build 新 Actor（NodeAffinity 调度）
  ⑤ LOAD         注册新 actor + 原子更新本地映射（build-before-swap）
  ⑥ RESTORE      仅此刻 kill 旧 actor + 注销旧注册（swap 已提交）
  ⑦ COMPLETED    幂等（migration_id 去重）；中途失败 → 旧 actor 仍在 → 无损回滚
```

## 5.4 场景 4：弹性伸缩（TP/PP 变化）

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  场景: 弹性伸缩（TP/PP 变化）                                                  │
│  参与者: 用户, ActorPoolManager, ExternalExecutor, Actors                      │
│                                                                                │
│  前置条件: vLLM 实例使用 TP=4, PP=2 (8 个 Actor)，需要扩展到 TP=8, PP=2 (16 个)│
│                                                                                │
│  主流程:                                                                       │
│  1. 用户调用 pool.acquire(tp_size=8, pp_size=2) 获取 16 个 Actor               │
│     - Pool 返回 8 个已有 Actor + 8 个新 Actor (如果有的话)                     │
│     - 或者 Pool 返回错误 (空闲 Actor 不足)                                     │
│  2. 用户创建新的 AsyncLLM (使用 16 个 Actor)                                   │
│  3. 旧的 vLLM 实例释放 Actor: pool.release(old_actors)                         │
│  4. 新的 vLLM 实例使用 16 个 Actor 初始化                                      │
│     - 重新初始化 NCCL/HCCl (world_size=16)                                     │
│     - 重新加载模型 (TP=8 分片)                                                 │
│     - 重新分配 KV Cache                                                        │
│                                                                                │
│  后置条件: vLLM 实例已扩展到 TP=8, PP=2                                        │
│                                                                                │
│  涉及的视图:                                                                   │
│  - 逻辑视图: ActorPoolManager.acquire(), ExternalExecutor._init_executor()     │
│  - 进程视图: NCCL/HCCl 重新初始化、模型重新加载                                │
│  - 开发视图: actor_pool_manager.py, external_executor.py                       │
│  - 物理视图: 新的 NCCL/HCCl 拓扑 (16 个 Worker)                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 5.5 场景 5：释放 Actor 回 Pool

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  场景: 释放 Actor 回 Pool                                                        │
│  参与者: 用户, ExternalExecutor, ActorPoolManager, Actors                        │
│                                                                                  │
│  前置条件: vLLM 实例不再需要，需要释放 Actor 回 Pool                             │
│                                                                                  │
│  主流程:                                                                         │
│  1. 用户调用 executor.shutdown() 或 pool.release(actors)                         │
│  2. ExternalExecutor.shutdown():                                                 │
│     a. 停止 busy loop                                                            │
│     b. 关闭 MessageQueue                                                         │
│     c. 调用 Actor.reset() (每个 Actor)                                           │
│  3. Actor.reset():                                                               │
│     a. worker.shutdown() (释放模型、KV Cache、分布式环境)                         │
│     b. worker = None                                                             │
│     c. vllm_config = None                                                        │
│     d. state = IDLE                                                              │
│  4. ActorPoolManager 更新 states[idx] = IDLE                                     │
│                                                                                  │
│  后置条件: Actor 回到 IDLE 状态，可以被重新租用                                   │
│                                                                                  │
│  耗时: ~5-10s                                                                    │
│                                                                                  │
│  涉及视图:                                                                       │
│  - 逻辑视图: ExternalExecutor.shutdown(), Actor.reset()                          │
│  - 进程视图: 资源释放、状态重置                                                   │
│  - 开发视图: vllm_external_executor/external_executor.py,                        │
│  - 物理视图: GPU 内存释放、NCCL/HCCl 销毁                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 5.6 场景 6：编译缓存懒加载共享（G6）

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  场景: 编译缓存懒加载共享                                                         │
│  参与者: ExternalExecutor, CacheManagerActor, Workers, Ray Object Store, NFS      │
│                                                                                  │
│  前置条件: 集群已启动 CacheManagerActor (由 ActorPoolManager.pre_start 创建)      │
│                                                                                  │
│  主流程 (每个 vLLM 实例初始化时):                                                 │
│  1. ExternalExecutor._handle_compilation_optimization():                          │
│     a. 计算 cache_hash (模型架构/TP/PP/DP/cudagraph 配置)                         │
│     b. 检查本地缓存 ~/.cache/vllm/torch_compile_cache/{hash}/                    │
│        - 命中 → 跳过编译，直接进入 Graph Capture                                 │
│     c. 未命中 → 向 CacheManagerActor.pull(hash) 拉取                             │
│        - 命中 → 解压到本地 → 跳过编译                                           │
│     d. 都未命中 → try_acquire_compile_lock(hash)                                │
│        - "acquired" → 编译 → 打包 → push(hash) → release_lock                  │
│        - "wait"     → 等待他人编译完成 → 再 pull                                │
│        - "done"     → 直接 pull                                                 │
│  2. 编译完成后 Workers 进入 Graph Capture 阶段                                    │
│                                                                                  │
│  后置条件: 所有节点共享一份编译产物，重复编译被消除                                │
│                                                                                  │
│  性能:                                                                           │
│  - 节点 1 (无缓存): 30-180s (编译)                                               │
│  - 节点 2+: 5-10s (拉取缓存)                                                    │
│                                                                                  │
│  涉及的视图:                                                                     │
│  - 逻辑视图: ExternalExecutor._handle_compilation_optimization()                 │
│  - 进程视图: CacheManagerActor pull/push/编译锁 交互 (见 2.3.5 时序)             │
│  - 开发视图: vllm_external_executor/cache_manager_actor.py                       │
│  - 物理视图: Ray Object Store (<50MB) + NFS (≥50MB) 混合存储                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 5.7 场景 7：存储权重加载（G7）

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  场景: 从存储加载模型权重                                                         │
│  参与者: ExternalWorkerActor, StorageCheckpointEngine, StorageBackend             │
│                                                                                  │
│  前置条件: checkpoint 已保存到存储 (NFS 目录或 Mooncake Store)                    │
│                                                                                  │
│  主流程:                                                                         │
│  1. Worker.initialize_worker() 创建模型结构 (dummy weights)                      │
│  2. Worker.load_model_from_storage(checkpoint_path, backend, config):            │
│     a. 创建 StorageCheckpointEngine (NFS/Mooncake)                               │
│     b. set_checkpoint(path)                                                      │
│     c. get_weights() 流式获取 (name, tensor)                                     │
│     d. param.data.copy_(tensor) 写入模型                                         │
│  3. 与 weight_transfer 相比:                                                     │
│     - 不需要 Trainer 在线 (存储是持久的)                                         │
│     - 适合多任务推理 (多次从存储加载不同权重)                                     │
│     - Mooncake 后端 RDMA ~9 GB/s                                                 │
│                                                                                  │
│  后置条件: 模型权重已就绪，可进入 KV Cache 初始化和编译阶段                       │
│                                                                                  │
│  涉及的视图:                                                                     │
│  - 逻辑视图: ExternalWorkerActor.load_model_from_storage()                       │
│  - 进程视图: 存储加载时序 (见 2.3.6)                                             │
│  - 开发视图: vllm_external_executor/storage_checkpoint_engine.py                 │
│  - 物理视图: NFSStorageBackend / MooncakeStoreBackend                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

# 6. 性能分析

## 6.1 初始化耗时对比

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              初始化耗时对比                                       │
│                                                                                  │
│  阶段                    RayExecutorV2      ExternalExecutor     节省            │
│  ──────────────────────────────────────────────────────────────────────────────  │
│  Actor 创建              5-10s              ~0s (从 Pool 获取)   5-10s          │
│  GPU 发现                ~1s                ~0.1s                ~0.9s          │
│  MessageQueue            ~0.5s              ~0.5s                0s             │
│  Worker 初始化           ~2s                ~0.5s (已预热)       ~1.5s          │
│  设备初始化 (NCCL)       3-5s               1-3s (已预热)        2-2s           │
│  模型加载                30-300s            5-30s (weight_transfer) 25-270s     │
│  KV Cache 初始化         1-2s               1-2s                 0s             │
│  算子编译 (torch.compile) 25-170s           ~0s (缓存命中)       25-170s        │
│  Graph Capture           5-10s              1-3s (配置优化)      4-7s           │
│  ──────────────────────────────────────────────────────────────────────────────  │
│  总计                    68-495s            7-36s                61-459s        │
│                                                                                  │
│  节省比例              -                  84-93%                -              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 6.2 运行时性能

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              运行时性能                                          │
│                                                                                  │
│  ExternalExecutor 与 RayExecutorV2 的数据通路完全一致:                            │
│                                                                                  │
│  控制面:                                                                         │
│  - 同节点: MessageQueue (SHM) → 延迟 < 1μs                                      │
│  - 跨节点: MessageQueue (TCP) → 延迟 ~10-100μs                                  │
│                                                                                  │
│  数据面:                                                                         │
│  - NCCL/HCCl AllReduce → 带宽 ~300 GB/s (NVLink)                                │
│  - NCCL/HCCl P2P → 带宽 ~30 GB/s (PCIe)                                         │
│                                                                                  │
│  结论: ExternalExecutor 的运行时性能与 RayExecutorV2 完全一致                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 6.3 编译缓存共享性能（G6）

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          编译缓存共享性能                                         │
│                                                                                  │
│  多节点部署 (无 CacheManagerActor):                                              │
│  ┌────────────────────────────────────────────────────────────────────┐          │
│  │  节点 1 首次启动: 30-180s (torch.compile)                          │          │
│  │  节点 2 首次启动: 30-180s (重复编译, 浪费)                         │          │
│  │  节点 3 首次启动: 30-180s (重复编译, 浪费)                         │          │
│  └────────────────────────────────────────────────────────────────────┘          │
│                                                                                  │
│  多节点部署 (有 CacheManagerActor + NFS):                                        │
│  ┌────────────────────────────────────────────────────────────────────┐          │
│  │  节点 1 首次启动: 30-180s (编译 + push)                             │          │
│  │  节点 2 首次启动: 5-10s  (pull + 解压, 跳过编译)                   │          │
│  │  节点 3 首次启动: 5-10s  (pull + 解压, 跳过编译)                   │          │
│  │  后续所有节点:    5-10s  (本地缓存命中)                             │          │
│  └────────────────────────────────────────────────────────────────────┘          │
│                                                                                  │
│  存储带宽参考 (perl benchmark):                                                  │
│  ┌────────────────────────────────────────┐                                     │
│  │  mooncake (RDMA, 2*8 H100):  9.44 GB/s │                                     │
│  │  NFS (TCP):                  ~1-3 GB/s  │                                     │
│  └────────────────────────────────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 6.4 存储权重加载性能（G7）

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          存储权重加载性能                                         │
│                                                                                  │
│  模型大小         | NFS (1-3 GB/s)           | Mooncake (9 GB/s)                  │
│  ──────────────────────────────────────────────────────────────────────────────  │
│  1B-3B  (6 GB)   | 2-6s                     | ~0.7s                              │
│  7B-13B (28 GB)  | 9-28s                    | ~3.1s                              │
│  30B-70B (140GB) | 47-140s                  | ~15.6s                             │
│                                                                                  │
│  相比磁盘加载 (30-300s) 和 weight_transfer (依赖 Trainer 在线):                  │
│  - NFS 存储加载: 持久化，无 Trainer 依赖                                         │
│  - Mooncake 存储加载: 持久化 + RDMA 高性能                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

# 7. 约束与限制

## 7.1 已知约束

| 约束 | 说明 | 影响 |
|------|------|------|
| **Actor 必须预启动** | ExternalExecutor 不创建 Actor，只从 Pool 获取 | 需要提前规划资源 |
| **模型加载三路径可选** | 磁盘加载 / weight_transfer / 存储加载 (G7) | 按场景选择，不再强依赖 weight_transfer |
| **数据通路固定** | 必须使用 MessageQueue（与 RayExecutorV2 一致） | 无法使用其他通信方式 |
| **模型架构变化需重建** | 如果模型架构变化，需要重新创建模型结构 | 跨架构切换较慢 |
| **CUDA Graph 不兼容** | 不同模型的 CUDA Graph 不兼容（涉及运行时状态） | 需要重新执行 Graph Capture；torch.compile 缓存可共享 (G6) |
| **空间换时间** | 编译缓存共享需要 Ray Object Store + NFS 存储 | 大缓存 (≥50MB) 走 NFS 压缩 |
| **缓存命中依赖配置一致** | cache_hash 取决于架构/TP/PP/DP/cudagraph 配置 | 配置不同则无法共享 |

## 7.2 已知问题

| 问题 | 影响 | 解决方案 |
|------|------|---------|
| **模型架构变化** | 需要重新创建模型结构 | 支持两种模式：同架构热切换 / 跨架构切换 |
| **Graph Capture 优化** | CUDA Graph 不能直接缓存，但可通过配置优化减少 Capture 时间 | 减少 batch size、使用 PIECEWISE 模式、确保 torch.compile 缓存启用 |
| **KV Cache 大小变化** | 不同模型/并行策略可能需要不同大小的 KV Cache | 每次重新分配 KV Cache |
| **分布式环境重建** | TP/PP 大小变化时需要重新初始化 NCCL/HCCl | 允许重新初始化，NPU 侧已预热所以很快 |
| **switch_model 时序** | 热切换 (G3) 需要调用方保证无 in-flight 请求 | executor 层负责模型重建 + 存储加载 + KV cache 重分配 (已实现) |
| **编译等待硬编码** | 编译锁等待用 time.sleep(5) 轮询 | 改为事件通知/等待机制 |

---

# 8. 待确认问题

| 编号 | 问题 | 选项 | 建议 |
|------|------|------|------|
| Q1 | weight_transfer 的触发方式？ | A) Trainer 主动推送 B) Worker 主动拉取 C) 改用存储加载 (G7) | 已实现 C；A/B 视场景 |
| Q2 | 模型结构是否可以在 Actor 预启动时就创建？ | A) 预启动时创建 (meta device) B) acquire 后创建 | 建议 B，更灵活 |
| Q3 | 如何优化 Graph Capture 时间？ | A) 减少 batch size B) 使用 PIECEWISE 模式 C) 确保 torch.compile 缓存 D) 以上全部 | 建议 D，综合优化 |
| Q4 | weight_transfer 失败时的处理策略？ | A) 回退到磁盘加载 B) 标记 Actor FAILED C) 回退到存储加载 | 建议 A/C |
| Q5 | Actor Pool 是否需要持久化？ | A) 需要 (跨 vLLM 实例) B) 不需要 (每次重建) | 建议 A |
| Q6 | 是否需要支持多 vLLM 实例共享同一 Pool？ | A) 需要 B) 不需要 | 建议 A，提高资源利用率 |
| Q7 | 编译锁等待机制？ | A) 轮询 (time.sleep) B) 事件通知 C) Ray 条件变量 | 建议 C |
| Q8 | Mooncake 后端是否需要 batch API？ | A) 单 tensor put/get B) batch_put_from_multi_buffers | 建议 B，提升吞吐 |

---

# 9. 附录

## 9.1 参考代码位置

| 模块 | 文件路径 |
|------|---------|
| Executor 基类 | `vllm/v1/executor/abstract.py` |
| RayExecutorV2 | `vllm/v1/executor/ray_executor_v2.py` |
| MultiprocExecutor | `vllm/v1/executor/multiproc_executor.py` |
| MessageQueue | `vllm/distributed/device_communicators/shm_broadcast.py` |
| WorkerWrapperBase | `vllm/v1/worker/worker_base.py` |
| GPUWorker | `vllm/v1/worker/gpu_worker.py` |
| GPUModelRunner | `vllm/v1/worker/gpu_model_runner.py` |
| EngineCore | `vllm/v1/engine/core.py` |
| AsyncLLM | `vllm/v1/engine/async_llm.py` |
| WeightTransferEngine | `vllm/distributed/weight_transfer/base.py` |
| ExternalExecutor (插件) | `multi-task-infer/vllm_external_executor/external_executor.py` |
| ActorPoolManager (插件) | `multi-task-infer/vllm_external_executor/actor_pool_manager.py` |
| ExternalWorkerActor (插件) | `multi-task-infer/vllm_external_executor/external_worker_actor.py` |
| CacheManagerActor (插件) | `multi-task-infer/vllm_external_executor/cache_manager_actor.py` |
| StorageCheckpointEngine (插件) | `multi-task-infer/vllm_external_executor/storage_checkpoint_engine.py` |
| verl CheckpointEngine 接口 | `verl/verl/checkpoint_engine/base.py` |
| verl MooncakeCheckpointEngine | `verl/verl/checkpoint_engine/mooncake_checkpoint_engine.py` |
| MooncakeDistributedStore API | `mooncake/store` (mooncake-transfer-engine 包) |

## 9.2 术语对照表

| 英文 | 中文 |
|------|------|
| Actor Pool | Actor 池 |
| Worker Lease | Worker 租用 |
| Worker Release | Worker 释放 |
| weight_transfer | 权重传输 |
| MessageQueue | 消息队列 |
| Placement Group | 放置组 |
| CUDA Graph | CUDA 图 |
| NCCL/HCCl | NVIDIA/Huawei 集合通信库 |
| CacheManagerActor | 缓存管理器 Actor |
| 编译锁 | Compilation Lock |
| 懒加载 | Lazy Loading |
| StorageCheckpointEngine | 存储 checkpoint 引擎 |
| Mooncake Store | Mooncake 分布式存储 |
