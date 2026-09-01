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
| G3 | 模型热切换 | 通过 weight_transfer 机制动态加载新模型权重，不重启进程 |
| G4 | 数据通路复用 | 复用 RayExecutorV2 的 MessageQueue 通信机制，保持性能一致 |
| G5 | 最小侵入 | 尽量以插件形式实现，减少对 vLLM 核心代码的修改 |

## 0.3 术语定义

| 术语 | 定义 |
|------|------|
| **Actor Pool** | 预启动的 Ray Actor 集合，每个 Actor 绑定一个 NPU/GPU 设备 |
| **ExternalExecutor** | 从 Actor Pool 获取 Worker 的 Executor 实现 |
| **ExternalWorkerActor** | 预启动的 Ray Actor，实现了 Worker 接口 |
| **Worker Lease** | Executor 从 Pool 中"租用"一组 Actor 的过程 |
| **Worker Release** | Executor 将 Actor 归还 Pool 的过程 |
| **weight_transfer** | vLLM 已有的权重热更新机制，支持 NCCL/IPC/sharded_rdt 后端 |

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
│  + initialize_kv_cache(config)      # 初始化 KV Cache                           │
│  + compile_or_warm_up_model()       # 编译/预热模型                              │
│  + run() → worker_busy_loop()       # 启动推理循环                               │
│  + reset()                          # 释放资源，回到 IDLE 状态                    │
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
│  ┌────────────────────┐  ┌────────────────────┐                                 │
│  │  Worker 适配层      │  │  通信层 (复用)      │                                 │
│  │  ────────────────  │  │  ────────────────  │                                 │
│  │                    │  │                    │                                 │
│  │  ExternalWorker    │  │  MessageQueue      │                                 │
│  │  Actor             │  │  - rpc_broadcast   │                                 │
│  │  - init_device()   │  │  - response_mqs    │                                 │
│  │  - load_model()    │  │  - SHM (同节点)     │                                 │
│  │  - run()           │  │  - TCP (跨节点)     │                                 │
│  └────────────────────┘  └────────────────────┘                                 │
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
    ) -> None:
        """预启动 Actor 池"""
    
    def acquire(
        self,
        tp_size: int,
        pp_size: int,
        node_constraint: dict[str, int] | None = None,
    ) -> list[ray.actor.ActorHandle]:
        """获取指定数量的空闲 Actor"""
    
    def release(self, actors: list[ray.actor.ActorHandle]) -> None:
        """释放 Actor 回池"""
    
    def get_idle_count(self) -> int:
        """获取空闲 Actor 数量"""
    
    def get_actor_states(self) -> dict[int, ActorState]:
        """获取所有 Actor 的状态"""
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

---

# 3. 开发视图（Development View）

> 关注代码组织、模块划分、依赖关系。

## 3.1 模块划分

```
vllm/
├── v1/
│   ├── executor/
│   │   ├── abstract.py              # Executor 基类 (不修改)
│   │   ├── uniproc_executor.py      # 单进程执行器 (不修改)
│   │   ├── multiproc_executor.py    # 多进程执行器 (不修改)
│   │   ├── ray_executor.py          # Ray V1 执行器 (不修改)
│   │   ├── ray_executor_v2.py       # Ray V2 执行器 (不修改)
│   │   └── external_executor.py     # ⭐ 新增: ExternalExecutor
│   │
│   ├── engine/
│   │   ├── async_llm.py             # ⭐ 修改: 添加 external_actors 参数
│   │   ├── core.py                  # ⭐ 修改: 传递 external_actors
│   │   ├── core_client.py           # ⭐ 修改: 传递 external_actors
│   │   └── utils.py                 # ⭐ 修改: 传递 external_actors
│   │
│   └── worker/
│       ├── gpu_worker.py            # (不修改)
│       └── gpu_model_runner.py      # (不修改)
│
└── external/                         # ⭐ 新增: 外部模块
    ├── __init__.py
    ├── actor_pool_manager.py        # ⭐ 新增: ActorPoolManager
    └── external_worker_actor.py     # ⭐ 新增: ExternalWorkerActor
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
│  外部依赖:                                                                       │
│  ├── ray (Actor 管理)                                                            │
│  ├── torch.distributed (NCCL/HCCl)                                              │
│  ├── vllm.distributed.weight_transfer (权重热更新)                               │
│  └── vllm.distributed.device_communicators.shm_broadcast (MessageQueue)         │
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

| 文件 | 说明 | 行数估计 |
|------|------|---------|
| `vllm/v1/executor/external_executor.py` | ExternalExecutor 实现 | ~300 |
| `vllm/external/__init__.py` | 外部模块入口 | ~10 |
| `vllm/external/actor_pool_manager.py` | ActorPoolManager 实现 | ~200 |
| `vllm/external/external_worker_actor.py` | ExternalWorkerActor 实现 | ~250 |

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
│  │  节点故障                    该节点所有 Actor 不可用  ActorPoolManager     │   │
│  │                              需要重新分配 Actor    检测到后标记 FAILED     │   │
│  │                                                   从其他节点 acquire      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

# 5. 场景视图（Scenarios / Use Case View）

> 用关键用例串联其他四个视图，展示系统如何工作。

## 5.1 场景 1：预启动 Actor Pool

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  场景: 预启动 Actor Pool                                                         │
│  参与者: 运维人员, ActorPoolManager, Ray Cluster, ExternalWorkerActor            │
│                                                                                  │
│  前置条件: Ray Cluster 已启动，GPU 资源可用                                      │
│                                                                                  │
│  主流程:                                                                         │
│  1. 运维人员调用 ActorPoolManager.pre_start(num_actors=8, devices_per_node=...)  │
│  2. ActorPoolManager 创建 Placement Group                                        │
│  3. ActorPoolManager 创建 8 个 ExternalWorkerActor                               │
│  4. 每个 Actor:                                                                  │
│     a. 绑定 GPU 设备                                                             │
│     b. Import 公共库 (torch, vllm, Worker, ModelRunner)                          │
│     c. 预热 NCCL/HCCl (创建临时 ProcessGroup 并销毁)                             │
│     d. 状态设为 IDLE                                                             │
│  5. ActorPoolManager 等待所有 Actor 就绪                                         │
│  6. ActorPoolManager 构建 node_mapping (node_id → actor_indices)                 │
│                                                                                  │
│  后置条件: 8 个 Actor 处于 IDLE 状态，可被租用                                   │
│                                                                                  │
│  耗时: ~10-20s                                                                   │
│                                                                                  │
│  涉及的视图:                                                                     │
│  - 逻辑视图: ActorPoolManager.pre_start()                                        │
│  - 进程视图: Actor 创建、预初始化                                                 │
│  - 开发视图: external/actor_pool_manager.py, external/external_worker_actor.py   │
│  - 物理视图: Placement Group、GPU 绑定                                           │
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
│  - 开发视图: external/external_executor.py                                       │
│  - 物理视图: MessageQueue (SHM/TCP)、NCCL/HCCl、weight_transfer                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 5.3 场景 3：模型热切换

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  场景: 模型热切换                                                                │
│  参与者: 用户, ExternalExecutor, Actors, weight_transfer Engine                  │
│                                                                                  │
│  前置条件: vLLM 实例正在运行模型 A，需要切换到模型 B                              │
│                                                                                  │
│  主流程:                                                                         │
│  1. 用户调用 executor.switch_model(model_b_config, weight_transfer_init_info)    │
│  2. ExternalExecutor 暂停推理 (停止调度新请求)                                   │
│  3. ExternalExecutor 调用 Actor.reset_model() (释放模型 A 的权重和 KV Cache)     │
│  4. ExternalExecutor 调用 Actor.load_model_via_weight_transfer(model_b_config)   │
│     a. 创建模型 B 的结构 (dummy weights)                                         │
│     b. 初始化 weight_transfer engine                                             │
│     c. 通过 weight_transfer 拉取模型 B 的真实权重                                │
│  5. ExternalExecutor 调用 Actor.initialize_kv_cache() (重新分配 KV Cache)        │
│  6. ExternalExecutor 调用 Actor.compile_or_warm_up_model() (Graph Capture)        │
│  7. ExternalExecutor 恢复推理                                                    │
│                                                                                  │
│  后置条件: vLLM 实例已切换到模型 B，可以处理推理请求                              │
│                                                                                  │
│  耗时: ~10-60s (相比重新创建实例 68-495s 大幅减少)                              │
│                                                                                  │
│  涉及的视图:                                                                     │
│  - 逻辑视图: ExternalExecutor.switch_model()                                     │
│  - 进程视图: 模型卸载、权重传输、KV Cache 重新分配                                │
│  - 开发视图: external/external_executor.py, weight_transfer 模块                 │
│  - 物理视图: weight_transfer 通信 (NCCL/IPC/sharded_rdt)                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 5.4 场景 4：弹性伸缩（TP/PP 变化）

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  场景: 弹性伸缩（TP/PP 变化）                                                    │
│  参与者: 用户, ActorPoolManager, ExternalExecutor, Actors                        │
│                                                                                  │
│  前置条件: vLLM 实例使用 TP=4, PP=2 (8 个 Actor)，需要扩展到 TP=8, PP=2 (16 个) │
│                                                                                  │
│  主流程:                                                                         │
│  1. 用户调用 pool.acquire(tp_size=8, pp_size=2) 获取 16 个 Actor                 │
│     - Pool 返回 8 个已有 Actor + 8 个新 Actor (如果有的话)                        │
│     - 或者 Pool 返回错误 (空闲 Actor 不足)                                       │
│  2. 用户创建新的 AsyncLLM (使用 16 个 Actor)                                     │
│  3. 旧的 vLLM 实例释放 Actor: pool.release(old_actors)                           │
│  4. 新的 vLLM 实例使用 16 个 Actor 初始化                                        │
│     - 重新初始化 NCCL/HCCl (world_size=16)                                       │
│     - 重新加载模型 (TP=8 分片)                                                   │
│     - 重新分配 KV Cache                                                          │
│                                                                                  │
│  后置条件: vLLM 实例已扩展到 TP=8, PP=2                                          │
│                                                                                  │
│  涉及的视图:                                                                     │
│  - 逻辑视图: ActorPoolManager.acquire(), ExternalExecutor._init_executor()       │
│  - 进程视图: NCCL/HCCl 重新初始化、模型重新加载                                   │
│  - 开发视图: external/actor_pool_manager.py, external/external_executor.py       │
│  - 物理视图: 新的 NCCL/HCCl 拓扑 (16 个 Worker)                                  │
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
│  涉及的视图:                                                                     │
│  - 逻辑视图: ExternalExecutor.shutdown(), Actor.reset()                          │
│  - 进程视图: 资源释放、状态重置                                                   │
│  - 开发视图: external/external_executor.py, external/external_worker_actor.py    │
│  - 物理视图: GPU 内存释放、NCCL/HCCl 销毁                                        │
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

---

# 7. 约束与限制

## 7.1 已知约束

| 约束 | 说明 | 影响 |
|------|------|------|
| **Actor 必须预启动** | ExternalExecutor 不创建 Actor，只从 Pool 获取 | 需要提前规划资源 |
| **weight_transfer 必须配置** | 模型加载依赖 weight_transfer 机制 | 需要配置 Trainer 端 |
| **数据通路固定** | 必须使用 MessageQueue（与 RayExecutorV2 一致） | 无法使用其他通信方式 |
| **模型架构变化需重建** | 如果模型架构变化，需要重新创建模型结构 | 跨架构切换较慢 |
| **CUDA Graph 不兼容** | 不同模型的 CUDA Graph 不兼容（涉及运行时状态） | 需要重新执行 Graph Capture |

## 7.2 已知问题

| 问题 | 影响 | 解决方案 |
|------|------|---------|
| **模型架构变化** | 需要重新创建模型结构 | 支持两种模式：同架构热切换 / 跨架构切换 |
| **Graph Capture 优化** | CUDA Graph 不能直接缓存，但可通过配置优化减少 Capture 时间 | 减少 batch size、使用 PIECEWISE 模式、确保 torch.compile 缓存启用 |
| **KV Cache 大小变化** | 不同模型/并行策略可能需要不同大小的 KV Cache | 每次重新分配 KV Cache |
| **分布式环境重建** | TP/PP 大小变化时需要重新初始化 NCCL/HCCl | 允许重新初始化，NPU 侧已预热所以很快 |

---

# 8. 待确认问题

| 编号 | 问题 | 选项 | 建议 |
|------|------|------|------|
| Q1 | weight_transfer 的触发方式？ | A) Trainer 主动推送 B) Worker 主动拉取 | 需要确认使用哪种方式 |
| Q2 | 模型结构是否可以在 Actor 预启动时就创建？ | A) 预启动时创建 (meta device) B) acquire 后创建 | 建议 B，更灵活 |
| Q3 | 如何优化 Graph Capture 时间？ | A) 减少 batch size B) 使用 PIECEWISE 模式 C) 确保 torch.compile 缓存 D) 以上全部 | 建议 D，综合优化 |
| Q4 | weight_transfer 失败时的处理策略？ | A) 回退到磁盘加载 B) 标记 Actor FAILED | 建议 A |
| Q5 | Actor Pool 是否需要持久化？ | A) 需要 (跨 vLLM 实例) B) 不需要 (每次重建) | 建议 A |
| Q6 | 是否需要支持多 vLLM 实例共享同一 Pool？ | A) 需要 B) 不需要 | 建议 A，提高资源利用率 |

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
