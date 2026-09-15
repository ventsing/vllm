# P2 后续事项（MVP 之后的可选能力清单）

> 这些能力在 MVP 阶段被**显式门控/禁用**，实现代码保留在当前仓库（未删除），
> 以便后续按需启用。本文档记录逐项规划、当前门控位置，以及「启用前必须先修」
> 的已知 bug。MVP 收敛范围与逐项现状见 `../MVP_TODO.md`。

## 一、总览

| 能力 | 对应目标 | 当前状态 | 门控位置（`vllm_external_executor/`） |
|------|---------|---------|--------------------------------------|
| 跨节点编译缓存 | G6 | 禁用（`cache_manager` 非 None 报错） | `external_executor.py` `__init__` |
| 完整模型权重预取 + 共享存储加载 | G7 | 禁用（决策层句柄非 None 报错） | `external_executor.py` `__init__` |
| 动态 TP/PP 重组 | G2 | 禁用（elastic EP 报错） | `external_executor.py` `_validate_mvp_scope` |
| 原地热切换 + 完整失败恢复 | G3 | 禁用（`switch_model` 抛 NotImplementedError） | `external_executor.py` `switch_model` |
| 同权重 KV 迁移一致性 / 映射 / 事务 | — | 禁用（`migrate_kv_cache_to` 未在 MVP 入口暴露） | `external_executor.py`（迁移路径） |
| 自动故障恢复 + 动态扩缩容 | — | 默认禁用（不调 `set_autoscaler`/`maybe_autoscale`） | `actor_pool_manager.py` |
| LoRA | — | 禁用（`lora_config.max_loras > 0` 报错） | `external_executor.py` `_validate_mvp_scope` |
| 分层存储（storage tiering） | — | 禁用（`tiered_cache` 非 None 报错） | `external_executor.py` `__init__` |

## 二、逐项说明

### 2.1 跨节点编译缓存（G6）

- 目标：`CacheManagerActor` 跨节点共享 torch.compile 缓存，避免重复编译；
  小缓存走 Ray Object Store，大缓存走 NFS + gzip 压缩。
- 启用前：对齐 vLLM **实际的**编译缓存 key、目录结构与 worker 分发方式
  （当前 `cache_manager_actor.py` 的 key/目录假设需与目标 vLLM 版本核对）。
- 入口：在 `ExternalExecutor` 构造时传入 `cache_manager`，并恢复
  `_handle_compilation_optimization` 的 lazy-loading 流程（该方法现已被 MVP
  改为「不调用、交给 EngineCore 标准编译」，仅保留实现）。

### 2.2 完整模型权重预取 + 共享存储加载（G7）

- 目标：`StorageCheckpointEngine` 从 NFS/Mooncake Store 加载权重，支持在
  任务切换前预取下一模型权重。
- 启用前：确认与 verl `CheckpointEngineWithCache` 接口兼容性，补
  `MooncakeStoreBackend`（RDMA）端到端验证。
- 入口：`ExternalExecutor.__init__` 的 `prefix_index`/`weight_ledger`/
  `heat_tracker`/`tiered_cache`/`prefetch_policy` 等决策层句柄。

### 2.3 动态 TP/PP 重组（G2）

- 目标：同一批 Actor 重新划分并行组，支持不同 TP/PP 形状。
- 现状：MVP 固定「一个租约 = tp_size × pp_size 个 Actor、rank 按租约顺序」。
- 启用前：明确「租约内 rank 分配 vs Actor 物理绑定」的映射，支持
  `acquire(tp, pp)` 返回可重组的 rank 布局。

### 2.4 原地热切换 + 完整失败恢复（G3）

- 目标：`switch_model` 在不释放 Actor 的前提下原子切换模型，含
  `PREPARING → GRACEFUL_PAUSE → CHECKPOINT → UNLOAD → LOAD → RESTORE →
  COMPLETED` 状态机与补偿式回滚。
- **启用前必修**：`external_worker_actor.py` 的 `switch_model` 重建
  `WorkerWrapperBase` 时仍用 `rpc_rank=self._local_rank` + 单元素 `all_kwargs`
  （与 2.4 同源 bug），需先改为 `rpc_rank=全局 rank` + 完整 `all_kwargs` 再撤
  `NotImplementedError` 门控。

### 2.5 同权重 KV 迁移一致性 / 映射 / 事务

- 目标：`migrate_kv_cache_to` 增量迁移同权重任务的 KV cache（prefix-cache
  感知、异构 block-id 重映射），恢复在途请求，支持可插拔传输
  （ray_object_store / mooncake_rdma / cuda_ipc）。
- 启用前：验证 block-id 重映射与事务一致性，补 `mooncake_rdma`/`cuda_ipc`
  传输的真机测试。

### 2.6 自动故障恢复 + 动态扩缩容

- 目标：`Autoscaler` 依据队列长度 / P99 延迟 / 资源利用率决定扩缩容；
  节点/actor 故障后自动重建并归还可用资源。
- **启用前必修**：修复 `_scale_down_actors` 删除列表中间元素造成的索引错位
  （当前用尾部删除规避），并复核扩容时的设备映射（新 actor 的
  `device_id` 与 `devices_per_node` 对齐）。
- 现状：`set_autoscaler`/`maybe_autoscale` 已实现但 MVP 不调用，默认禁用。

### 2.7 LoRA

- 目标：支持 base + adapter 的轻量切换（`WeightShareLedger` 已实现纯逻辑
  记账与 cost 估算）。
- 启用前：打通 `lora_config` 与 worker 的 LoRA 加载路径，验证
  `switch_model` 的 adapter-only 切换能复用 base 权重。

### 2.8 分层存储（storage tiering）

- 目标：HBM / DRAM / REMOTE 三层存储的换入换出决策（`StorageTier`/
  `TieredCache` 已实现纯逻辑）。
- 启用前：接入实际的 KV 块申请/回收路径，验证 tiering 决策与迁移状态机
  （`migration_orchestrator`）的联动。

## 三、启用流程（建议）

1. 修该能力的「启用前必修」bug。
2. 撤销对应门控（`external_executor.py` 的 `NotImplementedError`/
   `_validate_mvp_scope` 报错 / `cache_manager` 与决策层句柄拒绝）。
3. 补对应纯逻辑单测 + 真实 Ray 集成测试。
4. 在 `MVP_TODO.md` §六建立该能力的真机验证清单并逐项验证。