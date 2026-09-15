# ExternalExecutor 真机验证清单

> 对应 `PR_DESCRIPTION.md` 的 **Model evaluation** 章节。纯逻辑模块已在离线
> 环境验证（状态机、规划器、调度器、注册表、tiering/prefetch/索引/账本、orchestrator
> 接线），见 `PR_DESCRIPTION.md` 的 **Testing**。本清单覆盖**必须依赖 GPU +
> Ray +（RDMA 项依赖 Mooncake/IB）集群**的验证，逐项执行后在末尾 checklist 打勾。

## 前置环境

1. 按 `STARTUP_DEPENDENCIES.md` 装好依赖（vLLM、Ray、safetensors、本插件）。
2. 至少 **2 节点**，各 ≥1 GPU；RDMA 项（M5）需 Mooncake + InfiniBand/RoCE。
3. 可编辑安装 vLLM，插件可被 `vllm.general_plugins` 入口发现：

   ```bash
   cd vllm
   VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
   cd multi-task-infer
   uv pip install -e .
   ```

## 验证矩阵

| ID | 验证项 | 对应能力 | 硬件 |
|----|--------|---------|------|
| M1 | 基础推理正确性 | 预启动 Actor 池 + ExternalExecutor | 1 节点 1 GPU |
| M2 | 模型热切换 + 原子回滚 | `switch_model` 状态机 | 1 节点 1 GPU |
| M3 | 同布局 KV 迁移 token 一致 | 增量迁移 + 前缀命中 | 1 节点 2 GPU |
| M4 | 异构 remap 无别名 | 块 id 重映射 | 1 节点 2 GPU |
| M5 | RDMA 传输 bit-exact | Mooncake KV 传输 | 2 节点 + IB |
| M6 | CUDA IPC 同节点零拷贝 | `cuda_ipc` 传输 | 1 节点 2 GPU |
| M7 | 节点 kill 恢复 + 故障转移 | `NodeRegistryActor` + failover | 2 节点 |
| M8 | torch.compile 缓存共享 | `CacheManagerActor` | 1 节点 2 GPU |
| M9 | 决策层接线（预取 + tiering + 记账） | `MigrationOrchestrator` | 1 节点 2 GPU |
| M10 | LoRA 多任务共享 base（可选） | `WeightShareLedger` | 1 节点 1 GPU |
| M11 | 弹性扩缩容（队列/P99/利用率） | `Autoscaler` + `maybe_autoscale` | 2 节点 |

## 逐项步骤

### M1 基础推理正确性

**目的**：确认 `ExternalExecutor` 用预启动 Actor 池跑通普通推理，输出与
`RayExecutorV2` 基线逐 token 一致。

```bash
cd multi-task-infer
uv run python examples/basic_usage.py
```

**通过标准**：示例正常完成；同一 prompt、同一 seed 下输出与基线一致。

**失败排查**：检查 actor 是否成功 `acquire`、`get_info.remote()` 是否返回
`node_id`、MessageQueue 是否建立。

---

### M2 模型热切换 + 原子回滚

**目的**：模型 A→B 切换后 B 输出正确；注入一次 worker 切换失败，确认补偿回滚
使实例仍停留在 A（幂等 `migration_id` 重发直接返回缓存结果）。

**通过标准**：
1. 切换成功：B 的输出与「直接加载 B」一致，无 in-flight 请求被污染。
2. 失败回滚：已切换 worker 通过 `switch_model_rollback` 重载旧模型，executor
   `vllm_config` 引用恢复，后续请求仍在 A 上正确输出。

**失败排查**：核对 `MigrationPhase` 轨迹（日志 `Model switch complete`）、
`register_compensation` 逆序执行、scheduler gate 是否 resume。

---

### M3 同布局 KV 迁移 token 一致

**目的**：引擎 A 处理到一半的请求迁移到引擎 B，续跑输出与 A 连续跑一致
（`PR_DESCRIPTION.md` 里的核心 fidelity 属性）。

```bash
cd multi-task-infer
uv run python examples/kv_incremental_migration.py
```

**通过标准**：迁移后 decode 与不迁移连续 decode 逐 token 相同；前缀命中块
（`prefix_hits`）未走 `ship`。

**失败排查**：检查 `IncrementalKVPlanner` 的 `transfer`/`prefix_hits` 划分、
`block_mapping` 是否 1:1、transport 往返 tensor 形状/ dtype。

---

### M4 异构 remap 无别名

**目的**：`total_blocks` 提供的异构池下，源块 id 重映射到空闲目标 id，迁移后
不覆盖已占用块。

```bash
cd multi-task-infer
uv run python examples/kv_incremental_migration.py  # 异构分支
```

**通过标准**：`dst_occupied` 里的 id 从不被指派；迁移后目标块表无别名/错位；
输出与 M3 一致。

---

### M5 RDMA 传输 bit-exact

**目的**：`mooncake_rdma` 跳过 driver 做 peer-to-peer 张量传输，往返 bit-exact。

```bash
cd multi-task-infer
uv run --extra test pytest -q tests/test_storage_checkpoint_engine.py -m mooncake
uv run --extra test pytest -q tests/test_kv_transport.py -m mooncake
```

**通过标准**：put/get 往返张量逐字节相同；记录实测带宽供 reviewer 参考。

**失败排查**：确认 Mooncake 端口/IB 设备、`MooncakeRdmaTransport` 的
`_export`/`_import` 与 worker 的 `get_ipc_handle` 版本兼容。

---

### M6 CUDA IPC 同节点零拷贝

**目的**：`cuda_ipc` 传输在同节点 2 GPU 间零拷贝往返，`torch.from_ipc_handle`
API 兼容。

```bash
cd multi-task-infer
uv run --extra test pytest -q tests/test_kv_transport.py -k cuda_ipc
```

**通过标准**：往返 bit-exact；确认未发生 host 中转（可从显存带宽 / 时间侧证）。

---

### M7 节点 kill 恢复 + 故障转移

**目的**：kill 一个节点后 `NodeRegistryActor.detect_dead_nodes` 触发，
`recover_node` 重建该节点 actor 并重路由，服务不中断。

**通过标准**：
1. 心跳超时后 dead node 被识别。
2. `recover_node` 后该节点 actor 重新 `register`，新请求正常调度。

**失败排查**：核对 heartbeat 间隔、`detect_dead_nodes` 超时参数、actor 重建
时的 `acquire` 设备绑定。

---

### M8 torch.compile 缓存共享

**目的**：多引擎共享 compile cache，首个引擎编译后，同权重后续引擎命中缓存
不再重编译。

**通过标准**：第二引擎启动日志无 re-compile 记录；小缓存走 Ray Object Store、
大缓存走 NFS+压缩的路径与设计一致。

---

### M9 决策层接线（预取 + tiering + 记账）

**目的**：验证 orchestrator 在两条迁移路径上真正消费了决策层，而非空转。

**通过标准**：
1. `switch_model`：`phase_handlers` 按 GRACEFUL_PAUSE/CHECKPOINT/UNLOAD/RESTORE
   顺序触发；tiering 脚本产出 CHECKPOINT→REMOTE、UNLOAD→DRAM、LOAD→HBM 的
   `TierMove`；`WeightShareLedger` 注册 base+adapter 组合。
2. `migrate_kv_cache_to(prefetch=True)`：`_start_prefetch` 收到热的、目标未驻留
   前缀（PREPARING），`_await_prefetch` 在 LOAD 前被调；转移前缀以目标 actor 注册
   到 `GlobalPrefixIndex`。
3. 任一决策层动作失败时，补偿逆序清除 tier/prefix/ledger 记账（回滚一致性）。

> 注：`_start_prefetch`/`_await_prefetch` 默认为 no-op+log；本项验证需先在
> `ExternalExecutor` 子类覆盖二者为真实后台线程/ray task，或注入带记录的回调。

---

### M10 LoRA 多任务共享 base（可选）

**目的**：多个 adapter 共享同一 base 权重，`WeightShareLedger` 正确记录
holder 集合与 cost ratio，切换 adapter 不重载 base。

**通过标准**：`ledger.holders(base, adapter)` 准确；`cost_ratio` 反映 base
共享带来的权重加载节省；多 adapter 间切换仅重载 adapter 张量。

---

### M11 弹性扩缩容（队列/P99/利用率）

**目的**：`ActorPoolManager.maybe_autoscale` 按负载快照动态创建/销毁 Actor，
决策层（`Autoscaler`）输出正确的 target + reason，执行侧真正改变池大小。

**通过标准**：
1. 注入高队列/P99/利用率 → `maybe_autoscale` 返回 `SCALE_UP`，池新增
   `scale_step` 个 IDLE Actor，且新 Actor 注册进 registry、可被 `acquire`。
2. 三项负载都低 → `SCALE_DOWN`，尾部 IDLE Actor 被 `ray.kill` + 注销，池
   index/`node_mapping` 保持一致（无飘移）。
3. 冷却期内重复采样返回 `NONE`（无抖动）；时段窗口抬高 floor 或锚定
   target 时按预期收敛。
4. 默认 `_scale_up_actors`（`num_gpus=1` 普通调度，不绑 PG）能创建新 Actor；
   需 placement-group 亲和/排他时，注入 `scale_up_fn` 并按下方「操作手册」
   扩展 bundle。

**失败排查**：核对 `AutoscalingConfig` 水位 OR/AND 语义、`time_windows`
跨午夜判定、`_scale_down_actors` 从尾部移除是否破坏 `_actor_id_to_idx`。

**操作手册：placement-group 扩展（需 PG 亲和时）**

默认 `_scale_up_actors` 用 `num_gpus=1` 普通调度（无 placement group），新增
Actor 不占用 `pre_start` 按 `num_actors` 创建的 bundle，因此**不受 bundle
数量上限**；代价是新 Actor 由 Ray 任意放置——可能与既有 Actor 不在同一
故障域，或破坏 NCCL 拓扑亲和。

需要强亲和时，二选一：

1. **Ray ≥ 2.24 动态扩容**（推荐，不打断既有 Actor）：
   ```python
   import ray
   pool.placement_group.add_bundles([{"GPU": 1} for _ in range(n)])
   ray.get(pool.placement_group.ready(), timeout=30.0)
   # 在 scale_up_fn 里用 PlacementGroupSchedulingStrategy 绑定新增 bundle
   ```
2. **预留 headroom**：`pre_start` 时按 `max_actors` 建 bundle、初始只起
   `min_actors` 个 Actor，扩容时直接复用空闲 bundle（无需 `add_bundles`）。

Ray < 2.24 无 `add_bundles`，重建 PG 会打断既有 Actor，因此**推荐方案 2
（预留 headroom）或注入自定义 `scale_up_fn` 走普通调度**。

---

## 汇总 checklist

- [ ] M1 基础推理正确性
- [ ] M2 模型热切换 + 原子回滚
- [ ] M3 同布局 KV 迁移 token 一致
- [ ] M4 异构 remap 无别名
- [ ] M5 RDMA 传输 bit-exact
- [ ] M6 CUDA IPC 同节点零拷贝
- [ ] M7 节点 kill 恢复 + 故障转移
- [ ] M8 torch.compile 缓存共享
- [ ] M9 决策层接线（预取 + tiering + 记账）
- [ ] M10 LoRA 多任务共享 base（可选）
- [ ] M11 弹性扩缩容（队列/P99/利用率）

> 提交 PR 时，将勾选结果 + M5 实测带宽 + M3/M4 的 token 一致性截图贴进 PR
> 的 **Model evaluation** 章节；未通过的项需在 PR 里说明阻塞原因。
