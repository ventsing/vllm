# 多任务池化推理 MVP TODO（范围 + 现状盘点）

> 收敛基准：把当前「大而全」的 ExternalExecutor 插件收窄为**单节点、固定
> TP、不同完整模型顺序复用同一批 Actor**的可运行 MVP。本文档记录范围、
> 逐项现状盘点（哪些纯逻辑可离线、哪些需真机）与执行顺序。

## 一、范围

- 单节点；第一阶段 TP=1，第二阶段同节点 TP=2；PP 固定 1。
- 一个 Actor 同时只属于一个任务；任务完成后归还 Actor，再启动下一任务。
- 权重用本地模型目录，编译用 vLLM 原生缓存。
- **暂不启用**：动态扩缩容、在途请求迁移、跨节点 KV 共享、分层存储、LoRA。
- 对暂不支持的配置明确报错，避免进入未验证路径。

## 二、P0 现状盘点

| # | 项 | 现状（已核实代码位置） | 可离线 | 优先级 |
|---|----|----------------------|:---:|:---:|
| 2.1 | 设备绑定：统一 Ray 物理编号 vs 进程内逻辑编号 | `external_worker_actor.py:64,70` 直接把 `device_id` 当 `cuda:{id}`；`line 251` 有映射函数但未用于设备绑定 | 部分 | P0 |
| 2.2 | `_group_workers_by_node()` 不存在 | `external_executor.py:260` 调用，全仓无定义 → AttributeError | 是 | P0 |
| 2.3 | `vllm_config=None` 初始化顺序 | `external_worker_actor.py:81` 初始 None；`:335` `create_dist_init_method` 读 `.parallel_config` 会 NoneType | 是 | P0 |
| 2.4 | `rpc_rank` 与 `all_kwargs` 长度不匹配 | `:293-300` `init_worker(all_kwargs=[{1 个 dict}])`，TP>1 需 world_size 个 | 是 | P0 |
| 2.5 | READY 仅在真实初始化成功后发布 | `wait_for_ready` 在 __init__ 前就绪，未绑定 worker 初始化 | 真机 | P0 |
| 2.6 | 执行协议：`ResponseStatus` | `:927-933` 用 `(True,result)/(False,str)` 布尔元组，非 vLLM `ResponseStatus` | 是 | P0 |
| 2.7 | 执行协议：字符串/callable RPC 对齐 | `:924` `getattr(worker, method)` 假定 method 是字符串方法名 | 真机 | P0 |
| 3.1 | 租约原子化（select + grant 合并） | `actor_pool_manager.py:351-376` select 与 set_actor_state 分两步，并发可重复租用 | 是 | P0 |
| 3.2 | 心跳不覆盖租约状态 | `node_registry_actor.py:121-130` `heartbeat(state)` 直接 `reg.state=state` | 是 | P0 |
| 3.3 | lease 代次传递 + 过期释放隔离 | 当前 release 无 lease 校验，`set_actor_state` 无代次 | 是 | P0 |
| 4.1 | `run()` 永久循环阻塞心跳/stop/reset | `external_worker_actor.py:906` `while True: dequeue(indefinite=True)` | 真机 | P0 |
| 4.2 | reset 吞错误后标记 IDLE | `:942-953` `except: pass` 后仍 `state=IDLE` | 是 | P0 |
| 4.3 | shutdown 不 ray.kill / 幂等 | `actor_pool_manager.py:1021` 与 executor 侧待查 | 真机 | P0 |
| 5.1 | 唯一可运行 MVP 入口 | 需打通 AsyncLLM 路径 + 修示例参数 | 真机 | P0 |

**结论**：P0 里「可离线」的项集中在**租约/隔离/状态机**（3.1–3.3、2.2、2.3、
4.2、2.6 的状态判定部分）与**设备编号映射逻辑**；「真机」项集中在真实 Worker
初始化（2.4、2.5、2.7）、可停止执行循环（4.1）、进程/通信组生命周期（4.3）。

### 二.1 补充启动修复（静态审查对照 vLLM 参考实现新增，均已修复）

| # | 项 | 问题 | 修复 |
|---|----|------|------|
| 2.8 | worker 存活监控 | `run()` 立即返回，继承的 `start_worker_monitor` 以 `run_ref` 完成为「死亡」信号，启动即误判 shutdown | 覆盖为 `heartbeat` 轮询（`external_executor.py`） |
| 2.9 | 编译时序 | `_init_executor` 提前 `compile_or_warm_up_model`，在 EngineCore 分配 KV cache 前 capture CUDA Graph，重复且有害 | 删除提前调用，交给 EngineCore `_initialize_kv_caches` 标准流程 |

## 三、执行顺序

1. **P0 纯逻辑先行**（本仓库可离线修 + 单测）：3.1 原子租约、3.2 心跳不覆盖、
   3.3 lease 代次、4.2 reset 失败隔离、2.6 ResponseStatus 对齐。
2. **P0 代码修复**（真机集成 bug，py_compile 验证 + 留真机 checklist）：
   2.2 `_group_workers_by_node`、2.3 初始化顺序、2.4 `all_kwargs`、2.1 设备映射、
   4.1 可停止循环、4.3 资源所有权。
3. **P0 唯一入口**（5.1）：AsyncLLM 路径 + 示例修正 + try/finally。
4. **P1 测试 + 量化**（六、七）：纯逻辑单测先行，生命周期用真实 Ray 集成测试
   （真机），量化收益记录脚本。
5. **P2 后续**：跨节点缓存、权重预取、动态 TP/PP、热切换、KV 迁移一致性、
   扩缩容（修复缩容索引错位 + 扩容设备映射）、LoRA、分层存储。

## 四、进度 checklist

- [x] 2.1 设备编号映射统一（移除 `cuda:{device_id}` 直接绑定；绑定改由
      `init_device()` 经 `assigned_physical_gpu_ids`+`local_rank` 完成）
- [x] 2.2 修复 `_group_workers_by_node()` 调用（删除死调用，分组 Step 4 内联）
- [x] 2.3 初始化顺序（`create_dist_init_method(world_size)` 不读 self.vllm_config）
- [x] 2.4 `rpc_rank`/`all_kwargs` 对齐（完整 per-rank `all_kwargs`，`rpc_rank=rank`）
- [x] 2.5 READY 语义（`wait_for_init` 仅 worker + response MQ 就绪才 READY）
- [x] 2.6 `ResponseStatus` 对齐（响应用 `WorkerProc.ResponseStatus.SUCCESS/FAILURE`）
- [x] 2.7 字符串/callable RPC（`_execute_worker_rpc` 支持 str + bytes/cloudpickle）
- [x] 3.1 原子租约（`try_acquire` select+grant 合并，短缺无部分租用）
- [x] 3.2 心跳不覆盖租约（`heartbeat` 仅 liveness，不动 state/lease）
- [x] 3.3 lease 代次 + 过期释放隔离（`lease_generation` + `release_actors` 校验 lease_id）
- [x] 4.1 可停止执行循环（run 起后台 daemon 线程，`_stop_event`+dequeue timeout）
- [x] 4.2 reset 失败隔离（reset 抛异常 + 标 FAILED；release 侧 reset 成功才归还）
- [x] 4.3 shutdown 不 kill / 幂等（reset actor + 关 MQ，不 `ray.kill` 池资产）
- [x] 5.1 唯一 MVP 入口（`mvp_entry.run_mvp`：acquire → AsyncLLM → finally
      release；`validate_mvp_config` 显式报错；`examples/mvp_example.py`）
- [x] P1 纯逻辑测试齐全（13 个测试文件全部离线 PASS：租约/约束/决策层/迁移/
      调度/传输/存储；本轮补齐 weight_sharing + prefetch_policy）
- [ ] P1 真机集成测试 + 量化数字（第六/七节，待 GPU 运行 benchmark 脚本）
- [x] P2 显式报错/延后（executor 层已落地，见第八节；实现代码保留不动）

## 六、真机验证清单（GPU 环境执行；纯逻辑/代码修复已离线验证）

以下项已按 vLLM `RayExecutorV2`/`WorkerProc` 参考实现完成代码修复，并通过
`py_compile` + 离线逻辑校验，但**尚未在真实 GPU 上跑通**，需按顺序验证：

1. `examples/mvp_example.py`（TP=1，两个模型顺序复用同一批 Actor）能端到端
   出字，无 `AttributeError`/`ResponseStatus` 误判/`collective_rpc` 超时。
2. TP=2（`tp_size=2`）启动：`all_kwargs` 长度、`rpc_rank=rank`、设备映射
   （`assigned_physical_gpu_ids`）在多 worker 下正确。
3. 任务完成后 `pool.release` 归还租约，registry `free_gpus` 恢复，下一个
   任务能再次 `acquire` 同一批 Actor；中断（Ctrl-C / 异常）后 `finally` 归还。
4. 心跳线程不把租用中的 Actor 覆盖回 idle；`reset` 失败时 Actor 被隔离
   （标记 FAILED + `mark_actor_failed`），不被再次分配。
5. 启动收益量化：预热池 vs 冷启动 vLLM 的端到端时延对比（P1 第七节）。

## 七、P1 量化脚本（已交付 `examples/benchmark_startup.py`，待真机运行）

- 基线：`AsyncLLM.from_vllm_config` 冷启动（无池）→ 初始化 + 首次生成时延。
- 池化：`ActorPoolManager.pre_start`（摊薄）+ `run_mvp` → 每模型运行时延。
- 输出：init / prestart / run / first-gen / 端到端墙钟 + `speedup_e2e` 与
  `speedup_run` 两个实测加速比；JSON 可选。未运行的加速一律标「待验证」，
  不做未经验证的倍数声明。

## 五、遗留注记（P2 热切换/展示一致性，非 MVP 路径）

- `switch_model`（`external_worker_actor.py` 约 515-530 行）重建 WorkerWrapperBase
  时仍用 `rpc_rank=self._local_rank` + 单元素 `all_kwargs`，与 2.4 是同一 bug。
  该路径属于 P2「热切换」，MVP 用顺序复用不经过它；现已用 `NotImplementedError`
  门控（见第八节），P2 启用时必须先修该 bug 再撤门控。
- `get_info`/`wait_for_ready` 返回的 `physical_gpu_ids` 仍是 `[self.device_id]`
  （Ray control id），与 `get_node_and_physical_gpu_ids`（经
  `device_control_id_to_physical_device_id` 得到的 physical id）来源不一致。
  当前仅 registry 稳定身份使用 `device_id`，设备绑定不消费该字段，故不影响
  MVP；真机验证时若需要权威 physical id，统一到 `get_accelerator_ids`。

## 八、P2 显式报错/延后（已落地）

P2 未验证路径在 MVP 阶段被显式挡在入口之外；实现代码保留不动，P2 启用时
先修对应 bug 再撤门控：

| 能力 | 门控位置 | 行为 |
|------|---------|------|
| 热切换 `switch_model` | `external_executor.py` `switch_model` 入口 | `NotImplementedError` |
| 编译缓存共享 | `ExternalExecutor.__init__`（`cache_manager` 非 None） | `NotImplementedError` |
| 决策层共享句柄 | `__init__`（`prefix_index`/`weight_ledger`/`heat_tracker`/`tiered_cache`/`prefetch_policy` 非 None） | `NotImplementedError` |
| TP/PP/LoRA/KV-connector/elastic-EP | `ExternalExecutor._validate_mvp_scope`（复用 `validate_mvp_config`） | `ValueError` |
| 动态扩缩容 | 不调用 `set_autoscaler`/`maybe_autoscale`（默认禁用） | 无副作用 |

- P2 启用扩缩容前：修复 `_scale_down_actors` 缩容删除列表中间元素造成的索引
  错位（当前已用尾部删除规避），并复核扩容设备映射。
- P2 启用热切换前：修复 `switch_model` 的 `rpc_rank`/`all_kwargs`（见第五节）。
