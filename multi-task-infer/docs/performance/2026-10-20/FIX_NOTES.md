# 多任务 vLLM 资源池：修复与验证说明

测试环境：2026-10-20，现有 Ray 集群，16 张 Ascend910，Qwen3.5-9B，8 个 TP=2 实例。

## 1. 修复 Worker 退出时 Ascend 通信域残留

问题：池化 Worker 退出推理实例后进程继续存活。通用分布式清理会注销进程组，但 Ascend 后端仍可能通过 `_MC2` 等全局 coordinator 保留旧通信对象。只检查 `torch.distributed.is_initialized()` 或已注册进程组数量，无法证明后端引用已释放。

修改：在 NPU Worker 的 `_release_worker_resources()` 中，先调用已加载 Ascend 后端的 `destroy_ascend_model_parallel()`，随后调用通用 `cleanup_dist_env_and_memory(shutdown_ray=False)`。通过 `sys.modules` 查找后端，避免清理部分初始化失败的 Worker 时反而导入后端。

结果：释放后端通信对象后再执行通用清理，保持 Ray Actor 和设备绑定存活。该修复已原地安装到运行中的16个Actor，未重启Actor。

涉及代码：`multi-task-infer/vllm_external_executor/external_worker_actor.py`。扩展现有 reset 回归测试，验证后端清理先于通用清理，以及重复 reset 的行为。

边界：既有的模型/AOT产物清理、ACL图清理、TCPStore和消息队列清理沿用原实现；本次对该生产文件新增的核心逻辑是 Ascend 通信域清理顺序，不能将原有逻辑全部计作本次新增修复。

## 2. 新增 EngineCore 进程预热池

问题：只复用 Worker Actor 时，前端仍需新建 EngineCore、导入模块与插件、连接 Ray，启动路径仍约33秒。

修改：新增 `vllm/v1/engine/process_pool.py`，提供 `EngineCoreProcessPool`：

- 提前 spawn CPU 进程，导入 EngineCore、插件及指定模块，并连接同一 Ray 集群。
- 完成预热后才放入 ready 队列；任务到来时通过管道发送配置和启动参数。
- 一个预热进程只承载一次 EngineCore 生命周期，退出后不复用旧 engine 状态；取出后后台补充新的预热进程。
- 通过上下文启用，在 `CoreEngineProcManager` 中选择预热进程；未启用时仍走原启动路径。
- 代理原进程的 PID、sentinel 和生命周期接口，保留原有监控与关闭流程。
- 检查关键环境和 Ray 连接一致性；失败时处理管道、预热进程和超时。

涉及代码：`vllm/v1/engine/process_pool.py`、`vllm/v1/engine/utils.py`，测试扩展到 `tests/v1/engine/test_startup_watch_processes.py`。

使用范围：driver内本地池，spawn、external Worker Actors、DP=1、单本地Core，不支持NUMA绑定和multiprocessing tensor queue。池关闭仅管理它持有的CPU预热进程，不关闭Worker Actor池；已分配Core按引擎原生命周期关闭。

边界：当前池只提前准备CPU进程，不提前加载权重、建立本轮通信域、分配KV或捕获ACL图。每轮这些工作仍然执行。

## 3. BF16 图模式适配与编译缓存规避

测试中显式设置权重、Mamba缓存和SSM状态为BF16。该模型配置默认SSM状态为FP32，而当前安装的GDN算子需要BF16状态；这是当前后端的配置适配，不是修改模型算子实现。

Ascend图模式使用 `VLLM_COMPILE + FULL_DECODE_ONLY`，捕获batch 1/2/4。仅开启图而关闭编译时，后端workspace未初始化；改为后端支持的组合后通过。prefill不走ACL图，吞吐收益包含编译与decode图两部分，未独立隔离。

另外发现：持久化AOT编译缓存内固化了旧通信组名 `tp:11`，Actor重新建组后加载旧缓存会报 `Group tp:11 is destroyed`。

本次处理：测试脚本在空闲Actor中设置 `VLLM_DISABLE_COMPILE_CACHE=1` 并刷新环境读取缓存，禁用vLLM持久化编译缓存加载/保存，测试完成后恢复原设置。编译、图捕获及回放仍正常执行。

**这属于基准和复用测试的规避措施，尚未修复后端AOT缓存键、通信组重绑定或所有生产入口的跨engine缓存复用。**

## 4. 新增调试与性能验证工具

| 文件（位于 multi-task-infer/examples） | 用途 |
| --- | --- |
| `rolling_pool_test.py` | 构建或接入16个Actor；8个TP=2实例驻留，逐个替换实例，检查原Actor对和PID复用；替换时其余实例继续推理；可启用Core预热池 |
| `pool_diagnostics.py` | 检查进程组、vLLM/Ascend coordinator、TCPStore、执行循环、worker、消息队列、ACL图条目及全局图参数；退出后断言释放 |
| `profile_pool_test.py` | BF16 eager/图模式性能矩阵、启动耗时、独立profiler trace、输出token和清理快照；使用保留Actor池 |
| `compare_pool_startup.py` | 直接mp冷启动与池化启动对比，支持图模式与禁用编译缓存；记录引擎ready、driver到ready、首token和首请求 |
| `watch_pool_cleanup.py` | 调试阶段的外部观察脚本，含本次运行目录约定；不是通用生产监控组件 |

Actor保留策略由测试用RetainedPool保证：不执行Actor kill/rebuild，推理实例和EngineCore允许正常退出重建。报告未将测试脚本中的保留策略表述为全局改变ActorPoolManager的生产shutdown语义。

## 5. 验证结果

- 16卡、8×TP=2的实例驻留和8次滚动替换通过；72组、144条推理请求通过结果检查。接入Core预热池后同样完成滚动验证。
- 所有检查点Worker Actor ID/PID保持不变；最终16个Actor均为idle。
- 退出检查中，已注册进程组、活跃coordinator、TCPStore、执行循环和worker/消息队列引用均清空；图模式还检查ACL图条目和全局图参数释放。
- BF16 eager/graph各240条正式请求；固定提示的贪心输出token逐请求比较一致。此结果不代替模型精度评测。
- 实际图捕获与回放均有日志证据；每卡图捕获内存约274–300MiB。
- 相关联合回归测试57项通过；图模式测试阶段重跑Worker/Core两个测试文件，29项通过。两次结果不是86个独立测试。Ruff检查、相关格式检查和diff空白检查通过。

## 6. 性能结果与口径

| 启动方式 | 到ready耗时 | 说明 |
| --- | ---: | --- |
| 直接冷启动，eager | 60.32秒 | 此前3次均值，从创建引擎计时 |
| 仅Worker Actor池，eager | 33.32秒 | 此前3次均值，Core仍冷启动 |
| Worker+Core预热池，eager | 15.06秒 | 本轮8实例均值 |
| Worker+Core预热池，编译+decode图 | 16.53秒 | 本轮8实例均值，禁用vLLM持久化编译缓存 |
| 直接冷启动，编译+decode图 | 76.99秒 | 单个TP=2实例、单次补测，禁用编译缓存 |
| 同上，包含driver导入 | 92.03秒 | 与上行起点不同 |

以上均未清操作系统文件页缓存。16.53秒的前提是池已就绪；8个CPU Core本轮预热另耗18.74秒，Worker Actor池此前已建立。不能将首次整套部署耗时表述为16.53秒。

图模式输出吞吐相对eager提升约3.40–4.23倍，负载为128/512输入token、固定64输出token、总并发8/32，各3批；不是所有负载或最大吞吐的承诺。

详见[性能与启动拆解报告](bf16-graph-report.md)、[初版综合报告](pool-profile-report.md)、[逐实例启动拆解](bf16-startup-breakdown.json)。

## 7. 未解决项

- AOT磁盘缓存包含旧通信组的问题，目前通过测试入口禁用缓存规避，尚未做通用修复。
- 正常关闭时仍可能打印AsyncLLM output_handler的EngineDeadError；本次请求已完成且释放断言通过，但关闭日志竞态尚未修复。
- 16.53秒中前端/激活3.73秒、Worker/模型4.39秒、profile/KV/warmup6.61秒、ready收尾1.80秒。除显式计时阶段外，其余按秒级日志推算，仍需细粒度计时定位，尚未实施完整engine/KV/图常驻池化。
- 本次未评估多模态、长上下文、模型精度及不同硬件/后端组合。

当前说明仅整理本次代码与实测，不包含PR提交、发布或上游合并。
