# Qwen3.5-9B：16 卡池、8×TP=2 滚动复用与性能报告

测试日期：2026-10-20（UTC）。使用现有 Ray 集群 172.16.12.66:6925。

滚动测试状态：**passed**；修复后性能测试状态：**passed**。

## 环境与范围

16 张 Ascend910 NPU，每卡 64 GiB；8 个独立推理实例，每实例 TP=2、PP=1。
模型目录：`/opt/huawei/dataset/ccb_data/code/ventsing/downloaded_models/Qwen3.5-9B`。
torch 2.10.0+cpu、torch-npu 2.10.0.post4、Ray 2.48.0；CANN 9.1.0。
安装包标识 vLLM 0.28.0+empty；通过 `VLLM_VERSION=0.28.1` 选择当前源码对应的 Ascend 接口分支。
使用 BF16 权重、BF16 GDN 状态缓存、eager、纯文本、max_model_len=1024、
max_num_seqs=4、max_num_batched_tokens=2048、gpu_memory_utilization=0.65，关闭 prefix cache。
本报告主体为 eager 短请求实测；新增的 [BF16 eager/图模式对比报告](bf16-graph-report.md) 包含 8×TP=2 图模式性能、启动成本与缓存限制。尚未测量多模态、长上下文、最大吞吐或模型精度。

## 滚动复用与隔离验证

8 个实例同时驻留并完成并发生成，随后依次替换全部 8 个实例；
每次替换间隙，另外 7 个实例继续完成推理。共 72 组、144 条请求通过结果检查。
每次重建复用原来的 Actor 对，所有检查点的 16 个 Actor ID 和 PID 均保持不变，租约无重叠。

| 阶段 | 样本数 | 平均 ready 时间(s) | 最小(s) | 最大(s) |
| --- | ---: | ---: | ---: | ---: |
| 池内首次创建引擎 | 8 | 39.312 | 38.069 | 41.357 |
| 池内滚动重建 | 8 | 31.878 | 30.820 | 34.181 |
| 修复后性能测试建引擎 | 8 | 31.172 | 30.082 | 33.785 |

以上阶段均使用已预启动 Actor，不代表直接冷启动，也不包含池预启动成本。

## 无 profiler 的稳定态性能

先使用 32 请求预热，每请求 128 输入/64 输出 token。正式测试用固定长度 token ID 输入，
每组重复 3 个批次，temperature=0、ignore_eos=True、固定输出 64 token；通过 AsyncLLM 流式接口在客户端计时。
TTFT 为调用 generate 到首个非空输出的时间，包含客户端与调度开销；
TPOT=(末次输出时间−首次输出时间)/(输出 token 数−1)，并非逐个设备 token 的 kernel 时延。
吞吐=所有批次实际输出 token 数/批次墙钟时间之和；不含启动、预热、关闭及 trace 采集。

| 输入token | 每实例并发/总并发 | 请求数 | 输出token/s(16卡合计) | 请求/s | TTFT p50/p95(ms) | E2E p50/p95(ms) | TPOT p50/p95(ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1/8 | 24 | 129.13 | 2.018 | 76.10/81.88 | 3722.12/3967.10 | 57.86/61.71 |
| 128 | 4/32 | 96 | 471.47 | 7.367 | 216.56/234.48 | 4012.29/4346.85 | 60.67/65.38 |
| 512 | 1/8 | 24 | 128.59 | 2.009 | 77.66/80.66 | 3745.87/3979.41 | 58.25/61.91 |
| 512 | 4/32 | 96 | 399.04 | 6.235 | 230.57/315.23 | 4049.69/5392.67 | 60.92/81.97 |

512 输入、总并发 32 的三个批次存在明显波动：

- measure-512-4-0：4.380s，467.56 output token/s。
- measure-512-4-1：5.394s，379.66 output token/s。
- measure-512-4-2：5.622s，364.25 output token/s。

| 实例 | 此负载的平均请求耗时(ms) | 最慢请求(ms) |
| ---: | ---: | ---: |
| 0 | 4226.69 | 4355.57 |
| 1 | 3969.17 | 4005.05 |
| 2 | 4125.68 | 4379.49 |
| 3 | 3908.87 | 3946.27 |
| 4 | 3947.22 | 4003.78 |
| 5 | 4044.87 | 4096.36 |
| 6 | 4993.68 | 5620.92 |
| 7 | 4103.16 | 4273.63 |

slot 6 的尾延迟拉长了批次完成时间；当前数据不能判定其来源是主机调度、设备差异还是通信等待。
没有剔除慢批次；聚合吞吐包含全部正式样本。以下 trace 只采 slot 0，不能直接解释 slot 6 的变慢。

## NPU profile

在全部 8 个实例生成时，采集 slot 0 的两个 TP rank（128 输入、16 输出 token、总并发 8）；
CPU+NPU、ProfilerLevel.Level1、PipeUtilization，不记录调用栈或内存。trace 单独采集，其耗时不进入上表。
下表来自各 rank 的 op_statistic.csv。百分比是采集区间算子累计时长占比，可能存在重叠，不能解释为端到端延迟占比。

| rank | 算子类型 | 次数 | 累计时间(ms) | 占算子累计时长(%) |
| --- | --- | ---: | ---: | ---: |
| rank0 | allreduceAicpuKernel | 1040 | 150.700 | 47.846 |
| rank0 | MatMulV2 | 2448 | 105.528 | 33.504 |
| rank0 | Transpose | 384 | 6.742 | 2.14 |
| rank0 | fused_gdn_gating_kernel | 384 | 5.893 | 1.871 |
| rank0 | ConcatD | 384 | 5.735 | 1.821 |
| rank1 | allreduceAicpuKernel | 1040 | 148.669 | 47.702 |
| rank1 | MatMulV2 | 2448 | 106.077 | 34.036 |
| rank1 | Transpose | 384 | 6.545 | 2.1 |
| rank1 | ConcatD | 384 | 5.401 | 1.733 |
| rank1 | fused_gdn_gating_kernel | 384 | 5.185 | 1.664 |

trace 输出目录：`/tmp/vllm-profile-8x2-fixed/traces`，包含 trace_view.json、算子/kernel CSV 及通信分析。

## 退出时通信域释放

最初检查发现：通用清理虽令 torch.distributed.is_initialized()=False、PyTorch 进程组注册数=0，
Ascend 的 `_MC2` 仍引用带 cpu_group/device_group 字段的旧 GroupCoordinator；因此仅查 PyTorch 初始化标记不足以证明清理完整。
修复：在通用 cleanup_dist_env_and_memory 前调用已加载 Ascend 模块的 destroy_ascend_model_parallel。
修复已在原有 Actor 内加载，未重启或 kill Actor。退出检查逐实例执行并检查：
PyTorch 默认组/子组注册表、vLLM 进程组全局引用、残留 coordinator 的 CPU/device group 字段、TCPStore、执行线程、Worker 与两个消息队列。

修复后退出检查记录：8 个实例、16 个 Worker。最终 idle Actor 数：16。
Actor PID 前后一致：True。
已注册进程组总数：0；仍存活 coordinator 总数：0。

回归测试：39 passed。修复前与修复后的资源快照保存在 `/tmp/vllm-profile-8x2/cleanup_fix.json`。

关闭部分引擎时仍会记录 AsyncLLM output_handler 的 EngineDeadError；请求已完成、Actor 复用与通信域释放检查通过，但关闭时序日志问题尚未修复。

## 直接冷启动对照

同一对物理 NPU、相同模型及参数、交替执行资源池与标准 MultiprocExecutor。
两组均从全新的 driver 进程运行，但 ready 计时从配置构建前开始，不包含 driver 的 Python import。
池组的 Ray 连接/attach 另计；池创建成本不在单次启动计时内。冷启动组新建普通 Worker 进程，不销毁资源池中的 Actor。
权重文件页缓存保持热态，未清空系统缓存；因此这是 Worker/引擎进程冷启动对照，不是存储冷读性能对照。

| 方式 | 成功样本 | ready平均/最小/最大(s) | 启动至首token平均(s) | 启动至首请求完成平均(s) | 关闭平均(s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 资源池 | 3 | 33.318/33.034/33.630 | 33.680 | 34.103 | 8.925 |
| 直接冷启动(mp) | 3 | 60.317/59.891/60.831 | 60.780 | 61.186 | 11.622 |

按平均 ready 时间计算：池化启动减少 44.8%，速度比 1.81×。
样本数较少，仅报告均值与范围，不推断显著性或给出启动 p95。

池化首轮 33.63s 的日志阶段划分（秒级时间戳近似，非独立精确计时）：

| 阶段 | 近似耗时 |
| --- | ---: |
| 配置及前端初始化 | 2–3s |
| 新建 EngineCore、导入依赖及插件（15:54:50→15:55:10） | 20s |
| Worker 通信域重建及权重加载（15:55:10→15:55:14） | 4s |
| 内存 profiling、KV cache 初始化与预热（引擎日志实测） | 5.50s |
| 其余连接和同步 | 约1s |

资源池只复用 Worker Actor 进程及其已导入依赖；仍重新创建 EngineCore、Worker 对象、通信域、模型和 KV cache。
该样本约六成时间位于新 EngineCore 启动阶段；权重已有页缓存，加载并非主要耗时。
若进一步优化，需对 EngineCore/前端进程复用或其依赖导入做单独实验；本次没有实现这些优化。
对照是否完整：True；原始数据：`/tmp/vllm-startup-comparison-final/comparison.json`。

## EngineCore 预热入口池优化实测

实现位于 `vllm/v1/engine/process_pool.py`，通过 `EngineCoreProcessPool.activate()` 显式接入。
spawn 入口进程提前导入 EngineCore、通用插件和指定模块并连接 Ray；接收到模型配置后才启动引擎。
每个入口只接管一个引擎生命周期；引擎正常退出后由后台补充新的预热入口，避免复用残留 EngineCore 状态。
此池属于当前 driver，不支持跨 driver 共享进程句柄；当前支持 external actors、DP=1、无 NUMA binding、无 torch_shm tensor queue。

8 个入口并行预热耗时 **19.557s**，单独计入准备成本，不包含在后续实例 ready 时间中。
这会把导入开销移出请求关键路径，仍需在后台支付 CPU、内存和补充进程的成本。

| 启动方式 | 样本数 | ready平均(s) | 最小(s) | 最大(s) |
| --- | ---: | ---: | ---: | ---: |
| 直接冷启动(mp) | 3 | 60.317 | 59.891 | 60.831 |
| 仅 Worker Actor 池 | 3 | 33.318 | 33.034 | 33.630 |
| Worker池 + EngineCore预热入口：首次部署 | 8 | 15.043 | 14.343 | 16.514 |
| Worker池 + EngineCore预热入口：滚动重建 | 8 | 14.649 | 14.056 | 15.155 |

全部 16 个预热引擎的平均 ready 时间为 **14.846s**；相对仅 Worker 池减少 **55.4%**，相对直接冷启动减少 **75.4%**。
直接冷启动/仅 Worker 池样本来自同一物理卡对的3轮交替试验；预热入口数据来自8实例驻留及滚动试验。配置均为 TP=2、max_num_seqs=4，其主机并发背景不同，不能当作严格配对显著性检验。
优化后功能验证：**passed**；所有实际 EngineCore PID 均验证来自已预热队列；
16 个 Worker Actor ID/PID 与最初测试一致：**True**；退出通信域检查 **16 个实例/32 个 Worker快照**。
8 个实例并发推理和8次滚动替换再次验证输出一致；每次替换间隙其余7实例继续完成请求。
前述吞吐/延迟和NPU trace在EngineCore预热改动前采集；优化后本轮只重测启动、功能与退出资源，不将之前的吞吐数字当作后台补充进程时的性能结果。
联合回归：57 passed；Ruff、格式检查、git diff --check 通过。

预热池日志与结果：`/tmp/vllm-enginecore-prewarm-8x2`。预热入口宿主 PID：`188304`，保持运行。

复现本次预热池滚动测试：

```bash
PYTHONPATH="$PWD/multi-task-infer:$PYTHONPATH" .venv/bin/python \
  multi-task-infer/examples/rolling_pool_test.py \
  --registry-name rolling-8x2-139702 --engine-core-pool-size 8 \
  --max-num-seqs 4 --output-dir /tmp/vllm-enginecore-prewarm-8x2
```

### 14.85 秒的阶段拆解

对齐16个样本的事件时间和 EngineCore/Worker 日志；文本日志时区为UTC+8，已与epoch事件时间对齐。
总耗时及内存/缓存初始化段有直接计时；其余段用秒级日志边界估算，存在约1秒的边界误差，不能视为精确微基准。

| 阶段 | 平均(s) | 占总耗时 | 说明 |
| --- | ---: | ---: | --- |
| 前端配置和配置交接 | 3.61 | 24.3% | Actor租约、配置构建、AsyncLLM前端初始化、预热入口接管/反序列化与握手；尚无更细独立计时 |
| Worker与模型初始化 | 4.28 | 28.8% | MQ/Worker对象、通信域重建及模型构建加载 |
| 内存profiling、KV分配及预热 | 5.22 | 35.2% | 包括缓存布局、内存profile、KV分配、ATB/算子预热及CPU绑定尝试；日志有直接计时 |
| ready收尾 | 1.73 | 11.7% | 调度器/输出组件初始化、握手响应及前端返回；估算残差 |

Worker的模型加载总计时（每次取较慢TP rank）：平均 **3.60s**；
其中权重加载计时平均 **2.10s**。这些是Worker/模型段的子项，不能另加到14.85s上。
当前每卡仍按 gpu_memory_utilization=0.65 分配约30.84GiB KV缓存；CPU绑定在预热后仍会尝试，并记录CPU数量不足后跳过。
这些均属于下一步可独立计时的候选项；现有数据不能将5.22秒进一步精确分配给内存profile、缓存分配、ATB预热和CPU绑定。
当前eager模式未执行torch.compile或图捕获；19.56秒的入口池准备、关闭回收及后续生成均不在14.85秒内。
逐样本拆解数据：`/tmp/vllm-enginecore-prewarm-8x2/startup_breakdown.json`。

## 保留池与复现

宿主 PID：`139702`；registry：`rolling-8x2-139702`；namespace：`external-executor`。
16 个 Actor 保持存活、保留 Ray 的 16 卡资源预留；引擎退出后处于 IDLE，便于继续部署。不要结束宿主进程。
调试客户端必须使用不会销毁共享 Actor 的 RetainedPool；原版 ActorPoolManager.shutdown 会 kill Actor。

复现脚本位于仓库 multi-task-infer/examples：rolling_pool_test.py、profile_pool_test.py、compare_pool_startup.py；
需先加载 CANN 环境，用 `.venv/bin/python` 执行，并保留 `PYTHONPATH=$PWD/multi-task-infer:$PYTHONPATH`。

原始性能数据：`/tmp/vllm-profile-8x2-fixed/metrics.json`；滚动结果：`/tmp/vllm-rolling-8x2/result.json`；各目录 run.log 包含完整运行日志。

## 计时口径说明

直接启动包含 driver 导入为92.03秒；从创建引擎到ready为76.99秒。资源池已就绪时，接到配置后到ready平均16.53秒，范围15.68–18.73秒。因此可以说请求路径由约92秒缩短到约16–20秒，但两者起点不同：池化将导入和部分初始化提前完成，池预热成本没有消失。图测试的8个CPU Core预热另耗18.74秒，Worker Actor池此前已建立；首次部署总耗时不能直接按16–20秒计算。同口径创建引擎到ready的比较为76.99秒对16.53秒。
