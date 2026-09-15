# 推送和创建 PR 指南

## 当前状态

✅ 已完成并推送 `origin/feature/external-executor`：

- 48 files changed（+13952 / −2，实时核对：`git diff --stat origin/main...HEAD`）
- 25 commits（实时核对：`git log --oneline origin/main..HEAD | wc -l`）
- 核心 vLLM 修改仅 8 文件（+159 / −2，最小侵入）
- 插件 `vllm_external_executor/` 17 个模块 + `tests/` 10 个测试模块
- PR 描述 `PR_DESCRIPTION.md` + 真机验证清单 `HARDWARE_VALIDATION.md`

## 推送步骤

### 方式 1：使用 HTTPS（需要认证）

```bash
cd /home/ventsing/source/opensource/ai/llm/vllm

# 推送分支
git push -u origin feature/external-executor

# 如果提示输入用户名和密码：
# 用户名：你的 GitHub 用户名
# 密码：使用 Personal Access Token（不是 GitHub 密码）
```

**创建 Personal Access Token**：
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 勾选 `repo` 权限
4. 生成并复制 token
5. 推送时使用 token 作为密码

### 方式 2：使用 SSH（推荐）

```bash
# 1. 修改远程 URL 为 SSH
cd /home/ventsing/source/opensource/ai/llm/vllm
git remote set-url origin git@github.com:ventsing/vllm.git

# 2. 确保 SSH key 已添加到 GitHub
# 检查是否有 SSH key
ls -la ~/.ssh/id_rsa.pub

# 如果没有，生成一个
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 添加公钥到 GitHub
cat ~/.ssh/id_rsa.pub
# 复制输出，访问 https://github.com/settings/keys 添加

# 3. 测试 SSH 连接
ssh -T git@github.com
# 应该看到：Hi ventsing! You've successfully authenticated...

# 4. 推送分支
git push -u origin feature/external-executor
```

### 方式 3：使用 GitHub CLI

```bash
# 安装 GitHub CLI
# macOS
brew install gh

# Linux
sudo apt install gh

# 登录
gh auth login

# 推送并创建 PR
git push -u origin feature/external-executor
gh pr create --title "feat: ExternalExecutor — actor pool, model hot-switching, KV migration" \
             --body-file multi-task-infer/PR_DESCRIPTION.md
```

## 创建 PR

推送成功后，有两种方式创建 PR：

### 方式 1：通过 GitHub Web 界面

1. 访问 https://github.com/ventsing/vllm
2. 你应该能看到 "Compare & pull request" 按钮
3. 点击按钮，填写 PR 信息：
   - **Title**: `feat: ExternalExecutor — actor pool, model hot-switching, KV migration`
   - **Base branch**: `main`（或你的目标分支）
   - **Compare branch**: `feature/external-executor`
   - **Description**: 复制 `PR_DESCRIPTION.md` 的内容
4. 点击 "Create pull request"

### 方式 2：使用 GitHub CLI

```bash
cd /home/ventsing/source/opensource/ai/llm/vllm

gh pr create \
  --title "feat: ExternalExecutor — actor pool, model hot-switching, KV migration" \
  --body-file multi-task-infer/PR_DESCRIPTION.md \
  --base main \
  --head feature/external-executor
```

## PR 内容概览

### 核心功能

1. **Actor Pooling**：预启动 Ray Actor，跨 vLLM 实例复用（含跨节点注册、
   故障域调度、节点/actor 级故障转移）
2. **Model Hot-switching**：迁移状态机驱动的模型热切换（原子事务 + 幂等 +
   补偿回滚）
3. **Cross-engine KV Migration**：增量、前缀感知、异构块重映射的 KV 迁移，
   Pluggable transport（Ray / Mooncake RDMA / CUDA IPC）
4. **Compilation Cache Sharing**：CacheManagerActor 管理编译缓存
5. **Storage Checkpoint Engine**：从 NFS/Mooncake Store 加载模型权重
6. **Decision Layer**：三层存储 tiering / 全局前缀索引 / 权重共享账本 /
   异步预取（`migration_orchestrator.py` 接入迁移状态机）
7. **Elastic Autoscaling**：按队列/P99/利用率水位 + 时段窗口 + 冷却自动扩缩
   Actor 池（`autoscaling.py` + `maybe_autoscale`）

详见 `PR_DESCRIPTION.md` 与 `HARDWARE_VALIDATION.md`。

### 文件结构

```
multi-task-infer/
├── README.md                                    # 主文档
├── design.md                                    # 4+1 视图设计文档（含场景 5.4.1/7.4 弹性扩缩容）
├── PR_DESCRIPTION.md                            # PR 描述（提交时复制为 body）
├── HARDWARE_VALIDATION.md                       # 真机验证清单（M1-M11）
├── STORAGE_CHECKPOINT_ENGINE_DESIGN.md          # 存储后端设计
├── STARTUP_DEPENDENCIES.md                      # 启动依赖清单
├── PUSH_AND_PR_GUIDE.md                         # 本指南
├── pyproject.toml                               # 插件包配置
├── vllm_external_executor/                      # 插件代码（17 个模块）
│   ├── __init__.py
│   ├── external_worker_actor.py                 # 预启动的 Ray Actor（设备绑定 + KV 导出/导入）
│   ├── actor_pool_manager.py                    # Actor 池（启停/租借/故障转移/弹性扩缩容）
│   ├── external_executor.py                     # ExternalExecutor（迁移状态机 + 增量 KV）
│   ├── cluster_state.py                         # NodeInfo / ActorRegistration / GlobalScheduler（纯）
│   ├── node_registry_actor.py                   # 注册/心跳/统一视图/死检测
│   ├── migration.py                             # 迁移状态机 + 原子事务 + 幂等
│   ├── kv_migration.py                          # KV 增量 diff + 前缀感知
│   ├── kv_transport.py                          # Ray / Mooncake RDMA / CUDA IPC 传输
│   ├── storage_tier.py                          # 三级分层存储决策（纯）
│   ├── global_prefix_index.py                   # 跨 Actor 前缀索引（纯）
│   ├── weight_sharing.py                        # base+adapter 权重共享账本（纯）
│   ├── prefetch_policy.py                       # 访问热度 + 异步预取（纯）
│   ├── autoscaling.py                           # 弹性扩缩容决策（纯）
│   ├── migration_orchestrator.py                # 决策层 → 迁移状态机接线
│   ├── cache_manager_actor.py                   # 编译缓存共享
│   └── storage_checkpoint_engine.py             # NFS / Mooncake 存储后端
├── examples/                                    # 示例代码（3 个）
├── tests/                                       # 测试（10 个模块）
└── verify_dependencies.sh                       # 依赖验证脚本
```

> 完整分层细节以 `design.md` §3.4 文件清单为准。

### vLLM 核心修改（最小侵入）

仅修改 8 个文件（+159 / −2 行）：
- `vllm/v1/engine/async_llm.py` — 接受 `external_actors` 并传递（+5）
- `vllm/v1/engine/core_client.py` — 线程化 `external_actors`（+9）
- `vllm/v1/engine/utils.py` — 传递 `external_actors`（+4）
- `vllm/v1/engine/core.py` — 传参 + `bind_scheduler` 钩子（+17/−1）
- `vllm/v1/request.py` — 请求快照/恢复（+58）
- `vllm/v1/core/block_pool.py` — KV 快照读接口（+20）
- `vllm/v1/core/kv_cache_manager.py` — computed-token crop（+24）
- `vllm/v1/core/kv_cache_coordinator.py` — 请求→块表访问器（+24）

## 验证推送

```bash
# 检查分支是否推送成功
git branch -vv

# 应该看到（HEAD hash 以 `git rev-parse --short HEAD` 为准）：
# * feature/external-executor  <HEAD> [origin/feature/external-executor] ...

# 检查远程分支
git ls-remote origin feature/external-executor
```

## 常见问题

### Q: 推送时提示 "Authentication failed"

使用 Personal Access Token 而不是 GitHub 密码：
```bash
# 创建 token: https://github.com/settings/tokens
# 推送时使用 token 作为密码
```

### Q: SSH 连接失败

```bash
# 检查 SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

# 测试连接
ssh -T git@github.com
```

### Q: 推送后看不到 PR

确保推送到正确的仓库：
```bash
git remote -v
# 应该是：origin  git@github.com:ventsing/vllm.git

# 如果不对，修改
git remote set-url origin git@github.com:ventsing/vllm.git
```

## 下一步

推送成功后：
1. 在 GitHub 上创建 PR
2. 等待 CI 检查通过
3. 请求 reviewer 审核
4. 根据反馈修改代码
