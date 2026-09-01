# 推送和创建 PR 指南

## 当前状态

✅ 已完成：
- 代码整理和清理
- 创建分支 `feature/external-executor`
- 创建 commit（19 files changed, 5749 insertions）
- 创建 PR 描述文件 `PR_DESCRIPTION.md`

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
gh pr create --title "feat: ExternalExecutor plugin with actor pooling and storage backend" \
             --body-file multi-task-infer/PR_DESCRIPTION.md
```

## 创建 PR

推送成功后，有两种方式创建 PR：

### 方式 1：通过 GitHub Web 界面

1. 访问 https://github.com/ventsing/vllm
2. 你应该能看到 "Compare & pull request" 按钮
3. 点击按钮，填写 PR 信息：
   - **Title**: `feat: ExternalExecutor plugin with actor pooling and storage backend`
   - **Base branch**: `main`（或你的目标分支）
   - **Compare branch**: `feature/external-executor`
   - **Description**: 复制 `PR_DESCRIPTION.md` 的内容
4. 点击 "Create pull request"

### 方式 2：使用 GitHub CLI

```bash
cd /home/ventsing/source/opensource/ai/llm/vllm

gh pr create \
  --title "feat: ExternalExecutor plugin with actor pooling and storage backend" \
  --body-file multi-task-infer/PR_DESCRIPTION.md \
  --base main \
  --head feature/external-executor
```

## PR 内容概览

### 核心功能

1. **Actor Pooling**：预启动 Ray Actor，跨 vLLM 实例复用
2. **Compilation Cache Sharing**：CacheManagerActor 管理编译缓存
3. **Storage Checkpoint Engine**：从 NFS/Mooncake Store 加载模型权重

### 文件结构

```
multi-task-infer/
├── README.md                                    # 主文档
├── design.md                                    # 4+1 视图设计文档
├── STORAGE_CHECKPOINT_ENGINE_DESIGN.md          # 存储后端设计
├── STARTUP_DEPENDENCIES.md                      # 启动依赖清单
├── pyproject.toml                               # 插件包配置
├── vllm_external_executor/                      # 插件代码（5 个核心文件）
│   ├── __init__.py
│   ├── external_worker_actor.py                 # 预启动的 Ray Actor
│   ├── actor_pool_manager.py                    # Actor 池管理器
│   ├── external_executor.py                     # ExternalExecutor 实现
│   ├── cache_manager_actor.py                   # CacheManagerActor 实现
│   └── storage_checkpoint_engine.py             # 存储后端 checkpoint engine
├── examples/                                    # 示例代码
├── tests/                                       # 测试用例
└── verify_dependencies.sh                       # 依赖验证脚本
```

### vLLM 核心修改（最小侵入）

仅修改 4 个文件用于参数传递：
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/utils.py`
- `vllm/v1/engine/core.py`

## 验证推送

```bash
# 检查分支是否推送成功
git branch -vv

# 应该看到：
# * feature/external-executor  37d2754423 [origin/feature/external-executor] feat: ExternalExecutor...

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
