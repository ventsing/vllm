# ExternalExecutor 启动依赖清单

## 1. Python 依赖

### 核心依赖（必须）

```bash
# vLLM 本体
pip install vllm

# Ray（Actor 池化 + 分布式调度）
pip install ray[default]

# safetensors（NFS 后端权重序列化）
pip install safetensors

# 本插件
cd vllm/multi-task-infer
pip install -e .
```

### Mooncake 依赖（使用 Mooncake Store 后端时必须）

```bash
# Mooncake Transfer Engine
pip install mooncake-transfer-engine

# 如果 pip 安装失败，需要从源码编译：
# https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md
```

### 可选依赖

```bash
# 编译缓存压缩（大缓存场景）
pip install zstandard

# verl（如果需要注册到 verl 的 CheckpointEngineRegistry）
pip install verl
```

## 2. 系统服务依赖

### Mooncake Store 后端（必须启动以下服务）

#### 2.1 mooncake_master

Mooncake 的元数据管理和协调服务。**必须在所有客户端之前启动**。

```bash
# 启动 master（默认端口 50051）
mooncake_master --port 50051

# 如果需要磁盘卸载
mooncake_master --port 50051 --enable_offload=true

# 如果需要多租户隔离
mooncake_master --port 50051 --enable_multi_tenants=true
```

**验证**：
```bash
# 检查 master 是否可达
curl -s http://127.0.0.1:50051/health || echo "master not reachable"
```

#### 2.2 mooncake_client（standalone-store 模式可选）

仅在 `standalone-store` 模式下需要。`embedded` 模式下不需要。

```bash
# 启动 client（拥有 CPU 内存池和可选的 SSD 层）
mooncake_client \
    --port 50053 \
    --metadata_server "http://127.0.0.1:8080/metadata"

# 如果需要 SSD 卸载
MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/data/mooncake/ssd \
mooncake_client --port 50053 --enable_offload=true
```

#### 2.3 Mooncake 配置文件

创建 `mooncake_config.json`：

```json
{
    "mode": "embedded",
    "metadata_server": "P2PHANDSHAKE",
    "master_server_address": "127.0.0.1:50051",
    "global_segment_size": "80GB",
    "local_buffer_size": "4GB",
    "protocol": "rdma",
    "device_name": "",
    "enable_offload": false
}
```

| 参数 | 说明 | embedded 模式 | standalone-store 模式 |
|------|------|--------------|---------------------|
| `mode` | 拓扑模式 | 每个节点贡献内存 | 外部 client 拥有存储 |
| `metadata_server` | 元数据服务 | `P2PHANDSHAKE` | `P2PHANDSHAKE` |
| `master_server_address` | Master 地址 | `host:50051` | `host:50051` |
| `global_segment_size` | 每节点内存池 | > 0（如 80GB） | 必须为 0 |
| `local_buffer_size` | 本地传输缓冲 | 4GB | 4GB |
| `protocol` | 传输协议 | `rdma`（推荐）或 `tcp` | 同左 |
| `device_name` | RDMA 设备名 | 留空自动检测，或指定如 `mlx5_0` | 同左 |

设置环境变量：
```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

### NFS 后端（无额外服务依赖）

只需要一个共享文件系统挂载点：

```bash
# 确保 NFS 挂载正常
mount -t nfs server:/shared /shared
ls /shared/

# 或使用本地目录测试
mkdir -p /tmp/vllm_checkpoints
```

## 3. 硬件/网络依赖

### Mooncake Store 后端

| 需求 | 说明 |
|------|------|
| **RDMA 网卡**（推荐） | InfiniBand 或 RoCE v2，带宽 ≥ 100 Gbps |
| **RDMA 驱动** | `ibv_devinfo` 能看到设备 |
| **GPU 显存** | 每卡至少 `global_segment_size` 的 CPU 内存 |
| **网络连通** | 所有节点能访问 `master_server_address` |

验证 RDMA：
```bash
# 检查 RDMA 设备
ibv_devinfo

# 检查网卡
ibstat

# 如果没有 RDMA，可以用 TCP 模式（性能低但可用）
# 在 config 中设置 "protocol": "tcp"
```

### Ascend NPU 环境

```bash
# 检查 NPU 设备
npu-smi info

# Mooncake 支持 Ascend 直传
# 在 config 中设置 "protocol": "ascend_direct"
```

## 4. 启动顺序

### NFS 后端（简单）

```
1. 确保 NFS 挂载正常
2. 启动 Ray cluster（ray start --head / ray start --address=...）
3. 运行 ExternalExecutor
```

### Mooncake Store 后端

```
1. 启动 mooncake_master
   $ mooncake_master --port 50051

2. [可选] 启动 mooncake_client（standalone-store 模式）
   $ mooncake_client --port 50053

3. 设置环境变量
   $ export MOONCAKE_CONFIG_PATH=/path/to/config.json

4. 启动 Ray cluster
   $ ray start --head

5. 运行 ExternalExecutor
   $ python examples/basic_usage.py storage
```

## 5. 快速验证脚本

```bash
#!/bin/bash
# verify_dependencies.sh - 验证所有依赖是否就绪

echo "=== Python 依赖 ==="
python3 -c "import vllm; print(f'vllm: {vllm.__version__}')" 2>/dev/null || echo "❌ vllm not installed"
python3 -c "import ray; print(f'ray: {ray.__version__}')" 2>/dev/null || echo "❌ ray not installed"
python3 -c "import safetensors; print('safetensors: OK')" 2>/dev/null || echo "❌ safetensors not installed"
python3 -c "import torch; print(f'torch: {torch.__version__}')" 2>/dev/null || echo "❌ torch not installed"

echo ""
echo "=== Mooncake 依赖 ==="
python3 -c "from mooncake.store import MooncakeDistributedStore; print('mooncake: OK')" 2>/dev/null || echo "❌ mooncake not installed"

echo ""
echo "=== Mooncake 服务 ==="
if [ -n "$MOONCAKE_CONFIG_PATH" ]; then
    echo "MOONCAKE_CONFIG_PATH: $MOONCAKE_CONFIG_PATH"
    python3 -c "
import json
with open('$MOONCAKE_CONFIG_PATH') as f:
    c = json.load(f)
master = c.get('master_server_address', '127.0.0.1:50051')
host, port = master.split(':')
import socket
s = socket.socket()
s.settimeout(2)
try:
    s.connect((host, int(port)))
    print(f'mooncake_master ({master}): ✅ reachable')
except:
    print(f'mooncake_master ({master}): ❌ not reachable')
finally:
    s.close()
"
else
    echo "❌ MOONCAKE_CONFIG_PATH not set"
fi

echo ""
echo "=== RDMA ==="
if command -v ibv_devinfo &>/dev/null; then
    ibv_devinfo 2>/dev/null | head -5
else
    echo "⚠️  ibv_devinfo not found (RDMA tools not installed)"
fi

echo ""
echo "=== NFS ==="
if mount | grep -q nfs; then
    mount | grep nfs | head -3
else
    echo "⚠️  No NFS mounts found"
fi

echo ""
echo "=== GPU/NPU ==="
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA: {torch.cuda.device_count()} devices')
elif hasattr(torch, 'npu') and torch.npu.is_available():
    print(f'NPU: {torch.npu.device_count()} devices')
else:
    print('⚠️  No GPU/NPU found')
" 2>/dev/null

echo ""
echo "=== 运行测试 ==="
echo "  NFS 测试（无外部依赖）:"
echo "    python tests/test_storage_checkpoint_engine.py nfs"
echo ""
echo "  Mooncake Mock 测试（无需服务）:"
echo "    python tests/test_storage_checkpoint_engine.py mooncake_mock"
echo ""
echo "  Mooncake 集成测试（需要服务）:"
echo "    MOONCAKE_CONFIG_PATH=config.json python tests/test_storage_checkpoint_engine.py mooncake"
```

## 6. 最小可运行环境

### 仅 NFS 后端（最简）

```
- Python 3.10+
- vllm
- ray
- safetensors
- torch
- 本地磁盘或 NFS 挂载
```

### Mooncake Store 后端（完整）

```
- Python 3.10+
- vllm
- ray
- safetensors
- torch
- mooncake-transfer-engine
- mooncake_master 服务运行中
- RDMA 网络（推荐）或 TCP 回退
- MOONCAKE_CONFIG_PATH 环境变量
```

## 7. 常见问题

### Q: mooncake_master 启动失败？

```bash
# 检查端口是否被占用
lsof -i :50051

# 检查是否有旧进程
ps aux | grep mooncake_master
```

### Q: Mooncake put/get 返回错误？

```bash
# 检查 master 是否可达
python3 -c "
import socket
s = socket.socket()
s.settimeout(2)
s.connect(('127.0.0.1', 50051))
print('OK')
s.close()
"

# 检查 global_segment_size 是否足够
# 7B 模型约需 14GB 存储，建议设置 80GB
```

### Q: RDMA 不可用？

在配置中使用 TCP 模式：
```json
{
    "protocol": "tcp",
    "device_name": ""
}
```

性能会下降（~2 GB/s vs ~9 GB/s），但功能正常。

### Q: Ascend NPU 环境？

```bash
# Mooncake 支持 Ascend 直传
{
    "protocol": "ascend_direct",
    "device_name": ""
}

# 需要设置白名单
export HCCL_WHITELIST_DISABLE=0
export HCCL_WHITELIST_FILE=/path/to/whitelist.json

# CANN >= 8.5.0 需要
export HCCL_INTRA_ROCE_ENABLE=1
```
