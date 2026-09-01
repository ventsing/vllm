#!/bin/bash
# verify_dependencies.sh - 验证 ExternalExecutor 所有依赖是否就绪
#
# Usage:
#   bash verify_dependencies.sh
#   bash verify_dependencies.sh --mooncake   # 只检查 Mooncake 相关

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✅ $1${NC}"; }
fail() { echo -e "  ${RED}❌ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠️  $1${NC}"; }
section() { echo -e "\n${GREEN}=== $1 ===${NC}"; }

CHECK_MOONCAKE=false
CHECK_ALL=true

for arg in "$@"; do
    case "$arg" in
        --mooncake) CHECK_MOONCAKE=true; CHECK_ALL=false ;;
        --all) CHECK_ALL=true ;;
    esac
done

# ---- Python ----
section "Python 依赖"

python3 -c "import sys; print(f'  Python {sys.version}')" 2>/dev/null || fail "python3 not found"

python3 -c "import vllm; print(f'  vllm {vllm.__version__}')" 2>/dev/null \
    && pass "vllm" || fail "vllm not installed"

python3 -c "import ray; print(f'  ray {ray.__version__}')" 2>/dev/null \
    && pass "ray" || fail "ray not installed"

python3 -c "import torch; print(f'  torch {torch.__version__}')" 2>/dev/null \
    && pass "torch" || fail "torch not installed"

python3 -c "import safetensors; print('  safetensors OK')" 2>/dev/null \
    && pass "safetensors" || fail "safetensors not installed"

python3 -c "
from vllm_external_executor import (
    ExternalWorkerActor, ActorPoolManager,
    ExternalExecutor, StorageCheckpointEngine,
)
print('  vllm_external_executor OK')
" 2>/dev/null && pass "vllm_external_executor plugin" \
    || fail "vllm_external_executor not importable (run: pip install -e .)"

# ---- Mooncake ----
if $CHECK_ALL || $CHECK_MOONCAKE; then
    section "Mooncake 依赖"

    python3 -c "
from mooncake.store import MooncakeDistributedStore, ReplicateConfig
print('  mooncake.store OK')
" 2>/dev/null && pass "mooncake-transfer-engine" \
        || fail "mooncake-transfer-engine not installed (pip install mooncake-transfer-engine)"

    # Check config
    if [ -n "${MOONCAKE_CONFIG_PATH:-}" ]; then
        pass "MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH"

        # Parse master address and check connectivity
        python3 -c "
import json, socket, sys
with open('${MOONCAKE_CONFIG_PATH}') as f:
    c = json.load(f)
master = c.get('master_server_address', '127.0.0.1:50051')
host, port = master.rsplit(':', 1)
port = int(port)
s = socket.socket()
s.settimeout(3)
try:
    s.connect((host, port))
    print(f'  mooncake_master ({master}): reachable')
    sys.exit(0)
except Exception as e:
    print(f'  mooncake_master ({master}): NOT reachable ({e})')
    sys.exit(1)
finally:
    s.close()
" 2>/dev/null && pass "mooncake_master reachable" \
            || fail "mooncake_master not reachable"

        # Print config summary
        python3 -c "
import json
with open('${MOONCAKE_CONFIG_PATH}') as f:
    c = json.load(f)
print(f'  mode={c.get(\"mode\",\"embedded\")}')
print(f'  protocol={c.get(\"protocol\",\"tcp\")}')
print(f'  global_segment_size={c.get(\"global_segment_size\",\"?\")}')
print(f'  local_buffer_size={c.get(\"local_buffer_size\",\"?\")}')
" 2>/dev/null
    else
        fail "MOONCAKE_CONFIG_PATH not set"
        echo "    export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json"
    fi
fi

# ---- RDMA ----
if $CHECK_ALL || $CHECK_MOONCAKE; then
    section "RDMA / 网络"

    if command -v ibv_devinfo &>/dev/null; then
        devs=$(ibv_devinfo 2>/dev/null | grep -c "hca_id:" || true)
        if [ "$devs" -gt 0 ]; then
            pass "RDMA devices found: $devs"
            ibv_devinfo 2>/dev/null | grep -E "hca_id:|state:" | head -6 | sed 's/^/    /'
        else
            warn "No RDMA devices found (will use TCP fallback)"
        fi
    else
        warn "ibv_devinfo not found (RDMA tools not installed)"
        echo "    Mooncake TCP mode will work, but RDMA is recommended"
    fi
fi

# ---- GPU / NPU ----
section "GPU / NPU"

python3 -c "
import torch
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if n > 0 else 'unknown'
    print(f'  CUDA: {n} devices ({name})')
elif hasattr(torch, 'npu') and torch.npu.is_available():
    n = torch.npu.device_count()
    print(f'  NPU: {n} devices')
else:
    print('  No GPU/NPU found')
" 2>/dev/null

# ---- NFS ----
section "NFS / 共享存储"

nfs_count=$(mount | grep -c "nfs\|nfs4" || true)
if [ "$nfs_count" -gt 0 ]; then
    pass "NFS mounts found: $nfs_count"
    mount | grep -E "nfs|nfs4" | head -3 | sed 's/^/    /'
else
    warn "No NFS mounts found"
    echo "    NFS backend needs a shared directory"
fi

# ---- Ray ----
section "Ray Cluster"

python3 -c "
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True, logging_level='ERROR')
    ctx = ray.state.state.storage_client
    nodes = ray.nodes()
    alive = [n for n in nodes if n.get('Alive', False)]
    print(f'  Ray cluster: {len(alive)} alive nodes')
    for n in alive:
        ip = n.get('NodeManagerAddress', '?')
        gpus = n.get('Resources', {}).get('GPU', 0)
        print(f'    {ip}: GPU={gpus}')
    ray.shutdown()
except Exception as e:
    print(f'  Ray cluster: not connected ({e})')
    print('    Start: ray start --head')
" 2>/dev/null

# ---- Summary ----
section "测试命令"
echo ""
echo "  NFS 测试（无外部依赖）:"
echo "    python tests/test_storage_checkpoint_engine.py nfs"
echo ""
echo "  Mooncake Mock 测试（无需服务）:"
echo "    python tests/test_storage_checkpoint_engine.py mooncake_mock"
echo ""
echo "  Mooncake 集成测试（需要服务）:"
echo "    MOONCAKE_CONFIG_PATH=config.json python tests/test_storage_checkpoint_engine.py mooncake"
echo ""
echo "  全部测试:"
echo "    python tests/test_storage_checkpoint_engine.py all"
echo ""
