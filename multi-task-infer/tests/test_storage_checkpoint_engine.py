#!/usr/bin/env python3
"""
Test cases for StorageCheckpointEngine.

Includes:
1. NFS backend test (no external dependencies)
2. Mooncake backend test (requires Mooncake services)
3. Mock test for Mooncake (no services needed, validates interface)

Usage:
    # Run NFS test only (no dependencies)
    python test_storage_checkpoint_engine.py nfs

    # Run Mooncake mock test (no services needed)
    python test_storage_checkpoint_engine.py mooncake_mock

    # Run Mooncake integration test (requires Mooncake services)
    python test_storage_checkpoint_engine.py mooncake

    # Run all tests
    python test_storage_checkpoint_engine.py all
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vllm_external_executor.storage_checkpoint_engine import (
    CheckpointMetadata,
    NFSStorageBackend,
    StorageBackendFactory,
    StorageCheckpointEngine,
    TensorMeta,
)


def make_test_weights():
    """Generate test model weights."""
    weights = {
        "model.embed_tokens.weight": torch.randn(1000, 128),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(128),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 128),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(128, 128),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(128, 128),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(512, 128),
        "model.layers.0.mlp.up_proj.weight": torch.randn(512, 128),
        "model.layers.0.mlp.down_proj.weight": torch.randn(128, 512),
        "model.norm.weight": torch.randn(128),
        "lm_head.weight": torch.randn(1000, 128),
    }
    return weights


def weights_to_gen(weights):
    """Convert dict to generator."""
    for name, tensor in weights.items():
        yield name, tensor


def assert_tensors_equal(a, b, name):
    """Assert two tensors are equal."""
    assert a.shape == b.shape, (
        f"{name}: shape mismatch {a.shape} vs {b.shape}"
    )
    assert a.dtype == b.dtype, (
        f"{name}: dtype mismatch {a.dtype} vs {b.dtype}"
    )
    assert torch.allclose(a, b, atol=1e-6), (
        f"{name}: values differ"
    )


# ============================================================
# Test 1: NFS Backend
# ============================================================

def test_nfs_backend():
    """Test NFS storage backend."""
    print("=" * 60)
    print("Test: NFS Backend")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"base_path": tmpdir}
        checkpoint_path = "test-model/step_100"

        # 1. Create engine and save
        engine = StorageCheckpointEngine(
            backend="nfs",
            config=config,
            device="cpu",
        )
        engine.set_checkpoint(checkpoint_path)

        weights = make_test_weights()
        total_bytes = sum(t.nbytes for t in weights.values())
        print(f"  Saving {len(weights)} tensors, "
              f"{total_bytes / 1024:.1f} KB")

        start = time.monotonic()
        asyncio.run(engine.send_weights(
            weights_to_gen(weights),
            global_steps=100,
        ))
        save_time = time.monotonic() - start
        print(f"  Save time: {save_time:.3f}s")

        # 2. Verify checkpoint exists
        assert engine.exists(checkpoint_path), (
            "Checkpoint should exist"
        )
        print("  Checkpoint exists: OK")

        # 3. Verify files on disk
        ckpt_dir = os.path.join(tmpdir, checkpoint_path)
        assert os.path.exists(
            os.path.join(ckpt_dir, "model.safetensors")
        ), "model.safetensors should exist"
        assert os.path.exists(
            os.path.join(ckpt_dir, "metadata.json")
        ), "metadata.json should exist"
        print("  Files on disk: OK")

        # 4. Load metadata
        meta = engine.load_metadata(checkpoint_path)
        assert meta.model_name == "unknown"
        assert meta.global_steps == 100
        assert len(meta.tensors) == len(weights)
        print(f"  Metadata: {len(meta.tensors)} tensors, OK")

        # 5. Load weights and verify
        loaded = {}
        start = time.monotonic()
        for name, tensor in engine.get_weights():
            loaded[name] = tensor
        load_time = time.monotonic() - start
        print(f"  Load time: {load_time:.3f}s")

        assert len(loaded) == len(weights), (
            f"Expected {len(weights)} tensors, "
            f"got {len(loaded)}"
        )
        for name in weights:
            assert name in loaded, f"Missing: {name}"
            assert_tensors_equal(weights[name], loaded[name], name)
        print("  Weight verification: OK")

        # 6. Test delete
        engine.delete(checkpoint_path)
        assert not engine.exists(checkpoint_path), (
            "Checkpoint should be deleted"
        )
        print("  Delete: OK")

    print("  NFS Backend: PASSED\n")


# ============================================================
# Test 2: Mooncake Mock Test (no services needed)
# ============================================================

class MockMooncakeStore:
    """Mock MooncakeDistributedStore for testing."""

    def __init__(self):
        self._data: dict[str, bytes] = {}
        self._setup_done = False

    def setup(self, *args, **kwargs):
        self._setup_done = True
        return 0

    def put(self, key, data, config=None):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._data[key] = bytes(data)
        return 0

    def get(self, key):
        return self._data.get(key)

    def close(self):
        self._data.clear()


class MockReplicateConfig:
    """Mock ReplicateConfig."""

    def __init__(self):
        self.with_soft_pin = True
        self.preferred_segment = None


def test_mooncake_mock():
    """Test Mooncake backend with mock store."""
    print("=" * 60)
    print("Test: Mooncake Backend (Mock)")
    print("=" * 60)

    # Monkey-patch mooncake imports
    import unittest.mock as mock

    mock_module = mock.MagicMock()
    mock_module.MooncakeDistributedStore = MockMooncakeStore
    mock_module.ReplicateConfig = MockReplicateConfig

    with mock.patch.dict(
        "sys.modules",
        {"mooncake.store": mock_module, "mooncake": mock_module},
    ):
        # Re-import to pick up mocks
        from vllm_external_executor.storage_checkpoint_engine import (
            MooncakeStoreBackend,
            StorageCheckpointEngine,
        )

        config = {
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "master_server_address": "127.0.0.1:50051",
            "protocol": "tcp",
            "device_name": "",
            "global_segment_size": 64 * 1024 * 1024,
            "local_buffer_size": 8 * 1024 * 1024,
        }
        checkpoint_path = "test-model/step_200"

        # 1. Create engine
        backend = MooncakeStoreBackend()
        backend.initialize(config)
        assert backend._initialized, "Should be initialized"
        print("  Initialize: OK")

        # 2. Save weights
        engine = StorageCheckpointEngine(
            backend="mooncake",
            config=config,
            device="cpu",
        )
        engine.set_checkpoint(checkpoint_path)

        weights = make_test_weights()
        total_bytes = sum(t.nbytes for t in weights.values())
        print(f"  Saving {len(weights)} tensors, "
              f"{total_bytes / 1024:.1f} KB")

        start = time.monotonic()
        asyncio.run(engine.send_weights(
            weights_to_gen(weights),
            global_steps=200,
        ))
        save_time = time.monotonic() - start
        print(f"  Save time: {save_time:.3f}s")

        # 3. Verify exists
        assert engine.exists(checkpoint_path), (
            "Checkpoint should exist"
        )
        print("  Exists: OK")

        # 4. Verify keys in mock store
        mock_store = engine.storage.store
        expected_keys = (
            [f"ckpt:{checkpoint_path}:metadata"]
            + [
                f"ckpt:{checkpoint_path}:tensor:{name}"
                for name in weights
            ]
        )
        for key in expected_keys:
            assert key in mock_store._data, (
                f"Key missing: {key}"
            )
        print(f"  Store keys: {len(mock_store._data)} keys, OK")

        # 5. Load metadata
        meta = engine.load_metadata(checkpoint_path)
        assert meta.global_steps == 200
        assert len(meta.tensors) == len(weights)
        print(f"  Metadata: {len(meta.tensors)} tensors, OK")

        # 6. Load weights and verify
        loaded = {}
        start = time.monotonic()
        for name, tensor in engine.get_weights():
            loaded[name] = tensor
        load_time = time.monotonic() - start
        print(f"  Load time: {load_time:.3f}s")

        assert len(loaded) == len(weights)
        for name in weights:
            assert_tensors_equal(weights[name], loaded[name], name)
        print("  Weight verification: OK")

        # 7. Test tensor serialization roundtrip
        for name, orig in weights.items():
            raw = MooncakeStoreBackend._serialize_tensor(orig)
            tmeta = TensorMeta(
                name=name,
                shape=orig.shape,
                dtype=orig.dtype,
                storage_key="",
                size=orig.nbytes,
            )
            restored = MooncakeStoreBackend._deserialize_tensor(
                raw, tmeta
            )
            assert_tensors_equal(orig, restored, name)
        print("  Serialization roundtrip: OK")

    print("  Mooncake Mock: PASSED\n")


# ============================================================
# Test 3: Mooncake Integration (requires services)
# ============================================================

def test_mooncake_integration():
    """Test Mooncake backend with real services."""
    print("=" * 60)
    print("Test: Mooncake Backend (Integration)")
    print("=" * 60)

    # Check if mooncake is available
    try:
        from mooncake.store import (
            MooncakeDistributedStore,
            ReplicateConfig,
        )
    except ImportError:
        print("  SKIP: mooncake-transfer-engine not installed")
        print("  Install: pip install mooncake-transfer-engine")
        return

    # Check if master is reachable
    config_path = os.environ.get("MOONCAKE_CONFIG_PATH")
    if not config_path:
        print("  SKIP: MOONCAKE_CONFIG_PATH not set")
        print("  Set: export MOONCAKE_CONFIG_PATH=/path/to/config.json")
        return

    with open(config_path) as f:
        config = json.load(f)

    checkpoint_path = "test-model/integration_step"

    # 1. Create engine
    engine = StorageCheckpointEngine(
        backend="mooncake",
        config=config,
        device="cpu",
    )
    engine.set_checkpoint(checkpoint_path)
    print("  Initialize: OK")

    # 2. Save weights
    weights = make_test_weights()
    total_bytes = sum(t.nbytes for t in weights.values())
    print(f"  Saving {len(weights)} tensors, "
          f"{total_bytes / 1024:.1f} KB")

    start = time.monotonic()
    asyncio.run(engine.send_weights(
        weights_to_gen(weights),
        global_steps=999,
    ))
    save_time = time.monotonic() - start
    bw = total_bytes / save_time / (1024 ** 2) if save_time > 0 else 0
    print(f"  Save: {save_time:.3f}s, {bw:.1f} MB/s")

    # 3. Load and verify
    assert engine.exists(checkpoint_path)
    print("  Exists: OK")

    loaded = {}
    start = time.monotonic()
    for name, tensor in engine.get_weights():
        loaded[name] = tensor
    load_time = time.monotonic() - start
    bw = total_bytes / load_time / (1024 ** 2) if load_time > 0 else 0
    print(f"  Load: {load_time:.3f}s, {bw:.1f} MB/s")

    assert len(loaded) == len(weights)
    for name in weights:
        assert_tensors_equal(weights[name], loaded[name], name)
    print("  Weight verification: OK")

    print("  Mooncake Integration: PASSED\n")


# ============================================================
# Test 4: Cross-backend compatibility
# ============================================================

def test_cross_backend():
    """Test saving with NFS, loading with Mooncake mock."""
    print("=" * 60)
    print("Test: Cross-backend (NFS save -> verify interface)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save with NFS
        engine_nfs = StorageCheckpointEngine(
            backend="nfs",
            config={"base_path": tmpdir},
            device="cpu",
        )
        engine_nfs.set_checkpoint("cross-test/step_1")

        weights = make_test_weights()
        asyncio.run(engine_nfs.send_weights(
            weights_to_gen(weights),
            global_steps=1,
        ))

        # Verify interface consistency
        assert engine_nfs.exists("cross-test/step_1")
        meta = engine_nfs.load_metadata("cross-test/step_1")
        assert len(meta.tensors) == len(weights)

        loaded = {}
        for name, tensor in engine_nfs.get_weights():
            loaded[name] = tensor
        assert len(loaded) == len(weights)
        for name in weights:
            assert_tensors_equal(weights[name], loaded[name], name)

    print("  Cross-backend: PASSED\n")


# ============================================================
# Main
# ============================================================

def main():
    tests = {
        "nfs": test_nfs_backend,
        "mooncake_mock": test_mooncake_mock,
        "mooncake": test_mooncake_integration,
        "cross": test_cross_backend,
    }

    if len(sys.argv) < 2 or sys.argv[1] == "all":
        to_run = list(tests.keys())
    else:
        to_run = sys.argv[1:]

    passed = 0
    failed = 0
    skipped = 0

    for name in to_run:
        if name not in tests:
            print(f"Unknown test: {name}")
            print(f"Available: {', '.join(tests.keys())}")
            sys.exit(1)
        try:
            tests[name]()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
