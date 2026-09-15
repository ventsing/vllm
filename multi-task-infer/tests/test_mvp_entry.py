# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Unit tests for the MVP envelope validation (mvp_entry.validate_mvp_config).

These run without Ray/torch/vllm: mvp_entry only imports them lazily inside
run_mvp, so the constraint checker is importable and testable in isolation.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_EXEC_DIR = Path(__file__).resolve().parent.parent / "vllm_external_executor"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Register a stub package so mvp_entry's lazy imports resolve only when run_mvp
# is actually called (never in these tests).
sys.modules.setdefault(
    "vllm_external_executor", types.ModuleType("vllm_external_executor")
)
mvp = _load_module(
    "vllm_external_executor.mvp_entry",
    _EXEC_DIR / "mvp_entry.py",
)


def test_valid_tp1_pp1_passes():
    """TP=1 PP=1 single-node is within the MVP envelope."""
    mvp.validate_mvp_config(tp_size=1, pp_size=1)  # no raise


def test_valid_tp2_pp1_passes():
    """TP=2 PP=1 single-node is within the MVP envelope."""
    mvp.validate_mvp_config(tp_size=2, pp_size=1)  # no raise


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"tp_size": 4}, "TP in {1, 2}"),
        ({"tp_size": 1, "pp_size": 2}, "PP=1"),
        ({"tp_size": 1, "num_nodes": 2}, "single-node"),
        ({"tp_size": 1, "enable_lora": True}, "LoRA"),
        ({"tp_size": 1, "kv_transfer_config": {}}, "KV sharing"),
        ({"tp_size": 1, "elastic_ep": True}, "dynamic TP/PP"),
        ({"tp_size": 1, "enable_autoscaling": True}, "autoscaling"),
    ],
)
def test_unsupported_config_raises(kwargs, message):
    """Every out-of-envelope flag is rejected with a clear error."""
    with pytest.raises(ValueError, match=message):
        mvp.validate_mvp_config(**kwargs)
