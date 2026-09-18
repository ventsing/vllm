# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Unit tests for the MVP envelope validation (mvp_entry.validate_mvp_config).

These run without Ray/torch/vllm: mvp_entry only imports them lazily inside
run_mvp, so the constraint checker is importable and testable in isolation.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_EXEC_DIR = Path(__file__).resolve().parent.parent / "vllm_external_executor"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mvp = _load_module(
    "mvp_entry_under_test",
    _EXEC_DIR / "mvp_entry.py",
)


def test_valid_tp1_pp1_passes():
    """TP=1 PP=1 single-node is within the MVP envelope."""
    mvp.validate_mvp_config(tp_size=1, pp_size=1)  # no raise


def test_valid_tp2_pp1_passes():
    """TP=2 PP=1 single-node is within the MVP envelope."""
    mvp.validate_mvp_config(tp_size=2, pp_size=1)  # no raise


def test_valid_tp4_pp1_passes():
    """TP=4 PP=1 single-node is within the MVP envelope."""
    mvp.validate_mvp_config(tp_size=4, pp_size=1)  # no raise


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"tp_size": 0}, "positive"),
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


@pytest.mark.parametrize("shutdown_fails", [False, True])
@pytest.mark.parametrize("own_pool", [False, True])
def test_cleanup_waits_for_engine_and_returns_actors_even_on_shutdown_error(
    monkeypatch, shutdown_fails, own_pool
):
    from types import ModuleType, SimpleNamespace
    from unittest.mock import Mock

    events = []
    actors = [object()]
    pool = SimpleNamespace(
        pre_start=Mock(),
        acquire=Mock(return_value=actors),
        release=Mock(side_effect=lambda actors: events.append("release")),
        shutdown=Mock(side_effect=lambda: events.append("pool shutdown")),
    )

    class EngineArgs:
        def __init__(self, **kwargs):
            assert "cleanup_timeout" not in kwargs

        def create_engine_config(self):
            return object()

    class LLM:
        def __init__(self, **kwargs):
            pass

        def shutdown(self, timeout):
            events.append(("engine shutdown", timeout))
            if shutdown_fails:
                raise RuntimeError("shutdown failed")

    modules = {
        "vllm": {"SamplingParams": lambda **kwargs: object()},
        "vllm.engine": {},
        "vllm.engine.arg_utils": {"AsyncEngineArgs": EngineArgs},
        "vllm.v1": {},
        "vllm.v1.engine": {},
        "vllm.v1.engine.async_llm": {"AsyncLLM": LLM},
        "vllm_external_executor": {
            "ActorPoolManager": lambda: pool,
            "ExternalExecutor": object(),
        },
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__path__ = []
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    kwargs = {"pool": None if own_pool else pool, "cleanup_timeout": 60.0}
    if shutdown_fails:
        with pytest.raises(RuntimeError, match="shutdown failed"):
            mvp.run_mvp("test-model", [], **kwargs)
    else:
        assert mvp.run_mvp("test-model", [], **kwargs) == []
    expected = [("engine shutdown", 60.0), "release"]
    if own_pool:
        expected.append("pool shutdown")
    assert events == expected
    pool.release.assert_called_once_with(actors)
