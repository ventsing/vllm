# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pooled worker RPC output must be serializable without device events."""

import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.outputs import AsyncModelRunnerOutput

_PATH = (
    Path(__file__).resolve().parent.parent
    / "vllm_external_executor"
    / "external_worker_actor.py"
)
_spec = importlib.util.spec_from_file_location("external_worker_under_test", _PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class PendingOutput(AsyncModelRunnerOutput):
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def __reduce__(self):
        raise TypeError("cannot pickle 'Event' object")

    def get_output(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("device copy failed")
        return {"token_ids": [42]}


def make_actor(output):
    replies = []

    def enqueue(reply):
        replies.append(pickle.loads(pickle.dumps(reply)))

    actor = object.__new__(_module.ExternalWorkerActor)
    actor._rank = 0
    actor.worker = SimpleNamespace(sample_tokens=lambda: output)
    actor.worker_response_mq = SimpleNamespace(enqueue=enqueue)
    return actor, replies


@pytest.mark.parametrize("fail", [False, True])
def test_async_output_is_resolved_before_serializing_reply(fail):
    output = PendingOutput(fail=fail)
    actor, replies = make_actor(output)

    actor._execute_worker_rpc(("sample_tokens", (), {}, 0))

    assert output.calls == 1
    expected = (
        (WorkerProc.ResponseStatus.FAILURE, "device copy failed")
        if fail
        else (WorkerProc.ResponseStatus.SUCCESS, {"token_ids": [42]})
    )
    assert replies == [expected]


def test_non_output_rank_does_not_resolve_or_send_reply():
    output = PendingOutput()
    actor, replies = make_actor(output)

    actor._execute_worker_rpc(("sample_tokens", (), {}, 1))

    assert output.calls == 0
    assert replies == []


def test_reset_does_not_free_resources_while_worker_thread_is_alive():
    from threading import Event
    from unittest.mock import Mock

    actor = object.__new__(_module.ExternalWorkerActor)
    actor.state = _module.ActorState.RUNNING
    actor._stop_event = Event()
    actor._loop_thread = Mock()
    actor._loop_thread.is_alive.return_value = True
    actor._release_worker_resources = Mock()

    with pytest.raises(RuntimeError, match="Worker loop did not stop"):
        actor.reset()

    actor._release_worker_resources.assert_not_called()
    assert actor.state == _module.ActorState.FAILED
    with pytest.raises(RuntimeError, match="must be rebuilt"):
        actor.reset()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_reset_only_returns_to_idle_after_successful_cleanup(cleanup_fails):
    from threading import Event
    from unittest.mock import Mock

    actor = object.__new__(_module.ExternalWorkerActor)
    actor.state = _module.ActorState.RUNNING
    actor._stop_event = Event()
    actor._loop_thread = None
    actor._release_worker_resources = Mock(
        side_effect=RuntimeError("cleanup failed") if cleanup_fails else None
    )

    if cleanup_fails:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            actor.reset()
        assert actor.state == _module.ActorState.FAILED
    else:
        actor.reset()
        assert actor.state == _module.ActorState.IDLE
    actor._release_worker_resources.assert_called_once()


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_reset_collects_runner_and_layer_cache_before_device_cache_eviction(
    monkeypatch,
    device_type,
):
    """A pooled worker must release cyclic runners and config-owned KV caches."""
    import gc
    import weakref
    from threading import Event
    from unittest.mock import Mock

    from vllm.distributed import parallel_state
    from vllm.platforms import current_platform
    from vllm.v1.worker import workspace

    class Resource:
        def bytecode_hook(self, *args):
            pass

    config = SimpleNamespace(static_forward_context={})
    runner = Resource()
    runner.compilation_config = config
    runner.model = Resource()
    hooks = {0: runner.model.bytecode_hook, 1: object()}
    runner.model._bytecode_hook_handle = SimpleNamespace(
        remove=lambda: hooks.pop(0, None)
    )
    runner.kv_caches = [Resource()]
    config.static_forward_context["attention"] = runner.kv_caches[0]
    runner.cycle = runner
    refs = [weakref.ref(obj) for obj in (runner, runner.model, runner.kv_caches[0])]

    class Worker:
        device = "test-device"

        def __init__(self, model_runner):
            self.model_runner = model_runner

        def shutdown(self):
            assert self.model_runner.model is not None
            self.model_runner.model = None

    actor = object.__new__(_module.ExternalWorkerActor)
    actor.actor_id = "test-actor"
    actor.state = _module.ActorState.RUNNING
    actor._stop_event = Event()
    actor._loop_thread = None
    actor.worker = SimpleNamespace(worker=Worker(runner), device="test-device")
    actor.worker.shutdown = actor.worker.worker.shutdown
    actor.vllm_config = config
    actor.rpc_broadcast_mq = actor.worker_response_mq = None
    import sys
    from types import FunctionType, ModuleType

    class SerializableFunction(SimpleNamespace):
        pass

    # PyTorch AOT constructs a private globals copy, not the module dictionary.
    model_module = ModuleType("pooled_test_model")
    globals_copy = dict(vars(model_module))
    owned = SerializableFunction(
        example_inputs=[runner.model, runner.kv_caches[0]],
    )
    globals_copy["__compiled_fn_owned"] = owned
    foreign = object()
    globals_copy["__compiled_fn_foreign"] = foreign
    model_module.__compiled_fn_owned = foreign
    artifacts = SimpleNamespace(
        backend_id="__compiled_fn_owned",
        compiled_fn=owned,
        guard_manager=SimpleNamespace(model=runner.model),
    )
    runner.model.aot_compiled_fn = SimpleNamespace(
        fn=FunctionType((lambda: None).__code__, globals_copy),
        _artifacts=artifacts,
    )
    del owned
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    if device_type == "npu":
        graph_module = ModuleType("vllm_ascend.compilation.acl_graph")
        wrapper = SimpleNamespace(concrete_aclgraph_entries={1: runner.model})
        graph_module._acl_graph_wrappers = [wrapper]
        for name in (
            "_graph_params",
            "_draft_graph_params",
            "_draft_graph_prefill_params",
        ):
            setattr(
                graph_module,
                name,
                SimpleNamespace(
                    attn_params={1: list(runner.kv_caches)},
                    handles={},
                    events={},
                    workspaces={},
                ),
            )
        compiled = [runner]
        monkeypatch.setitem(sys.modules, graph_module.__name__, graph_module)
        monkeypatch.setattr(
            _module.torch, "_dynamo", SimpleNamespace(reset=compiled.clear)
        )
    del runner

    def evict_cache():
        assert all(ref() is None for ref in refs)

    device_module = SimpleNamespace(
        synchronize=Mock(),
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
        empty_cache=Mock(side_effect=evict_cache),
        mem_get_info=lambda: (100, 100),
    )
    monkeypatch.setattr(current_platform, "device_type", device_type)
    monkeypatch.setattr(current_platform, "set_device", lambda device: None)
    monkeypatch.setattr(_module.torch, device_type, device_module, raising=False)
    monkeypatch.setattr(
        parallel_state, "cleanup_dist_env_and_memory", lambda **kwargs: gc.collect()
    )
    monkeypatch.setattr(workspace, "reset_workspace_manager", lambda: None)

    actor.reset()

    assert actor.state == _module.ActorState.IDLE
    assert config.static_forward_context == {}
    assert set(hooks) == {1}
    assert "__compiled_fn_owned" not in globals_copy
    assert globals_copy["__compiled_fn_foreign"] is foreign
    assert model_module.__compiled_fn_owned is foreign
    assert artifacts.compiled_fn is None
    assert artifacts.guard_manager is None
    device_module.empty_cache.assert_called_once()
    if device_type == "npu":
        assert graph_module._graph_params is None
        assert graph_module._draft_graph_params is None
        assert graph_module._draft_graph_prefill_params is None
        assert wrapper.concrete_aclgraph_entries == {}
    actor.reset()
    assert actor.state == _module.ActorState.IDLE


def test_reset_diagnostics_identify_holder_without_keeping_model_alive():
    import gc
    import weakref
    from unittest.mock import Mock

    class Model:
        pass

    actor = object.__new__(_module.ExternalWorkerActor)
    actor.actor_id = "diagnostic-test"
    actor.worker = SimpleNamespace(
        worker=SimpleNamespace(model_runner=SimpleNamespace(model=Model()))
    )
    held = {"retained_model": actor.worker.worker.model_runner.model}
    model_ref = weakref.ref(held["retained_model"])
    refs = actor._capture_cleanup_refs()
    actor.worker = None
    logger = Mock()

    actor._log_cleanup_referrers(refs, logger)

    assert any("retained_model" in str(call) for call in logger.warning.call_args_list)
    assert len(logger.warning.call_args_list) <= 65
    held.clear()
    gc.collect()
    assert model_ref() is None


def test_reset_allocator_diagnostics_count_shared_storage_once(monkeypatch):
    import gc
    from unittest.mock import Mock

    storage = SimpleNamespace(data_ptr=lambda: 4096, nbytes=lambda: 8192)

    class Tensor:
        device = "npu:0"

        def untyped_storage(self):
            return storage

    tensors = [Tensor(), Tensor()]
    monkeypatch.setattr(_module.torch, "Tensor", Tensor, raising=False)
    monkeypatch.setattr(gc, "get_objects", lambda: tensors)
    actor = object.__new__(_module.ExternalWorkerActor)
    actor.actor_id = "storage-test"
    actor._log_cleanup_referrers = Mock()
    device = SimpleNamespace(
        synchronize=Mock(),
        empty_cache=Mock(),
        memory_stats=lambda: {},
        current_device=lambda: 0,
        memory_snapshot=lambda: [
            {"device": 0, "blocks": [{"state": "active_allocated", "size": 8192}]},
            {"device": 1, "blocks": [{"state": "active_allocated", "size": 9999}]},
        ],
    )
    logger = Mock()

    actor._log_allocator_diagnostics(device, "npu:0", logger)

    assert logger.warning.call_args_list[1].args[-1] == {"active_allocated": 8192}
    assert logger.warning.call_args_list[2].args[2:5] == (1, 8192, 0)
    assert len(actor._log_cleanup_referrers.call_args.args[0]) == 1


def test_reset_removes_nested_compile_hook_without_aot():
    import gc
    import weakref
    from unittest.mock import Mock

    class Layer:
        def bytecode_hook(self, *args):
            pass

    layer = Layer()
    hooks = {0: layer.bytecode_hook, 1: object()}
    layer._bytecode_hook_handle = SimpleNamespace(remove=lambda: hooks.pop(0, None))
    ref = weakref.ref(layer)
    model = SimpleNamespace()
    model.modules = lambda model=model, layer=layer: iter((model, layer))
    actor = object.__new__(_module.ExternalWorkerActor)
    actor.actor_id = "non-aot-test"
    actor.worker = SimpleNamespace(
        worker=SimpleNamespace(model_runner=SimpleNamespace(model=model))
    )

    actor._release_compiled_functions(Mock())

    assert set(hooks) == {1}
    assert layer._bytecode_hook_handle is None
    actor.worker = None
    del model, layer
    gc.collect()
    assert ref() is None
