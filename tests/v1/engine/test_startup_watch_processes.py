# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from multiprocessing import connection
from threading import Event
from types import SimpleNamespace

import pytest
import zmq

import vllm.platforms as platforms
from vllm.v1.engine import core as core_module
from vllm.v1.engine import utils as engine_utils
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    CoreEngine,
    CoreEngineLaunch,
    CoreEngineProcManager,
    EngineZmqAddresses,
    wait_for_engine_startup,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("is_rocm", "request_timeout", "manager_timeout", "process_timeout"),
    [
        (True, 0, 0, 15.0),
        (True, 0, 7, 7),
        (True, 0, None, None),
        (False, 0, 0, 0),
        (True, 7, 0, 0),
    ],
)
def test_engine_core_process_shutdown_timeout(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    request_timeout: float | None,
    manager_timeout: float | None,
    process_timeout: float | None,
):
    manager = object.__new__(CoreEngineProcManager)
    manager._request_shutdown_timeout = request_timeout
    manager.manager_stopped = Event()
    manager.processes = [object()]
    detach_results = iter((object(), None))
    manager._finalizer = SimpleNamespace(detach=lambda: next(detach_results))

    shutdown_calls = []
    monkeypatch.setattr(
        engine_utils,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: is_rocm),
    )
    monkeypatch.setattr(
        engine_utils,
        "shutdown",
        lambda processes, timeout: shutdown_calls.append((processes, timeout)),
    )

    manager.shutdown(timeout=manager_timeout)
    manager.shutdown(timeout=manager_timeout)

    assert manager.manager_stopped.is_set()
    assert shutdown_calls == [(manager.processes, process_timeout)]


@pytest.mark.parametrize(
    (
        "is_rocm",
        "shutdown_state",
        "has_work",
        "shutdown_timeout",
        "exit_code",
        "expected_calls",
    ),
    [
        (
            True,
            EngineShutdownState.SHUTTING_DOWN,
            False,
            0,
            None,
            ["shutdown", "freeze"],
        ),
        (False, EngineShutdownState.SHUTTING_DOWN, False, 0, None, ["shutdown"]),
        (True, EngineShutdownState.RUNNING, False, 0, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, True, 0, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, False, 7, None, ["shutdown"]),
        (True, EngineShutdownState.SHUTTING_DOWN, False, 0, 1, ["shutdown"]),
    ],
)
def test_freeze_gc_after_clean_rocm_engine_core_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    shutdown_state: EngineShutdownState,
    has_work: bool,
    shutdown_timeout: int,
    exit_code: int | None,
    expected_calls: list[str],
):
    calls: list[str] = []
    vllm_config = SimpleNamespace(shutdown_timeout=shutdown_timeout)
    proc = SimpleNamespace(
        shutdown_state=EngineShutdownState.RUNNING,
        has_work=lambda: has_work,
        vllm_config=vllm_config,
    )

    def run_busy_loop():
        proc.shutdown_state = shutdown_state
        raise SystemExit(exit_code)

    proc.run_busy_loop = run_busy_loop
    proc.shutdown = lambda: calls.append("shutdown")
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        numa_bind=False,
        reconfigure_for_independent_dp_rank=lambda: None,
    )
    vllm_config.parallel_config = parallel_config

    for name in (
        "maybe_register_config_serialize_by_value",
        "set_process_title",
        "maybe_init_worker_tracer",
        "decorate_logs",
    ):
        monkeypatch.setattr(core_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(core_module, "EngineCoreProc", lambda *args, **kwargs: proc)
    monkeypatch.setattr(
        core_module,
        "SignalCallback",
        lambda callback: SimpleNamespace(trigger=lambda: None, stop=lambda: None),
    )
    monkeypatch.setattr(core_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_rocm=lambda: is_rocm)
    )
    monkeypatch.setattr(core_module.gc, "freeze", lambda: calls.append("freeze"))

    with pytest.raises(SystemExit):
        EngineCoreProc.run_engine_core(vllm_config=vllm_config)

    assert calls == expected_calls


class _FinishedProcess:
    name = "RustFrontend"

    def __init__(self, sentinel):
        self.sentinel = sentinel

    @property
    def exitcode(self):
        return 1


def test_wait_for_engine_startup_reports_watched_process_exit():
    ctx = zmq.Context()
    handshake_socket = ctx.socket(zmq.ROUTER)
    recv, send = connection.Pipe(duplex=False)
    send.close()

    parallel_config = SimpleNamespace(
        data_parallel_size_local=1,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
    )

    try:
        launch = CoreEngineLaunch(
            engine_manager=None,
            coordinator=None,
            addresses=EngineZmqAddresses(inputs=[], outputs=[]),
            tensor_queue=None,
        )
        launch.watched_frontend_processes = [_FinishedProcess(recv)]
        with pytest.raises(RuntimeError) as exc_info:
            wait_for_engine_startup(
                handshake_socket,
                [CoreEngine()],
                parallel_config,  # type: ignore[arg-type]
                coordinated_dp=False,
                cache_config=None,  # type: ignore[arg-type]
                launch=launch,
            )
    finally:
        recv.close()
        handshake_socket.close(linger=0)
        ctx.term()

    assert "Frontend process failed during engine core initialization" in str(
        exc_info.value
    )
    assert "Failed frontend proc(s): {'RustFrontend': 1}" in str(exc_info.value)


@pytest.mark.parametrize("connected", [False, True])
def test_external_actor_handles_deserialize_after_ray_connect(monkeypatch, connected):
    """A spawned engine must join Ray before unpickling pooled actor handles."""
    import sys

    events: list[str] = []
    actors = [object()]

    def init(**kwargs):
        nonlocal connected
        assert kwargs == {"address": "cluster:6925", "namespace": "pool"}
        connected = True
        events.append("connect")

    def loads(payload):
        assert connected
        assert payload == b"actor-handles"
        events.append("deserialize")
        return actors

    def run(**kwargs):
        assert kwargs == {"external_actors": actors, "dp_rank": 0}
        events.append("run")

    expected = ([] if connected else ["connect"]) + ["deserialize", "run"]
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: connected,
            init=init,
            cloudpickle=SimpleNamespace(loads=loads),
        ),
    )
    monkeypatch.setattr(EngineCoreProc, "run_engine_core", run)
    engine_utils._run_engine_core_with_external_actors(
        external_actors_payload=b"actor-handles",
        ray_address="cluster:6925",
        ray_namespace="pool",
        dp_rank=0,
    )
    assert events == expected


@pytest.mark.parametrize("cancel", [False, True])
def test_prewarmed_core_waits_for_configuration_and_honors_cancellation(
    monkeypatch, cancel
):
    import sys

    from vllm import envs, plugins
    from vllm.v1.engine import process_pool

    events: list[str] = []

    def execute(**kwargs):
        assert kwargs == {"configuration": "assigned"}
        assert events[-1] == "close"
        events.append("engine")

    def receive():
        assert "ready" in events and "engine" not in events
        events.append("receive")
        return b"" if cancel else b"configuration"

    channel = SimpleNamespace(
        send=lambda message: events.append(message[0]),
        recv_bytes=receive,
        close=lambda: events.append("close"),
    )
    monkeypatch.setattr(
        process_pool.importlib, "import_module", lambda name: events.append(name)
    )
    monkeypatch.setattr(
        plugins, "load_general_plugins", lambda: events.append("plugins")
    )
    monkeypatch.setattr(envs, "disable_envs_cache", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            init=lambda **kwargs: events.append("ray"),
            cloudpickle=SimpleNamespace(
                loads=lambda payload: (execute, {"configuration": "assigned"}, {})
            ),
        ),
    )
    process_pool._warm_engine_entry(channel, "cluster", "pool", ("extra.module",))

    assert events[:4] == ["vllm.v1.engine.core", "plugins", "extra.module", "ray"]
    assert events[-1] == ("close" if cancel else "engine")


def test_prepared_core_is_single_use_and_preserves_process_monitoring(monkeypatch):
    from unittest.mock import Mock

    from vllm.v1.engine import process_pool

    monkeypatch.setattr(
        process_pool,
        "get_mp_context",
        lambda: SimpleNamespace(get_start_method=lambda: "spawn"),
    )
    pool = process_pool.EngineCoreProcessPool(1, "cluster", "pool")
    process = SimpleNamespace(pid=1234, sentinel=42, is_alive=lambda: True)
    channel = SimpleNamespace(send_bytes=Mock(), close=Mock())
    pool._ready.append((process, channel))
    monkeypatch.setattr(pool, "_replenish", Mock())

    prepared = pool.prepare(
        lambda: None, "EngineCore", {"ray_address": "cluster", "ray_namespace": "pool"}
    )
    assert prepared.pid == 1234 and prepared.sentinel == 42
    assert pool.idle_pids() == []
    channel.send_bytes.assert_not_called()
    prepared.start()
    with pytest.raises(RuntimeError, match="only start once"):
        prepared.start()
    channel.send_bytes.assert_called_once()
    channel.close.assert_called_once()
    pool._replenish.assert_called_once()


def test_prewarmed_core_rejects_environment_changes_before_leasing(monkeypatch):
    from vllm.v1.engine import process_pool

    monkeypatch.setattr(
        process_pool,
        "get_mp_context",
        lambda: SimpleNamespace(get_start_method=lambda: "spawn"),
    )
    pool = process_pool.EngineCoreProcessPool(1, "cluster", "pool")
    monkeypatch.setenv("VLLM_PLUGINS", "different-plugins")
    with pytest.raises(RuntimeError, match="environment changed"):
        pool.prepare(lambda: None, "EngineCore", {})
