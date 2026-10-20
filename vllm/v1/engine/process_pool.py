# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pre-imported, single-use EngineCore processes for external worker pools."""

import contextlib
import contextvars
import importlib
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, cast

from vllm.utils.system_utils import get_mp_context

_active_pool: contextvars.ContextVar["EngineCoreProcessPool | None"] = (
    contextvars.ContextVar("engine_core_process_pool", default=None)
)
_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "VLLM_PLUGINS",
    "VLLM_VERSION",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
)


def get_active_engine_core_pool() -> "EngineCoreProcessPool | None":
    return _active_pool.get()


def _warm_engine_entry(
    channel: Connection,
    ray_address: str,
    ray_namespace: str,
    preload_modules: tuple[str, ...],
) -> None:
    try:
        import ray
        from ray import cloudpickle

        from vllm import envs
        from vllm.plugins import load_general_plugins

        importlib.import_module("vllm.v1.engine.core")
        load_general_plugins()
        for module in preload_modules:
            importlib.import_module(module)
        ray.init(address=ray_address, namespace=ray_namespace)
        channel.send(("ready", os.getpid()))
        payload = channel.recv_bytes()
        if not payload:
            return
        target, kwargs, environment = cloudpickle.loads(payload)
        os.environ.update(environment)
        envs.disable_envs_cache()
    except EOFError:
        return
    except Exception as exc:
        with contextlib.suppress(BrokenPipeError, OSError):
            channel.send(("error", repr(exc)))
        raise
    finally:
        channel.close()
    target(**kwargs)


class _PreparedEngineProcess:
    def __init__(
        self,
        process: BaseProcess,
        channel: Connection,
        payload: bytes,
        replenish: Callable[[], None],
    ) -> None:
        self._process = process
        self._channel = channel
        self._payload: bytes | None = payload
        self._replenish = replenish
        self._started = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._process, name)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("A prepared EngineCore process can only start once")
        self._started = True
        try:
            assert self._payload is not None
            self._channel.send_bytes(self._payload)
        finally:
            self._channel.close()
            self._payload = None
            self._replenish()


class EngineCoreProcessPool:
    """Keep spawned processes ready to accept one EngineCore configuration.

    Processes import EngineCore and plugins and connect to Ray before entering
    the idle queue. Each lease owns one process for one engine lifetime; normal
    EngineCore shutdown remains unchanged. A replacement warms in the background.
    The pool is local to its owning driver and requires external worker actors,
    DP=1, no NUMA binding, and no multiprocessing tensor queue.

    Args:
        size: Number of idle prewarmed processes to maintain.
        ray_address: Ray cluster address shared with the external worker pool.
        ray_namespace: Namespace used by the driver's Ray connection.
        timeout: Maximum wait for prewarming or acquiring a ready process.
        preload_modules: Additional modules imported before readiness.
    """

    def __init__(
        self,
        size: int,
        ray_address: str,
        ray_namespace: str,
        timeout: float = 180.0,
        preload_modules: tuple[str, ...] = (),
    ) -> None:
        if size < 1 or timeout <= 0:
            raise ValueError("size and timeout must be positive")
        self._context = get_mp_context()
        if self._context.get_start_method() != "spawn":
            raise ValueError("EngineCore prewarming requires the spawn start method")
        self.size = size
        self.ray_address = ray_address
        self.ray_namespace = ray_namespace
        self.timeout = timeout
        self.preload_modules = preload_modules
        self._environment = {key: os.environ.get(key) for key in _ENVIRONMENT_KEYS}
        self._condition = threading.Condition()
        self._ready: deque[tuple[BaseProcess, Connection]] = deque()
        self._pending: dict[int, tuple[BaseProcess, Connection]] = {}
        self._closed = False
        self._started = False
        self._error: Exception | None = None

    def _add_ready(self) -> None:
        process = channel = None
        try:
            with self._condition:
                if self._closed:
                    return
                channel, child_channel = self._context.Pipe()
                process = self._context.Process(
                    target=_warm_engine_entry,
                    name="EngineCoreWarm",
                    kwargs={
                        "channel": child_channel,
                        "ray_address": self.ray_address,
                        "ray_namespace": self.ray_namespace,
                        "preload_modules": self.preload_modules,
                    },
                )
                try:
                    process.start()
                finally:
                    child_channel.close()
                assert process.pid is not None
                self._pending[process.pid] = (process, channel)
            if not channel.poll(self.timeout):
                raise TimeoutError("EngineCore prewarming timed out")
            status, detail = channel.recv()
            if status != "ready":
                raise RuntimeError(f"EngineCore prewarming failed: {detail}")
            with self._condition:
                self._pending.pop(process.pid, None)
                if not self._closed:
                    self._ready.append((process, channel))
                    self._condition.notify_all()
                    return
        except Exception as exc:
            with self._condition:
                if process is not None and process.pid is not None:
                    self._pending.pop(process.pid, None)
                if not self._closed:
                    self._error = exc
                self._condition.notify_all()
        if channel is not None:
            channel.close()
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    def _replenish(self) -> None:
        threading.Thread(target=self._add_ready, daemon=True).start()

    def pre_start(self) -> None:
        with self._condition:
            if self._started or self._closed:
                raise RuntimeError("EngineCore pool cannot be started twice")
            self._started = True
        for _ in range(self.size):
            self._replenish()
        try:
            self.wait_ready(self.size)
        except Exception:
            self.close()
            raise

    def wait_ready(self, count: int = 1) -> None:
        if not 1 <= count <= self.size:
            raise ValueError("count must be within pool capacity")
        deadline = time.monotonic() + self.timeout
        with self._condition:
            while len(self._ready) < count:
                if self._closed:
                    raise RuntimeError("EngineCore pool is closed")
                if self._error is not None:
                    raise RuntimeError("EngineCore prewarming failed") from self._error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No prewarmed EngineCore process available")
                self._condition.wait(remaining)

    def prepare(self, target: Callable, name: str, kwargs: dict) -> BaseProcess:
        from ray import cloudpickle

        if any(os.environ.get(k) != v for k, v in self._environment.items()):
            raise RuntimeError("EngineCore pool environment changed after prewarming")
        if (kwargs.get("ray_address"), kwargs.get("ray_namespace")) != (
            self.ray_address,
            self.ray_namespace,
        ):
            raise ValueError("EngineCore and worker pools must share a Ray connection")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("VLLM_") or key == "PYTORCH_NPU_ALLOC_CONF"
        }
        payload = cloudpickle.dumps((target, kwargs, environment))
        with self._condition:
            self.wait_ready()
            if self._closed:
                raise RuntimeError("EngineCore pool is closed")
            process, channel = self._ready.popleft()
        if not process.is_alive():
            channel.close()
            self._replenish()
            raise RuntimeError("Prewarmed EngineCore process died before acquisition")
        process.name = name
        return cast(
            BaseProcess,
            _PreparedEngineProcess(process, channel, payload, self._replenish),
        )

    @contextlib.contextmanager
    def activate(self) -> Iterator["EngineCoreProcessPool"]:
        """Use this pool for EngineCore launches in the current context."""
        token = _active_pool.set(self)
        try:
            yield self
        finally:
            _active_pool.reset(token)

    def idle_pids(self) -> list[int]:
        with self._condition:
            return [
                process.pid
                for process, _ in self._ready
                if process.is_alive() and process.pid is not None
            ]

    def close(self) -> None:
        """Stop idle warm processes; leased engines retain their own managers."""
        with self._condition:
            self._closed = True
            slots = list(self._ready) + list(self._pending.values())
            self._ready.clear()
            self._condition.notify_all()
        for process, channel in slots:
            with contextlib.suppress(BrokenPipeError, OSError):
                channel.send_bytes(b"")
            channel.close()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
