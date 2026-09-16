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
