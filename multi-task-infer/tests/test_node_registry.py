# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Unit tests for NodeRegistryActor's node-level liveness bookkeeping.

The registry delegates scheduling policy to ``cluster_state.GlobalScheduler``
but owns the liveness records that node-level failover relies on: node
heartbeat refresh, dead-node detection, and ``mark_node_failed`` returning the
lost actor ids. These run without Ray (the registry only imports Ray inside
``create_registry_actor``).
"""

import importlib.util
import sys
import time
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


# Register a stub package so `from vllm_external_executor.cluster_state import`
# inside node_registry_actor.py resolves to a module loaded directly from disk
# (bypassing vllm_external_executor/__init__.py, which imports torch).
sys.modules.setdefault(
    "vllm_external_executor", types.ModuleType("vllm_external_executor")
)
_load_module(
    "vllm_external_executor.cluster_state",
    _EXEC_DIR / "cluster_state.py",
)
registry_mod = _load_module(
    "vllm_external_executor.node_registry_actor",
    _EXEC_DIR / "node_registry_actor.py",
)

NodeRegistryActor = registry_mod.NodeRegistryActor


def make_registry():
    reg = NodeRegistryActor()
    reg.register_node("n0", ip="10.0.0.1", total_gpus=2)
    reg.register_node("n1", ip="10.0.0.2", total_gpus=2)
    reg.register_actor("n0-g0", "n0", 0)
    reg.register_actor("n0-g1", "n0", 1)
    reg.register_actor("n1-g0", "n1", 0)
    return reg


def test_node_heartbeat_keeps_node_alive():
    """A heartbeat refreshes last_heartbeat so a node is not reported dead."""
    reg = make_registry()

    assert reg.detect_dead_nodes(timeout=1.0) == []
    # Age the record past the timeout, then refresh it.
    reg._nodes["n0"].last_heartbeat = time.time() - 10.0
    reg.node_heartbeat("n0")
    assert reg.detect_dead_nodes(timeout=1.0) == []


def test_detect_dead_nodes_by_staleness():
    """A node whose heartbeat stalls past timeout is reported dead."""
    reg = make_registry()
    reg._nodes["n0"].last_heartbeat = time.time() - 100.0

    assert reg.detect_dead_nodes(timeout=30.0) == ["n0"]


def test_mark_node_failed_returns_lost_actor_ids():
    """mark_node_failed evicts the node and returns its actor ids for rebuild."""
    reg = make_registry()

    lost = reg.mark_node_failed("n0")

    assert sorted(lost) == ["n0-g0", "n0-g1"]
    assert "n0" not in {n.node_id for n in reg.list_nodes()}
    assert {a.actor_id for a in reg.list_actors()} == {"n1-g0"}


def test_unregister_node_returns_actor_ids():
    """Unregistering a node also returns its actor ids."""
    reg = make_registry()

    lost = reg.unregister_node("n0")

    assert sorted(lost) == ["n0-g0", "n0-g1"]
    assert "n0" not in {n.node_id for n in reg.list_nodes()}


def test_actor_failure_does_not_kill_node():
    """A single actor failure leaves the node and its other actor intact."""
    reg = make_registry()

    reg.mark_actor_failed("n0-g0")

    assert "n0" in {n.node_id for n in reg.list_nodes()}
    remaining = {a.actor_id for a in reg.list_actors()}
    assert "n0-g0" not in remaining
    assert "n0-g1" in remaining


def test_free_gpus_tracks_idle_actor_count():
    """free_gpus reflects idle actors, not total registered actors."""
    reg = make_registry()
    reg.set_actor_state("n0-g0", "leased", lease_id="l1")
    reg.set_actor_state("n0-g1", "leased", lease_id="l1")

    view = reg.get_global_view()
    by_id = {n["node_id"]: n for n in view["nodes"]}
    assert by_id["n0"]["free_gpus"] == 0
    assert by_id["n1"]["free_gpus"] == 2


def test_detect_dead_actors_uses_actor_heartbeat():
    """Actor liveness is tracked independently of node liveness."""
    reg = make_registry()
    reg._actors["n0-g0"].last_heartbeat = time.time() - 100.0

    assert reg.detect_dead_actors(timeout=30.0) == ["n0-g0"]
    # Node liveness is separate and still fresh.
    assert reg.detect_dead_nodes(timeout=30.0) == []


# ----------------------------------------------------------------- lease logic
def _by_id(reg):
    return {a.actor_id: a for a in reg.list_actors()}


def test_try_acquire_atomic_no_partial_lease():
    """A shortfall leases nothing (no partial lease left behind)."""
    reg = make_registry()

    granted = reg.try_acquire(4, "l1")  # only 3 actors exist

    assert granted == []
    assert all(a.state == "idle" for a in reg.list_actors())


def test_try_acquire_grants_and_bumps_generation():
    """try_acquire marks selected actors leased and bumps their generation."""
    reg = make_registry()

    granted = reg.try_acquire(2, "l1")

    assert sorted(granted) == ["n0-g0", "n1-g0"]  # spread across fault domains
    regs = _by_id(reg)
    assert regs["n0-g0"].state == "leased"
    assert regs["n0-g0"].lease_id == "l1"
    assert regs["n0-g0"].lease_generation == 1
    # The unselected actor stays idle.
    assert regs["n0-g1"].state == "idle"
    assert regs["n0-g1"].lease_id is None


def test_heartbeat_liveness_only_does_not_flip_lease():
    """A heartbeat refreshing liveness never mutates lease/state."""
    reg = make_registry()
    reg.try_acquire(1, "l1")  # leases n0-g0

    assert reg.heartbeat("n0-g0") is True

    regs = _by_id(reg)
    assert regs["n0-g0"].state == "leased"
    assert regs["n0-g0"].lease_id == "l1"
    assert regs["n0-g0"].lease_generation == 1


def test_release_requires_matching_lease():
    """Releasing with the wrong lease id leaves the actor leased."""
    reg = make_registry()
    reg.try_acquire(1, "l1")

    released = reg.release_actors(["n0-g0"], "wrong-lease")

    assert released == []
    assert _by_id(reg)["n0-g0"].state == "leased"


def test_stale_release_cannot_disturb_new_lease():
    """A delayed release from task A cannot free an actor re-leased to B."""
    reg = make_registry()
    reg.try_acquire(1, "task-A")
    reg.release_actors(["n0-g0"], "task-A")
    reg.try_acquire(1, "task-B")  # n0-g0 is now leased to task-B

    reg.release_actors(["n0-g0"], "task-A")  # stale duplicate from A

    regs = _by_id(reg)
    assert regs["n0-g0"].state == "leased"
    assert regs["n0-g0"].lease_id == "task-B"
    assert regs["n0-g0"].lease_generation == 2


def test_duplicate_release_is_noop():
    """Releasing an already-idle actor is a no-op."""
    reg = make_registry()
    reg.try_acquire(1, "l1")
    reg.release_actors(["n0-g0"], "l1")

    released_again = reg.release_actors(["n0-g0"], "l1")

    assert released_again == []
    assert _by_id(reg)["n0-g0"].state == "idle"


def test_release_restores_free_gpu_count():
    """Leasing and releasing move free_gpus in lockstep."""
    reg = make_registry()
    reg.try_acquire(2, "l1")
    view = reg.get_global_view()
    by_node = {n["node_id"]: n for n in view["nodes"]}
    assert by_node["n0"]["free_gpus"] == 1  # 2 total, 1 leased
    assert by_node["n1"]["free_gpus"] == 0  # 1 total, 1 leased

    reg.release_actors(["n0-g0", "n1-g0"], "l1")
    view = reg.get_global_view()
    by_node = {n["node_id"]: n for n in view["nodes"]}
    assert by_node["n0"]["free_gpus"] == 2
    assert by_node["n1"]["free_gpus"] == 1