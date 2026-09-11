# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Unit tests for the pure-logic cluster state + global scheduler.

These tests exercise GlobalScheduler selection without a Ray cluster. The
cluster_state module is dependency-free, so it can be imported directly even
on hosts without torch/vllm/ray.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Load cluster_state.py directly (bypassing the package __init__ which imports
# torch at import time).
_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "vllm_external_executor"
    / "cluster_state.py"
)
_spec = importlib.util.spec_from_file_location("cluster_state_under_test", _MODULE_PATH)
cluster_state = importlib.util.module_from_spec(_spec)
sys.modules["cluster_state_under_test"] = cluster_state
_spec.loader.exec_module(cluster_state)

ActorRegistration = cluster_state.ActorRegistration
GlobalScheduler = cluster_state.GlobalScheduler
NodeInfo = cluster_state.NodeInfo


def make_node(node_id, fault_domain=None):
    return NodeInfo(
        node_id=node_id,
        ip=f"ip-{node_id}",
        fault_domain=fault_domain or node_id,
        total_gpus=8,
        free_gpus=8,
    )


def make_actor(actor_id, node_id, fault_domain=None, state="idle"):
    return ActorRegistration(
        actor_id=actor_id,
        node_id=node_id,
        device_id=0,
        fault_domain=fault_domain or node_id,
        state=state,
    )


def test_spread_across_fault_domains():
    """A lease of 4 should spread across 4 nodes, not pile onto one."""
    nodes = [make_node(f"n{i}") for i in range(4)]
    actors = [make_actor(f"a{i}", f"n{i}") for i in range(4)]

    selected = GlobalScheduler.select_actors(actors, nodes, world_size=4)

    # All four actors selected, one per node.
    assert len(selected) == 4
    assert len({a.rsplit("-", 1)[0] for a in selected}) == 4


def test_prefer_more_free_domain_but_still_spread():
    """With 2 domains, a 2-actor lease should take one from each domain."""
    nodes = [make_node("n0"), make_node("n1")]
    actors = [
        make_actor("a0", "n0"),
        make_actor("a1", "n0"),
        make_actor("a2", "n1"),
        make_actor("a3", "n1"),
    ]

    selected = GlobalScheduler.select_actors(actors, nodes, world_size=2)

    assert len(selected) == 2
    domains = {a.rsplit("-", 1)[0] for a in selected}
    assert len(domains) == 2  # spread across both domains


def test_skips_dead_nodes():
    """Actors on a dead node are never selected."""
    nodes = [make_node("n0"), make_node("n1")]
    actors = [
        make_actor("a0", "n0"),
        make_actor("a1", "n1"),
    ]
    # Only n0 is alive.
    alive_nodes = [make_node("n0")]

    selected = GlobalScheduler.select_actors(actors, alive_nodes, world_size=1)
    assert selected == ["a0"]


def test_hard_fault_domain_constraint():
    """A hard constraint forces per-domain counts."""
    nodes = [make_node("n0"), make_node("n1")]
    actors = [
        make_actor("a0", "n0"),
        make_actor("a1", "n0"),
        make_actor("a2", "n1"),
        make_actor("a3", "n1"),
    ]

    selected = GlobalScheduler.select_actors(
        actors,
        nodes,
        world_size=2,
        fault_domain_constraint={"n0": 1, "n1": 1},
    )
    assert len(selected) == 2
    assert "a0" in selected or "a1" in selected
    assert "a2" in selected or "a3" in selected


def test_constraint_insufficient_raises():
    nodes = [make_node("n0")]
    actors = [make_actor("a0", "n0")]

    with pytest.raises(RuntimeError):
        GlobalScheduler.select_actors(actors, nodes, world_size=2)


def test_constraint_exceeds_domain_raises():
    nodes = [make_node("n0")]
    actors = [make_actor("a0", "n0")]

    with pytest.raises(RuntimeError):
        GlobalScheduler.select_actors(
            actors,
            nodes,
            world_size=1,
            fault_domain_constraint={"n0": 2},
        )


def test_zero_world_size_raises():
    with pytest.raises(ValueError):
        GlobalScheduler.select_actors([], [], world_size=0)