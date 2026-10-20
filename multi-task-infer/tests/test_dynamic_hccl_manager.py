# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Unit tests for dynamic_hccl_manager.plan_tp_partition.

The partition planner is pure logic (no torch/ray), so it is loaded directly
from the file and exercised without a device.
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


dhm = _load_module(
    "dynamic_hccl_manager_under_test",
    _EXEC_DIR / "dynamic_hccl_manager.py",
)
plan = dhm.plan_tp_partition


def test_tp2_partition_of_world8_is_contiguous_pairs():
    assert plan(8, 2) == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_tp4_partition_of_world8_is_two_quads():
    assert plan(8, 4) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_tp8_covers_the_whole_world():
    assert plan(8, 8) == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_tp1_partition_is_singletons():
    assert plan(8, 1) == [[0], [1], [2], [3], [4], [5], [6], [7]]


def test_world16_tp8_partition():
    assert plan(16, 8) == [list(range(8)), list(range(8, 16))]


@pytest.mark.parametrize(
    ("world_size", "tp_size"),
    [(8, 0), (8, 3), (0, 2), (-1, 2)],
)
def test_invalid_partition_raises(world_size, tp_size):
    with pytest.raises(ValueError):
        plan(world_size, tp_size)
