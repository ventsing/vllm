# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""Unit tests for weight_sharing.WeightShareLedger (pure logic)."""

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


ws = _load_module(
    "vllm_external_executor.weight_sharing",
    _EXEC_DIR / "weight_sharing.py",
)
WeightShareLedger = ws.WeightShareLedger


def test_register_creates_holder_and_ref():
    ledger = WeightShareLedger()
    lease = ledger.register("a0", "base-hash", "adapter-1")

    assert lease.base_hash == "base-hash"
    assert lease.adapter_config == "adapter-1"
    assert lease.holders == {"a0"}
    assert lease.refs == 1


def test_repeat_register_dedupes_holder_but_accumulates_ref():
    ledger = WeightShareLedger()
    ledger.register("a0", "base-hash")
    ledger.register("a0", "base-hash")  # same actor, same combination

    assert ledger.holders("base-hash") == {"a0"}
    assert ledger._by_base["base-hash"][None].refs == 2


def test_holders_returns_copy_isolation():
    ledger = WeightShareLedger()
    ledger.register("a0", "base-hash")

    holders = ledger.holders("base-hash")
    holders.add("a1")  # mutating the copy must not corrupt the ledger

    assert ledger.holders("base-hash") == {"a0"}


def test_unregister_drops_holder_and_returns_remaining():
    ledger = WeightShareLedger()
    ledger.register("a0", "base-hash")
    ledger.register("a1", "base-hash")

    assert ledger.unregister("a0", "base-hash") == 1
    assert ledger.holders("base-hash") == {"a1"}


def test_unregister_last_holder_cleans_combination():
    ledger = WeightShareLedger()
    ledger.register("a0", "base-hash")

    assert ledger.unregister("a0", "base-hash") == 0
    assert ledger.holders("base-hash") == set()
    assert ledger._by_base == {}


def test_holders_unknown_combination_is_empty():
    ledger = WeightShareLedger()

    assert ledger.holders("missing-base") == set()


def test_cost_ratio_quantifies_adapter_win():
    ledger = WeightShareLedger()

    # 1% adapter vs 100% base -> 0.01 means an adapter-only switch is ~100x
    # cheaper than a full reload.
    assert ledger.cost_ratio(base_bytes=1000, adapter_bytes=10) == pytest.approx(
        0.01
    )


def test_cost_ratio_rejects_non_positive_base():
    ledger = WeightShareLedger()

    with pytest.raises(ValueError):
        ledger.cost_ratio(base_bytes=0, adapter_bytes=10)