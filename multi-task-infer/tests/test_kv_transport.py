# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Unit tests for the KV transport abstraction (pure-logic + mocked ray).

``kv_transport.py`` imports torch/ray only lazily inside ``ship``, so the key
protocol, id remapping and the factory run offline. The relay orchestration is
exercised with a fake ``ray.get`` and stub actors instead of a live cluster.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "vllm_external_executor"
    / "kv_transport.py"
)
_spec = importlib.util.spec_from_file_location("kv_transport_under_test", _MODULE_PATH)
kvt = importlib.util.module_from_spec(_spec)
sys.modules["kv_transport_under_test"] = kvt
_spec.loader.exec_module(kvt)

KVBlockKey = kvt.KVBlockKey
KVTransport = kvt.KVTransport
KVTransportFactory = kvt.KVTransportFactory
RayObjectStoreTransport = kvt.RayObjectStoreTransport
remap_payload_block_ids = kvt.remap_payload_block_ids
resolve_dst_block_id = kvt.resolve_dst_block_id


def test_resolve_dst_block_id():
    """Identity when unmapped; explicit id otherwise."""
    assert resolve_dst_block_id(5, None) == 5
    assert resolve_dst_block_id(5, {}) == 5
    assert resolve_dst_block_id(5, {5: 9}) == 9
    assert resolve_dst_block_id(7, {5: 9}) == 7


def test_remap_payload_block_ids_in_place():
    """The relay rewrites each payload's block_id to its destination id."""
    payloads = [
        {"block_id": 5, "shards": {}},
        {"block_id": 7, "shards": {}},
    ]

    remap_payload_block_ids(payloads, {5: 9})

    assert payloads[0]["block_id"] == 9
    assert payloads[1]["block_id"] == 7


def test_kv_block_key_shape():
    """The RDMA locator carries source id + group count for the receiver."""
    key = KVBlockKey(store_key="k", block_id=3, num_groups=2)
    assert (key.store_key, key.block_id, key.num_groups) == ("k", 3, 2)


def test_factory_known_and_unknown():
    """Known names instantiate; unknown names raise ValueError."""
    assert KVTransportFactory.create("ray_object_store").name == "ray_object_store"
    assert KVTransportFactory.create("mooncake_rdma").name == "mooncake_rdma"
    with pytest.raises(ValueError):
        KVTransportFactory.create("no_such_transport")


def test_factory_register_custom():
    """A registered backend is creatable by name."""
    class Dummy(KVTransport):
        name = "dummy"

        def ship(self, *args, **kwargs):
            return 0

    KVTransportFactory.register("dummy", Dummy)
    assert KVTransportFactory.create("dummy").name == "dummy"


# --------------------------------------------------------------------------- relay
class _RemoteCall:
    """Stands in for a Ray ObjectRef (resolved lazily)."""

    def __init__(self, fn, args):
        self._fn = fn
        self._args = args
        self._resolved = False
        self._result = None

    def resolve(self):
        if not self._resolved:
            self._result = self._fn(*self._args)
            self._resolved = True
        return self._result


class _BoundMethod:
    def __init__(self, actor, name):
        self._actor = actor
        self._name = name

    def remote(self, *args):
        return _RemoteCall(getattr(self._actor, self._name), args)


class _StubActor:
    """Minimal worker actor: exports then records what it imports.

    The transfer methods resolve through ``__getattr__`` to the ``remote()``
    stub, so the actor exposes the same ``.remote()`` calling convention as a
    real Ray actor.
    """

    _TransferMethods = {
        "export_kv_blocks": "_export",
        "import_kv_blocks": "_import",
        "export_kv_blocks_to_store": "_export",
        "import_kv_blocks_from_store": "_import",
    }

    def __init__(self):
        self.imported = None

    def __getattr__(self, name):
        target = self._TransferMethods.get(name)
        if target is not None:
            return _BoundMethod(self, target)
        raise AttributeError(name)

    def _export(self, block_ids):
        return [{"block_id": i, "shards": {0: i}} for i in block_ids]

    def _import(self, payloads):
        self.imported = [p["block_id"] for p in payloads]
        return len(payloads)


class _FakeRayModule:
    @staticmethod
    def get(refs, timeout=None):
        if isinstance(refs, list):
            return [r.resolve() for r in refs]
        return refs.resolve()


class _StubExecutor:
    def __init__(self, num_workers):
        self.ray_worker_handles = [
            types.SimpleNamespace(actor=_StubActor()) for _ in range(num_workers)
        ]


@pytest.fixture
def fake_ray(monkeypatch):
    fake = types.ModuleType("ray")
    fake.get = _FakeRayModule.get
    monkeypatch.setitem(sys.modules, "ray", fake)
    return fake


def test_relay_transport_remaps_and_ships_rank_for_rank(fake_ray):
    """The relay transport exports, remaps dst ids, and imports per rank."""
    src = _StubExecutor(2)
    dst = _StubExecutor(2)
    transport = RayObjectStoreTransport()

    shipped = transport.ship(
        src, dst, [3, 7], block_mapping={3: 11, 7: 13}, timeout=1.0
    )

    assert shipped == 2
    assert dst.ray_worker_handles[0].actor.imported == [11, 13]
    assert dst.ray_worker_handles[1].actor.imported == [11, 13]


def test_relay_transport_identity_without_mapping(fake_ray):
    """Without a block_mapping, block ids pass through unchanged."""
    src = _StubExecutor(1)
    dst = _StubExecutor(1)
    transport = RayObjectStoreTransport()

    transport.ship(src, dst, [5], block_mapping=None, timeout=1.0)

    assert dst.ray_worker_handles[0].actor.imported == [5]