# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
KV-block data-plane transport backends.

Migrated KV blocks must cross nodes; the optimal path depends on the network:

- ``RayObjectStoreTransport`` (default): relay through the driver over Ray
  Object Store (TCP). No extra dependency; the driver gathers the exported
  payloads and forwards them to the destination workers.
- ``MooncakeRdmaTransport`` (optional): peer-to-peer RDMA. Source workers
  serialize each block payload and ``put`` it into a Mooncake store, returning
  only a small :class:`KVBlockKey` to the driver; destination workers ``get``
  the payload by key and import it. Tensors never pass through the driver.

The pure-logic surface (key protocol, destination-id resolution, factory) has
no torch/ray import and is unit-testable offline; the ``ship`` orchestration
imports Ray lazily.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MIGRATION_RPC_TIMEOUT = 300.0


@dataclass
class KVBlockKey:
    """Locator for one KV block staged in a remote store (RDMA backend).

    The driver only ever forwards this small object between processes; the
    tensor payload stays in the store.

    Attributes:
        store_key: Key the source worker used to ``put`` the payload.
        block_id: Source block id. Destination workers remap this through the
            migration ``block_mapping`` to find the physical destination.
        num_groups: Number of KV cache groups carried in the payload, used as
            a receiver-side sanity check.
    """

    store_key: str
    block_id: int
    num_groups: int = 0


def resolve_dst_block_id(
    src_block_id: int, block_mapping: dict[int, int] | None
) -> int:
    """Map a source block id to its destination id (identity when unmapped)."""
    if block_mapping is None:
        return src_block_id
    return block_mapping.get(src_block_id, src_block_id)


def remap_payload_block_ids(
    payloads: list[dict], block_mapping: dict[int, int] | None
) -> list[dict]:
    """Rewrite each payload's ``block_id`` to its destination id, in place.

    Used by the driver-relay transport: the destination worker writes each
    payload to whatever ``block_id`` it carries.
    """
    for payload in payloads:
        payload["block_id"] = resolve_dst_block_id(
            payload["block_id"], block_mapping
        )
    return payloads


class KVTransport(ABC):
    """Abstraction for moving KV-block tensors between worker ranks."""

    name: str = "abstract"

    @abstractmethod
    def ship(
        self,
        src_executor,
        dst_executor,
        src_block_ids: list[int],
        block_mapping: dict[int, int] | None,
        timeout: float = _MIGRATION_RPC_TIMEOUT,
    ) -> int:
        """Move ``src_block_ids`` rank-for-rank from source to destination.

        Args:
            src_executor: Source executor exposing ``ray_worker_handles``.
            dst_executor: Destination executor exposing ``ray_worker_handles``.
            src_block_ids: Physical block ids to move.
            block_mapping: Optional ``src_block_id -> dst_block_id`` remap.

        Returns:
            Number of blocks moved.
        """


class RayObjectStoreTransport(KVTransport):
    """Driver-relay transport over Ray Object Store (TCP)."""

    name = "ray_object_store"

    def ship(
        self,
        src_executor,
        dst_executor,
        src_block_ids,
        block_mapping=None,
        timeout=_MIGRATION_RPC_TIMEOUT,
    ) -> int:
        import ray

        exported = ray.get(
            [
                h.actor.export_kv_blocks.remote(src_block_ids)
                for h in src_executor.ray_worker_handles
            ],
            timeout=timeout,
        )
        for payloads in exported:
            remap_payload_block_ids(payloads, block_mapping)
        ray.get(
            [
                dh.actor.import_kv_blocks.remote(payloads)
                for dh, payloads in zip(
                    dst_executor.ray_worker_handles, exported
                )
            ],
            timeout=timeout,
        )
        return len(src_block_ids)


class MooncakeRdmaTransport(KVTransport):
    """Peer-to-peer RDMA transport via a Mooncake store.

    Source workers stage serialized payloads into the store and hand the driver
    only :class:`KVBlockKey` objects; destination workers pull by key. This
    keeps tensors off the driver process and lets Mooncake's transfer engine
    use RDMA where available.
    """

    name = "mooncake_rdma"

    def ship(
        self,
        src_executor,
        dst_executor,
        src_block_ids,
        block_mapping=None,
        timeout=_MIGRATION_RPC_TIMEOUT,
    ) -> int:
        import ray

        keys = ray.get(
            [
                h.actor.export_kv_blocks_to_store.remote(src_block_ids)
                for h in src_executor.ray_worker_handles
            ],
            timeout=timeout,
        )
        # `keys` is per-rank [[KVBlockKey, ...]]; block ids stay source ids,
        # and the destination worker remaps them via block_mapping on import.
        ray.get(
            [
                dh.actor.import_kv_blocks_from_store.remote(
                    rank_keys, block_mapping
                )
                for dh, rank_keys in zip(
                    dst_executor.ray_worker_handles, keys
                )
            ],
            timeout=timeout,
        )
        return len(src_block_ids)


class KVTransportFactory:
    """Create a transport backend by name."""

    _registry: dict[str, type[KVTransport]] = {
        RayObjectStoreTransport.name: RayObjectStoreTransport,
        MooncakeRdmaTransport.name: MooncakeRdmaTransport,
    }

    @classmethod
    def register(cls, name: str, transport_cls: type[KVTransport]) -> None:
        """Register a transport backend."""
        cls._registry[name] = transport_cls

    @classmethod
    def create(cls, name: str) -> KVTransport:
        """Instantiate a transport backend by name."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown KV transport: {name}. "
                f"Available: {sorted(cls._registry)}"
            )
        return cls._registry[name]()