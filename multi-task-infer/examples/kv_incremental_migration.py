#!/usr/bin/env python3
"""
Incremental KV migration + cross-node failover usage example.

Demonstrates the full cross-engine migration surface:

1. Same-layout migration (default ``ray_object_store`` transport, ids 1:1).
2. Heterogeneous migration (``total_blocks`` drives dst-id assignment).
3. RDMA transport (``mooncake_rdma``: peer-to-peer, tensors skip the driver).
4. Cross-engine request migration (snapshot + restore + block mount).
5. Node-failure failover (automatic heartbeat + manual ``recover_node``).

These functions require a live Ray cluster and a vLLM build with the plugin
installed; they are kept as runnable sketches so each call site shows the
exact argument contract. The pure-logic pieces they depend on
(``IncrementalKVPlanner``, ``KVTransportFactory``) are unit-tested without
Ray in ``tests/``.
"""

import ray


def _executor_of(llm):
    """Return the ExternalExecutor driving an ``LLM``/``AsyncLLM`` instance."""
    return llm.llm_engine.model_executor


def example_same_layout_migration(src_llm, dst_llm):
    """Migrate live KV from one executor to another, ids mapping 1:1.

    Both instances must have identical KV-cache layouts (same model, TP/PP,
    and block count). Only new/modified blocks move; prefix-cache hits reuse
    the destination's existing blocks at zero transfer cost.
    """
    src = _executor_of(src_llm)
    dst = _executor_of(dst_llm)

    # Default transport = ray_object_store (driver relay over Ray TCP).
    plan = src.migrate_kv_cache_to(dst)
    print(f"Migrated {plan.transferred_count} blocks; "
          f"prefix hits: {len(plan.prefix_hits)}")


def example_heterogeneous_migration(src_llm, dst_llm, dst_total_blocks):
    """Migrate into a pool with a different block count.

    ``total_blocks`` triggers ``assign_targets``: unique source blocks are
    remapped onto ascending free destination ids, recorded in
    ``plan.block_mapping``. The destination's already-occupied ids must be
    listed in ``dst_occupied`` so they are never reused.
    """
    src = _executor_of(src_llm)
    dst = _executor_of(dst_llm)

    dst_occupied = [b.block_id for b in dst.snapshot_kv_blocks()]
    plan = src.migrate_kv_cache_to(
        dst,
        total_blocks=dst_total_blocks,
        dst_occupied=dst_occupied,
    )
    print(f"Remapped {len(plan.block_mapping)} blocks: "
          f"{plan.block_mapping}")


def example_rdma_transport(src_llm, dst_llm, mooncake_config):
    """Migrate over RDMA via a Mooncake store (peer-to-peer).

    Source workers stage serialized block payloads into the Mooncake store and
    hand the driver only ``KVBlockKey`` objects; destination workers pull by
    key. Tensors never pass through the driver process. ``mooncake_config`` is
    pushed to every worker before the transfer.
    """
    src = _executor_of(src_llm)
    dst = _executor_of(dst_llm)

    for handle in src.ray_worker_handles + dst.ray_worker_handles:
        ray.get(
            handle.actor.configure_kv_transport.remote(mooncake_config)
        )

    plan = src.migrate_kv_cache_to(dst, transport="mooncake_rdma")
    print(f"Migrated {plan.transferred_count} blocks over RDMA")


def example_cross_engine_request_migration(src_llm, dst_llm):
    """Move resident requests, not just KV, to the destination engine.

    ``snapshot_requests`` exports each request's admission inputs, full token
    sequence, delivered output tokens and compute position; ``restore_requests``
    rebuilds the ``Request``, admits it, and mounts the migrated KV blocks.
    Call this after ``migrate_kv_cache_to`` so the destination block table
    points at the copied tensors.
    """
    src = _executor_of(src_llm)
    dst = _executor_of(dst_llm)

    snapshots = src.snapshot_requests()
    plan = src.migrate_kv_cache_to(dst)
    dst.restore_requests(snapshots, block_mapping=plan.block_mapping or None)
    print(f"Restored {len(snapshots)} requests on the destination engine")


def example_node_failover(pool):
    """Node failure: automatic heartbeat, plus a manual recovery fallback.

    The pool's background heartbeat thread already refreshes node liveness,
    detects stalled nodes, and calls ``recover_node`` automatically (rebuilding
    the dead node's actors on healthy nodes). ``check_health`` + ``recover_node``
    remain available as an explicit administrative path.
    """
    # Automatic path (no call needed): the heartbeat thread runs after
    # ``pre_start`` and rebuilds actors when a node stalls beyond
    # ``node_heartbeat_timeout``.

    # Manual path: inspect and force-recover a specific node.
    health = pool.check_health()
    for node_id in health["dead_nodes"]:
        rebuilt = pool.recover_node(node_id)
        print(f"Recovered node {node_id}: rebuilt {len(rebuilt)} actors")


if __name__ == "__main__":
    print(
        "This module is a runnable sketch. Instantiate two LLM/AsyncLLM "
        "engines and pass them to the example_* functions; see the "
        "docstrings for the exact call contract."
    )
