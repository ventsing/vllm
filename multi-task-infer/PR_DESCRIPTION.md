# [ExternalExecutor] Pre-started Ray actor pool, model hot-switching, and cross-node KV migration

> PR description draft — copy into GitHub when opening the PR. Branch:
> `feature/external-executor` (pushed to the `ventsing/vllm` fork).

## Summary

Adds an `ExternalExecutor` plugin for vLLM V1 that:

- Runs workers on a **pre-started Ray Actor pool** (no per-engine Ray actor
  startup cost), with a detached `NodeRegistryActor`, fault-domain-aware
  `GlobalScheduler`, and automatic **actor-level and node-level failover**.
- Supports **model hot-switching** and dynamic TP/PP via the executor, with
  `StorageCheckpointEngine` for weight loading (`NFSStorageBackend` over TCP,
  `MooncakeStoreBackend` over RDMA) and `weight_transfer` as an alternative.
- Shares **torch.compile caches** across engines (small caches via Ray Object
  Store, large via NFS + gzip compression).
- Migrates a live engine's **KV cache incrementally** to another engine
  (prefix-cache aware; heterogeneous block-id remapping), restores in-flight
  requests, and moves KV tensors over a **pluggable transport**
  (`ray_object_store` default, `mooncake_rdma` peer-to-peer, `cuda_ipc`
  same-node zero-copy).
- Ships the **storage-base decision layer** for production deployment: a
  three-tier hierarchy (HBM/DRAM/REMOTE) with swap decisions, a
  cross-actor prefix index, a base+adapter weight-share ledger, and
  access-heat prefetch ranking (pure-logic, unit-tested). These are wired to
  the migration state machine via `migration_orchestrator.py` and consumed by
  both `ExternalExecutor` migration paths: `switch_model` injects per-phase
  execution callbacks (pause / checkpoint / switch / restore) through
  `phase_handlers` and consumes the tiering script + `WeightShareLedger`
  bookkeeping (base+adapter); `migrate_kv_cache_to(prefetch=...)` drives the
  async prefetch hook (hot, non-resident prefixes nominated at PREPARING and
  awaited at LOAD) and registers transferred prefixes on the shared
  `GlobalPrefixIndex`. Both paths get compensation-based rollback of their own
  bookkeeping.

The plugin lives entirely under `multi-task-infer/` and hooks vLLM through the
existing `vllm.general_plugins` entry point; core changes are intentionally
minimal.

## Why this is not a duplicate

- Existing vLLM executors (`RayExecutorV2`, `ExternalExecutor` variants in-tree)
  create workers per engine and do not pool actors across engine lifetimes.
- This PR adds **cross-engine KV migration** (incremental, prefix-cache aware,
  heterogeneous remap) plus a transport abstraction, which the in-tree
  `ExternalExecutor`/`RayExecutorV2` do not implement.
- Maintainer must confirm with `gh pr list` (commands in the pre-submit
  checklist below); no duplicate was identified from the checked-out history.

## Core changes (minimal, 8 files, +159 / −2 lines)

| File | Change |
|------|--------|
| `vllm/v1/engine/async_llm.py` | +5 — accept `external_actors` and pass to the engine client |
| `vllm/v1/engine/core_client.py` | +9 — thread `external_actors` through MP clients |
| `vllm/v1/engine/utils.py` | +4 — thread `external_actors` into `launch_core_engines` |
| `vllm/v1/engine/core.py` | +17/−1 — pass `external_actors` to the executor; call `bind_scheduler(self.scheduler)` when the executor exposes it |
| `vllm/v1/request.py` | +58 — `Request.from_engine_core_request` / `to_engine_core_request` / `restore_running_state` (request snapshot/restore for cross-engine migration) |
| `vllm/v1/core/block_pool.py` | +20 — read accessors for KV-block snapshot (block hash / group id) |
| `vllm/v1/core/kv_cache_manager.py` | +24 — `get_block_ids_for_computed_tokens` (per-group crop to computed tokens) |
| `vllm/v1/core/kv_cache_coordinator.py` | +24 — request→block-table accessor for request restore |

The `bind_scheduler` hook is a no-op for stock executors; the plugin is the
only caller. `external_actors` is `None` for stock paths, so behavior is
unchanged.

## Plugin layout (all new, under `multi-task-infer/`)

```
vllm_external_executor/
  external_executor.py          # ExternalExecutor + incremental KV migration + transport
  actor_pool_manager.py         # ActorPoolManager: pre-start/acquire/release + failover
  external_worker_actor.py      # ExternalWorkerActor: device bind + export/import KV
  cluster_state.py              # NodeInfo / ActorRegistration / GlobalScheduler (pure)
  node_registry_actor.py        # NodeRegistryActor: registration + heartbeat + dead detection
  migration.py                  # MigrationPhase state machine + atomic transactions
  kv_migration.py               # IncrementalKVPlanner (prefix-cache aware diff)
  kv_transport.py               # KVTransport ABC + Ray/Mooncake/CUDA-IPC backends + factory
  storage_tier.py               # HBM/DRAM/REMOTE tiering + swap decisions (pure)
  global_prefix_index.py        # cross-actor prefix index, keyed by weight_hash (pure)
  weight_sharing.py             # base+adapter weight-share ledger + cost ratio (pure)
  prefetch_policy.py            # access heat + async prefetch decisions (pure)
  migration_orchestrator.py     # decision layer -> state machine wiring
  cache_manager_actor.py        # compile-cache sharing (G6)
  storage_checkpoint_engine.py  # NFS / Mooncake backends (G7)
tests/                          # 9 test modules (pytest, pure-logic where possible)
examples/                       # basic usage + incremental migration/failover sketches
design.md                       # full design doc (4+1 view)
```

## Usage

```python
# Two engines share a pre-started pool; migrate live KV between them.
src = src_llm.llm_engine.model_executor
dst = dst_llm.llm_engine.model_executor

# Same-layout (ids 1:1), default Ray Object Store transport:
plan = src.migrate_kv_cache_to(dst)

# Heterogeneous pool (assign dst ids):
plan = src.migrate_kv_cache_to(dst, total_blocks=N, dst_occupied=occupied)

# RDMA peer-to-peer (tensors skip the driver):
plan = src.migrate_kv_cache_to(dst, transport="mooncake_rdma")

# Move requests, not just KV:
dst.restore_requests(src.snapshot_requests(), block_mapping=plan.block_mapping)
```

See `examples/kv_incremental_migration.py` for all five paths.

## Testing

**What ran in this environment** (no `torch`/`vllm`/`ray`/`pytest`/`uv` —
only the pure-logic modules are importable via `importlib`):

```bash
# 1. Syntax + import-surface check on every touched module:
python3 -m py_compile \
  vllm/v1/request.py vllm/v1/core/{kv_cache_manager,kv_cache_coordinator,block_pool}.py \
  vllm/v1/engine/core.py \
  multi-task-infer/vllm_external_executor/*.py multi-task-infer/tests/*.py
# -> COMPILE OK

# 2. Pure-logic regression (state machine + compensation rollback, KV planner,
#    transport factory + mocked-ray relay, scheduler, node registry heartbeat,
#    tiering, prefix index, weight ledger, prefetch + orchestrator wiring):
# -> FULL REGRESSION: all pure-logic modules PASS
#    SWITCH_MODEL WIRING (phase_handlers + external sm): PASS
#    KV MIGRATION SEMANTICS (prefetch + prefix index): PASS
```

**What must run on a full environment** (GPU + Ray + Mooncake for RDMA):

```bash
cd multi-task-infer
uv run --extra test pytest -q tests/test_global_scheduler.py \
    tests/test_migration.py tests/test_kv_migration.py \
    tests/test_kv_transport.py tests/test_node_registry.py \
    tests/test_storage_tier.py tests/test_global_prefix_index.py
uv run --extra test pytest -q tests/test_storage_checkpoint_engine.py -m "not mooncake"
# RDMA (only on a Mooncake + IB cluster):
uv run --extra test pytest -q tests/test_storage_checkpoint_engine.py -m mooncake
```

## Model evaluation

This PR adds orchestration/infrastructure only — it does not change weight
loading math, sampling, or the scheduler's token accounting. The correctness
property to verify on hardware is **migration fidelity**: engine A's output on
a request migrated to engine B must be identical to running the request to
completion on A. Pending hardware validation:

- [ ] Same-layout migration: migrated vs. uninterrupted decode are token-identical.
- [ ] Heterogeneous remap: dst-id assignment never aliases an occupied block.
- [ ] RDMA transport: Mooncake put/get round-trips tensors bit-exact.
- [ ] Node kill: `recover_node` rebuilds the dead node's actors and re-routes.

The full per-item hardware checklist (environment, commands, pass criteria,
and a summary checkbox table for M1–M10) lives in
`multi-task-infer/HARDWARE_VALIDATION.md`.

## Known limitations

- `MooncakeRdmaTransport`, `CudaIpcTransport` and `MooncakeStoreBackend`
  worker paths are skeleton-verified offline only; RDMA bandwidth,
  GPU-direct staging and `torch.from_ipc_handle` API compatibility need real
  hardware (Mooncake + IB for RDMA; same-node GPU pair for CUDA IPC).
- GPUDirect Storage (GDS) is not implemented — documented as a future
  `StorageBackend` (requires `cufile` + GPUDirect storage).
- The five "advanced storage" capabilities ship as **pure-logic decision
  modules** with tests: `storage_tier` (tiering), `global_prefix_index`
  (cross-actor prefix reuse), `weight_sharing` (base+adapter ledger),
  `prefetch_policy` (async prefetch ranking). `migration_orchestrator.py`
  wires the tiering/prefetch decisions into the state machine (async prefetch
  hook + tiered checkpoint movement + compensation rollback), and
  `ExternalExecutor.migrate_kv_cache_to` consumes that script (prefetch
  nomination + `GlobalPrefixIndex` registration). The
  `_start_prefetch`/`_await_prefetch` callbacks are no-op+log by default and
  the physical `ship()`/tensor moves still need a GPU runtime to execute; the
  decision layer is offline-tested.
- Cross-model prefix sharing is valid only for identical `weight_hash`
  (same base + adapter); KV tensors are weight-dependent.
- CUDA Graph capture is not shared across models (documented; re-captured on
  switch).

## Pre-submit checklist (maintainer must run)

- [ ] `gh pr list --repo vllm-project/vllm --state open --search "ExternalExecutor actor pool"`
- [ ] `gh pr list --repo vllm-project/vllm --state open --search "KV cache migration"`
- [ ] `pre-commit run --all-files` / `ruff` clean on the plugin directory.
- [ ] Re-run the full-env test commands above and paste output.

> **AI assistance**: generated with the assistance of an AI coding agent
> (DeepSeek); the submitting human must review every changed line and run the
> full-environment tests. All pure-logic modules were verified offline as
> described in Testing.
