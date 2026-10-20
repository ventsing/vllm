# Scheme B: cross-model hot-switch plan (ExternalExecutor)

Map of the three-tier hot-switch design onto the current code, the gaps, and
the phased landing order. This is a working plan, not a spec: the harder
pieces (dynamic communication sub-split) are staged last because they touch
core `parallel_state` and cannot be exercised without a device.

## Current coverage

| Scheme B layer / means | Status | Where |
| --- | --- | --- |
| EngineCore + ExternalExecutor (control) | done | `vllm_external_executor/external_executor.py` |
| ZMQ + Shm broadcast data path | done | `MessageQueue` (worker `initialize_worker`) |
| Pre-started resident Ray actor pool | done | `actor_pool_manager.py` |
| Explicit unload teardown (graph/model/KV release) | done | `external_worker_actor.py::_release_worker_resources` |
| ZMQ reset between rounds | done | `reset()` + `_release_worker_resources` |
| Same-world_size in-place `switch_model` | code exists, **entry gated off** | worker `switch_model`; executor `switch_model` raises `NotImplementedError` |
| Capture trim + AOT cache | landed this turn | `mvp_entry.py` (`cudagraph_capture_sizes`, `ACL_OP_COMPILER_CACHE_MODE`) |
| FastSwappableModelRunner (`unload`/`reload`) | missing (logic inlined) | — |
| DynamicHCCLManager (World Group resident + sub-group split) | missing | — |
| `switch_model_and_topology` / `reconfigure` entry | missing | — |
| Mooncake P2P weight zero-copy | P2, out of scope | `storage_checkpoint_engine.py` |

## Landing order

1. **P0 — capture trim + AOT cache** (landed): `cudagraph_capture_sizes=
   [1,2,4,8]` + `ACL_OP_COMPILER_CACHE_MODE=enable`. Independent, verifiable
   now on the real machine.
2. **P1 — FastSwappableModelRunner**: extract the worker's teardown/rebuild
   sequence (`_release_worker_resources` + `switch_model` body) into a named
   `unload()` / `reload(new_config)` pair. No behavior change; it is the seam
   that `switch_model_and_topology` drives.
3. **P2 — DynamicHCCLManager**: resident global World Group + cached
   `new_group` sub-split (details below).
4. **P3 — `switch_model_and_topology` / `reconfigure`**: enable the executor
   entry, wire `reset_channel → unload → get_or_create_subgroup → reload`.
5. **P4 — verify TP change** (1↔2↔4) on the real machine; reconcile prefill
   capture-off knob with `vllm_ascend`.

## DynamicHCCLManager design notes

Goal: `init_process_group`/`destroy_process_group` runs once per process; TP
re-grouping becomes an in-memory `new_group` over a resident World Group.
Lifecycle owner landed in `dynamic_hccl_manager.py` (thin; delegates to
`vllm.distributed.parallel_state`).

- **World Group** is the torch default process group, `world_size = node
  device count`, `rank = device id`. Created once at pool start (single node
  uses `file://`, which owns no TCP listen socket) and destroyed only by
  `destroy_distributed_environment()` at actor shutdown.
- **TP sub-group** derives from `initialize_model_parallel(tp, pp)`, which
  already does the right collective: `GroupCoordinator.__init__` iterates the
  whole `group_ranks` partition and calls `new_group` for *every* group, so
  every resident rank (including idle ones) participates, and each rank keeps
  only the group containing it. No hand-rolled barrier or `split_group`
  shim is required; the contiguous partition `[[0,1],[2,3],...]` falls out of
  rank-ordered chunking when `rank == device id`.
- **Teardown split** is the crux: `reset` must call `destroy_model_parallel()`
  (TP/PP/DP/EP, keeps World) instead of `cleanup_dist_env_and_memory`, whose
  `destroy_distributed_environment()` kills the resident World Group and the
  torch default process group. `cleanup_dist_env_and_memory` minus the World
  teardown is exactly `DynamicHCCLManager.release_tp_group`.
- **Cache** keyed by `tp_size`: the partition is deterministic from World rank
  ordering, so a repeated topology reuses the held subgroup without a fresh
  `new_group`.

Integration (next step): pool start calls `ensure_world_group` once with
`world_size == device count`; per task `acquire_tp_group(tp)`; reset calls
`release_tp_group`; actor shutdown calls `teardown`. Two call sites hard-code
``default group == TP x PP`` and must be decoupled: `gpu_worker.py` passes
`parallel_config.world_size` (TP x PP) to `init_distributed_environment`, and
the executor passes `world_size == TP` to `create_dist_init_method`. The
resident World Group spans device count while `parallel_config.world_size`
stays TP x PP for the model's own bookkeeping — closest to vLLM's DP split
(`world_size_across_dp = device count`), which the MVP currently disables, so
this needs a device-backed iteration rather than a blind wiring.

Open items needing device access: whether each carved subgroup opens its own
HCCL communicator (vs reusing the World Group's), and the exact `vllm_ascend`
flag that disables prefill capture.