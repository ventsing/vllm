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

## Communication domain: rebuild, not resident

Node-local Ascend fact (real-device confirmed): once `hccl` has done its first
handshake, each device's `hccp` process stays resident, so **destroy +
`init_process_group` for a new topology costs ~300ms**, not the multi-second
first-handshake cost. That removes the resident-World-Group subgroup machinery
from the plan.

- `switch_model_and_topology(unload)` runs the existing full teardown —
  `cleanup_dist_env_and_memory`, which destroys both the model-parallel groups
  and the torch default process group. That is fine: the per-round rebuild is
  the trade-off, and it is ~300ms.
- `reload(new_tp)` then calls `create_dist_init_method(world_size = new_tp)`
  (single node → `file://`, no TCP port) and re-runs the normal
  `initialize_worker` / `init_device` / `initialize_model_parallel(new_tp)`
  path. No `gpu_worker.py` world-size decoupling is required: the default
  group stays `TP x PP` exactly as vLLM expects.
- The only module retained from the earlier resident-world draft is
  `dynamic_hccl_manager.py::plan_tp_partition`, a pure contiguous-partition
  helper (kept for neutral rank planning and future multi-node/EP).

Expected switch cost after this change is dominated by model weight load and
graph capture — the comm rebuild (~300ms) is not the bottleneck.

Open item needing device access: the exact `vllm_ascend` flag that disables
prefill capture.