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
| Capture trim + AOT cache | done | `mvp_entry.py` (`cudagraph_capture_sizes`, `ACL_OP_COMPILER_CACHE_MODE`) |
| FastSwappableModelRunner (`unload`/`reload`) | done | worker `switch_model_and_topology` (unload=`_release_worker_resources`, reload=`initialize_worker`) |
| DynamicHCCLManager (resident World Group) | dropped | rebuild ~300ms; `dynamic_hccl_manager.py` keeps only `plan_tp_partition` |
| `switch_model_and_topology` / `reconfigure` entry | worker landed; **executor pending** | worker `switch_model_and_topology`; executor `reconfigure` next |
| Mooncake P2P weight zero-copy | P2, out of scope | `storage_checkpoint_engine.py` |

## Landing order

1. **P0 — capture trim + AOT cache** (landed).
2. **P1 — comm domain** (landed): rebuild per topology, ~300ms after first
   `hccl` handshake; no resident World Group.
3. **P1 — FastSwappableModelRunner seam** (landed): worker
   `switch_model_and_topology` = stop loop → `_release_worker_resources` →
   `initialize_worker`.
4. **P3 — executor `reconfigure`** (next): re-run `_init_executor` Steps 2-8
   after swapping `vllm_config`/`parallel_config`/`world_size`/handles; the
   executor closes its MQ handles and broadcasts
   `switch_model_and_topology` instead of `initialize_worker`.
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