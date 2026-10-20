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

- **World Group** is the default process group, `world_size = node device
  count`, `rank = device id`. Created once at pool start (single node uses
  `file://`, which owns no TCP listen socket). Never destroyed.
- **TP sub-group** for a task on ranks `R` is `torch.distributed.new_group(R)`
  over the World Group. Keyed by `tuple(sorted(R))`, cached so a reused
  topology does not re-create.
- **Collective constraint**: `new_group` is a collective over the *default*
  group, so every resident actor must enter the call with the same `R`, even
  the idle ones that will not run the model. Prefer vLLM's split-group /
  stateless group path (`VLLM_DISTRIBUTED_USE_SPLIT_GROUP`,
  `_init_process_group_for_split_group`, `StatelessGroupCoordinator`) which
  exists precisely to carve sub-groups without full-world participation, over
  hand-rolling `new_group` barriers.
- vLLM's `initialize_model_parallel(tp, pp)` derives TP groups from the
  default-group rank ordering (`[g0,g1],[g2,g3],...`), which matches our
  contiguous-device requirement when `rank == device id`. The sub-split API
  should reuse that grouping rather than re-derive it.

Open items needing device access: whether the two-process `new_group` captures
HCCL comm per sub-group (vs reusing the World Group communicator), and the
exact `vllm_ascend` flag that disables prefill capture.