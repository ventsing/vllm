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
| `switch_model_and_topology` / `reconfigure` entry | done | worker `switch_model_and_topology`; executor `reconfigure` + `_init_worker_group` |
| EngineCore `reconfigure` (KV re-init + scheduler rebuild) | done | `vllm/v1/engine/core.py` (`reconfigure` + `_setup_model_specific`) |
| Client tokenizer/renderer swap | done | `AsyncLLM.reconfigure` (`vllm/v1/engine/async_llm.py`) |
| End-to-end driver | done | `mvp_entry.py::run_hot_switch_mvp` (calls `llm.reconfigure`) |
| Mooncake P2P weight zero-copy | P2, out of scope | `storage_checkpoint_engine.py` |

## Landing order

1. **P0 — capture trim + AOT cache** (landed).
2. **P1 — comm domain** (landed): rebuild per topology, ~300ms after first
   `hccl` handshake; no resident World Group.
3. **P1 — FastSwappableModelRunner seam** (landed): worker
   `switch_model_and_topology` = stop loop → `_release_worker_resources` →
   `initialize_worker`.
4. **P3 — executor `reconfigure`** (landed): `_init_executor`'s worker-group
   steps (MQ + dist init + `initialize_worker` + device + model) moved into
   `_init_worker_group(use_switch=...)`; `reconfigure` resets leaving actors,
   closes executor MQs, swaps `vllm_config`/`parallel_config`/`world_size`/
   handles, then re-runs the group via
   `switch_model_and_topology` (in-place for reused actors).
5. **P3 — EngineCore `reconfigure`** (landed): `EngineCore.__init__`'s
   per-model setup (KV cache + scheduler + connectors + batch queue +
   prefix-cache hasher) extracted into `_setup_model_specific`; a new
   `EngineCore.reconfigure` swaps `vllm_config`, reconfigures the executor in
   place, and re-runs `_setup_model_specific` (KV re-profile + capture +
   scheduler rebuild). Reachable via `call_utility_async("reconfigure", …)`.
6. **P3 — client tokenizer/renderer swap** (landed): `AsyncLLM.reconfigure`
   rebuilds `renderer`/`input_processor`/`output_processor` + config refs for
   the new model, then delegates to `EngineCore.reconfigure` — distinct
   tokenizers are supported.
7. **P3 — end-to-end driver** (landed): `run_hot_switch_mvp` boots one
   AsyncLLM on a `max_tp` lease and reconfigures through TP `1/2/4` without
   releasing actors.
8. **P4 — device verify** (next): run `examples/hot_switch_bench.py` and
   `examples/hot_switch_mvp.py` on the real machine — confirm `init_device`
   ≈ 300ms, TP 4→2 rank-shrink works, and the whole switch is seconds-level.

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

Prefill capture on Ascend is not a separate flag — it rides vLLM's generic
`CompilationConfig.cudagraph_mode` (see the [vLLM-Ascend graph-mode
guide][ascend-graph-mode]): `NONE` = eager (no capture), `PIECEWISE` =
prefill/mixed capture, `FULL_DECODE_ONLY` = decode-only (no prefill),
`FULL_AND_PIECEWISE` = v1 default. Scheme B pins
`SCHEME_B_CUDAGRAPH_MODE = "FULL_DECODE_ONLY"` so each switch skips prefill
re-capture and only replays the trimmed decode ladder (`1,2,4,8`);
`--capture-trim` enables both.

[ascend-graph-mode]: https://raw.githubusercontent.com/vllm-project/vllm-ascend/70501212812e1e9842d74a0327c87c355268f849/docs/source/user_guide/feature_guide/graph_mode.md

## Device verification

```bash
# 1. Raw mechanism (no scheduler): timing across TP 4 -> 2 -> 1.
#    (bench already pins enforce_eager=True, so it measures teardown/comm/
#     weight-load only — no graph capture, no --capture-trim flag.)
python examples/hot_switch_bench.py

# 2. End-to-end (live AsyncLLM + EngineCore reconfigure): TP 4 -> 2 -> 1.
#    --capture-trim = decode-only capture FULL_DECODE_ONLY + sizes (1,2,4,8).
python examples/hot_switch_mvp.py --capture-trim
```

> End-to-end needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1`: `reconfigure` ships a
> `VllmConfig` + Ray actor handles across the typed msgpack utility channel,
> which only supports `msgspec.Struct`/primitives natively — the flag enables
> the pickle fallback on the trusted client↔EngineCore boundary.
> `run_hot_switch_mvp` sets it before constructing the engine; `hot_switch_bench`
> does not need it (it calls `executor.reconfigure` in-process).

Phase-1 optional: `--skip-memory-profile` + `--kv-cache-memory-bytes <bytes>`
skips the 2-3s dummy-forward memory profile on every switch and derives
`num_gpu_blocks` from the static per-GPU budget instead. Intended for repeat
switches where the budget is already known; pair with `--capture-trim`
(`FULL_DECODE_ONLY`) so decode capture is handled separately, not by the
profile run.

Check against these expectations:

* `init_device` (communication-domain rebuild) ≈ 300ms, not the multi-second
  first handshake — confirms the "rebuild, not resident" decision.
* `load_model` (weight load) dominates the switch; `teardown+reinit` is small.
* No `EADDRINUSE` / port collisions across rounds (single node uses `file://`:
  `create_dist_init_method` chooses `file://{tmp}/vllm_dist_{uuid}`).
* TP `4→2` rank-table shrink and `2→1` both complete without re-acquiring
  actors; `run_hot_switch_mvp` outputs valid text for every (model, tp).