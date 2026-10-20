#!/usr/bin/env python3
"""On-device benchmark: ``ExternalExecutor.reconfigure`` across TP transitions.

Constructs a *raw* :class:`ExternalExecutor` (no AsyncLLM / EngineCore) over a
pre-started actor pool and times ``reconfigure`` as TP shrinks (``4 -> 2 -> 1``
by default). This isolates the scheme-B switch mechanism from the
scheduler/KV-cache re-init that lives in EngineCore:

* worker teardown (graph/model/MessageQueue/communication-domain release),
* communication-domain rebuild (:meth:`init_device`, the ~300ms Ascend claim),
* model reload (:meth:`load_model`, weights — the switch bottleneck).

The executor logs a per-phase breakdown itself
(``worker-group phases: teardown+reinit=... init_device=... load_model=...``);
this script adds wall-clock totals per step. Generation end-to-end (scheduler +
KV cache for the new model) is exercised separately through ``run_mvp``.
"""

import argparse
import time

import ray
from vllm.engine.arg_utils import AsyncEngineArgs

from vllm_external_executor import ActorPoolManager, ExternalExecutor

MODEL = "/opt/huawei/dataset/downloaded_models/Qwen3.5-4B"


def build_config(model: str, tp_size: int):
    """Build a single-node VllmConfig for ``model`` at ``tp_size``."""
    return AsyncEngineArgs(
        model=model,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        enforce_eager=True,
    ).create_engine_config()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--devices", type=int, default=4,
                        help="Number of actors to pre-start (>= max TP)")
    parser.add_argument("--sequence", default="4,2,1",
                        help="Comma-separated TP sizes to step through")
    args = parser.parse_args()

    tp_sequence = [int(x) for x in args.sequence.split(",")]
    if any(x < 1 for x in tp_sequence):
        parser.error("TP sizes must be positive")
    num_actors = max(args.devices, max(tp_sequence))

    ray.init()

    pool = ActorPoolManager()
    pool.pre_start(
        num_actors=num_actors,
        devices_per_node=list(range(num_actors)),
        warmup_distributed=True,
    )
    try:
        boot_tp = tp_sequence[0]
        actors = pool.acquire(tp_size=boot_tp, pp_size=1)
        executor = ExternalExecutor(
            build_config(args.model, boot_tp),
            external_actors=actors,
        )
        print(f"[boot] TP={boot_tp} model loaded")

        try:
            # ``reconfigure`` reuses ``actors[:tp]`` in place and resets the
            # trailing actors (they go IDLE). Shrinking keeps this simple: the
            # same leased batch serves every step, and the pool lease covers it.
            for tp in tp_sequence[1:]:
                t0 = time.perf_counter()
                executor.reconfigure(
                    build_config(args.model, tp),
                    new_actors=actors[:tp],
                )
                dt = time.perf_counter() - t0
                print(f"[switch] TP={tp}: {dt:.3f}s total")
        finally:
            t0 = time.perf_counter()
            executor.shutdown()
            print(f"[shutdown] executor: {time.perf_counter() - t0:.3f}s")
            # shutdown() already reset workers; release() is idempotent and
            # returns the lease to the pool.
            pool.release(actors)
    finally:
        pool.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()