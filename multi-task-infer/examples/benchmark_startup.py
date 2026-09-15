#!/usr/bin/env python3
"""Quantify the pooled-executor startup benefit (P1).

Measures wall-clock time for two paths and reports the *measured* ratio:

- cold: plain ``AsyncLLM`` with the default Ray executor (worker actors created
  inside the engine).
- pooled: ``ActorPoolManager.pre_start`` (amortized) then ``run_mvp``, which
  acquires pre-started actors and runs one model.

Run from a single-node GPU host:

    python examples/benchmark_startup.py --model facebook/opt-125m --tp 1
"""

from __future__ import annotations

import argparse
import time

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]


def _cold_start(model: str, tp_size: int) -> dict[str, float]:
    """Baseline: cold-start vLLM with its default Ray executor."""
    import asyncio

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=model,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
    )
    vllm_config = engine_args.create_engine_config()

    t0 = time.perf_counter()
    llm = AsyncLLM.from_vllm_config(vllm_config, log_stats=False)
    init_s = time.perf_counter() - t0

    sampling = SamplingParams(max_tokens=32)

    async def _first_token() -> None:
        async for _ in llm.generate(PROMPTS[0], sampling, request_id="cold-0"):
            pass

    t0 = time.perf_counter()
    asyncio.run(_first_token())
    first_gen_s = time.perf_counter() - t0
    llm.shutdown()
    return {
        "init_s": init_s,
        "first_gen_s": first_gen_s,
        "total_s": init_s + first_gen_s,
    }


def _pooled(model: str, tp_size: int, warmup: bool) -> dict[str, float]:
    """Pooled: pre-start actors (amortized) then run one model."""
    from vllm_external_executor.mvp_entry import run_mvp
    from vllm_external_executor import ActorPoolManager

    t0 = time.perf_counter()
    pool = ActorPoolManager()
    pool.pre_start(
        num_actors=tp_size,
        devices_per_node=list(range(tp_size)),
        warmup_distributed=warmup,
    )
    prestart_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_mvp(model, PROMPTS, tp_size=tp_size, pool=pool)
    run_s = time.perf_counter() - t0

    pool.shutdown()
    return {
        "prestart_s": prestart_s,
        "run_s": run_s,
        "total_s": prestart_s + run_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1, choices=(1, 2))
    parser.add_argument(
        "--warmup", action="store_true", help="NCCL warmup at pre_start"
    )
    parser.add_argument("--json", action="store_true", help="Also emit JSON")
    args = parser.parse_args()

    import ray

    ray.init()

    cold = _cold_start(args.model, args.tp)
    pooled = _pooled(args.model, args.tp, args.warmup)

    # The pooled run reuses actors whose creation cost was paid in pre_start;
    # report both the end-to-end ratio and the run-vs-cold ratio.
    total = cold["total_s"]
    speedup_e2e = total / pooled["total_s"] if pooled["total_s"] else float("nan")
    speedup_run = total / pooled["run_s"] if pooled["run_s"] else float("nan")

    print(f"=== startup benefit (model={args.model}, tp={args.tp}) ===")
    print(
        f"cold   : init={cold['init_s']:.2f}s "
        f"first={cold['first_gen_s']:.2f}s total={cold['total_s']:.2f}s"
    )
    print(
        f"pooled : prestart={pooled['prestart_s']:.2f}s "
        f"run={pooled['run_s']:.2f}s total={pooled['total_s']:.2f}s"
    )
    print(f"speedup (cold/pooled-total) = {speedup_e2e:.2f}x")
    print(f"speedup (cold/pooled-run)   = {speedup_run:.2f}x")

    if args.json:
        import json

        print(
            json.dumps(
                {
                    "cold": cold,
                    "pooled": pooled,
                    "speedup_e2e": speedup_e2e,
                    "speedup_run": speedup_run,
                }
            )
        )

    ray.shutdown()


if __name__ == "__main__":
    main()
