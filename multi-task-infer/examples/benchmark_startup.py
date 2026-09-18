#!/usr/bin/env python3
"""Quantify the pooled-executor startup benefit for sequential reuse (P1).

Measures wall-clock time for two paths over the *same* model sequence and
reports the *measured* ratio:

- cold: each model cold-starts a fresh ``AsyncLLM`` (default Ray executor,
  worker actors created inside the engine).
- pooled: ``ActorPoolManager.pre_start`` once (amortized), then ``run_mvp`` for
  each model over the same reused actor batch.

Run from a single-node GPU host:

    python examples/benchmark_startup.py \\
        --models facebook/opt-125m,facebook/opt-350m --tp 1
"""

from __future__ import annotations

import argparse
import time

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]
DEFAULT_MODELS = "facebook/opt-125m,facebook/opt-350m"


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


def _pooled_sequence(
    models: list[str], tp_size: int, warmup: bool
) -> dict[str, float]:
    """Pooled: pre-start once, then run each model over the reused actors."""
    from vllm_external_executor import ActorPoolManager
    from vllm_external_executor.mvp_entry import run_mvp

    t0 = time.perf_counter()
    pool = ActorPoolManager()
    pool.pre_start(
        num_actors=tp_size,
        devices_per_node=list(range(tp_size)),
        warmup_distributed=warmup,
    )
    prestart_s = time.perf_counter() - t0

    run_s: list[float] = []
    for model in models:
        t0 = time.perf_counter()
        run_mvp(model, PROMPTS, tp_size=tp_size, pool=pool)
        run_s.append(time.perf_counter() - t0)

    pool.shutdown()
    run_total = sum(run_s)
    return {
        "prestart_s": prestart_s,
        "run_s": run_s,
        "run_total_s": run_total,
        "total_s": prestart_s + run_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default=DEFAULT_MODELS, help="comma-separated model ids"
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--warmup", action="store_true", help="NCCL warmup at pre_start"
    )
    parser.add_argument("--json", action="store_true", help="Also emit JSON")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    import ray

    ray.init()

    cold = [_cold_start(model, args.tp) for model in models]
    pooled = _pooled_sequence(models, args.tp, args.warmup)

    cold_total = sum(c["total_s"] for c in cold)
    speedup_e2e = cold_total / pooled["total_s"] if pooled["total_s"] else float("nan")
    speedup_run = (
        cold_total / pooled["run_total_s"] if pooled["run_total_s"] else float("nan")
    )

    print(f"=== sequential-reuse startup benefit (tp={args.tp}) ===")
    for model, c in zip(models, cold):
        print(f"  cold   {model}: total={c['total_s']:.2f}s")
    print(f"  pooled prestart={pooled['prestart_s']:.2f}s")
    for model, r in zip(models, pooled["run_s"]):
        print(f"  pooled {model}: run={r:.2f}s")
    print(f"cold total {cold_total:.2f}s vs pooled total {pooled['total_s']:.2f}s")
    print(f"speedup (cold/pooled-total) = {speedup_e2e:.2f}x")
    print(f"speedup (cold/pooled-run)   = {speedup_run:.2f}x")

    if args.json:
        import json

        print(
            json.dumps(
                {
                    "models": models,
                    "cold": cold,
                    "cold_total_s": cold_total,
                    "pooled": pooled,
                    "speedup_e2e": speedup_e2e,
                    "speedup_run": speedup_run,
                }
            )
        )

    ray.shutdown()


if __name__ == "__main__":
    main()
