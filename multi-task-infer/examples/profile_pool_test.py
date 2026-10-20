# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Profile eight TP=2 engines using the pool retained by rolling_pool_test.py."""

import argparse
import asyncio
import contextlib
import json
import os
import time
import traceback
import uuid
from pathlib import Path

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_VERSION", "0.28.1")

import ray
from pool_diagnostics import assert_released, inspect_workers
from rolling_pool_test import RetainedPool, emit
from vllm_external_executor import ExternalExecutor
from vllm_external_executor.mvp_entry import _detach_stuck_mm_warmup

from vllm import SamplingParams
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    ProfilerConfig,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.tokenizers import get_tokenizer
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.process_pool import EngineCoreProcessPool


def graph_snapshots(actors):
    def probe(actor):
        import sys

        import torch

        config = actor.vllm_config
        module = sys.modules.get("vllm_ascend.compilation.acl_graph")
        entries = []
        if module is not None:
            for wrapper in module._acl_graph_wrappers:
                for descriptor, entry in wrapper.concrete_aclgraph_entries.items():
                    if entry.aclgraph is not None:
                        entries.append(str(descriptor))
        worker = actor.worker.worker
        return {
            "actor_id": actor.actor_id,
            "pid": __import__("os").getpid(),
            "dtype": str(config.model_config.dtype),
            "ssm_dtype": config.cache_config.mamba_ssm_cache_dtype,
            "mode": str(config.compilation_config.cudagraph_mode),
            "captured_graphs": entries,
            "graph_memory_bytes": getattr(worker, "npugraph_memory_bytes", None),
            "allocated_bytes": torch.npu.memory_allocated(),
        }

    return ray.get([actor.__ray_call__.remote(probe) for actor in actors], timeout=180)


def set_compile_cache(actor, value):
    """Avoid loading serialized collectives tied to an earlier engine lease."""
    from vllm import envs

    assert actor.worker is None
    previous = os.environ.get("VLLM_DISABLE_COMPILE_CACHE")
    if value is None:
        os.environ.pop("VLLM_DISABLE_COMPILE_CACHE", None)
    else:
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = value
    clear = getattr(envs.__getattr__, "cache_clear", None)
    if clear is not None:
        clear()
    return previous


async def run(args, pool):
    output_dir = Path(args.output_dir)
    engines = {}
    leases = {}
    result = {
        "status": "running",
        "requests": [],
        "batches": [],
        "startup": [],
        "shutdown": [],
        "errors": [],
        "config": vars(args),
    }
    handles = list(pool.actors)
    pids_before = ray.get(
        [a.__ray_call__.remote(lambda _: __import__("os").getpid()) for a in handles]
    )
    result["actor_ids"] = [a._actor_id.hex() for a in handles]
    result["pids_before"] = pids_before
    tokenizer = get_tokenizer(args.model)
    text = (
        "Explain how a reusable pool of inference workers helps serve independent "
        "language model requests. Discuss scheduling, isolation, and latency. "
    ) * 100
    token_ids = tokenizer.encode(text)

    async def request(slot, phase, input_tokens, output_tokens, index):
        started = time.perf_counter()
        first = None
        final = None
        async for output in engines[slot].generate(
            {"prompt_token_ids": token_ids[:input_tokens]},
            SamplingParams(temperature=0, max_tokens=output_tokens, ignore_eos=True),
            request_id=f"{phase}-{slot}-{index}-{uuid.uuid4().hex}",
        ):
            now = time.perf_counter()
            if output.outputs and output.outputs[0].token_ids and first is None:
                first = now
            final = output
        ended = time.perf_counter()
        assert final is not None and final.finished and first is not None
        generated = len(final.outputs[0].token_ids)
        assert generated == output_tokens, (generated, output_tokens)
        assert len(final.prompt_token_ids) == input_tokens
        row = {
            "slot": slot,
            "phase": phase,
            "input_tokens": input_tokens,
            "output_tokens": generated,
            "ttft_ms": (first - started) * 1000,
            "latency_ms": (ended - started) * 1000,
            "tpot_ms": (ended - first) * 1000 / max(generated - 1, 1),
            "output_token_ids": list(final.outputs[0].token_ids),
        }
        result["requests"].append(row)
        return row

    async def batch(phase, input_tokens, output_tokens, per_instance):
        started = time.perf_counter()
        rows = await asyncio.gather(
            *(
                request(slot, phase, input_tokens, output_tokens, index)
                for slot in engines
                for index in range(per_instance)
            )
        )
        seconds = time.perf_counter() - started
        row = {
            "phase": phase,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "concurrency_per_instance": per_instance,
            "requests": len(rows),
            "seconds": seconds,
            "output_tokens_per_second": sum(r["output_tokens"] for r in rows) / seconds,
            "requests_per_second": len(rows) / seconds,
        }
        result["batches"].append(row)
        emit("profile_batch", **row)

    try:
        for slot in range(args.instances):
            actors = pool.acquire(tp_size=2, pp_size=1)
            leases[slot] = actors
            started = time.perf_counter()

            def build(slot=slot, actors=actors):
                config = AsyncEngineArgs(
                    model=args.model,
                    tensor_parallel_size=2,
                    served_model_name=f"profile-slot-{slot}",
                    dtype="bfloat16",
                    enforce_eager=args.mode == "eager",
                    max_model_len=1024,
                    max_num_seqs=4,
                    max_num_batched_tokens=2048,
                    gpu_memory_utilization=0.65,
                    enable_prefix_caching=False,
                    language_model_only=True,
                    mamba_ssm_cache_dtype="bfloat16",
                    mamba_cache_dtype="bfloat16",
                    compilation_config=CompilationConfig(
                        mode=(
                            CompilationMode.VLLM_COMPILE
                            if args.mode == "graph"
                            else CompilationMode.NONE
                        ),
                        cudagraph_mode=(
                            CUDAGraphMode.FULL_DECODE_ONLY
                            if args.mode == "graph"
                            else CUDAGraphMode.NONE
                        ),
                        cudagraph_capture_sizes=[1, 2, 4],
                        max_cudagraph_capture_size=4,
                    ),
                    profiler_config=ProfilerConfig(
                        profiler="torch",
                        torch_profiler_dir=str(output_dir / "traces"),
                        torch_profiler_with_stack=False,
                        torch_profiler_with_memory=False,
                        ignore_frontend=True,
                    ),
                ).create_engine_config()
                expected = (
                    CUDAGraphMode.FULL_DECODE_ONLY
                    if args.mode == "graph"
                    else CUDAGraphMode.NONE
                )
                assert config.compilation_config.cudagraph_mode == expected
                return AsyncLLM(
                    vllm_config=config,
                    executor_class=ExternalExecutor,
                    log_stats=False,
                    external_actors=actors,
                )

            engines[slot] = await asyncio.to_thread(build)
            seconds = time.perf_counter() - started
            result["startup"].append({"slot": slot, "seconds": seconds})
            emit("profile_engine_ready", slot=slot, seconds=seconds)
        await batch("warmup", 128, 64, 4)
        for input_tokens in (128, 512):
            for concurrency in (1, 4):
                for repeat in range(args.repeats):
                    await batch(
                        f"measure-{input_tokens}-{concurrency}-{repeat}",
                        input_tokens,
                        64,
                        concurrency,
                    )
        # Keep profiler overhead outside the latency/throughput samples above.
        # Trace both TP ranks of one instance while all eight engines generate.
        await engines[0].start_profile("pool-8x2-slot-0")
        try:
            await batch("trace", 128, 16, 1)
        finally:
            await engines[0].stop_profile()
        snapshots = await asyncio.to_thread(
            graph_snapshots, [a for actors in leases.values() for a in actors]
        )
        result["graph_state"] = snapshots
        for snapshot in snapshots:
            assert snapshot["dtype"] == "torch.bfloat16", snapshot
            if args.mode == "graph":
                assert snapshot["captured_graphs"], snapshot
            else:
                assert not snapshot["captured_graphs"], snapshot
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["errors"].append(traceback.format_exc())
        emit("profile_error", traceback=traceback.format_exc())
    finally:
        for slot in list(leases):
            started = time.perf_counter()
            try:
                engine = engines.pop(slot, None)
                if engine is not None:
                    await asyncio.to_thread(_detach_stuck_mm_warmup, engine)
                    await asyncio.to_thread(engine.shutdown, 180.0)
                actors = leases.pop(slot)
                await asyncio.to_thread(pool.release, actors)
                cleanup = await asyncio.to_thread(inspect_workers, actors)
                assert_released(cleanup)
                result.setdefault("communication_release", []).append(
                    {"slot": slot, "workers": cleanup}
                )
                result["shutdown"].append(
                    {"slot": slot, "seconds": time.perf_counter() - started}
                )
            except Exception:
                result["status"] = "failed"
                result["errors"].append(traceback.format_exc())
        try:
            pids_after = ray.get(
                [
                    a.__ray_call__.remote(lambda _: __import__("os").getpid())
                    for a in handles
                ]
            )
            result["pids_after"] = pids_after
            assert pids_before == pids_after, "Actor processes changed"
            regs = ray.get(pool.registry.list_actors.remote())
            assert len(regs) == 16 and all(r.state == "idle" for r in regs)
            result["final_idle_actors"] = 16
        except Exception:
            result["status"] = "failed"
            result["errors"].append(traceback.format_exc())
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        emit(
            "profile_complete",
            status=result["status"],
            metrics=str(output_dir / "metrics.json"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", default="/tmp/vllm-rolling-8x2")
    parser.add_argument("--output-dir", default="/tmp/vllm-profile-8x2")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--engine-core-pool-size", type=int, default=0)
    parser.add_argument(
        "--model",
        default=(
            "/opt/huawei/dataset/ccb_data/code/ventsing/downloaded_models/Qwen3.5-9B"
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.instances <= 8 or args.repeats < 1:
        parser.error("instances must be 1..8 and repeats must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Wait for the rolling workload to release leases, never competing with it.
    result_path = Path(args.pool_dir) / "result.json"
    while not result_path.exists():
        time.sleep(5)
    rolling = json.loads(result_path.read_text())
    if rolling["status"] != "passed":
        raise RuntimeError("Rolling test failed; inspect it before profiling")
    connection = json.loads((Path(args.pool_dir) / "connection.json").read_text())
    ray.init(
        address="auto",
        runtime_env={
            "env_vars": {
                key: os.environ[key]
                for key in (
                    "PYTHONPATH",
                    "VLLM_NO_USAGE_STATS",
                    "VLLM_VERSION",
                    "VLLM_WORKER_MULTIPROC_METHOD",
                )
            }
        },
    )
    pool = RetainedPool()
    pool.attach(registry_name=connection["registry_name"])
    core_pool = None
    old_cache_settings = ray.get(
        [actor.__ray_call__.remote(set_compile_cache, "1") for actor in pool.actors]
    )
    try:
        if args.engine_core_pool_size:
            context = ray.get_runtime_context()
            core_pool = EngineCoreProcessPool(
                args.engine_core_pool_size,
                context.gcs_address,
                context.namespace,
                preload_modules=("vllm_external_executor",),
            )
            started = time.perf_counter()
            core_pool.pre_start()
            (output_dir / "prewarm.json").write_text(
                json.dumps(
                    {
                        "seconds": time.perf_counter() - started,
                        "pids": core_pool.idle_pids(),
                    },
                    indent=2,
                )
            )
        with core_pool.activate() if core_pool else contextlib.nullcontext():
            asyncio.run(run(args, pool))
    finally:
        ray.get(
            [
                actor.__ray_call__.remote(set_compile_cache, value)
                for actor, value in zip(pool.actors, old_cache_settings)
            ]
        )
        if core_pool is not None:
            core_pool.close()
        # This job owns no actors. The rolling driver retains the entire pool.
        pool.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
