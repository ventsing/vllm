# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare retained TP=2 actors with fresh multiprocessing workers."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_VERSION", "0.28.1")
PROCESS_STARTED = time.perf_counter()


async def trial(args):
    import ray
    from pool_diagnostics import assert_released, inspect_workers
    from rolling_pool_test import RetainedPool
    from vllm_external_executor import ExternalExecutor
    from vllm_external_executor.mvp_entry import _detach_stuck_mm_warmup

    from vllm import SamplingParams
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    pool = None
    actors = None
    engine = None
    row = {
        "backend": args.backend,
        "mode": args.mode,
        "disable_compile_cache": args.disable_compile_cache,
        "status": "failed",
    }
    try:
        if args.backend == "pooled":
            connection = json.loads(Path(args.pool_dir, "connection.json").read_text())
            attach_start = time.perf_counter()
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
            row["attach_seconds"] = time.perf_counter() - attach_start
        started = time.perf_counter()
        if pool is not None:
            actors = pool.acquire(tp_size=2, pp_size=1)
        config = AsyncEngineArgs(
            model=args.model,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            distributed_executor_backend="mp" if pool is None else None,
            served_model_name=f"startup-{args.backend}",
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
        ).create_engine_config()
        kwargs = {
            "vllm_config": config,
            "log_stats": False,
            "executor_class": MultiprocExecutor,
        }
        if pool is not None:
            kwargs.update(executor_class=ExternalExecutor, external_actors=actors)
        engine = await asyncio.to_thread(AsyncLLM, **kwargs)
        row["ready_seconds"] = time.perf_counter() - started
        row["driver_to_ready_seconds"] = time.perf_counter() - PROCESS_STARTED
        print(json.dumps({"event": "startup_ready", **row}), flush=True)
        first = None
        output = None
        async for item in engine.generate(
            "The capital of France is",
            SamplingParams(temperature=0, max_tokens=8),
            request_id=f"startup-{args.backend}",
        ):
            if first is None and item.outputs and item.outputs[0].token_ids:
                first = time.perf_counter()
            output = item
        row["first_token_seconds"] = first - started
        row["first_request_seconds"] = time.perf_counter() - started
        row["output"] = output.outputs[0].text
        assert row["output"].lstrip().startswith("Paris"), row
        row["status"] = "passed"
    except Exception:
        row["error"] = traceback.format_exc()
    finally:
        stop = time.perf_counter()
        try:
            if engine is not None:
                await asyncio.to_thread(_detach_stuck_mm_warmup, engine)
                await asyncio.to_thread(engine.shutdown, 180.0)
            if actors is not None:
                await asyncio.to_thread(pool.release, actors)
                cleanup = inspect_workers(actors)
                assert_released(cleanup)
                row["communication_release"] = cleanup
        except Exception:
            row["status"] = "failed"
            row["cleanup_error"] = traceback.format_exc()
        row["shutdown_seconds"] = time.perf_counter() - stop
        Path(args.result).write_text(json.dumps(row, indent=2))
        if pool is not None:
            pool.shutdown()
            ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pooled", "cold"))
    parser.add_argument("--result")
    parser.add_argument("--pool-dir", default="/tmp/vllm-rolling-8x2")
    parser.add_argument("--profile-dir", default="/tmp/vllm-profile-8x2-fixed")
    parser.add_argument("--output-dir", default="/tmp/vllm-startup-comparison")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--disable-compile-cache", action="store_true")
    parser.add_argument(
        "--model",
        default=(
            "/opt/huawei/dataset/ccb_data/code/ventsing/downloaded_models/Qwen3.5-9B"
        ),
    )
    args = parser.parse_args()
    if args.disable_compile_cache:
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    if args.backend:
        asyncio.run(trial(args))
        return
    profile_result = Path(args.profile_dir, "metrics.json")
    while not profile_result.exists():
        time.sleep(5)
    assert json.loads(profile_result.read_text())["status"] == "passed"
    import ray

    connection = json.loads(Path(args.pool_dir, "connection.json").read_text())
    ray.init(address="auto")
    registry = ray.get_actor(
        connection["registry_name"], namespace=connection["namespace"]
    )
    regs = ray.get(registry.list_actors.remote())
    assert len(regs) == 16 and all(reg.state == "idle" for reg in regs)
    handles = ray.get(registry.get_actor_handles.remote())
    pair = sorted(regs, key=lambda reg: reg.device_id)[:2]
    physical = ray.get(
        [handles[reg.actor_id].get_node_and_physical_gpu_ids.remote() for reg in pair]
    )
    devices = ",".join(str(ids[0]) for _, ids in physical)
    ray.shutdown()
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(args.repeats):
        order = ("pooled", "cold") if repeat % 2 == 0 else ("cold", "pooled")
        for backend in order:
            result_path = directory / f"{repeat}-{backend}.json"
            env = dict(os.environ)
            if backend == "cold":
                env["ASCEND_RT_VISIBLE_DEVICES"] = devices
            command = [
                sys.executable,
                "-u",
                __file__,
                "--backend",
                backend,
                "--model",
                args.model,
                "--pool-dir",
                args.pool_dir,
                "--result",
                str(result_path),
                "--mode",
                args.mode,
            ]
            if args.disable_compile_cache:
                command.append("--disable-compile-cache")
            launched = time.perf_counter()
            with (directory / f"{repeat}-{backend}.log").open("w") as log:
                completed = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600
                )
            row = json.loads(result_path.read_text())
            row.update(
                repeat=repeat,
                process_exit_code=completed.returncode,
                process_total_seconds=time.perf_counter() - launched,
            )
            rows.append(row)
            (directory / "comparison.json").write_text(
                json.dumps(
                    {
                        "devices": devices,
                        "rows": rows,
                        "complete": len(rows) == args.repeats * 2,
                    },
                    indent=2,
                )
            )
            print(json.dumps(row), flush=True)
            assert row["status"] == "passed", row


if __name__ == "__main__":
    main()
