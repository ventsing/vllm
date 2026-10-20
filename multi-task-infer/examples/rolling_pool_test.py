# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise eight TP=2 engines rolling over a retained 16-device actor pool.

Run with the CANN environment loaded and this plugin on PYTHONPATH. The driver
stays alive after the test to retain the pool; neither success nor failure kills
actors. Results and connection details are written to --output-dir.
"""

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
import vllm_external_executor.actor_pool_manager as pool_module
from pool_diagnostics import assert_released, inspect_workers
from vllm_external_executor import ActorPoolManager, ExternalExecutor
from vllm_external_executor.mvp_entry import _detach_stuck_mm_warmup
from vllm_external_executor.node_registry_actor import NodeRegistryActor

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.process_pool import (
    EngineCoreProcessPool,
    get_active_engine_core_pool,
)

PROMPTS = ["The capital of France is", "1 + 1 ="]


class RetainedPool(ActorPoolManager):
    """Keep actor ownership with this driver, including on test failures."""

    def _start_heartbeat(self):
        # The test explicitly probes every actor; automatic rebuild would
        # invalidate the actor-survival invariant this test is checking.
        pass

    def shutdown(self):
        self._hb_stop.set()

    def rebuild_actor(self, *args, **kwargs):
        raise RuntimeError("Actor rebuild is forbidden in the rolling test")


def emit(event, **fields):
    record = {"event": event, "time": time.time(), **fields}
    print(json.dumps(record, default=str), flush=True)
    return record


async def run_test(args, pool, output_dir):
    engines = {}
    leases = {}
    records = []
    baseline = {}

    async def snapshot(expected_idle):
        regs = await asyncio.to_thread(ray.get, pool.registry.list_actors.remote())
        assert len(regs) == 16, regs
        idle = sum(reg.state == "idle" for reg in regs)
        assert idle == expected_idle, (idle, expected_idle)
        assert all(reg.state in ("idle", "leased", "running") for reg in regs)
        pids = await asyncio.to_thread(
            ray.get,
            [
                actor.__ray_call__.remote(lambda _: __import__("os").getpid())
                for actor in pool.actors
            ],
        )
        identity = [
            (actor._actor_id.hex(), pid) for actor, pid in zip(pool.actors, pids)
        ]
        if baseline:
            assert identity == baseline["identity"], "Actor process changed"
        else:
            baseline["identity"] = identity
        active_ids = [
            actor._actor_id.hex() for actors in leases.values() for actor in actors
        ]
        assert len(active_ids) == len(set(active_ids)), "Overlapping leases"
        return {"idle": idle, "actors": identity}

    async def create(slot, generation):
        started = time.monotonic()
        core_pool = get_active_engine_core_pool()
        warm_pids = core_pool.idle_pids() if core_pool else []
        actors = pool.acquire(tp_size=2, pp_size=1)
        leases[slot] = actors

        def build():
            config = AsyncEngineArgs(
                model=args.model,
                tensor_parallel_size=2,
                pipeline_parallel_size=1,
                served_model_name=f"rolling-slot-{slot}-generation-{generation}",
                enforce_eager=True,
                max_model_len=1024,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=0.65,
                enable_prefix_caching=False,
                language_model_only=True,
                mamba_ssm_cache_dtype="bfloat16",
            ).create_engine_config()
            return AsyncLLM(
                vllm_config=config,
                executor_class=ExternalExecutor,
                log_stats=False,
                external_actors=actors,
            )

        engines[slot] = await asyncio.to_thread(build)
        manager = engines[slot].engine_core.resources.engine_manager
        core_pids = [process.pid for process in manager.processes]
        if core_pool:
            assert all(pid in warm_pids for pid in core_pids), (core_pids, warm_pids)
        records.append(
            emit(
                "engine_ready",
                slot=slot,
                generation=generation,
                seconds=time.monotonic() - started,
                actors=[actor._actor_id.hex() for actor in actors],
                engine_core_pids=core_pids,
                prewarmed=core_pool is not None,
            )
        )

    async def close(slot):
        llm = engines.pop(slot, None)
        if llm is not None:
            await asyncio.to_thread(_detach_stuck_mm_warmup, llm)
            await asyncio.to_thread(llm.shutdown, 180.0)
        actors = leases.pop(slot, None)
        if actors:
            await asyncio.to_thread(pool.release, actors)
            cleanup = await asyncio.to_thread(inspect_workers, actors)
            assert_released(cleanup)
            records.append(emit("communication_released", slot=slot, workers=cleanup))

    async def infer(slot, phase):
        start = time.monotonic()
        outputs = []
        for i, prompt in enumerate(PROMPTS):
            final = None
            async for output in engines[slot].generate(
                prompt,
                SamplingParams(temperature=0, max_tokens=8),
                request_id=f"{phase}-slot-{slot}-prompt-{i}-{uuid.uuid4().hex}",
            ):
                final = output
            assert final is not None and final.finished
            outputs.append(final.outputs[0].text)
        assert outputs[0].lstrip().startswith("Paris"), outputs
        assert outputs[1].lstrip().startswith("2"), outputs
        key = f"slot-{slot}"
        if key in baseline:
            assert outputs == baseline[key], (baseline[key], outputs)
        else:
            baseline[key] = outputs
        records.append(
            emit(
                "inference_pass",
                slot=slot,
                phase=phase,
                seconds=time.monotonic() - start,
                outputs=outputs,
            )
        )

    status = "failed"
    error = None
    try:
        await snapshot(16)
        for slot in range(8):
            await create(slot, 0)
            await snapshot(16 - 2 * (slot + 1))
        await asyncio.gather(*(infer(slot, "initial") for slot in engines))
        records.append(emit("eight_instances_pass", **await snapshot(0)))
        for generation in range(1, args.cycles + 1):
            for slot in range(8):
                old_ids = [actor._actor_id.hex() for actor in leases[slot]]
                await close(slot)
                await snapshot(2)
                # Probe all seven surviving engines while the eighth is absent.
                phase = f"cycle-{generation}-without-{slot}"
                await asyncio.gather(*(infer(other, phase) for other in engines))
                await create(slot, generation)
                new_ids = [actor._actor_id.hex() for actor in leases[slot]]
                assert new_ids == old_ids, (old_ids, new_ids)
                await infer(slot, f"cycle-{generation}-replacement-{slot}")
                records.append(
                    emit(
                        "roll_pass",
                        slot=slot,
                        cycle=generation,
                        reused_actors=new_ids,
                        **await snapshot(0),
                    )
                )
        status = "passed"
    except Exception:
        error = traceback.format_exc()
        emit("test_error", traceback=error)
    finally:
        for slot in list(leases):
            try:
                await close(slot)
            except Exception:
                status = "failed"
                error = (error or "") + traceback.format_exc()
                emit("cleanup_error", slot=slot, traceback=traceback.format_exc())
        try:
            final = await snapshot(16)
        except Exception:
            status = "failed"
            error = (error or "") + traceback.format_exc()
            final = None
        result = {
            "status": status,
            "error": error,
            "records": records,
            "final": final,
            "cycles": args.cycles,
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2))
        emit(
            "test_complete",
            status=status,
            final=final,
            result=str(output_dir / "result.json"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=(
            "/opt/huawei/dataset/ccb_data/code/ventsing/downloaded_models/Qwen3.5-9B"
        ),
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--output-dir", default="/tmp/vllm-rolling-8x2")
    parser.add_argument("--registry-name")
    parser.add_argument("--engine-core-pool-size", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    args = parser.parse_args()
    if args.cycles < 1 or args.engine_core_pool_size < 0:
        parser.error("--cycles must be positive and pool size nonnegative")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_module.RPC_TIMEOUT = 180.0
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
    available = ray.available_resources()
    if args.registry_name is None and available.get("NPU", 0) < 16:
        raise RuntimeError(f"Need 16 free NPUs: {available}")
    registry_name = args.registry_name or f"rolling-8x2-{os.getpid()}"
    pool = RetainedPool()
    connection = {
        "driver_pid": os.getpid(),
        "registry_name": registry_name,
        "namespace": "external-executor",
        "ray_address": "auto",
    }
    (output_dir / "connection.json").write_text(json.dumps(connection, indent=2))
    core_pool = None
    try:
        if args.registry_name:
            pool.attach(registry_name=args.registry_name)
        else:
            registry = (
                ray.remote(NodeRegistryActor)
                .options(name=registry_name, namespace="external-executor")
                .remote()
            )
            pool.pre_start(
                16, list(range(16)), warmup_distributed=False, registry=registry
            )
        emit("pool_ready", **connection)
        if args.engine_core_pool_size:
            context = ray.get_runtime_context()
            core_pool = EngineCoreProcessPool(
                args.engine_core_pool_size,
                context.gcs_address,
                context.namespace,
                preload_modules=("vllm_external_executor",),
            )
            started = time.monotonic()
            core_pool.pre_start()
            warmup = emit(
                "engine_core_pool_ready",
                seconds=time.monotonic() - started,
                pids=core_pool.idle_pids(),
            )
            (output_dir / "engine_core_pool.json").write_text(
                json.dumps(warmup, indent=2)
            )
        with core_pool.activate() if core_pool else contextlib.nullcontext():
            asyncio.run(run_test(args, pool, output_dir))
        if core_pool:
            core_pool.wait_ready(core_pool.size)
            emit("engine_core_pool_retained", pids=core_pool.idle_pids())
    except Exception:
        error = traceback.format_exc()
        (output_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": error,
                },
                indent=2,
            )
        )
        emit("startup_error", traceback=error)
    emit("pool_retained", **connection)
    # Keep the owning Ray job and all actor handles alive after the test.
    # Do not call pool.shutdown(), ray.shutdown(), or ray.kill().
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
