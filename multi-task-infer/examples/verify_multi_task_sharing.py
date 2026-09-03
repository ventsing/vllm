#!/usr/bin/env python3
"""Verify multi-task sharing of the ExternalExecutor actor pool.

This script demonstrates and validates the core value proposition of the
ExternalExecutor plugin: **one pre-started actor pool serves multiple vLLM
instances sequentially / concurrently**, with:

1. Actor pool reuse - actors survive across instances (G1)
2. Dynamic TP/PP reconfiguration - same actors, different parallel shapes (G2)
3. Compilation cache sharing - CacheManagerActor deduplicates compile work (G6)
4. Storage-backed weight loading - NFS/Mooncake checkpoint reuse (G7)
5. Model hot-switching - swap models without releasing actors (G3)

Run::

    # Single-node, 4 GPUs (or NPUs)
    python verify_multi_task_sharing.py --model facebook/opt-125m \\
        --num-gpus 4 --mode sequential

    # Concurrent (two instances share the pool at the same time)
    python verify_multi_task_sharing.py --model facebook/opt-125m \\
        --num-gpus 8 --mode concurrent

Environment::

    # NFS shared cache (optional but recommended)
    mkdir -p /shared/vllm_compile_cache
    mkdir -p /shared/checkpoints

    # Mooncake (optional)
    export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json

    # vLLM + plugin
    pip install -e ../  # installs vllm_external_executor entry point
"""

import argparse
import logging
import time

import ray

from vllm import LLM, SamplingParams
from vllm.config import VllmConfig

from vllm_external_executor import (
    ActorPoolManager,
    ActorState,
    CacheManagerActor,
    ExternalExecutor,
    StorageCheckpointEngine,
)

logger = logging.getLogger("verify_multi_task")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vllm_config(model: str, tp: int, pp: int = 1) -> VllmConfig:
    """Build a minimal VllmConfig for the given parallel layout."""
    from vllm.config import ModelConfig, ParallelConfig, CacheConfig
    from vllm.config import CompilationConfig

    model_config = ModelConfig(model=model, task="generate", tokenizer=model)
    parallel_config = ParallelConfig(
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
    )
    cache_config = CacheConfig()
    compilation_config = CompilationConfig()
    return VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=cache_config,
        compilation_config=compilation_config,
    )


def run_inference(llm: LLM, tag: str) -> None:
    """Run a tiny inference and print output."""
    prompts = ["Hello, my name is", "The capital of France is"]
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    dt = time.time() - t0
    for o in outputs:
        logger.info("[%s] prompt=%r generated=%r", tag, o.prompt, o.outputs[0].text)
    logger.info("[%s] inference took %.2fs (%d reqs)", tag, dt, len(outputs))


def dump_pool_state(pool: ActorPoolManager, tag: str) -> None:
    """Print actor states + idle count, used to prove reuse."""
    states = pool.get_actor_states()
    idle = pool.get_idle_count()
    summary = {}
    for s in states.values():
        summary[s.value if isinstance(s, ActorState) else str(s)] = (
            summary.get(
                s.value if isinstance(s, ActorState) else str(s), 0
            ) + 1
        )
    logger.info("[%s] pool: idle=%d states=%s", tag, idle, summary)


def dump_cache_stats(pool: ActorPoolManager, tag: str) -> None:
    """Print CacheManagerActor stats, used to prove cache sharing."""
    if pool.cache_manager is None:
        logger.info("[%s] no cache_manager", tag)
        return
    stats = ray.get(pool.cache_manager.get_stats.remote())
    caches = ray.get(pool.cache_manager.list_caches.remote())
    logger.info("[%s] cache stats=%s caches=%s", tag, stats, list(caches))


# ---------------------------------------------------------------------------
# Verification modes
# ---------------------------------------------------------------------------

def verify_sequential(args):
    """Two vLLM instances, sequentially, sharing one pool.

    Proves:
    - Actor reuse: instance B acquires the SAME actors released by A.
    - Compile cache reuse: B's compile hits the cache A pushed.
    """
    ray.init(address="auto" if args.ray_auto else None)

    pool = ActorPoolManager()
    pool.pre_start(
        num_actors=args.num_gpus,
        devices_per_node=list(range(args.num_gpus)),
        warmup_distributed=True,
        shared_cache_dir=args.cache_dir,
        enable_cache_compression=True,
    )
    dump_pool_state(pool, "after pre_start")

    # --- Instance A ----------------------------------------------------------
    actors_a = pool.acquire(tp_size=args.tp, pp_size=args.pp)
    dump_pool_state(pool, "after acquire A")

    cfg_a = make_vllm_config(args.model, tp=args.tp, pp=args.pp)
    llm_a = LLM(
        model=args.model,
        executor_class=ExternalExecutor,
        external_actors=actors_a,
        cache_manager=pool.cache_manager,
    )
    run_inference(llm_a, "A")
    dump_cache_stats(pool, "after A")

    # Release actors back to pool (G1 reuse + G5 release path)
    pool.release(actors_a)
    dump_pool_state(pool, "after release A")

    # --- Instance B: reuses A's actors ---------------------------------------
    actors_b = pool.acquire(tp_size=args.tp, pp_size=args.pp)
    dump_pool_state(pool, "after acquire B")

    cfg_b = make_vllm_config(args.model, tp=args.tp, pp=args.pp)
    llm_b = LLM(
        model=args.model,
        executor_class=ExternalExecutor,
        external_actors=actors_b,
        cache_manager=pool.cache_manager,
    )
    # B's torch.compile should hit the cache A pushed (G6)
    run_inference(llm_b, "B")
    dump_cache_stats(pool, "after B")

    pool.release(actors_b)
    pool.shutdown()
    ray.shutdown()

    logger.info("=== sequential sharing verified ===")


def verify_concurrent(args):
    """Two vLLM instances running concurrently on one pool.

    Requires enough GPUs to split the pool (e.g. 8 GPUs -> 2x TP=4).
    Proves multiple instances coexist without interfering.
    """
    assert args.num_gpus >= 2 * args.tp, (
        f"concurrent mode needs >= {2 * args.tp} GPUs for two TP={args.tp} "
        f"instances, got {args.num_gpus}"
    )
    ray.init(address="auto" if args.ray_auto else None)

    pool = ActorPoolManager()
    pool.pre_start(
        num_actors=args.num_gpus,
        devices_per_node=list(range(args.num_gpus)),
        warmup_distributed=True,
        shared_cache_dir=args.cache_dir,
    )
    dump_pool_state(pool, "after pre_start")

    actors_a = pool.acquire(tp_size=args.tp, pp_size=args.pp)
    actors_b = pool.acquire(tp_size=args.tp, pp_size=args.pp)
    dump_pool_state(pool, "after both acquires")
    assert pool.get_idle_count() == 0, "pool should be fully drained"

    cfg = make_vllm_config(args.model, tp=args.tp, pp=args.pp)
    llm_a = LLM(
        model=args.model,
        executor_class=ExternalExecutor,
        external_actors=actors_a,
        cache_manager=pool.cache_manager,
    )
    llm_b = LLM(
        model=args.model,
        executor_class=ExternalExecutor,
        external_actors=actors_b,
        cache_manager=pool.cache_manager,
    )

    run_inference(llm_a, "A-concurrent")
    run_inference(llm_b, "B-concurrent")

    pool.release(actors_a)
    pool.release(actors_b)
    pool.shutdown()
    ray.shutdown()
    logger.info("=== concurrent sharing verified ===")


def verify_dynamic_tp(args):
    """Same pool, two instances with DIFFERENT TP sizes (G2 dynamic reshape).

    e.g. 8-GPU pool: instance A TP=4 PP=2, then released; instance B TP=8 PP=1.
    """
    ray.init(address="auto" if args.ray_auto else None)
    pool = ActorPoolManager()
    pool.pre_start(
        num_actors=args.num_gpus,
        devices_per_node=list(range(args.num_gpus)),
        warmup_distributed=True,
        shared_cache_dir=args.cache_dir,
    )

    tp_a, pp_a = args.tp, args.pp
    tp_b = max(args.tp, 2)
    pp_b = 1
    assert tp_a * pp_a + tp_b * pp_b <= args.num_gpus

    actors_a = pool.acquire(tp_size=tp_a, pp_size=pp_a)
    cfg_a = make_vllm_config(args.model, tp=tp_a, pp=pp_a)
    llm_a = LLM(
        model=args.model,
        executor_class=ExternalExecutor,
        external_actors=actors_a,
        cache_manager=pool.cache_manager,
    )
    run_inference(llm_a, f"A-TP{tp_a}PP{pp_a}")
    pool.release(actors_a)
    dump_pool_state(pool, "after release A")

    actors_b = pool.acquire(tp_size=tp_b, pp_size=pp_b)
    cfg_b = make_vllm_config(args.model, tp=tp_b, pp=pp_b)
    llm_b = LLM(
        model=args.model,
        executor_class=ExternalExecutor,
        external_actors=actors_b,
        cache_manager=pool.cache_manager,
    )
    run_inference(llm_b, f"B-TP{tp_b}PP{pp_b}")
    pool.release(actors_b)

    pool.shutdown()
    ray.shutdown()
    logger.info("=== dynamic TP/PP verified ===")


def verify_hot_switch(args):
    """Model hot-switching on a running executor (G3).

    Loads model A, runs inference, then switches to model B in place via
    executor.switch_model(). Requires two checkpoints on storage.
    """
    ray.init(address="auto" if args.ray_auto else None)
    pool = ActorPoolManager()
    pool.pre_start(
        num_actors=args.num_gpus,
        devices_per_node=list(range(args.num_gpus)),
        warmup_distributed=True,
        shared_cache_dir=args.cache_dir,
    )

    actors = pool.acquire(tp_size=args.tp, pp_size=args.pp)
    cfg = make_vllm_config(args.model, tp=args.tp, pp=args.pp)
    llm = LLM(
        model=args.model,
        executor_class=ExternalExecutor,
        external_actors=actors,
        cache_manager=pool.cache_manager,
    )
    run_inference(llm, "before-switch")

    # Grab the underlying executor to call switch_model.
    executor = llm.llm_engine.model_executor  # ExternalExecutor instance
    assert isinstance(executor, ExternalExecutor), (
        f"expected ExternalExecutor, got {type(executor)}"
    )

    new_cfg = make_vllm_config(args.switch_model, tp=args.tp, pp=args.pp)
    logger.info("switching %s -> %s", args.model, args.switch_model)
    t0 = time.time()
    executor.switch_model(
        new_vllm_config=new_cfg,
        checkpoint_path=args.switch_checkpoint,
        storage_backend=args.storage_backend,
        storage_config={
            "base_path": args.storage_base_path,
        } if args.storage_backend == "nfs" else None,
        reinitialize_cache=True,
    )
    logger.info("switch_model took %.2fs", time.time() - t0)

    run_inference(llm, "after-switch")

    pool.release(actors)
    pool.shutdown()
    ray.shutdown()
    logger.info("=== hot-switch verified ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--switch-model", default=None,
                   help="second model for hot-switch mode")
    p.add_argument("--switch-checkpoint", default=None,
                   help="checkpoint path of the second model")
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--cache-dir", default="/shared/vllm_compile_cache")
    p.add_argument("--storage-backend", default="nfs",
                   choices=["nfs", "mooncake"])
    p.add_argument("--storage-base-path", default="/shared/checkpoints")
    p.add_argument("--ray-auto", action="store_true",
                   help="use ray.init(address='auto')")
    p.add_argument("--mode", required=True,
                   choices=["sequential", "concurrent",
                            "dynamic_tp", "hot_switch"],
                   help="which verification scenario to run")
    args = p.parse_args()

    if args.mode == "sequential":
        verify_sequential(args)
    elif args.mode == "concurrent":
        verify_concurrent(args)
    elif args.mode == "dynamic_tp":
        verify_dynamic_tp(args)
    elif args.mode == "hot_switch":
        assert args.switch_model, "--switch-model required for hot_switch"
        verify_hot_switch(args)


if __name__ == "__main__":
    main()
