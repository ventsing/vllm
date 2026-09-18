#!/usr/bin/env python3
"""Multi-process actor-pool smoke test.

One process owns the pool (``--mode owner``): it pre-starts actors and stays
alive, since the actors are not detached (they die with the owner). Other
processes (``--mode worker``) attach to the same registry and concurrently
acquire -> run -> release actors without ever creating actors themselves.

Run in separate shells against the shared Ray cluster::

    # shell 1: owner stays alive and prints the pool view
    python examples/multi_process_pool_test.py --mode owner \
        --ray-address 172.16.15.120:6925 --num-actors 4 --devices 0,1,2,3

    # shell 2/3/...: each worker attaches and runs N rounds
    python examples/multi_process_pool_test.py --mode worker \
        --ray-address 172.16.15.120:6925 --model /path/to/model \
        --tp-size 1 --rounds 3

Concurrency correctness (atomic lease, generation-gated release) is exercised
by running several workers at once; total ``--tp-size`` across live workers
must not exceed ``--num-actors``.
"""

import argparse
import time

import ray

from vllm_external_executor import ActorPoolManager
from vllm_external_executor.actor_pool_manager import REGISTRY_ACTOR_NAME
from vllm_external_executor.mvp_entry import run_mvp

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]


def _run_owner(args: argparse.Namespace) -> None:
    pool = ActorPoolManager()
    pool.pre_start(
        num_actors=args.num_actors,
        devices_per_node=args.devices,
        warmup_distributed=True,
    )
    print(
        f"[owner] pool ready: registry={REGISTRY_ACTOR_NAME!r} "
        f"actors={args.num_actors} devices={args.devices}"
    )
    try:
        while True:
            view = ray.get(pool.registry.get_global_view.remote())
            print(f"[owner] {view['num_actors']} actors: {view['fault_domains']}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("[owner] shutting down")
    finally:
        pool.shutdown()


def _run_worker(args: argparse.Namespace) -> None:
    pool = ActorPoolManager()
    pool.attach(registry_name=REGISTRY_ACTOR_NAME)
    print(
        f"[worker {args.worker_id}] attached to {len(pool.actors)} actors"
    )
    for index in range(args.rounds):
        outputs = run_mvp(
            args.model,
            PROMPTS,
            tp_size=args.tp_size,
            pool=pool,
            enforce_eager=args.enforce_eager,
        )
        print(f"[worker {args.worker_id}] round {index + 1}/{args.rounds}:")
        for prompt, text in zip(PROMPTS, outputs):
            print(f"    {prompt!r} -> {text!r}")
    print(f"[worker {args.worker_id}] done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("owner", "worker"), required=True
    )
    parser.add_argument(
        "--ray-address", default="172.16.15.120:6925",
        help="Shared Ray cluster address",
    )
    parser.add_argument("--num-actors", type=int, default=4)
    parser.add_argument(
        "--devices", type=lambda s: [int(x) for x in s.split(",")],
        default=[0, 1, 2, 3],
    )
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--worker-id", default="0")
    parser.add_argument(
        "--model",
        default="/opt/huawei/dataset/downloaded_models/Qwen3.5-4B",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    ray.init(address=args.ray_address)
    try:
        if args.mode == "owner":
            _run_owner(args)
        else:
            _run_worker(args)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
