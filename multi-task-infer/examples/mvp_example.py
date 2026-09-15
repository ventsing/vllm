#!/usr/bin/env python3
"""Minimal MVP example: sequential model reuse over one pre-started actor pool.

Each call to :func:`run_mvp` acquires actors, runs one model, and returns the
actors in ``finally``; the *same* actors then serve the next model without a
new pool or actor creation.
"""

import ray

from vllm_external_executor.mvp_entry import run_mvp

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]

# Phase 1 (TP=1) / phase 2 (TP=2): switch tp_size to 2 once TP=2 is validated.
TP_SIZE = 1
MODELS = [
    "facebook/opt-125m",
    "facebook/opt-350m",
]


def main() -> None:
    ray.init()

    for model in MODELS:
        outputs = run_mvp(model, PROMPTS, tp_size=TP_SIZE)
        print(f"=== {model} (TP={TP_SIZE}) ===")
        for prompt, text in zip(PROMPTS, outputs):
            print(f"  {prompt!r} -> {text!r}")

    ray.shutdown()


if __name__ == "__main__":
    main()
