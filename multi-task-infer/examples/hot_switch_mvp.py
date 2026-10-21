#!/usr/bin/env python3
"""End-to-end scheme-B hot switch: one live AsyncLLM across TP transitions.

Drives :func:`run_hot_switch_mvp`: boots on a ``max_tp`` actor lease and calls
``AsyncLLM.reconfigure`` through every ``(model, tp)`` pair without releasing
actors or recreating the EngineCore. Distinct tokenizers are handled by the
client-side tokenizer/renderer swap.
"""

import argparse

import ray

from vllm_external_executor import ActorPoolManager
from vllm_external_executor.mvp_entry import (
    SCHEME_B_CUDAGRAPH_MODE,
    SCHEME_B_DECODE_CAPTURE_SIZES,
    run_hot_switch_mvp,
)

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]

# (model_path, tp_size); max TP drives the pre-started actor count.
SEQUENCE = [
    ("/opt/huawei/dataset/downloaded_models/Qwen3.5-4B", 4),
    ("/opt/huawei/dataset/downloaded_models/Qwen3.5-4B", 2),
    ("/opt/huawei/dataset/downloaded_models/Qwen3.5-4B", 1),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        help="Repeat as model:tp (e.g. /path/to/model:4)",
    )
    parser.add_argument(
        "--capture-trim",
        action="store_true",
        help="Scheme B: decode-only capture, 4 sizes (1,2,4,8)",
    )
    parser.add_argument(
        "--skip-memory-profile",
        action="store_true",
        help="Skip the dummy-forward memory profile on every switch and "
        "derive num_gpu_blocks from --kv-cache-memory-bytes instead",
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=None,
        help="Static per-GPU KV cache budget in bytes (required along with "
        "--skip-memory-profile)",
    )
    args = parser.parse_args()

    sequence = SEQUENCE
    if args.model:
        sequence = []
        for entry in args.model:
            model, _, tp = entry.rpartition(":")
            sequence.append((model, int(tp)))
    if not sequence:
        parser.error("empty (model, tp) sequence")

    ray.init()
    max_tp = max(tp for _, tp in sequence)

    pool = ActorPoolManager()
    pool.pre_start(
        num_actors=max_tp,
        devices_per_node=list(range(max_tp)),
        warmup_distributed=True,
    )
    try:
        outputs = run_hot_switch_mvp(
            sequence,
            PROMPTS,
            pool=pool,
            cudagraph_capture_sizes=(
                SCHEME_B_DECODE_CAPTURE_SIZES if args.capture_trim else None
            ),
            cudagraph_mode=(
                SCHEME_B_CUDAGRAPH_MODE if args.capture_trim else None
            ),
            skip_memory_profile=args.skip_memory_profile,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        )
        for (model, tp), texts in zip(sequence, outputs):
            print(f"=== {model} (TP={tp}) ===")
            for prompt, text in zip(PROMPTS, texts):
                print(f"  {prompt!r} -> {text!r}")
    finally:
        pool.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
