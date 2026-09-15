# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""
Single-node sequential-reuse MVP entry point.

This is the *one* supported way to run the pooled executor during the MVP
phase: pre-start an actor pool, acquire ``TP x PP`` actors, run one model's
inference through :class:`AsyncLLM` with :class:`ExternalExecutor`, then return
the actors in ``finally`` so the next model can reuse the same batch.

Out-of-scope configuration is rejected up front by :func:`validate_mvp_config`
instead of silently entering an untested path.
"""

from __future__ import annotations

# Kept free of torch/ray/vllm imports so the constraint checks stay unit-testable
# without a cluster (see tests/test_mvp_entry.py).


def validate_mvp_config(
    tp_size: int,
    pp_size: int = 1,
    *,
    num_nodes: int = 1,
    enable_lora: bool = False,
    kv_transfer_config: object | None = None,
    elastic_ep: bool = False,
    enable_autoscaling: bool = False,
) -> None:
    """Reject configurations the MVP does not support yet.

    Raises:
        ValueError: If any argument falls outside the MVP envelope
            (single-node, fixed TP in {1, 2}, PP=1, no LoRA / KV sharing /
            elastic EP / autoscaling).
    """
    if pp_size != 1:
        raise ValueError(f"MVP supports PP=1 only, got pp_size={pp_size}")
    if tp_size not in (1, 2):
        raise ValueError(f"MVP supports TP in {{1, 2}}, got tp_size={tp_size}")
    if num_nodes != 1:
        raise ValueError(f"MVP is single-node only, got num_nodes={num_nodes}")
    if enable_lora:
        raise ValueError("MVP does not support LoRA (enable_lora must be False)")
    if kv_transfer_config is not None:
        raise ValueError(
            "MVP does not support KV sharing (kv_transfer_config must be None)"
        )
    if elastic_ep:
        raise ValueError("MVP does not support dynamic TP/PP (elastic_ep=False)")
    if enable_autoscaling:
        raise ValueError(
            "MVP does not enable autoscaling (enable_autoscaling must be False)"
        )


def run_mvp(
    model: str,
    prompts: list[str],
    *,
    tp_size: int = 1,
    pp_size: int = 1,
    max_tokens: int = 32,
    temperature: float = 0.8,
    top_p: float = 0.95,
    devices: list[int] | None = None,
    pool: object | None = None,
    pool_size: int | None = None,
    warmup_distributed: bool = True,
    enable_lora: bool = False,
    kv_transfer_config: object | None = None,
    elastic_ep: bool = False,
    enable_autoscaling: bool = False,
    **engine_kwargs,
) -> list[str]:
    """Run one model over the pooled actors and return generated texts.

    Single entry for the MVP: acquire -> AsyncLLM(ExternalExecutor) ->
    generate -> ``finally`` release, so actors are returned to the pool even on
    error and the same batch can be reused for the next model.

    Args:
        model: Local model directory or HuggingFace model id.
        prompts: Prompts to generate from.
        tp_size: Tensor parallelism (1 or 2 in the MVP).
        pp_size: Pipeline parallelism (1 in the MVP).
        max_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        devices: Per-node device indices for the pool; defaults to [0, 1, ...].
        pool: Optional pre-existing :class:`ActorPoolManager` (reused, not
            shut down on return). When None, a pool is created and shut down
            after use.
        pool_size: Number of actors to pre-start; defaults to tp_size.
        warmup_distributed: Whether to warm up NCCL/HCCl at actor creation.
        engine_kwargs: Extra ``AsyncEngineArgs`` keyword arguments.

    Returns:
        Generated text for each prompt, in input order.

    Raises:
        ValueError: If the configuration is outside the MVP envelope.
    """
    validate_mvp_config(
        tp_size,
        pp_size,
        enable_lora=enable_lora,
        kv_transfer_config=kv_transfer_config,
        elastic_ep=elastic_ep,
        enable_autoscaling=enable_autoscaling,
    )

    import asyncio

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm_external_executor import ActorPoolManager, ExternalExecutor

    own_pool = pool is None
    if own_pool:
        if devices is None:
            devices = list(range(pool_size or tp_size))
        pool = ActorPoolManager()
        pool.pre_start(
            num_actors=pool_size or tp_size,
            devices_per_node=devices,
            warmup_distributed=warmup_distributed,
        )

    actors = pool.acquire(tp_size=tp_size, pp_size=pp_size)
    try:
        engine_args = AsyncEngineArgs(
            model=model,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            **engine_kwargs,
        )
        vllm_config = engine_args.create_engine_config()
        llm = AsyncLLM(
            vllm_config=vllm_config,
            executor_class=ExternalExecutor,
            log_stats=True,
            external_actors=actors,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        async def _generate() -> list[str]:
            texts: list[str] = []
            for i, prompt in enumerate(prompts):
                final = None
                async for output in llm.generate(
                    prompt,
                    sampling_params,
                    request_id=f"mvp-{i}",
                ):
                    final = output
                texts.append(final.outputs[0].text if final else "")
            return texts

        return asyncio.run(_generate())
    finally:
        pool.release(actors)
        if own_pool:
            pool.shutdown()
