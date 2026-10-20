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

import os

# Kept free of torch/ray/vllm imports so the constraint checks stay unit-testable
# without a cluster (see tests/test_mvp_entry.py).

# Scheme B trims decode graph capture to four high-frequency batch sizes
# instead of vLLM's default ladder (1..max_num_seqs). Fewer capture sizes means
# fewer dummy forwards and less private graph memory per hot-switch.
SCHEME_B_DECODE_CAPTURE_SIZES = (1, 2, 4, 8)


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
            (single-node, positive TP, PP=1, no LoRA / KV sharing / elastic
            EP / autoscaling).
    """
    if pp_size != 1:
        raise ValueError(f"MVP supports PP=1 only, got pp_size={pp_size}")
    if tp_size < 1:
        raise ValueError(f"tp_size must be positive, got tp_size={tp_size}")
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


def _detach_stuck_mm_warmup(llm, grace: float = 5.0) -> None:
    """Stop a stuck background multimodal warmup from blocking shutdown.

    vLLM's renderer joins its background MM warmup future in ``shutdown``
    with no timeout. On NPU that warmup can stall after many engine
    lifetimes, which would hang pool release and the next round. Detach the
    future so ``renderer.shutdown`` skips the join; the pooled actors are
    released either way.
    """
    renderer = getattr(llm, "renderer", None)
    future = getattr(renderer, "_mm_warmup_future", None) if renderer else None
    if future is None or future.done():
        return
    try:
        future.result(timeout=grace)
    except Exception:
        # Timed out or the warmup failed: detach so shutdown cannot hang.
        pass
    renderer._mm_warmup_future = None


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
    cleanup_timeout: float = 60.0,
    warmup_distributed: bool = True,
    enable_lora: bool = False,
    kv_transfer_config: object | None = None,
    elastic_ep: bool = False,
    enable_autoscaling: bool = False,
    cudagraph_capture_sizes: list[int] | tuple[int, ...] | None = None,
    **engine_kwargs,
) -> list[str]:
    """Run one model over the pooled actors and return generated texts.

    Single entry for the MVP: acquire -> AsyncLLM(ExternalExecutor) ->
    generate -> ``finally`` release, so actors are returned to the pool even on
    error and the same batch can be reused for the next model.

    Args:
        model: Local model directory or HuggingFace model id.
        prompts: Prompts to generate from.
        tp_size: Tensor parallelism (single-node; upper bound is the node's
            device count).
        pp_size: Pipeline parallelism (1 in the MVP).
        max_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        devices: Per-node device indices for the pool; defaults to [0, 1, ...].
        pool: Optional pre-existing :class:`ActorPoolManager` (reused, not
            shut down on return). When None, a pool is created and shut down
            after use.
        pool_size: Number of actors to pre-start; defaults to tp_size.
        cleanup_timeout: Seconds to wait for EngineCore resource teardown.
        warmup_distributed: Whether to warm up NCCL/HCCl at actor creation.
        cudagraph_capture_sizes: Decode graph capture batch sizes. Scheme B
            uses :data:`SCHEME_B_DECODE_CAPTURE_SIZES` to cut capture time and
            graph memory; None keeps vLLM's default ladder.
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

    # Each AsyncLLM creates a continuous usage reporter. The MVP reuses a
    # process across many engine lifetimes, so disable telemetry before any
    # vLLM imports or child EngineCore processes are created.
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    # Ascend AOT kernel cache: reuse CANN-compiled operators across models
    # instead of JIT-compiling them on every load (no-op on non-Ascend).
    os.environ.setdefault("ACL_OP_COMPILER_CACHE_MODE", "enable")

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
    llm = None
    try:
        if cudagraph_capture_sizes is not None:
            engine_kwargs["cudagraph_capture_sizes"] = list(
                cudagraph_capture_sizes
            )
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

        async def _run() -> list[str]:
            try:
                return await _generate()
            finally:
                # Shutdown while the event loop is still alive so MPClient's
                # asyncio ZMQ sockets close on the live loop (via
                # close_sockets_and_tasks) instead of the "loop closed"
                # fallback. Otherwise the zmq context lingers and its
                # __del__ -> destroy -> term hangs on a later engine's GC.
                if llm is not None:
                    _detach_stuck_mm_warmup(llm)
                    await asyncio.to_thread(llm.shutdown, cleanup_timeout)

        return asyncio.run(_run())
    finally:
        try:
            pool.release(actors)
        finally:
            if own_pool:
                pool.shutdown()


def run_hot_switch_mvp(
    models: list[tuple[str, int]],
    prompts: list[str],
    *,
    pool: object | None = None,
    devices: list[int] | None = None,
    max_tokens: int = 32,
    temperature: float = 0.8,
    top_p: float = 0.95,
    cleanup_timeout: float = 60.0,
    warmup_distributed: bool = True,
    cudagraph_capture_sizes: list[int] | tuple[int, ...] | None = None,
    **engine_kwargs,
) -> list[list[str]]:
    """Run a sequence of ``(model, tp_size)`` over one live AsyncLLM.

    Scheme-B hot switch: boots with a ``max_tp`` actor lease, then for every
    subsequent model calls ``EngineCore.reconfigure`` through
    ``AsyncMPClient.call_utility_async`` using the first ``tp`` actors of that
    lease. Shrinking resets the trailing actors; growing re-initializes them in
    place. The EngineCore process and the actor pool stay alive across the
    whole sequence (no release -> acquire round-trip).

    Client-side caveat: the *same* AsyncLLM tokenizer/renderer is reused across
    switches, so every model must share one tokenizer (true for the same model
    at different TP, the current 4B test). Swapping a different tokenizer on
    the client is a separate AsyncLLM-level follow-up.

    Args:
        models: Ordered ``(model_path, tp_size)`` pairs to serve in sequence.
        prompts: Prompts to generate from, for each model.
        pool: Optional pre-existing :class:`ActorPoolManager` (reused, not shut
            down on return).
        devices: Per-node device indices; defaults to ``[0, .., max_tp)``.
        max_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        cleanup_timeout: Seconds to wait for EngineCore teardown.
        warmup_distributed: Whether to warm up NCCL/HCCl at actor creation.
        cudagraph_capture_sizes: Decode capture sizes; None keeps vLLM default.
        engine_kwargs: Extra ``AsyncEngineArgs`` keyword arguments.

    Returns:
        One generated-text list per model, in input order.

    Raises:
        ValueError: If ``models`` is empty or any topology is out of scope.
    """
    if not models:
        raise ValueError("models must be non-empty")
    for _, tp in models:
        validate_mvp_config(tp, 1)
    max_tp = max(tp for _, tp in models)

    import asyncio

    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("ACL_OP_COMPILER_CACHE_MODE", "enable")

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm_external_executor import ActorPoolManager, ExternalExecutor

    if cudagraph_capture_sizes is not None:
        engine_kwargs["cudagraph_capture_sizes"] = list(cudagraph_capture_sizes)

    def _config(model: str, tp: int):
        return AsyncEngineArgs(
            model=model,
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            **engine_kwargs,
        ).create_engine_config()

    own_pool = pool is None
    if own_pool:
        if devices is None:
            devices = list(range(max_tp))
        pool = ActorPoolManager()
        pool.pre_start(
            num_actors=max_tp,
            devices_per_node=devices,
            warmup_distributed=warmup_distributed,
        )

    # One max-TP lease serves every topology in the sequence via prefixes.
    actors = pool.acquire(tp_size=max_tp, pp_size=1)
    llm = None
    try:
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
                    request_id=f"hs-{i}",
                ):
                    final = output
                texts.append(final.outputs[0].text if final else "")
            return texts

        async def _run() -> list[list[str]]:
            nonlocal llm
            first_model, first_tp = models[0]
            llm = AsyncLLM(
                vllm_config=_config(first_model, first_tp),
                executor_class=ExternalExecutor,
                log_stats=True,
                external_actors=actors[:first_tp],
            )
            results = [await _generate()]
            for model, tp in models[1:]:
                await llm.engine_core.call_utility_async(
                    "reconfigure", _config(model, tp), actors[:tp]
                )
                results.append(await _generate())
            return results

        async def _run_guarded() -> list[list[str]]:
            try:
                return await _run()
            finally:
                if llm is not None:
                    _detach_stuck_mm_warmup(llm)
                    await asyncio.to_thread(llm.shutdown, cleanup_timeout)

        return asyncio.run(_run_guarded())
    finally:
        try:
            pool.release(actors)
        finally:
            if own_pool:
                pool.shutdown()
