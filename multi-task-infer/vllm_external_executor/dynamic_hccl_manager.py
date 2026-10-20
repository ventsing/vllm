# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""Resident World Group + cached TP subgroup carve-out.

The pooled executor currently rebuilds the whole distributed environment every
round: ``init_process_group`` with ``world_size == TP``, then a full teardown
that destroys both the model-parallel groups and the World Group. That is what
makes TP changes and multi-round reuse expensive and port-leaky.

This module owns the scheme-B alternative:

- create **one** World Group spanning every pooled device (``rank == device
  id``, ``world_size == device count``) at pool start, and keep it resident
  until actor shutdown;
- carve each task's TP subgroup out of that resident World Group via vLLM's
  ``initialize_model_parallel`` (which already performs the full-world
  collective ``new_group`` correctly), and on reset destroy only the
  model-parallel groups.

It is deliberately thin: it delegates to ``vllm.distributed.parallel_state``
and only owns the lifecycle contract plus per-topology subgroup caching. The
caller (the pooled worker actor) still owns device binding, MessageQueue
lifecycle, and weight loading.
"""

from __future__ import annotations

from typing import Any


def plan_tp_partition(world_size: int, tp_size: int) -> list[list[int]]:
    """Return the contiguous TP partition of a resident World Group.

    World ranks are device ids, so vLLM's rank-ordered chunking yields
    contiguous device groups, which is what HCCS requires: TP=2 over an
    8-device world is ``[[0, 1], [2, 3], [4, 5], [6, 7]]``.

    Args:
        world_size: Resident World Group size (node device count).
        tp_size: Tensor-parallel size of the task.

    Returns:
        The rank partition, one contiguous run per TP group.

    Raises:
        ValueError: If ``tp_size`` is non-positive or does not divide
            ``world_size``.
    """
    if tp_size < 1:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if world_size % tp_size:
        raise ValueError(
            f"tp_size={tp_size} does not divide world_size={world_size}; "
            "contiguous TP partition is undefined"
        )
    return [
        list(range(start, start + tp_size))
        for start in range(0, world_size, tp_size)
    ]


class DynamicHCCLManager:
    """Lifecycle owner for the resident World Group and carved TP subgroups.

    Instances are per actor (one device each). The World Group is initialized
    exactly once per process and is *not* destroyed between rounds; only the
    model-parallel groups are torn down on reset. This keeps the rendezvous
    (file:// on a single node) alive across the whole pool lifetime and removes
    the per-round ``destroy_process_group`` / port churn.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        backend: str = "hccl",
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self._world_ready = False
        self._tp_group: Any | None = None
        self._tp_size: int | None = None

    @property
    def world_ready(self) -> bool:
        return self._world_ready

    def ensure_world_group(
        self,
        distributed_init_method: str,
        timeout: Any | None = None,
    ) -> None:
        """Initialize the resident World Group once (idempotent).

        Must be called with the *same* ``distributed_init_method`` on every
        actor of the pool. Single-node pools pass a ``file://`` method (no TCP
        listen socket), so there is no port to leak across rounds.
        """
        import torch.distributed as dist

        from vllm.distributed.parallel_state import init_distributed_environment

        if not dist.is_initialized():
            init_distributed_environment(
                world_size=self.world_size,
                rank=self.rank,
                distributed_init_method=distributed_init_method,
                local_rank=self.local_rank,
                backend=self.backend,
                timeout=timeout,
            )
        self._world_ready = True

    def acquire_tp_group(self, tp_size: int, pp_size: int = 1) -> Any:
        """Carve the task's TP subgroup from the resident World Group.

        Reuses the cached subgroup when the topology is unchanged; otherwise
        releases the previous subgroup and re-carves. Every actor in the World
        Group must call this for the same ``tp_size`` in the same order (it is
        a collective), including idle actors that hold but do not use their
        subgroup.
        """
        if not self._world_ready:
            raise RuntimeError("ensure_world_group must be called first")
        if self._tp_size == tp_size and self._tp_group is not None:
            return self._tp_group

        if self._tp_group is not None:
            self.release_tp_group()

        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_group,
            initialize_model_parallel,
        )

        initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
        )
        self._tp_group = get_tensor_model_parallel_group()
        self._tp_size = tp_size
        return self._tp_group

    def release_tp_group(self) -> None:
        """Destroy model-parallel groups; keep the World Group resident.

        Mirrors ``cleanup_dist_env_and_memory`` minus the World teardown:
        only ``destroy_model_parallel`` plus device cache eviction run here.
        """
        import gc

        from vllm.distributed.parallel_state import destroy_model_parallel
        from vllm.platforms import current_platform

        gc.unfreeze()
        destroy_model_parallel()
        gc.collect()
        if not current_platform.is_cpu():
            current_platform.empty_cache()
        self._tp_group = None
        self._tp_size = None

    def teardown(self) -> None:
        """Destroy the World Group (actor shutdown only, never between rounds)."""
        from vllm.distributed.parallel_state import destroy_distributed_environment

        destroy_distributed_environment()
        self._world_ready = False
        self._tp_group = None
        self._tp_size = None
