# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""HCCL topology helpers for dynamic TP.

Node-local Ascend fact: once ``hccl`` has done one initial handshake, each
device's ``hccp`` communication process stays resident, so tearing down and
re-initializing the communication domain for a *new* topology costs on the
order of ~300ms, not the multi-second first-handshake cost. The scheme-B hot
switch therefore does **not** keep a resident World Group or carve cached
``new_group`` sub-groups; it simply destroys the domain on unload and rebuilds
it from the new ``tp_size`` on reload.

The only piece worth keeping out of that exchange is the contiguous rank
partition, which stays useful for neutral rank planning (and for future
multi-node/EP where contiguous device placement still matters).
"""

from __future__ import annotations


def plan_tp_partition(world_size: int, tp_size: int) -> list[list[int]]:
    """Return the contiguous TP partition of a device world.

    Ranks are device ids, so vLLM's rank-ordered chunking yields contiguous
    device groups, which is what HCCS requires: TP=2 over an 8-device world is
    ``[[0, 1], [2, 3], [4, 5], [6, 7]]``.

    Args:
        world_size: Number of devices in the world.
        tp_size: Tensor-parallel size of one group.

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