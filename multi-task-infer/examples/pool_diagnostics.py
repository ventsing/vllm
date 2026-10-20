# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inspect communication and execution resources without terminating actors."""

import ray


def inspect_workers(actors):
    """Return snapshots captured inside each actor's control thread."""

    def probe(actor):
        import os
        import sys

        import torch

        import vllm.distributed.parallel_state as ps

        group_names = (
            "_WORLD",
            "_INNER_DP_WORLD",
            "_TP",
            "_ETP",
            "_PP",
            "_DP",
            "_EP",
            "_EPLB",
            "_PCP",
            "_DCP",
        )
        active = [name for name in group_names if getattr(ps, name, None) is not None]
        world = torch.distributed.distributed_c10d._world
        acl_graph = sys.modules.get("vllm_ascend.compilation.acl_graph")
        graph_entries = sum(
            len(wrapper.concrete_aclgraph_entries)
            for wrapper in getattr(acl_graph, "_acl_graph_wrappers", ())
        )
        thread = actor._loop_thread
        coordinators = []
        for name, ref in ps._groups.items():
            group = ref()
            if group is not None:
                comm = getattr(group, "device_communicator", None)
                coordinators.append(
                    {
                        "name": name,
                        "device_group_present": hasattr(group, "device_group"),
                        "cpu_group_present": hasattr(group, "cpu_group"),
                        "communicator_type": type(comm).__name__ if comm else None,
                        "communicator_process_groups": [
                            key
                            for key, value in vars(comm).items()
                            if isinstance(value, torch.distributed.ProcessGroup)
                        ]
                        if comm
                        else [],
                    }
                )
        return {
            "actor_id": actor.actor_id,
            "pid": os.getpid(),
            "state": actor.state.value,
            "distributed_initialized": torch.distributed.is_initialized(),
            "registered_process_groups": len(world.pg_map),
            "active_vllm_groups": active,
            "live_group_coordinators": sum(
                ref() is not None for ref in ps._groups.values()
            ),
            "group_coordinators": coordinators,
            "tcp_store_present": actor._dist_init_store is not None,
            "loop_alive": thread is not None and thread.is_alive(),
            "worker_present": actor.worker is not None,
            "rpc_mq_present": actor.rpc_broadcast_mq is not None,
            "response_mq_present": actor.worker_response_mq is not None,
            "graph_entries": graph_entries,
            "graph_params_present": (
                getattr(acl_graph, "_graph_params", None) is not None
            ),
            "allocated_bytes": (
                torch.npu.memory_allocated()
                if hasattr(torch, "npu") and torch.npu.is_initialized()
                else None
            ),
        }

    return ray.get([actor.__ray_call__.remote(probe) for actor in actors], timeout=180)


def assert_released(rows):
    """Require idle actors to retain no process groups or task resources."""
    for row in rows:
        assert row["state"] == "idle", row
        assert not row["distributed_initialized"], row
        assert row["registered_process_groups"] == 0, row
        assert not row["active_vllm_groups"], row
        assert row["graph_entries"] == 0 and not row["graph_params_present"], row
        for group in row["group_coordinators"]:
            assert not group["device_group_present"], row
            assert not group["cpu_group_present"], row
        for key in (
            "tcp_store_present",
            "loop_alive",
            "worker_present",
            "rpc_mq_present",
            "response_mq_present",
        ):
            assert not row[key], row
