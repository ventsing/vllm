# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Observe communication cleanup in a retained pool without changing leases."""

import json
import os
import time
import traceback
from pathlib import Path

import ray
from pool_diagnostics import assert_released, inspect_workers


def main():
    directory = Path("/tmp/vllm-profile-8x2")
    connection = json.loads(Path("/tmp/vllm-rolling-8x2/connection.json").read_text())
    ray.init(address="auto")
    registry = ray.get_actor(
        connection["registry_name"], namespace=connection["namespace"]
    )
    handles = ray.get(registry.get_actor_handles.remote())
    seen = set()
    records = []
    errors = []
    try:
        while True:
            regs = ray.get(registry.list_actors.remote(), timeout=30)
            candidates = [
                reg
                for reg in regs
                if (reg.actor_id, reg.lease_generation, reg.state) not in seen
            ]
            if candidates:
                rows = inspect_workers([handles[reg.actor_id] for reg in candidates])
                for reg, row in zip(candidates, rows):
                    if row["state"] == "idle" and reg.state == "idle":
                        try:
                            assert_released([row])
                        except AssertionError:
                            errors.append(traceback.format_exc())
                    row["lease_generation"] = reg.lease_generation
                    records.append(row)
                    seen.add((reg.actor_id, reg.lease_generation, reg.state))
                    print(json.dumps(row), flush=True)
            if (directory / "metrics.json").exists():
                final = inspect_workers(list(handles.values()))
                assert_released(final)
                result = {
                    "status": "passed" if not errors else "failed",
                    "errors": errors,
                    "observations": records,
                    "final": final,
                    "observer_pid": os.getpid(),
                }
                (directory / "communication_cleanup.json").write_text(
                    json.dumps(result, indent=2)
                )
                return
            time.sleep(1)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
