# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyright: Copyright contributors to the vLLM project

"""Explicit teardown for vLLM shared-memory :class:`MessageQueue` objects.

``MessageQueue.shutdown`` only flags the queue; its ZMQ sockets, IPC socket
files and ``/dev/shm`` ring buffer are normally reclaimed when the worker
*process* exits. A pooled worker actor keeps its process alive across tasks, so
every released queue would otherwise leak sockets, file descriptors and shared
memory until the actor is killed. :func:`close_message_queue` closes those
resources explicitly and is safe to call on both the writer and reader ends.
"""


def close_message_queue(mq) -> None:
    """Close a MessageQueue's ZMQ sockets and shared-memory backing store.

    Best-effort: each step ignores exceptions so a partially-initialized or
    already-closed queue still cleans up whatever it exposes.
    """
    try:
        mq.shutdown()
    except Exception:
        pass

    for name in ("local_socket", "remote_socket"):
        sock = getattr(mq, name, None)
        if sock is not None:
            try:
                sock.close(linger=0)
            except Exception:
                pass
            setattr(mq, name, None)

    spin = getattr(mq, "_spin_condition", None)
    if spin is not None:
        for name in (
            "local_notify_socket",
            "write_cancel_socket",
            "read_cancel_socket",
        ):
            sock = getattr(spin, name, None)
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
                setattr(spin, name, None)
        del spin
        mq._spin_condition = None

    buf = getattr(mq, "buffer", None)
    if buf is not None:
        shm = getattr(buf, "shared_memory", None)
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
            if getattr(buf, "is_creator", False):
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
                # Prevent ShmRingBuffer.__del__ from unlinking again.
                buf.is_creator = False
        del buf
        mq.buffer = None

    context = getattr(mq, "_context", None)
    if context is not None:
        try:
            context.destroy(linger=0)
        except Exception:
            pass
        mq._context = None
