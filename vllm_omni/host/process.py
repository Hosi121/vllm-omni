# SPDX-License-Identifier: Apache-2.0
"""Portable worker ownership; termination reports whether resources drained."""

from __future__ import annotations

import os
import signal
import subprocess


def spawn_worker(argv: list[str], **kwargs) -> subprocess.Popen:
    if os.name == "nt":
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs.setdefault("start_new_session", True)
    return subprocess.Popen(argv, **kwargs)


def terminate_worker(process: subprocess.Popen, timeout: float = 5.0) -> bool:
    """Kill descendants and parent, retaining the caller's resource charge on failure."""
    import psutil

    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        process.wait(timeout=timeout)
        if os.name == "nt":
            return False  # absent parent cannot prove that descendants exited
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return False  # an orphaned group was found; retain its charge until verified
    except psutil.Error:
        return False
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            return False
    try:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=timeout)
        _, alive = psutil.wait_procs(children, timeout=timeout)
        return not alive
    except (OSError, subprocess.TimeoutExpired):
        return False
