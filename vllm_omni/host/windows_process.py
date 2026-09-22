# SPDX-License-Identifier: Apache-2.0
"""Standalone native-Windows process identity and bounded tree retirement.

This file is executed by the worker interpreter from WSL; it imports no engine
or third-party packages. A launcher PID is never treated as a Windows PID.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from ctypes import wintypes


def _api():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    return kernel


def _created(kernel, handle) -> int:
    times = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
        raise ctypes.WinError(ctypes.get_last_error())
    return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime


def identity() -> dict:
    kernel = _api()
    handle = kernel.OpenProcess(0x1000, False, os.getpid())  # QUERY_LIMITED_INFORMATION
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return {"pid": os.getpid(), "created_filetime": _created(kernel, handle)}
    finally:
        kernel.CloseHandle(handle)


def retire(pid: int, created_filetime: int) -> bool:
    kernel = _api()
    handle = kernel.OpenProcess(0x1000 | 0x100000, False, pid)  # query + synchronize
    if not handle:
        return False  # parent absence alone says nothing about its descendants
    try:
        if _created(kernel, handle) != created_filetime:
            return False  # stale identity must not kill a reused PID
        result = subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return result.returncode == 0 and kernel.WaitForSingleObject(handle, 5000) == 0
    finally:
        kernel.CloseHandle(handle)


if __name__ == "__main__":
    try:
        print(json.dumps({"drained": retire(int(sys.argv[1]), int(sys.argv[2]))}))
    except Exception as exc:
        print(json.dumps({"drained": False, "error": str(exc)}))
        sys.exit(1)
