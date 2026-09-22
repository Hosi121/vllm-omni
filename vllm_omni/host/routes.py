# SPDX-License-Identifier: Apache-2.0
"""Deployment-owned worker routes, independent of model and physical device."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from vllm_omni.edge.local.external.launch import Route, is_wsl, resolve


def resolve_route(spec: str | Mapping) -> Route:
    """Names retain compatibility; explicit profiles have no machine-local defaults."""
    if isinstance(spec, str):
        return resolve(spec)
    required = {"name", "interpreter", "worker", "ep", "os_domain"}
    if not required <= set(spec) or set(spec) - required:
        raise ValueError(f"explicit route requires exactly {sorted(required)}")
    if spec["os_domain"] not in {"windows", "posix"}:
        raise ValueError("route os_domain must be windows or posix")
    windows = spec["os_domain"] == "windows"
    interpreter = Path(os.path.expandvars(str(spec["interpreter"]))).expanduser()
    worker = Path(os.path.expandvars(str(spec["worker"]))).expanduser()
    reason = ""
    if windows and os.name != "nt" and not is_wsl():
        reason = "Windows route requires Windows or WSL interop on this machine"
    elif not windows and os.name == "nt":
        reason = "POSIX route needs a POSIX controller; no implicit remote/WSL launch"
    elif not interpreter.is_file() or not worker.is_file():
        reason = "configured interpreter or worker script is missing"
    return Route(str(spec["name"]), str(interpreter), str(worker), str(spec["ep"]), windows, not reason, reason)
