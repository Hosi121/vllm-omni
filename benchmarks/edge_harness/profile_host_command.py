# SPDX-License-Identifier: Apache-2.0
"""Run one explicit benchmark command with Windows host and whole-GPU telemetry."""

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil


class PowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_byte),
        ("BatteryFlag", ctypes.c_byte),
        ("BatteryLifePercent", ctypes.c_byte),
        ("SystemStatusFlag", ctypes.c_byte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cwd", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    a.out.mkdir(parents=True, exist_ok=False)
    status = {
        "command": command,
        "cwd": str(a.cwd),
        "host": platform.platform(),
        "scope": "Windows RAM includes WSL; GPU counters are whole-device, not model-exclusive",
        "sample_interval_s": 1,
        "start_unix": time.time(),
        "status": "running",
    }
    if os.name == "nt":
        status["power_scheme"] = subprocess.run(
            ["powercfg", "/getactivescheme"], capture_output=True, text=True, errors="replace"
        ).stdout
    log = (a.out / "command.log").open("w", encoding="utf-8")
    child = subprocess.Popen(command, cwd=a.cwd, stdout=log, stderr=subprocess.STDOUT)
    status["pid"] = child.pid
    (a.out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    with (a.out / "telemetry.jsonl").open("w", encoding="utf-8") as f:
        while child.poll() is None:
            sample = {
                "unix": time.time(),
                "monotonic": time.monotonic(),
                "host_memory": psutil.virtual_memory()._asdict(),
                "host_cpu_percent": psutil.cpu_percent(),
            }
            if os.name == "nt":
                power = PowerStatus()
                if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)):
                    sample["power"] = {name: getattr(power, name) for name, _ in power._fields_}
            try:
                gpu = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,name,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,pstate",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                sample["gpu_csv"] = gpu.stdout.strip()
                sample["gpu_returncode"] = gpu.returncode
            except Exception as error:
                sample["gpu_error"] = repr(error)
            f.write(json.dumps(sample) + "\n")
            f.flush()
            time.sleep(1)
    log.close()
    status.update(status="exited", returncode=child.returncode, end_unix=time.time())
    (a.out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status), flush=True)
    return child.returncode


if __name__ == "__main__":
    sys.exit(main())
