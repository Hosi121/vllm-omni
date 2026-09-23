#!/usr/bin/env python3
"""Capture one local process's raw output, exit code, elapsed time and memory.

This wrapper samples RSS and private bytes; the values are observed maxima, not
proof of the true loading peak between samples.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.sample_interval_ms < 10:
        parser.error("a command and sample interval >= 10 ms are required")
    command = args.command[1:] if args.command[0] == "--" else args.command
    for path in (args.stdout, args.stderr, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with args.stdout.open("wb") as stdout, args.stderr.open("wb") as stderr:
        child = subprocess.Popen(command, cwd=args.cwd, stdout=stdout, stderr=stderr)
        process = psutil.Process(child.pid)
        samples = 0
        peak_rss = 0
        peak_private = 0
        while child.poll() is None:
            try:
                memory = process.memory_info()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            peak_rss = max(peak_rss, memory.rss)
            peak_private = max(peak_private, getattr(memory, "private", 0))
            samples += 1
            time.sleep(args.sample_interval_ms / 1000)
        exit_code = child.wait()
    report = {
        "command": command,
        "cwd": str(args.cwd),
        "exit_code": exit_code,
        "wall_s": time.perf_counter() - started,
        "memory_sample_interval_ms": args.sample_interval_ms,
        "memory_samples": samples,
        "peak_sampled_rss_bytes": peak_rss,
        "peak_sampled_private_bytes": peak_private,
        "measurement_note": "sampled process values, not guaranteed loading peaks",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
