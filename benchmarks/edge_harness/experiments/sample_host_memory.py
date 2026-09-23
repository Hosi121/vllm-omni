#!/usr/bin/env python3
"""Sample host RAM/swap and a named process tree during a local profile."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    if not 0.1 <= args.interval_s <= 10 or args.max_seconds <= 0:
        parser.error("invalid sample interval or duration")

    root = psutil.Process(args.pid)
    birth = root.create_time()
    deadline = time.monotonic() + args.max_seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        while time.monotonic() < deadline:
            try:
                if not root.is_running() or root.create_time() != birth or root.status() == psutil.STATUS_ZOMBIE:
                    break
                tree = [root, *root.children(recursive=True)]
                rss = 0
                for process in tree:
                    try:
                        rss += process.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                ram = psutil.virtual_memory()
                swap = psutil.swap_memory()
                out.write(json.dumps({
                    "epoch_s": time.time(),
                    "available_ram_bytes": ram.available,
                    "used_swap_bytes": swap.used,
                    "summed_process_tree_rss_bytes": rss,
                    "process_count": len(tree),
                }) + "\n")
                out.flush()
            except psutil.NoSuchProcess:
                break
            time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
