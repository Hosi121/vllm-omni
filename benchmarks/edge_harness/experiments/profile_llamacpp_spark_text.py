#!/usr/bin/env python3
"""Check two complete Spark text requests and profile a resident llama.cpp stage.

The target llama-server must already be running on the intended local device.
This measures HTTP request wall time, including tokenization and delivery, not
Omni pipeline time. The caller must retain the server launch and placement log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import urllib.request
from urllib.parse import urlparse
from pathlib import Path


def _read_json(url: str, body: dict | None = None, timeout: float = 60) -> dict:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18173")
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--profile-case", choices=("short", "inventory_120"),
                        default="inventory_120")
    parser.add_argument("--cache-prompt", action="store_true",
                        help="Permit prompt/KV reuse across serial requests")
    parser.add_argument("--cancel-check", action="store_true",
                        help="Close a live stream and verify the following request")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats positive")
    actual_sha = _sha256(args.model_file)
    if actual_sha != args.expected_model_sha256.lower():
        raise ValueError("The GGUF differs from the pinned model artifact")

    base = args.endpoint.rstrip("/")
    if urlparse(base).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("This single-device probe requires a local llama-server")
    health = _read_json(base + "/health")
    if health.get("status") != "ok":
        raise RuntimeError(f"Server not ready: {health}")
    props = _read_json(base + "/props")
    if Path(props.get("model_path", "")).resolve() != args.model_file.resolve():
        raise RuntimeError("The server did not load the pinned GGUF path")
    if props.get("total_slots") != 1:
        raise RuntimeError("Expected one server slot for the serial profile")

    inventory = "Here is an inventory listing. " + " ".join(
        f"Item {i}: the code for warehouse district {i} is {1000 + 7 * i}."
        for i in range(1, 121)
    ) + " What is the code for warehouse district 73? Answer with just the number."
    cases = [
        ("short", "What is the capital of France? Answer in one word.", "Paris"),
        ("inventory_120", inventory, "1511"),
    ]

    def run(name: str, prompt: str, expected: str) -> dict:
        started = time.perf_counter()
        result = _read_json(base + "/v1/chat/completions", {
            "model": props["model_alias"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 96,
            "stream": False,
            "cache_prompt": args.cache_prompt,
        })
        wall_s = time.perf_counter() - started
        choice = result["choices"][0]
        content = choice["message"]["content"].strip()
        row = {
            "case": name,
            "wall_s": wall_s,
            "content": content,
            "expected": expected,
            "correct": content == expected,
            "finish_reason": choice["finish_reason"],
            "usage": result.get("usage"),
        }
        if not row["correct"] or row["finish_reason"] != "stop":
            raise RuntimeError(f"Complete-response gate failed: {row}")
        return row

    checks = [run(*case) for case in cases]
    profile_case = next(case for case in cases if case[0] == args.profile_case)
    warmups = [run(*profile_case) for _ in range(args.warmups)]
    measured = [run(*profile_case) for _ in range(args.repeats)]
    times = [row["wall_s"] for row in measured]
    cancellation = None
    if args.cancel_check:
        body = json.dumps({
            "model": props["model_alias"],
            "messages": [{"role": "user", "content": "Count from 1 to 500, separated by spaces."}],
            "temperature": 0,
            "max_tokens": 512,
            "stream": True,
            "cache_prompt": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            base + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        first_delta = None
        first_delta_wall_s = None
        with urllib.request.urlopen(request, timeout=30) as response:
            for _ in range(100):
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data: {"):
                    continue
                event = json.loads(line[6:])
                delta = event["choices"][0].get("delta", {}).get("content")
                if delta:
                    first_delta = delta
                    first_delta_wall_s = time.perf_counter() - started
                    break
        if not first_delta:
            raise RuntimeError("Did not observe a live token before stream cancellation")
        recovery = run(*cases[0])
        cancellation = {
            "first_delta": first_delta,
            "stream_closed_after_first_content": True,
            "first_delta_wall_s": first_delta_wall_s,
            "recovery": recovery,
        }
    report = {
        "scope": "Standalone llama.cpp full-model text path; no Omni stage binding",
        "platform": platform.platform(),
        "model_path": str(args.model_file),
        "model_sha256": actual_sha,
        "server": {
            "endpoint": base,
            "build_info": props.get("build_info"),
            "model_alias": props["model_alias"],
            "model_ftype": props.get("model_ftype"),
            "total_slots": props["total_slots"],
        },
        "quality_checks": checks,
        "warmups": warmups,
        "samples": measured,
        "cancellation": cancellation,
        "profile": {
            "case": args.profile_case,
            "cache_prompt": args.cache_prompt,
            "repeats": args.repeats,
            "concurrency": 1,
            "p50_wall_s": _percentile(times, 0.5),
            "p95_wall_s": _percentile(times, 0.95),
            "min_wall_s": min(times),
            "max_wall_s": max(times),
            "cached_prompt_tokens": [
                row["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                for row in measured
            ],
        },
        "status": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "profile": report["profile"]}, indent=2))


if __name__ == "__main__":
    main()
