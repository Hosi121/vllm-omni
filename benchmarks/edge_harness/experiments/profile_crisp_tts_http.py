#!/usr/bin/env python3
"""Profile complete serial text-to-WAV requests against a resident CrispASR server.

The caller must separately retain and inspect the server's actual placement log.
HTTP success and a nonzero WAV do not establish speech quality.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import io
import json
import math
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path


def _rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18175/v1/audio/speech")
    parser.add_argument("--text", default="Hello from the local computer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("warmups must be nonnegative and repeats positive")

    import psutil

    process = psutil.Process(args.server_pid) if args.server_pid else None
    samples: list[dict] = []
    done = threading.Event()

    def sample_memory() -> None:
        if process is None:
            return
        while not done.is_set():
            try:
                memory = process.memory_full_info()
                samples.append({
                    "unix": time.time(),
                    "working_set_bytes": memory.rss,
                    "private_bytes": getattr(memory, "private", None),
                    "host_available_bytes": psutil.virtual_memory().available,
                })
            except psutil.Error:
                break
            done.wait(0.25)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    payload = json.dumps({"input": args.text, "response_format": "wav", "seed": args.seed}).encode()

    def request_one(*, save_wav: bool) -> dict:
        request = urllib.request.Request(
            args.endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as response:
            status = response.status
            wav_bytes = response.read()
        wall_s = time.perf_counter() - started
        if status != 200:
            raise RuntimeError(f"speech endpoint returned HTTP {status}")
        with wave.open(io.BytesIO(wav_bytes), "rb") as audio:
            channels, width, sample_rate, frames = (
                audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()
            )
            pcm = audio.readframes(frames)
        if channels != 1 or width != 2 or sample_rate <= 0 or not frames:
            raise RuntimeError("expected nonempty 16-bit mono WAV")
        values = array.array("h")
        values.frombytes(pcm)
        if sys.byteorder != "little":
            values.byteswap()
        rms = math.sqrt(sum(value * value for value in values) / len(values)) / 32768
        if rms <= 1e-5:
            raise RuntimeError("speech WAV is effectively silent")
        if save_wav:
            args.output_wav.parent.mkdir(parents=True, exist_ok=True)
            args.output_wav.write_bytes(wav_bytes)
        return {
            "wall_s": wall_s,
            "sample_rate": sample_rate,
            "frames": frames,
            "duration_s": frames / sample_rate,
            "rms_pcm16_normalized": rms,
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "sha256": hashlib.sha256(wav_bytes).hexdigest(),
            "bytes": len(wav_bytes),
        }

    try:
        warmups = [request_one(save_wav=False) for _ in range(args.warmups)]
        measured = [request_one(save_wav=i == 0) for i in range(args.repeats)]
    finally:
        done.set()
        sampler.join(timeout=2)

    walls = [row["wall_s"] for row in measured]
    report = {
        "scope": "complete serial HTTP text-to-WAV requests in one resident server; speech quality and placement assessed separately",
        "endpoint": args.endpoint,
        "text": args.text,
        "seed": args.seed,
        "concurrency": 1,
        "warmups": warmups,
        "measured": measured,
        "nearest_rank_p50_wall_s": _rank(walls, 0.5),
        "nearest_rank_p95_wall_s": _rank(walls, 0.95),
        "all_same_wav_sha256": len({row["sha256"] for row in measured}) == 1,
        "all_same_pcm_sha256": len({row["pcm_sha256"] for row in measured}) == 1,
        "server_pid": args.server_pid,
        "memory_samples": samples,
        "sampled_max_working_set_bytes": max((row["working_set_bytes"] for row in samples), default=None),
        "sampled_max_private_bytes": max((row["private_bytes"] for row in samples if row["private_bytes"] is not None), default=None),
        "sampled_min_host_available_bytes": min((row["host_available_bytes"] for row in samples), default=None),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"memory_samples", "measured", "warmups"}}, indent=2))


if __name__ == "__main__":
    main()
