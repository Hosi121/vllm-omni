#!/usr/bin/env python3
"""Run the pinned Qwen3-TTS CustomVoice checkpoint on native Windows CPU.

This uses Qwen's PyTorch wrapper as a standalone backend feasibility check.
It does not prove Omni streaming, playback, or speech intelligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--text", default="Hello from the local computer.")
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.threads <= 0 or args.warmups < 0 or args.repeats <= 0:
        parser.error("Token/thread/repeat counts must be positive and warmups nonnegative")
    checkpoint = args.model_dir / "model.safetensors"
    tokenizer_weights = args.model_dir / "speech_tokenizer" / "model.safetensors"
    if not checkpoint.is_file() or not tokenizer_weights.is_file():
        raise FileNotFoundError("The exact talker and speech-tokenizer weights are required")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from qwen_tts import Qwen3TTSModel

    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(args.model_dir.resolve()),
        device_map="cpu",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    startup_s = time.perf_counter() - started
    model_devices = sorted({parameter.device.type for parameter in model.model.parameters()})
    if model_devices != ["cpu"]:
        raise RuntimeError(f"TTS model was not wholly on CPU: {model_devices}")
    if args.speaker.lower() not in model.get_supported_speakers():
        raise ValueError("The requested speaker is not supported by this checkpoint")

    def generate_one(*, save_wav: bool) -> dict:
        torch.manual_seed(42)
        started = time.perf_counter()
        wavs, sample_rate = model.generate_custom_voice(
            text=args.text,
            language="English",
            speaker=args.speaker,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            non_streaming_mode=True,
        )
        request_s = time.perf_counter() - started
        if len(wavs) != 1 or sample_rate <= 0:
            raise RuntimeError("Expected one waveform with a valid sample rate")
        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if not wav.size or not np.isfinite(wav).all():
            raise RuntimeError("TTS returned an empty or non-finite waveform")
        rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
        if rms <= 1e-5:
            raise RuntimeError("TTS waveform is effectively silent")
        if save_wav:
            args.output_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(args.output_wav, wav, sample_rate, subtype="PCM_16")
        return {
            "request_s": request_s,
            "sample_rate": sample_rate,
            "samples": int(wav.size),
            "duration_s": float(wav.size / sample_rate),
            "rms": rms,
            "peak_abs": float(np.max(np.abs(wav))),
            "nonzero_samples": int(np.count_nonzero(wav)),
            "float32_sha256": hashlib.sha256(wav.tobytes()).hexdigest(),
        }

    warmups = [generate_one(save_wav=False) for _ in range(args.warmups)]
    measured = [generate_one(save_wav=index == 0) for index in range(args.repeats)]
    first = measured[0]
    latencies = sorted(row["request_s"] for row in measured)
    report = {
        "scope": "one complete native Windows CPU text-to-WAV request via Qwen's standalone PyTorch wrapper; no Omni streaming or speech-quality claim",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_dir": str(args.model_dir.resolve()),
        "talker_sha256": _sha256(checkpoint),
        "speech_tokenizer_sha256": _sha256(tokenizer_weights),
        "model_devices": model_devices,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "threads": args.threads,
        "text": args.text,
        "speaker": args.speaker,
        "language": "English",
        "max_new_tokens": args.max_new_tokens,
        "seed": 42,
        "startup_s": startup_s,
        "request_s": first["request_s"],
        "sample_rate": first["sample_rate"],
        "samples": first["samples"],
        "duration_s": first["duration_s"],
        "rms": first["rms"],
        "peak_abs": first["peak_abs"],
        "nonzero_samples": first["nonzero_samples"],
        "wav_sha256": _sha256(args.output_wav),
        "warmups": warmups,
        "measured": measured,
        "profile": {
            "repeats": args.repeats,
            "concurrency": 1,
            "p50_request_s": latencies[math.ceil(0.5 * len(latencies)) - 1],
            "p95_request_s": latencies[math.ceil(0.95 * len(latencies)) - 1],
            "all_same_float32_audio": len({row["float32_sha256"] for row in measured}) == 1,
        },
        "status": "passed",
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
