#!/usr/bin/env python3
"""Verify the same prompted MiniCPM-o request across CPU and Radeon runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def language_tokens(path: Path) -> list[int]:
    matches = re.findall(r"LLM->TTS:.*?token_ids=\[([^]]+)\]", path.read_text(
        encoding="utf-8", errors="replace"
    ))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one logged text chunk: {path}")
    return [int(value) for value in matches[0].split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, required=True)
    parser.add_argument("--radeon-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu, gpu = (read_json(path / "full_report.json") for path in
                (args.cpu_dir, args.radeon_dir))
    if (cpu["expected_placement"], gpu["expected_placement"]) != (
        "cpu", "radeon-hybrid"
    ):
        raise RuntimeError("incorrect device placement")
    if any(item["status"] != "scoped_complete_path_pass" for item in (cpu, gpu)):
        raise RuntimeError("one complete path failed")
    for key in ("artifacts", "model_revision", "input_image_sha256",
                "input_audio_sha256", "response_text"):
        if cpu[key] != gpu[key]:
            raise RuntimeError(f"cross-device mismatch: {key}")
    cpu_tokens = language_tokens(args.cpu_dir / "full.stdout.log")
    gpu_tokens = language_tokens(args.radeon_dir / "full.stdout.log")
    if cpu_tokens != gpu_tokens:
        raise RuntimeError("language token IDs differ")
    input_asr = read_json(args.radeon_dir / "input_asr.json")
    cpu_asr = read_json(args.cpu_dir / "output_asr.json")
    gpu_asr = read_json(args.radeon_dir / "output_asr.json")
    if input_asr["word_error_rate"] != 0 or any(
        item["word_error_rate"] != 0 for item in (cpu_asr, gpu_asr)
    ):
        raise RuntimeError("input or output speech failed the ASR proxy")
    if cpu["response_text"] != "The square in the image is red.":
        raise RuntimeError("visual question was not answered correctly")
    report = {
        "scope": "one synthetic spoken visual question; standalone CPU vs Radeon hybrid",
        "status": "scoped_prompted_path_pass",
        "checkpoint_revision": cpu["model_revision"],
        "same_artifact_set": True,
        "same_input_image_audio": True,
        "spoken_input": input_asr["reference_text"],
        "input_asr_wer": input_asr["word_error_rate"],
        "response_text": cpu["response_text"],
        "language_token_ids": cpu_tokens,
        "cpu": {
            "cold_wall_s": cpu["sampled_process_profile"]["wall_s"],
            "audio_duration_s": cpu["audio_duration_s"],
            "pcm_sha256": cpu["pcm_sha256"],
            "output_asr_wer": cpu_asr["word_error_rate"],
        },
        "radeon_hybrid": {
            "cold_wall_s": gpu["sampled_process_profile"]["wall_s"],
            "audio_duration_s": gpu["audio_duration_s"],
            "pcm_sha256": gpu["pcm_sha256"],
            "output_asr_wer": gpu_asr["word_error_rate"],
        },
        "quality_limit": "one synthetic red square and one spoken question",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
