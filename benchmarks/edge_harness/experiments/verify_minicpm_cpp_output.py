#!/usr/bin/env python3
"""Verify a complete MiniCPM-o C++ text/audio result and join its PCM chunks.

The upstream CLI currently writes wav_N.wav but its optional merger searches
for tts_output_chunk_N.wav. This verifier checks the actual bounded sequence,
then joins PCM frames without changing sample values.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import re
import wave
from pathlib import Path


ARTIFACTS = (
    "MiniCPM-o-4_5-Q4_K_M.gguf",
    "audio/MiniCPM-o-4_5-audio-F16.gguf",
    "vision/MiniCPM-o-4_5-vision-F16.gguf",
    "tts/MiniCPM-o-4_5-tts-F16.gguf",
    "tts/MiniCPM-o-4_5-projector-F16.gguf",
    "token2wav-gguf/encoder.gguf",
    "token2wav-gguf/flow_matching.gguf",
    "token2wav-gguf/flow_extra.gguf",
    "token2wav-gguf/hifigan2.gguf",
    "token2wav-gguf/prompt_cache.gguf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--merged-wav", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--platform-label", default="WSL CPU")
    parser.add_argument(
        "--expected-placement", choices=("cpu", "radeon-hybrid"), default="cpu"
    )
    parser.add_argument("--process-profile", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve(strict=True)
    log = args.run_log.read_text(encoding="utf-8", errors="replace")
    required_log = (
        "Media type: 2 (omni: audio+vision)",
        "Token2Wav device: cpu",
        "Token2Wav: initialized successfully",
        "stream_prefill(index=1): processing user audio:",
        "stream_prefill(index=1): user audio embedding:",
        "stream_prefill: prefilled 1 vision chunks",
    )
    placement_log = {
        "cpu": (
            "GPU layers: 0",
            "vision_ctx: vision using CPU backend",
        ),
        "radeon-hybrid": (
            "GPU layers: 99",
            "using device Vulkan0 (AMD Radeon(TM) 890M Graphics)",
            "offloaded 37/37 layers to GPU",
            "vision_ctx: vision using Vulkan0 backend",
            "TTS model: loading with n_gpu_layers=0",
            "Token2Wav: non-CUDA backend, vocoder using CPU",
        ),
    }[args.expected_placement]
    missing_log = [item for item in required_log + placement_log if item not in log]
    if missing_log:
        raise RuntimeError(f"missing complete-path/placement evidence: {missing_log}")
    process_profile = None
    if args.process_profile:
        process_profile = json.loads(args.process_profile.read_text(encoding="utf-8"))
        if process_profile["exit_code"] != 0:
            raise RuntimeError("profiled process exited with an error")
    elif "Exit status: 0" not in log:
        raise RuntimeError("/usr/bin/time did not report exit status 0")
    wave_dir = output_dir / "round_000" / "tts_wav"
    if not (wave_dir / "generation_done.flag").is_file():
        raise RuntimeError("Token2Wav did not finish")
    chunk_paths = sorted(
        wave_dir.glob("wav_*.wav"),
        key=lambda path: int(re.fullmatch(r"wav_(\d+)\.wav", path.name).group(1)),
    )
    if not 0 < len(chunk_paths) <= 256:
        raise RuntimeError("no bounded audio chunk sequence")
    if [path.name for path in chunk_paths] != [f"wav_{i}.wav" for i in range(len(chunk_paths))]:
        raise RuntimeError("audio chunk sequence has a gap or duplicate")

    pcm_chunks: list[bytes] = []
    chunks: list[dict] = []
    square_sum = 0
    frames_total = 0
    for path in chunk_paths:
        with wave.open(str(path), "rb") as source:
            fmt = (source.getnchannels(), source.getsampwidth(), source.getframerate())
            if fmt != (1, 2, 24000):
                raise RuntimeError(f"invalid PCM16 mono 24 kHz chunk: {path.name}, {fmt}")
            frames = source.getnframes()
            if not 0 < frames <= 24000 * 10:
                raise RuntimeError(f"invalid bounded chunk length: {path.name}, {frames}")
            pcm = source.readframes(frames)
        if len(pcm) != frames * 2:
            raise RuntimeError(f"truncated audio chunk: {path.name}")
        samples = array.array("h")
        samples.frombytes(pcm)
        square_sum += sum(int(value) ** 2 for value in samples)
        frames_total += frames
        pcm_chunks.append(pcm)
        chunks.append({"name": path.name, "frames": frames, "sha256": sha256(path)})
    if frames_total > 24000 * 120 or square_sum == 0:
        raise RuntimeError("assembled audio is too long or silent")
    pcm = b"".join(pcm_chunks)
    args.merged_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.merged_wav), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24000)
        target.writeframes(pcm)

    text_dir = output_dir / "round_000" / "llm_debug"
    text_paths = sorted(
        text_dir.glob("chunk_*/llm_text.txt"),
        key=lambda path: int(path.parent.name.removeprefix("chunk_")),
    )
    response_text = "".join(path.read_text(encoding="utf-8") for path in text_paths).strip()
    if not response_text:
        raise RuntimeError("language stage returned no text")
    memory_match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
    wall_match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", log)
    model_dir = args.model_dir.resolve(strict=True)
    artifacts = {}
    for name in ARTIFACTS:
        path = model_dir / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    report = {
        "scope": f"standalone MiniCPM-o 4.5 GGUF audio+image input to text+speech on {args.platform_label}",
        "status": "scoped_complete_path_pass",
        "expected_placement": args.expected_placement,
        "quality": "synthetic red-square/tone response not established as semantically correct",
        "model_revision": args.model_revision,
        "artifacts": artifacts,
        "runtime_binary_sha256": sha256(args.runtime_bin),
        "input_image_sha256": sha256(args.input_image),
        "input_audio_sha256": sha256(args.input_audio),
        "response_text": response_text,
        "audio_chunks": chunks,
        "audio_frames": frames_total,
        "audio_duration_s": frames_total / 24000,
        "audio_rms_pcm16": math.sqrt(square_sum / frames_total),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "merged_wav_sha256": sha256(args.merged_wav),
        "peak_rss_kib_from_time": int(memory_match.group(1)) if memory_match else None,
        "wall_clock_from_time": wall_match.group(1) if wall_match else None,
        "sampled_process_profile": process_profile,
        "upstream_merge_name_mismatch": "TTS: no valid WAV files to merge" in log,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "artifacts"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
