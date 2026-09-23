#!/usr/bin/env python3
"""Check generated Mandarin speech against its own text with local Whisper.

Character error rate is an intelligibility proxy, not a task-quality or
speaker-fidelity score. It does not say whether the model understood its input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def characters(value: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]", value.casefold())


def character_error_rate(reference: str, actual: str) -> float:
    expected, got = characters(reference), characters(actual)
    previous = list(range(len(got) + 1))
    for position, item in enumerate(expected, 1):
        current = [position]
        for index, candidate in enumerate(got, 1):
            current.append(min(
                current[-1] + 1,
                previous[index] + 1,
                previous[index - 1] + (item != candidate),
            ))
        previous = current
    return previous[-1] / max(len(expected), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import librosa
    import numpy as np
    from opencc import OpenCC
    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    audio, sample_rate = sf.read(args.wav, dtype="float32", always_2d=False)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("expected finite mono audio")
    audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    reference = args.reference_file.read_text(encoding="utf-8").strip()
    if not characters(reference):
        raise ValueError("reference text has no Mandarin or alphanumeric characters")
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(args.model_dir), local_files_only=True
    ).to("cpu")
    load_s = time.perf_counter() - started
    if {parameter.device.type for parameter in model.parameters()} != {"cpu"}:
        raise RuntimeError("Whisper was not wholly on CPU")
    started = time.perf_counter()
    transcripts = []
    segment_frames = 25 * 16000
    with torch.no_grad():
        for offset in range(0, len(audio_16k), segment_frames):
            features = processor(
                audio_16k[offset:offset + segment_frames],
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            ids = model.generate(
                features.input_features, attention_mask=features.attention_mask,
                language="zh", task="transcribe",
            )
            transcripts.append(
                processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            )
    infer_s = time.perf_counter() - started
    transcript = "".join(transcripts)
    traditional_to_simplified = OpenCC("t2s")
    normalized_reference = traditional_to_simplified.convert(reference)
    normalized_transcript = traditional_to_simplified.convert(transcript)
    report = {
        "scope": "Whisper Mandarin ASR proxy for generated speech; no input-task or perceptual-quality claim",
        "asr_model": "openai/whisper-tiny",
        "asr_revision": args.model_revision,
        "asr_weights_sha256": sha256(args.model_dir / "model.safetensors"),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "wav_sha256": sha256(args.wav),
        "wav_duration_s": len(audio) / sample_rate,
        "reference_text": reference,
        "transcript": transcript,
        "asr_segments_25_s": len(transcripts),
        "character_error_rate_raw": character_error_rate(reference, transcript),
        "character_error_rate_t2s_normalized": character_error_rate(
            normalized_reference, normalized_transcript
        ),
        "normalization": "OpenCC 0.1.7 t2s; punctuation ignored",
        "load_s": load_s,
        "inference_s": infer_s,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
