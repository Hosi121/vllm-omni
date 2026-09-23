#!/usr/bin/env python3
"""Transcribe a generated WAV with a separately pinned local ASR checkpoint.

ASR text agreement is only a proxy for intelligibility, never a speech-quality
or speaker-similarity evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _word_error_rate(reference: str, actual: str) -> float:
    expected, got = _words(reference), _words(actual)
    previous = list(range(len(got) + 1))
    for index, token in enumerate(expected, start=1):
        current = [index]
        for position, candidate in enumerate(got, start=1):
            current.append(min(
                current[-1] + 1,
                previous[position] + 1,
                previous[position - 1] + (token != candidate),
            ))
        previous = current
    return previous[-1] / max(len(expected), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    wav, sample_rate = sf.read(args.wav, dtype="float32", always_2d=False)
    if wav.ndim != 1 or sample_rate <= 0 or not wav.size or not np.isfinite(wav).all():
        raise ValueError("Expected finite mono audio")
    audio_16k = librosa.resample(wav, orig_sr=sample_rate, target_sr=16000)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(args.model_dir), local_files_only=True
    ).to("cpu")
    load_s = time.perf_counter() - started
    if {parameter.device.type for parameter in model.parameters()} != {"cpu"}:
        raise RuntimeError("ASR model was not wholly on CPU")
    features = processor(
        audio_16k, sampling_rate=16000, return_tensors="pt",
        return_attention_mask=True,
    )
    started = time.perf_counter()
    with torch.no_grad():
        token_ids = model.generate(
            features.input_features, attention_mask=features.attention_mask
        )
    inference_s = time.perf_counter() - started
    transcript = processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
    report = {
        "scope": "local Whisper tiny.en ASR proxy for generated speech; no perceptual quality claim",
        "asr_model": "openai/whisper-tiny.en",
        "asr_revision": args.model_revision,
        "asr_weights_sha256": _sha256(args.model_dir / "model.safetensors"),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "wav_sha256": _sha256(args.wav),
        "input_sample_rate": sample_rate,
        "input_duration_s": wav.size / sample_rate,
        "reference_text": args.reference_text,
        "transcript": transcript,
        "word_error_rate": _word_error_rate(args.reference_text, transcript),
        "asr_load_s": load_s,
        "asr_inference_s": inference_s,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
