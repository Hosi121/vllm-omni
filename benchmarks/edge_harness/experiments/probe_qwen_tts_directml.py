#!/usr/bin/env python3
"""Probe whole-model Qwen3-TTS on a named native Windows DirectML adapter.

This is a standalone feasibility run; it does not claim Omni integration or
that every operator executes on the target adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
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
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--expected-device-name", default="AMD Radeon(TM) 890M")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--text", default="Hello from the local computer.")
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--trace-cat", action="store_true")
    parser.add_argument("--workaround-empty-int64-cat", action="store_true")
    parser.add_argument("--subtalker-greedy", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.device_index < 0 or args.threads <= 0 or args.max_new_tokens <= 0:
        parser.error("device index must be nonnegative; threads and token limit positive")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import soundfile as sf
    import torch
    import torch_directml
    import transformers
    from qwen_tts import Qwen3TTSModel

    torch.set_num_threads(args.threads)
    names = [torch_directml.device_name(i).rstrip("\x00") for i in range(torch_directml.device_count())]
    if args.device_index >= len(names) or args.expected_device_name not in names[args.device_index]:
        raise RuntimeError(f"requested DirectML adapter is absent: index={args.device_index}, adapters={names}")
    device = torch_directml.device(args.device_index)
    report = {
        "scope": "standalone real-weight Qwen3-TTS DirectML feasibility; no Omni, streaming, or all-operator placement claim",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_directml": getattr(torch_directml, "__version__", "unknown"),
        "transformers": transformers.__version__,
        "adapters": names,
        "requested_index": args.device_index,
        "target_device": str(device),
        "model_dir": str(args.model_dir.resolve()),
        "talker_sha256": _sha256(args.model_dir / "model.safetensors"),
        "speech_tokenizer_sha256": _sha256(args.model_dir / "speech_tokenizer" / "model.safetensors"),
        "dtype": args.dtype,
        "attention": "sdpa",
        "threads": args.threads,
        "text": args.text,
        "speaker": args.speaker,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty_override": args.repetition_penalty,
        "workaround_empty_int64_cat": args.workaround_empty_int64_cat,
        "subtalker_greedy": args.subtalker_greedy,
        "seed": 42,
        "status": "started",
    }
    try:
        started = time.perf_counter()
        model = Qwen3TTSModel.from_pretrained(
            str(args.model_dir.resolve()),
            device_map="cpu",
            dtype=getattr(torch, args.dtype),
            attn_implementation="sdpa",
            local_files_only=True,
        )
        report["cpu_load_s"] = time.perf_counter() - started
        report["phase"] = "move_to_directml"
        started = time.perf_counter()
        model.model.to(device)
        report["device_move_s"] = time.perf_counter() - started
        report["model_devices"] = sorted({str(parameter.device) for parameter in model.model.parameters()})
        if report["model_devices"] != [str(device)]:
            raise RuntimeError(f"model parameters not wholly on DirectML target: {report['model_devices']}")
        # Qwen's wrapper captures its device at construction, before this move.
        model.device = device
        report["phase"] = "generate"
        if args.trace_cat or args.workaround_empty_int64_cat:
            original_cat = torch.cat

            def traced_cat(tensors, *cat_args, **cat_kwargs):
                tensors = tuple(tensors)
                dim = cat_kwargs.get("dim", cat_args[0] if cat_args else 0)
                axis = dim % tensors[0].ndim if tensors else dim
                if (
                    args.workaround_empty_int64_cat
                    and len(tensors) == 2
                    and tensors[0].device == device
                    and tensors[1].device == device
                    and tensors[0].dtype == torch.int64
                    and tensors[1].dtype == torch.int64
                    and tensors[0].ndim == tensors[1].ndim
                    and tensors[0].numel() == 0
                    and tensors[0].shape[axis] == 0
                    and all(tensors[0].shape[i] == tensors[1].shape[i] for i in range(tensors[0].ndim) if i != axis)
                ):
                    print("EMPTY_INT64_CAT_IDENTITY", file=sys.stderr, flush=True)
                    return tensors[1].clone()
                try:
                    return original_cat(tensors, *cat_args, **cat_kwargs)
                except Exception:
                    print(
                        "CAT_FAILURE " + repr([(str(x.device), str(x.dtype), tuple(x.shape)) for x in tensors]),
                        file=sys.stderr,
                        flush=True,
                    )
                    raise

            torch.cat = traced_cat
        torch.manual_seed(42)
        started = time.perf_counter()
        generation_overrides = {}
        if args.repetition_penalty is not None:
            generation_overrides["repetition_penalty"] = args.repetition_penalty
        if args.subtalker_greedy:
            generation_overrides["subtalker_dosample"] = False
        wavs, sample_rate = model.generate_custom_voice(
            text=args.text,
            language="English",
            speaker=args.speaker,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            non_streaming_mode=True,
            **generation_overrides,
        )
        report["request_s"] = time.perf_counter() - started
        if len(wavs) != 1 or sample_rate <= 0:
            raise RuntimeError("expected one waveform at a valid sample rate")
        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if not wav.size or not np.isfinite(wav).all():
            raise RuntimeError("waveform is empty or non-finite")
        report["rms"] = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
        report["peak_abs"] = float(np.max(np.abs(wav)))
        report["nonzero_samples"] = int(np.count_nonzero(wav))
        if report["rms"] <= 1e-5:
            raise RuntimeError("waveform is effectively silent")
        args.output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output_wav, wav, sample_rate, subtype="PCM_16")
        report.update(
            status="passed",
            phase="complete",
            sample_rate=sample_rate,
            samples=int(wav.size),
            duration_s=float(wav.size / sample_rate),
            wav_sha256=_sha256(args.output_wav),
        )
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if isinstance(exc, UnicodeDecodeError):
            report["backend_error_bytes_hex"] = exc.object.hex()
            report["backend_error_cp936"] = exc.object.decode("cp936", errors="replace")
        raise
    finally:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
