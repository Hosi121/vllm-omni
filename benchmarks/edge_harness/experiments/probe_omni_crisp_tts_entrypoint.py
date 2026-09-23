#!/usr/bin/env python3
"""Verify public AsyncOmni audio delivery from the pinned hybrid TTS stage."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model_config_dir", "server_bin", "server_sha256", "talker_file", "talker_sha256",
        "codec_file", "codec_sha256", "punc_file", "punc_sha256", "server_log", "output_report",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args()

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_CRISP_HYBRID_PIPELINE

    backend = {
        "name": "external.crisp.tts.v1",
        "server_bin": args.server_bin,
        "server_sha256": args.server_sha256,
        "talker_file": args.talker_file,
        "talker_sha256": args.talker_sha256,
        "codec_file": args.codec_file,
        "codec_sha256": args.codec_sha256,
        "punc_file": args.punc_file,
        "punc_sha256": args.punc_sha256,
        "log_file": args.server_log,
        "expected_gpu_name": "AMD Radeon(TM) 890M Graphics",
        "ggml_vk_visible_devices": "1",
        "memory_overhead_bytes": 5 << 30,
        "max_text_bytes": 4096,
        "max_wav_bytes": 4 << 20,
        "max_audio_s": 10,
        "voice": "ryan",
        "seed": 42,
    }
    report = {"entrypoint": "AsyncOmni.generate", "pipeline": QWEN3_TTS_CRISP_HYBRID_PIPELINE.model_type}
    engine = None
    try:
        with tempfile.TemporaryDirectory(prefix="omni-crisp-tts-") as temporary:
            deployment = Path(temporary) / "deploy.yaml"
            deployment.write_text(yaml.safe_dump({
                "pipeline": QWEN3_TTS_CRISP_HYBRID_PIPELINE.model_type,
                "async_chunk": False,
                "stages": [{
                    "stage_id": 0,
                    "backend": backend,
                    "resource_budget": {
                        "capacities": {"host_ram": 16 << 30},
                        "demands": {"host_ram": 8 << 30},
                    },
                }],
            }), encoding="utf-8")
            started = time.perf_counter()
            engine = AsyncOmni(
                model=str(Path(args.model_config_dir).resolve(strict=True)),
                deploy_config=str(deployment), stage_init_timeout=180, init_timeout=240,
            )
            report["startup_s"] = time.perf_counter() - started
            outputs = []
            started = time.perf_counter()
            async for output in engine.generate(
                {"text": "Hello from the local computer.", "voice": "ryan", "seed": 42},
                request_id="public-tts-1",
            ):
                if output.error:
                    raise RuntimeError(output.error)
                audio = output.multimodal_output["audio"]
                pcm = np.rint(audio.numpy() * 32768).astype("<i2").tobytes()
                outputs.append({
                    "request_id": output.request_id,
                    "sample_rate": output.multimodal_output["sr"],
                    "frames": len(audio),
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "stage_event": output.custom_output.get("stage_event"),
                })
            report["request_wall_s"] = time.perf_counter() - started
            report["outputs"] = outputs
            assert len(outputs) == 1 and outputs[0]["request_id"] == "public-tts-1"
            assert outputs[0]["sample_rate"] == 24000 and outputs[0]["frames"] == 61440
            assert outputs[0]["pcm_sha256"] == "57f60228e2b456f6dfefc7e49df862c7bff6ba7cf39231ef1824f615948a1539"
            report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        destination = Path(args.output_report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
