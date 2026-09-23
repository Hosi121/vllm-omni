#!/usr/bin/env python3
"""Check a pinned Spark GGUF through the public AsyncOmni complete-request API."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import yaml


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model_file", "server_bin", "model_sha256", "server_sha256", "device", "server_log", "output_report"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--expected-device-name")
    parser.add_argument("--model-config-dir", help="Existing Spark HF config directory for public resolution")
    args = parser.parse_args()

    from vllm_omni.model_executor.models.spark2_5.pipeline import SPARK2_5_GGUF_TEXT_PIPELINE
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    pipeline = SPARK2_5_GGUF_TEXT_PIPELINE
    backend = {
        "name": "external.llamacpp.text.v1",
        "model_file": args.model_file,
        "model_sha256": args.model_sha256,
        "server_bin": args.server_bin,
        "server_sha256": args.server_sha256,
        "log_file": args.server_log,
        "device": args.device,
        "expected_device_name": args.expected_device_name,
        "context_tokens": 4096,
        "max_new_tokens": 96,
        "max_io_bytes": 1 << 20,
        "memory_overhead_bytes": 2 << 30,
    }
    report = {"device": args.device, "entrypoint": "AsyncOmni.generate"}
    engine = None
    try:
        with tempfile.TemporaryDirectory(prefix="omni-spark-gguf-") as temporary:
            directory = Path(temporary)
            if args.model_config_dir:
                model_config_dir = Path(args.model_config_dir).resolve(strict=True)
            else:
                (directory / "config.json").write_text('{"model_type":"bert"}', encoding="utf-8")
                model_config_dir = directory
            deployment = directory / "deploy.yaml"
            deployment.write_text(yaml.safe_dump({
                "pipeline": pipeline.model_type,
                "async_chunk": False,
                "stages": [{
                    "stage_id": 0,
                    "backend": backend,
                    "resource_budget": {
                        "capacities": {"host_ram": 12 << 30},
                        "demands": {"host_ram": 4 << 30},
                    },
                }],
            }), encoding="utf-8")
            started = time.perf_counter()
            engine = AsyncOmni(
                model=str(model_config_dir), deploy_config=str(deployment),
                stage_init_timeout=120, init_timeout=180,
            )
            report["startup_s"] = time.perf_counter() - started
            outputs = []
            started = time.perf_counter()
            async for output in engine.generate(
                {"text": "What is the capital of France? Answer in one word.", "max_tokens": 96},
                request_id="public-spark-1",
            ):
                outputs.append({
                    "request_id": output.request_id,
                    "text": output.outputs[0].text if output.outputs else None,
                    "error": output.error,
                    "stage_event": output.custom_output.get("stage_event"),
                })
            report["request_wall_s"] = time.perf_counter() - started
            report["outputs"] = outputs
            assert len(outputs) == 1 and outputs[0]["request_id"] == "public-spark-1"
            assert not outputs[0]["error"] and outputs[0]["text"].strip() == "Paris"
            report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
