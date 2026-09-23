#!/usr/bin/env python3
"""Audit an RB3 Spark attention-layer fallback against the FP32 source."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_spark_qualcomm_attention import (
    FIXTURE_SHA256,
    SOURCE_ONNX_SHA256,
    compare,
    nearest_rank,
    read_outputs,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-name", default="Dragonwing RB3 Gen 2 Vision Kit")
    parser.add_argument("--compute-unit", choices=("cpu", "npu"), required=True)
    parser.add_argument("--compile-options", required=True)
    for name in (
        "source-onnx", "fixture", "source-cpu-output", "compile-report",
        "inference-report", "profile-report", "device-output", "output-report",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    compile_report = json.loads(args.compile_report.read_text(encoding="utf-8"))
    inference = json.loads(args.inference_report.read_text(encoding="utf-8"))
    profile = json.loads(args.profile_report.read_text(encoding="utf-8"))
    if any(report.get("status") != "SUCCESS" for report in (compile_report, inference, profile)):
        raise RuntimeError("compile, inference and profile must all succeed")
    if compile_report.get("source_model_id") != "mn14dw6zn" or compile_report.get("options") != args.compile_options:
        raise RuntimeError("compile did not use the pinned FP32 Spark source and requested route")
    target_id = compile_report["target_model_id"]
    if inference.get("model_id") != target_id or profile.get("model_id") != target_id:
        raise RuntimeError("inference/profile did not use the compiled artifact")
    if inference.get("input_dataset_id") != "d7m83gjy2":
        raise RuntimeError("inference did not use the pinned cross-device fixture")
    for report in (compile_report, inference, profile):
        if report.get("device", {}).get("name") != args.device_name:
            raise RuntimeError("job did not use the exact requested device")
    requested = f"--compute_unit {args.compute_unit}"
    if inference.get("options") != requested or profile.get("options") != requested:
        raise RuntimeError("inference/profile requested a different compute unit")
    files = {
        name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in {
            "source_onnx": args.source_onnx,
            "fixture": args.fixture,
            "source_cpu_output": args.source_cpu_output,
            "device_output": args.device_output,
        }.items()
    }
    if files["source_onnx"]["sha256"] != SOURCE_ONNX_SHA256 or files["fixture"]["sha256"] != FIXTURE_SHA256:
        raise RuntimeError("source or fixture differs from the pinned experiment")
    output = read_outputs(args.device_output, hub=True)
    source = read_outputs(args.source_cpu_output, hub=False)
    execution = profile["profile"]
    times_us = execution["execution_summary"]["all_inference_times"]
    if not times_us or any(not isinstance(value, int) or value <= 0 for value in times_us):
        raise RuntimeError("missing or invalid device samples")
    units = Counter(row.get("compute_unit") for row in execution["execution_detail"])
    if units[args.compute_unit.upper()] <= 0:
        raise RuntimeError("profile has no rows attributed to the requested compute unit")
    report = {
        "scope": "one real Spark-X2.5 attention layer at one static 1024-token cache shape; no generation E2E",
        "status": f"component_executed_on_{args.compute_unit}_quality_unqualified",
        "device": profile["device"],
        "compile_job_id": compile_report["job_id"],
        "inference_job_id": inference["job_id"],
        "profile_job_id": profile["job_id"],
        "target_model_id": target_id,
        "input_dataset_id": inference["input_dataset_id"],
        "compile_options": args.compile_options,
        "files": files,
        "source_cpu_vs_device": compare(source, output),
        "profile": {
            "sample_count": len(times_us),
            "sample_min_us": min(times_us),
            "nearest_rank_p50_us": nearest_rank(times_us, 0.50),
            "nearest_rank_p95_us": nearest_rank(times_us, 0.95),
            "reported_estimated_inference_time_us": execution["execution_summary"]["estimated_inference_time"],
            "reported_peak_memory_bytes": execution["execution_summary"]["estimated_inference_peak_memory"],
            "compute_unit_row_counts": dict(units),
        },
        "limits": [
            "No next-token or task-level numerical tolerance is established.",
            "No complete prefill, KV/ring, sampling, continuous generation or Omni deployment is tested.",
            "Hosted component latency is not a complete request or sustained thermal profile.",
        ],
    }
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "profile": report["profile"]}, indent=2))


if __name__ == "__main__":
    main()
