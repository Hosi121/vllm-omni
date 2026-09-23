#!/usr/bin/env python3
"""Audit a real Spark attention-layer AI Hub result without claiming model E2E."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


OUTPUT_NAMES = ("hidden", "k_new_0", "v_new_0")
SOURCE_ONNX_SHA256 = "4e2b1c8d791b1ee0e9b75f3322a20a743c0bf8c40f9b98edd963f2944d6300bb"
QUANTIZED_ZIP_SHA256 = "703821ca8ace5672228919ea2add4901def4887748c381b12ffc8b259665c553"
FIXTURE_SHA256 = "5a439e6c6d148f1720938770daa110039a39edde014f1c1592c823c7d7dfc8ed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_outputs(path: Path, *, hub: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        actual_names = tuple(data.files)
        expected_names = tuple(f"output_{i}__0" for i in range(3)) if hub else OUTPUT_NAMES
        if actual_names != expected_names:
            raise ValueError(f"{path}: output names/order differ: {actual_names}")
        values = {name: np.asarray(data[key]) for name, key in zip(OUTPUT_NAMES, expected_names)}
    shapes = ((1, 1, 2048), (1, 2, 1, 256), (1, 2, 1, 256))
    for name, shape in zip(OUTPUT_NAMES, shapes):
        value = values[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"{path}: {name} violates FP32 finite layer-output contract")
    return values


def compare(reference: dict[str, np.ndarray], actual: dict[str, np.ndarray]) -> dict:
    result = {}
    for name in OUTPUT_NAMES:
        left = reference[name].astype(np.float64).ravel()
        right = actual[name].astype(np.float64).ravel()
        if not np.linalg.norm(left):
            raise ValueError(f"{name}: zero reference cannot define relative L2")
        result[name] = {
            "relative_l2": float(np.linalg.norm(left - right) / np.linalg.norm(left)),
            "max_abs": float(np.max(np.abs(left - right))),
            "cosine": float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))),
        }
    return result


def nearest_rank(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-name", required=True)
    for name in (
        "source-onnx", "quantized-zip", "fixture", "source-cpu-output",
        "quantized-cpu-output", "s25-output", "s25-inference-report", "device-output", "compile-report",
        "inference-report", "profile-report", "output-report",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    compile_report = json.loads(args.compile_report.read_text(encoding="utf-8"))
    inference = json.loads(args.inference_report.read_text(encoding="utf-8"))
    s25_inference = json.loads(args.s25_inference_report.read_text(encoding="utf-8"))
    profile = json.loads(args.profile_report.read_text(encoding="utf-8"))
    if compile_report.get("status") != "SUCCESS" or inference.get("status") != "SUCCESS" or profile.get("status") != "SUCCESS":
        raise RuntimeError("QNN DLC compile, device inference and profile must all succeed")
    if (compile_report.get("source_quantized_model_id") != "mmd5e3v3q"
            or compile_report.get("options") != "--target_runtime qnn_dlc --quantize_io"):
        raise RuntimeError("compile did not use the pinned quantized Spark layer and QNN DLC layout")
    target = compile_report["target_model_id"]
    if inference["model_id"] != target or profile["model_id"] != target:
        raise RuntimeError("device inference/profile used a different compiled artifact")
    if inference["input_dataset_id"] != "d7m83gjy2":
        raise RuntimeError("device inference did not use the pinned cross-device fixture")
    if (s25_inference.get("status") != "SUCCESS" or s25_inference.get("model_id") != "mnowl6xpq"
            or s25_inference.get("input_dataset_id") != inference["input_dataset_id"]
            or s25_inference.get("device", {}).get("name") != "Samsung Galaxy S25"):
        raise RuntimeError("historical S25 output lacks matching model/device/dataset provenance")
    if inference["device"]["name"] != args.device_name or profile["device"]["name"] != args.device_name:
        raise RuntimeError("jobs did not execute on the exact requested device")
    if inference["options"] != "--compute_unit npu" or profile["options"] != "--compute_unit npu":
        raise RuntimeError("jobs did not request NPU placement")

    execution = profile["profile"]
    times_us = execution["execution_summary"]["all_inference_times"]
    if not times_us or any(not isinstance(value, int) or value <= 0 for value in times_us):
        raise RuntimeError("missing or invalid device samples")
    units = Counter(row.get("compute_unit") for row in execution["execution_detail"])
    if units["NPU"] <= 0:
        raise RuntimeError("profile has no attributed NPU execution rows")

    source = read_outputs(args.source_cpu_output, hub=False)
    quantized = read_outputs(args.quantized_cpu_output, hub=False)
    s25 = read_outputs(args.s25_output, hub=True)
    device_output = read_outputs(args.device_output, hub=True)
    files = {
        name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in {
            "source_onnx": args.source_onnx,
            "quantized_onnx_zip": args.quantized_zip,
            "fixture": args.fixture,
            "source_cpu_output": args.source_cpu_output,
            "quantized_cpu_output": args.quantized_cpu_output,
            "s25_output": args.s25_output,
            "device_output": args.device_output,
        }.items()
    }
    for name, expected in {
        "source_onnx": SOURCE_ONNX_SHA256,
        "quantized_onnx_zip": QUANTIZED_ZIP_SHA256,
        "fixture": FIXTURE_SHA256,
    }.items():
        if files[name]["sha256"] != expected:
            raise RuntimeError(f"{name} hash differs from pinned layer artifact or fixture")
    report = {
        "scope": "one real Spark-X2.5 full-attention decoder layer at one static 1024-token cache shape; no generation E2E",
        "status": "component_executed_on_npu_quality_unqualified",
        "device": profile["device"],
        "compile_job_id": compile_report["job_id"],
        "inference_job_id": inference["job_id"],
        "profile_job_id": profile["job_id"],
        "target_model_id": target,
        "input_dataset_id": inference["input_dataset_id"],
        "s25_prior_inference_job_id": s25_inference["job_id"],
        "files": files,
        "source_vs_quantized_cpu": compare(source, quantized),
        "quantized_cpu_vs_device_npu": compare(quantized, device_output),
        "source_cpu_vs_device_npu": compare(source, device_output),
        "s25_npu_vs_device_npu": compare(s25, device_output),
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
            "No task-level numerical tolerance or next-token quality gate is established.",
            "Hub component jobs do not prove device-local prefill, KV/ring update, sampling, continuous generation or Omni deployment.",
            "Profile sample latency is one component, not complete-request or sustained thermal performance.",
        ],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "profile": report["profile"]}, indent=2))


if __name__ == "__main__":
    main()
