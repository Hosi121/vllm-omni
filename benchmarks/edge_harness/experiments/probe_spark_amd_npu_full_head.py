#!/usr/bin/env python3
"""Reassemble four real-weight Spark output-head shards on one Windows NPU.

RMS normalization and QDQ boundary nodes may execute on CPU. This probes only
the output head; it does not qualify generation, KV state or an Omni handoff.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

from probe_spark_amd_npu_lm_head import compare, sha256


def percentile_nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--ep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    data = np.load(args.activations)
    inputs, references = data["x"], data["hidden_out"]
    if inputs.shape != (4, 1, 1, 2048) or references.shape != (4, 1, 131072):
        raise ValueError(f"Unexpected real-activation layout: {inputs.shape}, {references.shape}")

    started = time.perf_counter()
    cpu = ort.InferenceSession(str(args.source_model), providers=["CPUExecutionProvider"])
    cpu_load_s = time.perf_counter() - started
    cpu_rows = []
    for i in range(len(inputs)):
        started = time.perf_counter()
        actual = cpu.run(None, {"x": inputs[i]})[0]
        row = compare(references[i], actual)
        row.update(index=i, wall_s=time.perf_counter() - started)
        cpu_rows.append(row)
    cpu.run(None, {"x": inputs[0]})
    cpu_times = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        cpu.run(None, {"x": inputs[0]})
        cpu_times.append(time.perf_counter() - started)
    del cpu
    gc.collect()

    source = onnx.load(str(args.source_model))
    norm = numpy_helper.to_array(next(item for item in source.graph.initializer if item.name == "norm"))
    del source

    ep_dir = args.ep_dir.resolve()
    os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(str(ep_dir))
    ort.register_execution_provider_library("vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
    devices = [
        device for device in ort.get_ep_devices()
        if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")
    ]
    if not devices:
        raise RuntimeError("VitisAI EP did not expose an NPU device")

    sessions = []
    shard_load_s = []
    shard_artifacts = []
    for offset in (0, 32768, 65536, 98304):
        folder = args.shard_root / f"shard_{offset}"
        artifact = folder / "spark_lm_head_a16w8_qdq.onnx"
        manifest = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if (manifest["vocab_shard_offset"] != offset or manifest["vocab_shard_size"] != 32768
                or manifest["quantized_sha256"] != sha256(artifact)
                or manifest["npu_node_events"] == 0):
            raise ValueError(f"Unqualified or changed output-head shard: {folder}")
        options = ort.SessionOptions()
        options.add_provider_for_devices(devices, {})
        options.enable_profiling = True
        options.profile_file_prefix = str(args.output_dir / f"shard_{offset}_profile")
        started = time.perf_counter()
        session = ort.InferenceSession(str(artifact), sess_options=options)
        shard_load_s.append(time.perf_counter() - started)
        sessions.append(session)
        shard_artifacts.append({
            "offset": offset,
            "quantized_sha256": manifest["quantized_sha256"],
            "providers": session.get_providers(),
        })

    def execute(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(1, 2048)
        normalized = flat * (
            1.0 / np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + np.float32(1e-6))
        )
        normalized = np.ascontiguousarray(normalized * norm)
        return np.concatenate(
            [session.run(None, {session.get_inputs()[0].name: normalized})[0]
             for session in sessions],
            axis=-1,
        )

    npu_rows = []
    for i in range(len(inputs)):
        started = time.perf_counter()
        actual = execute(inputs[i])
        row = compare(references[i], actual)
        row.update(
            index=i,
            wall_s=time.perf_counter() - started,
            shard_checks=[
                {"offset": offset, **compare(
                    references[i, :, offset : offset + 32768],
                    actual[:, offset : offset + 32768],
                )}
                for offset in (0, 32768, 65536, 98304)
            ],
        )
        npu_rows.append(row)
    execute(inputs[0])
    npu_times = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        execute(inputs[0])
        npu_times.append(time.perf_counter() - started)

    placement = []
    for artifact, session in zip(shard_artifacts, sessions, strict=True):
        profile_path = Path(session.end_profiling())
        events = json.loads(profile_path.read_text(encoding="utf-8"))
        providers = [
            (event.get("args") or {}).get("provider")
            for event in events if event.get("cat") == "Node"
        ]
        placement.append({
            "offset": artifact["offset"],
            "npu_node_events": providers.count("vitisai"),
            "cpu_node_events": providers.count("CPUExecutionProvider"),
            "profile_file": str(profile_path),
        })

    report = {
        "scope": "Spark-X2.5-1.7B complete 131072-logit output head only; CPU RMS normalization plus four NPU MatMul shards, no full generation",
        "hardware": "AMD Ryzen AI 9 HX 370 NPU",
        "os": platform.platform(),
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "source_model_sha256": sha256(args.source_model),
        "activation_sha256": sha256(args.activations),
        "source_shape": list(inputs.shape[1:]),
        "output_shape": list(references.shape[1:]),
        "cpu_load_s": cpu_load_s,
        "shard_load_s": shard_load_s,
        "shard_artifacts": shard_artifacts,
        "placement": placement,
        "cpu_reference_rows": cpu_rows,
        "npu_rows": npu_rows,
        "timing_protocol": f"one warmup then {args.repeats} serial repeats of captured activation 0, host wall time including normalization, NPU transfers, four sessions and concatenation",
        "cpu_wall_s": cpu_times,
        "npu_wall_s": npu_times,
        "cpu_wall_p50_s": percentile_nearest_rank(cpu_times, 0.5),
        "cpu_wall_p95_s": percentile_nearest_rank(cpu_times, 0.95),
        "npu_wall_p50_s": percentile_nearest_rank(npu_times, 0.5),
        "npu_wall_p95_s": percentile_nearest_rank(npu_times, 0.95),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "placement": placement,
        "npu_rows": npu_rows,
        "cpu_wall_p50_s": report["cpu_wall_p50_s"],
        "npu_wall_p50_s": report["npu_wall_p50_s"],
    }, indent=2))
    if not all(item["npu_node_events"] > 0 for item in placement):
        raise RuntimeError("At least one full-head shard fell back entirely to CPU")
    if not all(item["top1_match"] and item["finite"] and item["snr_db"] > 40
               for item in npu_rows):
        raise RuntimeError("Full-vocabulary NPU head failed parity on real activations")


if __name__ == "__main__":
    main()
