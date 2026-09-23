#!/usr/bin/env python3
"""Probe real InternVLA Cosmos encoder ONNX placement on the HX370 AMD NPU.

This is a fixed-shape component probe. A successful output without VitisAI
node events is a CPU fallback, not NPU support or a policy E2E result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import traceback
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(reference, actual) -> dict:
    import numpy as np

    if actual.shape != reference.shape or actual.dtype != np.float32 or not np.isfinite(actual).all():
        raise ValueError("Cosmos latent shape, dtype or finiteness differs from contract")
    diff = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(reference.astype(np.float64))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "fixtures", "export-report", "quantized", "ep-dir", "output-report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--prequantized", action="store_true")
    parser.add_argument("--cpu-only", action="store_true", help="Run numerical gate without loading VitisAI")
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--candidate-layout", default="Conv+MatMul A16W8 QDQ")
    parser.add_argument("--op-types", default="Conv,MatMul")
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--max-relative-l2", type=float, default=1e-3)
    args = parser.parse_args()
    if args.prequantized and not args.candidate_sha256:
        parser.error("prequantized candidate requires a pinned SHA256")
    if args.max_relative_l2 <= 0:
        parser.error("numerical gate must be positive")
    op_types = args.op_types.split(",")
    if not op_types or any(name not in {"Conv", "MatMul"} for name in op_types):
        parser.error("quantized op types must be Conv, MatMul or both")

    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )

    source = args.model.resolve(strict=True)
    fixture_path = args.fixtures.resolve(strict=True)
    export = json.loads(args.export_report.read_text(encoding="utf-8"))
    if export.get("status") != "passed":
        raise RuntimeError("source Cosmos ONNX export lacks a passed CPU gate")
    if sha256(source) != export["artifact"]["onnx"]["sha256"]:
        raise RuntimeError("Cosmos ONNX hash differs from pinned export")
    external = source.with_name(source.name + ".data")
    if (sha256(external) != export["artifact"]["external_data"][0]["sha256"]
            or sha256(fixture_path) != export["fixtures"]["sha256"]):
        raise RuntimeError("Cosmos external weights or fixture hash differs from pinned export")
    model = onnx.load(str(source), load_external_data=True)
    if [item.name for item in model.graph.input] != ["pixels"]:
        raise RuntimeError("Cosmos ONNX input differs from fixed layout")
    with np.load(fixture_path, allow_pickle=False) as data:
        cases = [(name, np.asarray(data[f"{name}_pixels"]), np.asarray(data[f"{name}_reference"]))
                 for name in ("ramp", "pattern")]
    if any(pixels.shape != (6, 3, 256, 256) or reference.shape != (6, 16, 32, 32)
           for _, pixels, reference in cases):
        raise RuntimeError("Cosmos fixture layout differs from fixed contract")

    report = {
        "scope": "real InternVLA Cosmos CI8x8 fixed-shape encoder component only; no action-policy E2E claim",
        "hardware": "Ryzen AI 9 HX 370 AMD NPU", "os": platform.platform(),
        "onnx": onnx.__version__, "onnxruntime": ort.__version__,
        "source_sha256": sha256(source), "external_weights_sha256": sha256(external),
        "fixtures_sha256": sha256(fixture_path), "input_shape": [6, 3, 256, 256],
        "output_shape": [6, 16, 32, 32],
        "quantization": args.candidate_layout,
        "quantized_op_types": op_types, "per_channel": args.per_channel,
        "max_relative_l2_gate": args.max_relative_l2,
    }
    try:
        cpu = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
        report["source_cpu_parity"] = {
            name: compare(reference, cpu.run(["latent"], {"pixels": pixels})[0])
            for name, pixels, reference in cases
        }

        class Reader(CalibrationDataReader):
            def __init__(self) -> None:
                self._iter = iter({"pixels": pixels} for _, pixels, _ in cases)

            def get_next(self):
                return next(self._iter, None)

        if args.prequantized:
            args.quantized.resolve(strict=True)
            if sha256(args.quantized) != args.candidate_sha256.lower():
                raise RuntimeError("prequantized candidate hash differs from pinned SHA256")
        else:
            args.quantized.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            quantize_static(
                str(source), str(args.quantized), Reader(),
                quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt16,
                weight_type=QuantType.QInt8, per_channel=args.per_channel,
                op_types_to_quantize=op_types, calibrate_method=CalibrationMethod.MinMax,
            )
            report["quantize_s"] = time.perf_counter() - started
        report["quantized_sha256"] = sha256(args.quantized)
        report["quantized_bytes"] = args.quantized.stat().st_size
        candidate_cpu = ort.InferenceSession(str(args.quantized), providers=["CPUExecutionProvider"])
        report["quantized_cpu_parity"] = {
            name: compare(reference, candidate_cpu.run(["latent"], {"pixels": pixels})[0])
            for name, pixels, reference in cases
        }
        if any(row["relative_l2"] > args.max_relative_l2
               for row in report["quantized_cpu_parity"].values()):
            raise RuntimeError("quantized Cosmos artifact failed the CPU numerical gate before NPU compilation")
        if args.cpu_only:
            report["status"] = "passed_cpu_component"
            return

        ep_dir = args.ep_dir.resolve(strict=True)
        os.environ["PATH"] = str(ep_dir) + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(str(ep_dir))
        ort.register_execution_provider_library("vitisai", str(ep_dir / "onnxruntime_vitisai_ep.dll"))
        devices = [device for device in ort.get_ep_devices()
                   if device.ep_name == "vitisai" and str(device.device.type).endswith("NPU")]
        report["npu_devices"] = len(devices)
        if not devices:
            raise RuntimeError("VitisAI EP did not expose an NPU device")
        options = ort.SessionOptions()
        options.add_provider_for_devices(devices, {})
        options.enable_profiling = True
        options.profile_file_prefix = str(args.output_report.with_suffix("")) + "_profile"
        started = time.perf_counter()
        session = ort.InferenceSession(str(args.quantized), sess_options=options)
        report["session_load_s"] = time.perf_counter() - started
        report["session_providers"] = session.get_providers()
        rows = []
        for name, pixels, reference in cases:
            started = time.perf_counter()
            actual = session.run(["latent"], {"pixels": pixels})[0]
            rows.append({"fixture": name, "wall_s": time.perf_counter() - started,
                         **compare(reference, actual)})
        report["npu_rows"] = rows
        profile_path = Path(session.end_profiling())
        report["profile_path"] = str(profile_path)
        events = json.loads(profile_path.read_text(encoding="utf-8"))
        providers = [(event.get("args") or {}).get("provider")
                     for event in events if event.get("cat") == "Node"]
        report["npu_node_events"] = providers.count("vitisai")
        report["cpu_node_events"] = providers.count("CPUExecutionProvider")
        if report["npu_node_events"] == 0:
            raise RuntimeError("quantized Cosmos graph returned output but no NPU node executed")
        report["status"] = "passed_component"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "traceback"}, indent=2))


if __name__ == "__main__":
    main()
