#!/usr/bin/env python3
"""Qualify one real Spark output-head graph on the Windows Ryzen AI NPU.

This is a model-component probe, not a Spark generation or Omni pipeline pass.
The calibration file contains real pre-final-norm activations captured from the
matching Spark-X2.5-1.7B checkpoint and reference output-head logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(reference: np.ndarray, actual: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape:
        raise ValueError(f"Output shape changed: {reference.shape} vs {actual.shape}")
    error = reference.astype(np.float64) - actual.astype(np.float64)
    norm = np.linalg.norm(reference.astype(np.float64))
    return {
        "snr_db": float(20 * np.log10(norm / max(np.linalg.norm(error), 1e-12))),
        "max_abs_diff": float(np.max(np.abs(error))),
        "top1_match": bool(np.argmax(reference) == np.argmax(actual)),
        "finite": bool(np.isfinite(actual).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--ep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardware-label", default="AMD Ryzen AI 9 HX 370 NPU")
    parser.add_argument("--vocab-shard-size", type=int, default=0,
                        help="Probe the first N real output-head columns; 0 keeps the full head")
    parser.add_argument("--vocab-shard-offset", type=int, default=0,
                        help="Starting output column for a real-weight vocabulary shard")
    parser.add_argument("--signed-activations", action="store_true",
                        help="Use QInt16 rather than QUInt16 activations")
    parser.add_argument("--flatten-input", action="store_true",
                        help="Use equivalent [1, 2048] input and remove final singleton-axis Gather")
    parser.add_argument("--batch-repeats", type=int, default=1,
                        help="Repeat each real activation N times in a batch (requires --flatten-input)")
    parser.add_argument("--matmul-only", action="store_true",
                        help="Move exact RMS normalization to the CPU side and probe only real-weight MatMul")
    parser.add_argument("--composite-shards", action="store_true",
                        help="Build four named 32768-column MatMuls and concatenate in one ONNX graph")
    parser.add_argument("--composite-with-norm", action="store_true",
                        help="Keep the checkpoint RMS norm before four named MatMul shards")
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.calibration)
    inputs = data["x"]
    references = data["hidden_out"]
    if inputs.shape != (4, 1, 1, 2048) or references.shape != (4, 1, 131072):
        raise ValueError(f"Unexpected calibration layout: {inputs.shape}, {references.shape}")
    if (args.vocab_shard_size < 0 or args.vocab_shard_offset < 0
            or args.vocab_shard_offset + (args.vocab_shard_size or 131072) > references.shape[-1]):
        raise ValueError("The requested vocabulary shard must fit within 131072 columns")
    if args.batch_repeats < 1 or (args.batch_repeats > 1 and not args.flatten_input):
        raise ValueError("batch-repeats must be positive and requires --flatten-input")
    if args.matmul_only and not args.flatten_input:
        raise ValueError("matmul-only requires --flatten-input")
    if args.composite_shards and (not args.matmul_only or args.vocab_shard_size
                                  or args.vocab_shard_offset or args.batch_repeats != 1):
        raise ValueError("composite-shards requires matmul-only/flatten-input and a full batch-1 head")
    if args.composite_with_norm and (args.matmul_only or args.flatten_input
                                     or args.vocab_shard_size or args.vocab_shard_offset
                                     or args.batch_repeats != 1):
        raise ValueError("composite-with-norm requires the original full head and batch-1 input")
    input_name = (f"normalized_x_offset_{args.vocab_shard_offset}"
                  if args.matmul_only else "x")

    prepared_model = args.model
    if args.composite_with_norm:
        from onnx import TensorProto, helper, numpy_helper

        prepared_model = args.output_dir / "spark_lm_head_norm_4shards.onnx"
        if not prepared_model.exists():
            original = onnx.load(str(args.model))
            if [node.op_type for node in original.graph.node[9:11]] != ["Mul", "Mul"]:
                raise ValueError("Unexpected original RMS-norm graph")
            weight = numpy_helper.to_array(next(
                item for item in original.graph.initializer if item.name == "onnx::MatMul_24"
            ))
            nodes = list(original.graph.node[:11])
            nodes.append(helper.make_node(
                "Reshape", ["/Mul_1_output_0", "flat_shape"], ["normalized_x_flat"],
                name="spark_lm_head_flatten_norm",
            ))
            initializers = [
                next(item for item in original.graph.initializer if item.name == "norm"),
                numpy_helper.from_array(np.array([1, 2048], dtype=np.int64), name="flat_shape"),
            ]
            outputs = []
            for offset in (0, 32768, 65536, 98304):
                output_name = f"logits_offset_{offset}"
                weight_name = f"weight_offset_{offset}"
                nodes.append(helper.make_node(
                    "MatMul", ["normalized_x_flat", weight_name], [output_name],
                    name=f"spark_lm_head_matmul_offset_{offset}",
                ))
                initializers.append(numpy_helper.from_array(
                    np.ascontiguousarray(weight[:, offset : offset + 32768]), name=weight_name
                ))
                outputs.append(output_name)
            nodes.append(helper.make_node(
                "Concat", outputs, ["logits_concatenated"], axis=1,
                name="spark_lm_head_concat",
            ))
            graph = helper.make_graph(
                nodes, "SparkRealWeightOutputHeadNormFourShards",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2048])],
                [helper.make_tensor_value_info("logits_concatenated", TensorProto.FLOAT,
                                               [1, 131072])],
                initializer=initializers,
            )
            prepared = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
            prepared.ir_version = min(prepared.ir_version, 10)
            onnx.checker.check_model(prepared)
            onnx.save(prepared, str(prepared_model))
            del original, weight
    elif args.matmul_only:
        from onnx import TensorProto, helper, numpy_helper

        prepared_model = args.output_dir / (
            "spark_lm_head_matmul_4shards.onnx" if args.composite_shards else
            f"spark_lm_head_matmul_offset{args.vocab_shard_offset}"
            f"_size{args.vocab_shard_size or 131072}"
            f"_batch{args.batch_repeats}.onnx"
        )
        model = onnx.load(str(args.model))
        norm = numpy_helper.to_array(next(item for item in model.graph.initializer if item.name == "norm"))
        weight = numpy_helper.to_array(next(
            item for item in model.graph.initializer if item.name == "onnx::MatMul_24"
        ))
        inputs = inputs.reshape(len(inputs), 1, 2048)
        inputs = inputs * (1.0 / np.sqrt(np.mean(inputs**2, axis=-1, keepdims=True) + np.float32(1e-6)))
        inputs = np.ascontiguousarray(inputs * norm)
        inputs = np.repeat(inputs, args.batch_repeats, axis=1)
        references = np.repeat(references, args.batch_repeats, axis=1)
        if args.vocab_shard_size:
            end = args.vocab_shard_offset + args.vocab_shard_size
            references = references[..., args.vocab_shard_offset : end]
            weight = weight[:, args.vocab_shard_offset : end]
        if not prepared_model.exists():
            offsets = (0, 32768, 65536, 98304) if args.composite_shards else (args.vocab_shard_offset,)
            nodes = []
            initializers = []
            outputs = []
            for offset in offsets:
                output_name = f"logits_offset_{offset}"
                weight_name = f"weight_offset_{offset}"
                shard_weight = (weight[:, offset : offset + 32768]
                                if args.composite_shards else weight)
                nodes.append(helper.make_node(
                    "MatMul", [input_name, weight_name], [output_name],
                    name=f"spark_lm_head_matmul_offset_{offset}",
                ))
                initializers.append(numpy_helper.from_array(
                    np.ascontiguousarray(shard_weight), name=weight_name
                ))
                outputs.append(output_name)
            final_output = outputs[0]
            if args.composite_shards:
                final_output = "logits_concatenated"
                nodes.append(helper.make_node("Concat", outputs, [final_output], axis=1,
                                              name="spark_lm_head_concat"))
            graph = helper.make_graph(
                nodes,
                "SparkRealWeightOutputHeadFourShards" if args.composite_shards else
                f"SparkRealWeightOutputHeadMatMulOffset{args.vocab_shard_offset}",
                [helper.make_tensor_value_info(input_name, TensorProto.FLOAT,
                                               [args.batch_repeats, 2048])],
                [helper.make_tensor_value_info(final_output, TensorProto.FLOAT,
                                               [args.batch_repeats, references.shape[-1]])],
                initializer=initializers,
            )
            prepared = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
            prepared.ir_version = min(prepared.ir_version, 10)
            onnx.checker.check_model(prepared)
            onnx.save(prepared, str(prepared_model))
        del model, weight
    elif args.vocab_shard_size or args.flatten_input:
        from onnx import numpy_helper

        suffix = (f"offset{args.vocab_shard_offset}_size{args.vocab_shard_size}"
                  if args.vocab_shard_size else "full")
        if args.flatten_input:
            suffix += f"_flat_batch{args.batch_repeats}"
            inputs = inputs.reshape(len(inputs), 1, 2048)
            inputs = np.repeat(inputs, args.batch_repeats, axis=1)
            references = np.repeat(references, args.batch_repeats, axis=1)
        prepared_model = args.output_dir / f"spark_lm_head_{suffix}.onnx"
        if args.vocab_shard_size:
            references = references[..., args.vocab_shard_offset : args.vocab_shard_offset + args.vocab_shard_size]
        if not prepared_model.exists():
            model = onnx.load(str(args.model))
            if args.vocab_shard_size:
                weight = next(item for item in model.graph.initializer if item.name == "onnx::MatMul_24")
                matrix = numpy_helper.to_array(weight)
                if matrix.shape != (2048, 131072):
                    raise ValueError(f"Unexpected output-head layout: {matrix.shape}")
                weight.CopyFrom(numpy_helper.from_array(
                    np.ascontiguousarray(matrix[:, args.vocab_shard_offset :
                                            args.vocab_shard_offset + args.vocab_shard_size]),
                    name=weight.name
                ))
                model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value = args.vocab_shard_size
            if args.flatten_input:
                if [node.op_type for node in model.graph.node[-2:]] != ["Constant", "Gather"]:
                    raise ValueError("Expected final singleton-axis Gather")
                del model.graph.input[0].type.tensor_type.shape.dim[1]
                model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = args.batch_repeats
                model.graph.output[0].name = model.graph.node[-3].output[0]
                model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = args.batch_repeats
                del model.graph.node[-2:]
            onnx.checker.check_model(model)
            onnx.save(model, str(prepared_model))
            del model

    source_session = ort.InferenceSession(str(prepared_model), providers=["CPUExecutionProvider"])
    source_rows = []
    for index in range(len(inputs)):
        got = source_session.run(None, {input_name: inputs[index]})[0]
        source_rows.append(compare(references[index], got))
    if not all(row["top1_match"] and row["snr_db"] > 100 for row in source_rows):
        raise RuntimeError("Source ONNX graph failed real-activation reference check")

    opset21 = args.output_dir / "spark_lm_head_opset21.onnx"
    quantized = args.output_dir / "spark_lm_head_a16w8_qdq.onnx"
    if not opset21.exists():
        model = onnx.load(str(prepared_model))
        if [(item.domain, item.version) for item in model.opset_import] == [("", 18)]:
            model = onnx.version_converter.convert_version(model, 21)
        elif [(item.domain, item.version) for item in model.opset_import] != [("", 21)]:
            raise ValueError("Expected the recorded opset-18 head or prepared opset-21 MatMul")
        onnx.checker.check_model(model)
        onnx.save(model, str(opset21))
        del model

    class RealActivationReader(CalibrationDataReader):
        def __init__(self) -> None:
            self.index = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self.index == len(inputs):
                return None
            sample = {input_name: inputs[self.index]}
            self.index += 1
            return sample

        def rewind(self) -> None:
            self.index = 0

    if not quantized.exists():
        quantize_static(
            str(opset21),
            str(quantized),
            RealActivationReader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt16 if args.signed_activations else QuantType.QUInt16,
            weight_type=QuantType.QInt8,
            per_channel=False,
            op_types_to_quantize=["MatMul"],
            calibrate_method=CalibrationMethod.MinMax,
        )

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
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {})
    options.enable_profiling = True
    options.profile_file_prefix = str(args.output_dir / "spark_lm_head_npu_profile")
    started = time.perf_counter()
    session = ort.InferenceSession(str(quantized), sess_options=options)
    load_s = time.perf_counter() - started
    rows = []
    for index in range(len(inputs)):
        started = time.perf_counter()
        result = session.run(None, {input_name: inputs[index]})[0]
        row = compare(references[index], result)
        row.update(index=index, wall_s=time.perf_counter() - started)
        rows.append(row)
    profile_path = Path(session.end_profiling())
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    node_providers = [
        (event.get("args") or {}).get("provider")
        for event in profile if event.get("cat") == "Node"
    ]
    report = {
        "scope": "Spark-X2.5-1.7B output head only; no prefill/decode/KV/sampling or full generation",
        "vocab_shard_size": args.vocab_shard_size or 131072,
        "vocab_shard_offset": args.vocab_shard_offset,
        "flatten_input": args.flatten_input,
        "batch_repeats": args.batch_repeats,
        "matmul_only": args.matmul_only,
        "composite_shards": args.composite_shards,
        "composite_with_norm": args.composite_with_norm,
        "hardware": args.hardware_label,
        "os": platform.platform(),
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "source_model_sha256": sha256(args.model),
        "prepared_model_sha256": sha256(prepared_model),
        "calibration_sha256": sha256(args.calibration),
        "opset21_sha256": sha256(opset21),
        "quantized_sha256": sha256(quantized),
        "quantization": (
            "opset 21, MatMul only, QDQ, "
            f"{'QInt16' if args.signed_activations else 'QUInt16'} activations/QInt8 weights, "
            "MinMax on four real activations"
        ),
        "input_shape": list(inputs.shape[1:]),
        "output_shape": list(references.shape[1:]),
        "source_cpu_reference": source_rows,
        "npu_devices": len(devices),
        "session_providers": session.get_providers(),
        "npu_node_events": node_providers.count("vitisai"),
        "cpu_node_events": node_providers.count("CPUExecutionProvider"),
        "load_s": load_s,
        "npu_rows": rows,
        "profile_file": str(profile_path),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["npu_node_events"] == 0:
        raise RuntimeError("Graph returned output but no node executed on the NPU")
    if not all(row["top1_match"] and row["finite"] for row in rows):
        raise RuntimeError("NPU output failed finite/top-1 parity on real activations")


if __name__ == "__main__":
    main()
