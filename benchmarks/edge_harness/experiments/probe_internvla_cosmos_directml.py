#!/usr/bin/env python3
"""Probe InternVLA's real Cosmos image encoder on CPU and one DirectML GPU.

This loads the unchanged Omni model source as an isolated experiment because
the installed DirectML PyTorch environment is separate from the vLLM runtime.
It tests one model component, not the full action policy or robot-task quality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
import types
from pathlib import Path


def _rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_omni_cosmos_module(source_dir: Path):
    # Keep the model implementation unchanged without importing the unrelated
    # vLLM engine in a separate, older Torch/DirectML experiment environment.
    package_name = "internvla_cosmos_probe"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_dir)]
    sys.modules[package_name] = package
    name = f"{package_name}.model_cosmos"
    spec = importlib.util.spec_from_file_location(name, source_dir / "model_cosmos.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load unchanged InternVLA Cosmos source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dml-index", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("invalid warmup or measured count")

    import numpy as np
    import psutil
    import torch
    import torch_directml
    from safetensors import safe_open

    source_dir = Path(__file__).resolve().parents[3] / "vllm_omni" / "diffusion" / "models" / "internvla_a1"
    source = _load_omni_cosmos_module(source_dir)
    device_name = torch_directml.device_name(args.dml_index).rstrip("\x00")
    if device_name != "AMD Radeon(TM) 890M Graphics":
        raise RuntimeError(f"refusing unverified DirectML device: {device_name!r}")
    dml = torch_directml.device(args.dml_index)
    encoder = args.encoder.resolve(strict=True)
    report = {
        "scope": "InternVLA real Cosmos CI8x8 image encoder component; synthetic patterned 256x256 input; no full policy or robot-task claim",
        "source_files_sha256": {
            name: _sha256(source_dir / name)
            for name in ("model_cosmos.py", "cosmos_ci_torch.py")
        },
        "encoder_sha256": _sha256(encoder),
        "device_name": device_name,
        "torch": torch.__version__,
        "torch_directml": __import__("importlib.metadata", fromlist=["version"]).version("torch-directml"),
        "input_shape": [1, 3, 256, 256],
        "input_dtype": "float32",
        "warmups": args.warmups,
        "repeats": args.repeats,
    }
    try:
        torch.manual_seed(42)
        image = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        image[:, 0, 48:208, 48:208] = 1
        image[:, 1, 48:208, 48:208] = -1
        image[:, 2, :, :] = torch.linspace(-1, 1, 256)[None, None, :]

        started = time.perf_counter()
        cpu_model = source.load_cosmos_component(encoder, component="encoder", device="cpu")
        report["cpu_load_s"] = time.perf_counter() - started
        with safe_open(str(encoder), framework="pt", device="cpu") as checkpoint:
            checkpoint_keys = set(checkpoint.keys())
            checkpoint_buffers = {
                key: checkpoint.get_tensor(key)
                for key in checkpoint_keys
                if key in {"encoder.patcher._arange", "encoder.patcher.wavelets"}
            }
        model_keys = set(cpu_model.state_dict())
        report["checkpoint_tensor_count"] = len(checkpoint_keys)
        report["encoder_state_tensor_count"] = len(model_keys)
        report["checkpoint_missing_keys"] = sorted(model_keys - checkpoint_keys)
        report["checkpoint_unexpected_keys"] = sorted(checkpoint_keys - model_keys)
        model_buffers = dict(cpu_model.named_buffers())
        report["nonpersistent_buffer_comparison"] = {
            key: {
                "checkpoint_shape": list(value.shape),
                "model_shape": list(model_buffers[key].shape),
                "checkpoint_dtype": str(value.dtype),
                "model_dtype": str(model_buffers[key].dtype),
                "max_abs_error": float((value.float() - model_buffers[key].float()).abs().max()),
            }
            for key, value in checkpoint_buffers.items()
        }
        if (report["checkpoint_missing_keys"] or
            set(report["checkpoint_unexpected_keys"]) != set(checkpoint_buffers)):
            raise RuntimeError("Cosmos encoder checkpoint/state keys do not match")
        started = time.perf_counter()
        dml_model = source.load_cosmos_component(encoder, component="encoder", device="cpu").to(dml)
        report["dml_load_and_transfer_s"] = time.perf_counter() - started
        if {parameter.device.type for parameter in cpu_model.parameters()} != {"cpu"}:
            raise RuntimeError("CPU reference encoder placement differs")
        if {parameter.device.type for parameter in dml_model.parameters()} != {"privateuseone"}:
            raise RuntimeError("DirectML encoder weights were not on the declared GPU")

        def execute(model, input_tensor, expected_device_type):
            with torch.no_grad():
                output = model(input_tensor)
                if not isinstance(output, torch.Tensor):
                    output = output[0]
                if output.device.type != expected_device_type:
                    raise RuntimeError(f"encoder output on {output.device}, expected {expected_device_type}")
                return output.cpu().float().contiguous()

        cpu_times: list[float] = []
        dml_times: list[float] = []
        cpu_output = dml_output = None
        for index in range(args.warmups + args.repeats):
            started = time.perf_counter()
            cpu_output = execute(cpu_model, image, "cpu")
            cpu_wall = time.perf_counter() - started
            started = time.perf_counter()
            dml_output = execute(dml_model, image.to(dml), "privateuseone")
            dml_wall = time.perf_counter() - started
            if index >= args.warmups:
                cpu_times.append(cpu_wall)
                dml_times.append(dml_wall)
        assert cpu_output is not None and dml_output is not None
        if cpu_output.shape != dml_output.shape or not torch.isfinite(dml_output).all():
            raise RuntimeError("DirectML encoder returned wrong shape or non-finite output")
        reference = cpu_output.numpy()
        actual = dml_output.numpy()
        difference = reference - actual
        report.update({
            "status": "passed",
            "output_shape": list(reference.shape),
            "cpu_output_sha256_float32": hashlib.sha256(reference.tobytes()).hexdigest(),
            "dml_output_sha256_float32": hashlib.sha256(actual.tobytes()).hexdigest(),
            "max_abs_error": float(np.max(np.abs(difference))),
            "relative_l2_error": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
            "cosine_similarity": float(np.dot(reference.ravel(), actual.ravel()) / (
                np.linalg.norm(reference) * np.linalg.norm(actual)
            )),
            "cpu_nearest_rank_p50_wall_s": _rank(cpu_times, 0.5),
            "cpu_nearest_rank_p95_wall_s": _rank(cpu_times, 0.95),
            "dml_nearest_rank_p50_wall_s": _rank(dml_times, 0.5),
            "dml_nearest_rank_p95_wall_s": _rank(dml_times, 0.95),
            "cpu_times_s": cpu_times,
            "dml_times_s": dml_times,
            "process_rss_bytes_after": psutil.Process().memory_info().rss,
        })
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if not key.endswith("times_s")}, indent=2))


if __name__ == "__main__":
    main()
