#!/usr/bin/env python3
"""Export the real fixed-shape InternVLA Cosmos encoder for an NPU placement probe.

The export and CPU numerical gate are component evidence only. This script does
not claim VitisAI placement or a complete InternVLA action-policy request.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
import types
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    encoder = args.encoder.resolve(strict=True)
    source_dir = Path(__file__).resolve().parents[3] / "vllm_omni" / "diffusion" / "models" / "internvla_a1"
    package = types.ModuleType("internvla_cosmos_onnx_export")
    package.__path__ = [str(source_dir)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.model_cosmos", source_dir / "model_cosmos.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned Cosmos source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class LatentOnly(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            result = self.inner(pixels)
            return result if isinstance(result, torch.Tensor) else result[0]

    model = LatentOnly(module.load_cosmos_component(encoder, component="encoder", device="cpu")).eval()
    fixtures = {
        "ramp": torch.linspace(-1, 1, 6 * 3 * 256 * 256, dtype=torch.float32).view(6, 3, 256, 256),
        "pattern": torch.zeros((6, 3, 256, 256), dtype=torch.float32),
    }
    fixtures["pattern"][:, 0, 64:192, 64:192] = 0.75
    fixtures["pattern"][:, 1, 32:96, 32:96] = -0.5
    report = {
        "scope": "real InternVLA Cosmos CI8x8 encoder; fixed [6,3,256,256] FP32; component only",
        "encoder_sha256": sha256(encoder),
        "source_sha256": {name: sha256(source_dir / name) for name in ("model_cosmos.py", "cosmos_ci_torch.py")},
        "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
        "input_shape": [6, 3, 256, 256], "input_dtype": "float32",
        "output_shape": [6, 16, 32, 32], "opset": 21,
    }
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        torch.onnx.export(
            model, (fixtures["ramp"],), args.output,
            input_names=["pixels"], output_names=["latent"], opset_version=21,
            dynamo=True, external_data=True, optimize=False, verify=False,
        )
        report["export_s"] = time.perf_counter() - started
        graph = onnx.load(str(args.output), load_external_data=True)
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        if [item.name for item in session.get_inputs()] != ["pixels"]:
            raise RuntimeError("ONNX input name differs from fixed contract")
        report["cpu_parity"] = {}
        fixture_arrays = {}
        with torch.no_grad():
            for name, pixels in fixtures.items():
                source = model(pixels).detach().float().cpu().numpy()
                actual = session.run(["latent"], {"pixels": pixels.numpy()})[0]
                if source.shape != (6, 16, 32, 32) or actual.shape != source.shape:
                    raise RuntimeError("Cosmos latent shape differs from fixed contract")
                if not np.isfinite(actual).all():
                    raise RuntimeError("ONNX Cosmos latent contains nonfinite values")
                diff = actual - source
                report["cpu_parity"][name] = {
                    "max_abs": float(np.max(np.abs(diff))),
                    "relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(source)),
                }
                fixture_arrays[f"{name}_pixels"] = pixels.numpy()
                fixture_arrays[f"{name}_reference"] = source
        args.fixtures.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.fixtures, **fixture_arrays)
        report["fixtures"] = {"path": str(args.fixtures.resolve()),
                              "sha256": sha256(args.fixtures), "bytes": args.fixtures.stat().st_size}
        data_files = sorted(args.output.parent.glob(args.output.name + ".data*"))
        report["artifact"] = {
            "onnx": {"path": str(args.output.resolve()), "sha256": sha256(args.output),
                     "bytes": args.output.stat().st_size},
            "external_data": [
                {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
                for path in data_files
            ],
        }
        report["node_op_types"] = sorted({node.op_type for node in graph.graph.node})
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "traceback"}, indent=2))


if __name__ == "__main__":
    main()
