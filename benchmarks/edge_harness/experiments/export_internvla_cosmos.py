#!/usr/bin/env python3
"""Export the unchanged InternVLA Cosmos encoder as a fixed-shape graph."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()

    import torch

    source_dir = Path(__file__).resolve().parents[3] / "vllm_omni" / "diffusion" / "models" / "internvla_a1"
    package = types.ModuleType("internvla_cosmos_export")
    package.__path__ = [str(source_dir)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.model_cosmos", source_dir / "model_cosmos.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned Cosmos source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    encoder_path = args.encoder.resolve(strict=True)
    encoder_model = module.load_cosmos_component(encoder_path, component="encoder", device="cpu")

    class LatentOnly(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            result = self.inner(pixels)
            return result if isinstance(result, torch.Tensor) else result[0]

    model = LatentOnly(encoder_model).eval()
    image = torch.linspace(-1, 1, args.batch_size * 3 * 256 * 256, dtype=torch.float32).view(
        args.batch_size, 3, 256, 256
    )
    started = time.perf_counter()
    with torch.no_grad():
        reference = model(image)
        exported = torch.export.export(model, (image,), strict=False)
        actual = exported.module()(image)
    export_s = time.perf_counter() - started
    reference_tensor = reference
    actual_tensor = actual
    if not torch.allclose(reference_tensor, actual_tensor, atol=1e-5, rtol=1e-5):
        raise RuntimeError("exported CPU graph does not match source module")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, args.output)
    report = {
        "status": "passed",
        "scope": "fixed-shape Cosmos CI8x8 encoder graph; no InternVLA action-policy claim",
        "encoder_sha256": sha256(encoder_path),
        "source_sha256": {name: sha256(source_dir / name) for name in ("model_cosmos.py", "cosmos_ci_torch.py")},
        "torch": torch.__version__,
        "input_shape": list(image.shape),
        "input_dtype": str(image.dtype),
        "output_shape": list(reference_tensor.shape),
        "max_abs_export_error": float((reference_tensor - actual_tensor).abs().max()),
        "export_s": export_s,
        "artifact_sha256": sha256(args.output),
        "artifact_bytes": args.output.stat().st_size,
        "artifact_path": str(args.output.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
