#!/usr/bin/env python3
"""Compare complete InternVLA actions on CPU with a Radeon Cosmos handoff.

The policy and action sampler remain on CPU. Only its Cosmos encoder is
replaced by Omni's existing external torch-DirectML graph worker. The test
uses synthetic observations and does not establish robot-task quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cosmos-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats must be positive")

    os.environ["INTERNVLA_A1_COSMOS_DIR"] = str(args.cosmos_dir.resolve(strict=True))
    os.environ["INTERNVLA_A1_PROCESSOR_DIR"] = str(args.processor_dir.resolve(strict=True))
    os.environ["HF_HUB_OFFLINE"] = "1"

    import numpy as np
    import psutil
    import torch

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.registry import initialize_model
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.edge.local.external.client import ExternalWorker
    from vllm_omni.edge.local.external.launch import ROUTE_TORCH_DML, resolve
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    graph = args.graph.resolve(strict=True)
    export_report = json.loads(args.export_report.read_text(encoding="utf-8"))
    if export_report["artifact_sha256"] != sha256(graph):
        raise RuntimeError("exported graph hash differs from the pinned manifest")
    if export_report["input_shape"] != [6, 3, 256, 256] or export_report["input_dtype"] != "torch.float32":
        raise RuntimeError("wrong Cosmos export shape or precision")
    model_dir = args.model_dir.resolve(strict=True)
    checkpoint = model_dir / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    report = {
        "scope": "InternVLA real Place_Markpen checkpoint, synthetic zero camera/state observations and zero diffusion noise; CPU BF16 policy with FP32 Radeon Cosmos encoder; no robot-task quality claim",
        "model_checkpoint_sha256": sha256(checkpoint),
        "graph_sha256": export_report["artifact_sha256"],
        "graph_export": export_report,
        "policy_torch": torch.__version__,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "status": "running",
    }
    worker = None
    try:
        config = OmniDiffusionConfig(
            model=str(model_dir),
            model_class_name="InternVLAA1Pipeline",
            dtype=torch.bfloat16,
            custom_pipeline_args={
                "device": "cpu", "dtype": "bfloat16", "compile_model": False,
                "enable_regional_compile": False, "enable_warmup": False,
                "strict_load": True, "processor_model_name": str(args.processor_dir.resolve()),
            },
        )
        started = time.perf_counter()
        pipeline = initialize_model(config)
        report["policy_load_s"] = time.perf_counter() - started
        if pipeline.runtime_mode() != "real_checkpoint_loaded":
            raise RuntimeError(f"unexpected policy mode: {pipeline.runtime_mode()}")
        if {p.device.type for p in pipeline.policy.parameters()} != {"cpu"}:
            raise RuntimeError("policy did not load wholly on CPU")
        report["policy_parameters_device"] = "cpu"
        inputs = pipeline._build_fake_batch_inputs()
        noise = torch.zeros((1, pipeline.config.chunk_size, pipeline.config.max_action_dim), dtype=torch.float32)

        route = resolve(ROUTE_TORCH_DML)
        if not route.available:
            raise RuntimeError(f"DirectML route unavailable: {route.reason}")
        worker = ExternalWorker(route)
        started = time.perf_counter()
        hello = worker.start()
        load_report = worker.load(graph, example_inputs={"pixels": np.zeros((6, 3, 256, 256), dtype=np.float32)})
        report["worker_start_and_graph_load_s"] = time.perf_counter() - started
        report["worker_hello"] = {key: hello.get(key) for key in ("torch", "torch_directml", "available_providers", "executable")}
        report["worker_load"] = load_report.to_dict()
        if (load_report.device_name != "AMD Radeon(TM) 890M Graphics" or
            load_report.fraction_on_target != 1.0 or
            load_report.outputs[0]["shape"] != [6, 16, 32, 32]):
            raise RuntimeError("Radeon placement or output shape not verified")

        class ExternalCosmosEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.timings = []

            def forward(self, pixels: torch.Tensor) -> torch.Tensor:
                if pixels.device.type != "cpu" or tuple(pixels.shape) != (6, 3, 256, 256):
                    raise RuntimeError(f"unexpected encoder input: {pixels.device}, {tuple(pixels.shape)}")
                # DirectML BF16 aborts this runtime; the FP32 graph is an
                # explicit artifact change, then latents return as policy BF16.
                array = pixels.detach().float().contiguous().numpy()
                output, timing = worker.run({"pixels": array})
                self.calls += 1
                self.timings.append(timing.to_dict())
                latent = output["out_0"]
                if latent.shape != (6, 16, 32, 32) or not np.isfinite(latent).all():
                    raise RuntimeError("bad DirectML Cosmos latent")
                return torch.from_numpy(latent).to(pixels.dtype)

        external_encoder = ExternalCosmosEncoder()
        cosmos = pipeline.policy.model.cosmos
        cpu_encoder = cosmos._enc_model

        def actions(encoder: torch.nn.Module) -> np.ndarray:
            cosmos._enc_model = encoder
            result = pipeline.forward(DiffusionRequestBatch(requests=[
                OmniDiffusionRequest(
                    prompt="",
                    sampling_params=OmniDiffusionSamplingParams(extra_args={
                        "batch_inputs": inputs, "noise": noise, "decode_image": False,
                    }),
                    request_id="internvla-hybrid-synthetic-observation",
                )
            ]))
            if result.error:
                raise RuntimeError(result.error)
            tensor = result.output["payload"]["actions"]
            if tensor.device.type != "cpu" or tuple(tensor.shape) != (1, 50, 32):
                raise RuntimeError(f"bad policy action shape/device: {tensor.shape} {tensor.device}")
            if not torch.isfinite(tensor).all():
                raise RuntimeError("policy actions nonfinite")
            return tensor.detach().float().cpu().contiguous().numpy()

        cpu_wall = []
        hybrid_wall = []
        errors = []
        reference = actual = None
        for index in range(args.warmups + args.repeats):
            started = time.perf_counter()
            reference = actions(cpu_encoder)
            cpu_s = time.perf_counter() - started
            started = time.perf_counter()
            actual = actions(external_encoder)
            hybrid_s = time.perf_counter() - started
            if index >= args.warmups:
                cpu_wall.append(cpu_s)
                hybrid_wall.append(hybrid_s)
                errors.append({
                    "max_abs": float(np.max(np.abs(reference - actual))),
                    "relative_l2": float(np.linalg.norm(reference - actual) / np.linalg.norm(reference)),
                    "cosine": float(np.dot(reference.ravel(), actual.ravel()) / (
                        np.linalg.norm(reference) * np.linalg.norm(actual)
                    )),
                })
        assert reference is not None and actual is not None
        cosmos._enc_model = cpu_encoder
        report.update({
            "status": "passed",
            "action_shape": list(actual.shape),
            "cpu_action_sha256_last": hashlib.sha256(reference.tobytes()).hexdigest(),
            "hybrid_action_sha256_last": hashlib.sha256(actual.tobytes()).hexdigest(),
            "cpu_wall_s": cpu_wall,
            "hybrid_wall_s": hybrid_wall,
            "per_request_action_error": errors,
            "max_action_abs_error": max(e["max_abs"] for e in errors),
            "max_action_relative_l2_error": max(e["relative_l2"] for e in errors),
            "min_action_cosine": min(e["cosine"] for e in errors),
            "cpu_p50_wall_s": rank(cpu_wall, .5),
            "cpu_p95_wall_s": rank(cpu_wall, .95),
            "hybrid_p50_wall_s": rank(hybrid_wall, .5),
            "hybrid_p95_wall_s": rank(hybrid_wall, .95),
            "external_encoder_calls": external_encoder.calls,
            "external_encoder_timings": external_encoder.timings,
            "worker_stats": worker.stats(),
            "parent_rss_bytes_after": psutil.Process().memory_info().rss,
            "wsl_available_bytes_after": psutil.virtual_memory().available,
        })
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if worker is not None:
            worker.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key not in (
            "cpu_wall_s", "hybrid_wall_s", "per_request_action_error", "external_encoder_timings"
        )}, indent=2))


if __name__ == "__main__":
    main()
