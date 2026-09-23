#!/usr/bin/env python3
"""Real-checkpoint InternVLA action smoke with synthetic observations.

This probes loading and one full policy forward. Synthetic images and state do
not establish robot-task quality, physical units, or control safety.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cosmos-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actions-output", type=Path, help="Optional float32 .npy actions for reference comparison")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ["INTERNVLA_A1_COSMOS_DIR"] = str(args.cosmos_dir.resolve())
    os.environ["INTERNVLA_A1_PROCESSOR_DIR"] = str(args.processor_dir.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"

    import torch

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.registry import initialize_model
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    model_dir = args.model_dir.resolve()
    if not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing real InternVLA checkpoint: {model_dir / 'model.safetensors'}")

    config = OmniDiffusionConfig(
        model=str(model_dir),
        model_class_name="InternVLAA1Pipeline",
        dtype=torch.bfloat16,
        custom_pipeline_args={
            "device": args.device,
            "dtype": "bfloat16",
            "compile_model": False,
            "enable_regional_compile": False,
            "enable_warmup": False,
            "strict_load": True,
            "processor_model_name": str(args.processor_dir.resolve()),
        },
    )
    started = time.perf_counter()
    pipeline = initialize_model(config)
    if pipeline.runtime_mode() != "real_checkpoint_loaded":
        raise RuntimeError(f"Unexpected InternVLA runtime mode: {pipeline.runtime_mode()}")
    policy_devices = sorted({parameter.device.type for parameter in pipeline.policy.parameters()})
    requested_device_type = torch.device(args.device).type
    if policy_devices != [requested_device_type]:
        raise RuntimeError(f"InternVLA policy ran on {policy_devices}, not {requested_device_type}")
    startup_s = time.perf_counter() - started

    batch_inputs = pipeline._build_fake_batch_inputs()
    noise = torch.zeros(
        (1, pipeline.config.chunk_size, pipeline.config.max_action_dim),
        device=args.device,
        dtype=torch.float32,
    )
    if args.device.startswith("cuda"):
        torch.accelerator.synchronize()
        torch.accelerator.reset_peak_memory_stats()
    started = time.perf_counter()
    output = pipeline.forward(
        DiffusionRequestBatch(
            requests=[
                OmniDiffusionRequest(
                    prompt="",
                    sampling_params=OmniDiffusionSamplingParams(
                        extra_args={"batch_inputs": batch_inputs, "noise": noise, "decode_image": False}
                    ),
                    request_id="internvla-real-weight-synthetic-observation",
                )
            ]
        )
    )
    if args.device.startswith("cuda"):
        torch.accelerator.synchronize()
    forward_s = time.perf_counter() - started
    if output.error:
        raise RuntimeError(output.error)
    result = output.output
    actions = result.get("payload", {}).get("actions") if isinstance(result, dict) else None
    if not isinstance(actions, torch.Tensor) or actions.shape != (
        1,
        pipeline.config.chunk_size,
        pipeline.config.max_action_dim,
    ):
        raise RuntimeError(f"Unexpected action output: {type(actions).__name__}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("InternVLA returned non-finite actions")
    if actions.device.type != requested_device_type:
        raise RuntimeError(f"InternVLA actions were on {actions.device}, not {requested_device_type}")

    action_array = actions.detach().float().cpu().contiguous().numpy()
    action_bytes = action_array.tobytes()
    if args.actions_output is not None:
        import numpy as np

        args.actions_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.actions_output, action_array)
    report = {
        "scope": "real checkpoint, synthetic zero observations and noise; no task-quality claim",
        "model_dir": str(model_dir),
        "runtime_mode": pipeline.runtime_mode(),
        "device": args.device,
        "policy_devices": policy_devices,
        "actions_device": str(actions.device),
        "dtype": str(actions.dtype),
        "action_shape": list(actions.shape),
        "finite": True,
        "action_sha256_float32": hashlib.sha256(action_bytes).hexdigest(),
        "first_action_prefix": actions[0, 0, :8].detach().float().cpu().tolist(),
        "startup_s": startup_s,
        "forward_s": forward_s,
        "cuda_peak_allocated_bytes": (
            torch.accelerator.max_memory_allocated() if args.device.startswith("cuda") else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
