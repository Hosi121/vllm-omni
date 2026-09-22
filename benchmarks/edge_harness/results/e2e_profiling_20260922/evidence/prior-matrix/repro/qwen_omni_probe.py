"""Bounded, real-checkpoint public Omni startup/text probe (no model substitution)."""

import argparse
import json
import os
import platform
import time
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    from vllm_omni.windows.aio import install_selector_policy

    install_selector_policy()
    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni

    config = dict(
        model=a.model,
        language_model_only=True,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        enforce_eager=True,
        gpu_memory_utilization=0.80,
        cpu_offload_gb=2,
        dtype="bfloat16",
        stage_init_timeout=120,
    )
    record = {
        "platform": platform.platform(),
        "config": config,
        "status": "starting",
        "scope": "public Omni, short text only; explicit 2 GiB native offload",
        "start": time.time(),
    }
    engine = None
    try:
        engine = Omni(**config)
        outputs = engine.generate(
            "Explain local inference in one sentence.", SamplingParams(temperature=0, max_tokens=16)
        )
        record["outputs"] = [str(output) for output in outputs]
        if not outputs:
            raise RuntimeError("No output")
        record["status"] = "completed"
    except Exception as error:
        record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if engine is not None:
            engine.close()
        record["end"] = time.time()
        a.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(record["status"], flush=True)


if __name__ == "__main__":
    main()
