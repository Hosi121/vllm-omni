# SPDX-License-Identifier: Apache-2.0
"""Run a verified graph bundle through the public Omni API on one machine.

The bundle manifest identifies graph_file and example_inputs_file. This example
uses those fixed-bucket examples as inputs; real pipelines supply model-owned
input/handoff functions. All memory arguments are controller ceilings/estimates.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from omni_stage_contracts import ArtifactManifest
from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.entrypoints.omni import Omni


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--route", default="ort-cpu")
    parser.add_argument("--python", dest="interpreter")
    parser.add_argument("--windows-worker", action="store_true")
    parser.add_argument("--ep", default="cpu", choices=["cpu", "dml", "vitisai"])
    parser.add_argument("--allowed-provider", action="append", default=[])
    parser.add_argument("--min-fraction", type=float, default=1.0)
    parser.add_argument("--capacity-mib", type=int, default=1024)
    parser.add_argument("--stage-mib", type=int, default=512)
    parser.add_argument("--max-io-mib", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    manifest_path = args.manifest.resolve()
    manifest = ArtifactManifest.read(manifest_path)
    with np.load(manifest_path.parent / manifest.metadata["example_inputs_file"], allow_pickle=False) as samples:
        inputs = {name: samples[name] for name in samples.files}
    register_pipeline(
        PipelineConfig(
            model_type="omni_external_graph",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="graph",
                    execution_type=StageExecutionType.GRAPH,
                    final_output=True,
                    final_output_type="latent",
                ),
            ),
        )
    )
    route = args.route
    if args.interpreter:
        import vllm_omni.edge.local.external.worker_ort as worker_module

        route = {
            "name": args.route,
            "interpreter": args.interpreter,
            "worker": worker_module.__file__,
            "os_domain": "windows" if args.windows_worker else "posix",
            "ep": args.ep,
        }
    backend = {
        "name": "external.graph.v1",
        "route": route,
        "manifest": str(manifest_path),
        "min_fraction_on_target": args.min_fraction,
        "max_io_bytes": args.max_io_mib << 20,
        "profile_prefix": str(args.report.resolve().with_suffix(".placement")),
    }
    if args.allowed_provider:
        backend["allowed_providers"] = args.allowed_provider
    result = {
        "schema_version": 1,
        "evidence": "B",
        "platform": platform.platform(),
        "python": sys.executable,
        "workload": {"repeats": args.repeats, "concurrency": 1},
        "artifact": str(manifest_path),
        "samples_s": [],
        "started_unix": time.time(),
    }
    engine = None
    try:
        with tempfile.TemporaryDirectory(prefix="omni-graph-") as tmp:
            deploy = Path(tmp) / "deploy.yaml"
            deploy.write_text(
                yaml.safe_dump(
                    {
                        "pipeline": "omni_external_graph",
                        "async_chunk": False,
                        "stages": [
                            {
                                "stage_id": 0,
                                "backend": backend,
                                "resource_budget": {
                                    "capacities": {"machine:ram": args.capacity_mib << 20},
                                    "demands": {"machine:ram": args.stage_mib << 20},
                                },
                            }
                        ],
                    }
                )
            )
            started = time.perf_counter()
            engine = Omni(model=str(manifest_path.parent), deploy_config=str(deploy))
            result["startup_s"] = time.perf_counter() - started
            client = engine.engine.stage_pools[0].stage_client
            result["execution_plan"] = client.execution_plan
            outputs = None
            for _ in range(args.repeats):
                started = time.perf_counter()
                (output,) = engine.generate({"tensors": inputs}, use_tqdm=False)
                result["samples_s"].append(time.perf_counter() - started)
                outputs = output.custom_output["tensors"]
            result["worker_stats"] = client._worker.stats()
            result["finite"] = all(np.isfinite(v).all().item() for v in outputs.values())
            args.report.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.report.with_suffix(".npz"), **outputs)
            result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if engine is not None:
            engine.shutdown()
            result["ledger_after_shutdown"] = engine.engine._runtime.resource_ledger.snapshot()
        result["ended_unix"] = time.time()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
