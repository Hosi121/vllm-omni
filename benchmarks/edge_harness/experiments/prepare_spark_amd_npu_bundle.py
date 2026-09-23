#!/usr/bin/env python3
"""Package the qualified Spark output-head graph for Omni's graph stage.

The model and quantized graph stay in local storage; the manifest records their
hashes and refuses a graph that did not pass the component placement gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    args = parser.parse_args()

    report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    if not report.get("composite_with_norm") or report["npu_node_events"] < 4:
        raise ValueError("The graph is not a placement-verified complete output head")
    if not all(row["finite"] and row["top1_match"] and row["snr_db"] > 40
               for row in report["npu_rows"]):
        raise ValueError("The output head failed real-activation numerical checks")
    if digest(args.graph) != report["quantized_sha256"]:
        raise ValueError("Graph hash differs from the validated artifact")
    if digest(args.activations) != report["calibration_sha256"]:
        raise ValueError("Calibration/reference hash differs from the validated inputs")

    with np.load(args.activations, allow_pickle=False) as data:
        x = data["x"][0]
    if x.shape != (1, 1, 2048) or x.dtype != np.float32:
        raise ValueError(f"Unexpected real-activation shape or dtype: {x.shape}, {x.dtype}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph = args.output_dir / "graph.onnx"
    examples = args.output_dir / "inputs.npz"
    validation = args.output_dir / "validation_report.json"
    shutil.copy2(args.graph, graph)
    shutil.copy2(args.validation_report, validation)
    np.savez(examples, x=x)
    files = {path.name: digest(path) for path in (graph, examples, validation)}
    manifest = {
        "schema_version": 1,
        "component": "spark_x2_5_1_7b_output_head_amd_npu",
        "files": files,
        "metadata": {
            "graph_file": graph.name,
            "example_inputs_file": examples.name,
            "checkpoint_revision": args.checkpoint_revision,
            "precision": "A16W8 QDQ, FP32 I/O",
            "layout": "[1,1,2048] pre-final-norm input -> [1,131072] logits",
            "exporter": "probe_spark_amd_npu_lm_head.py --composite-with-norm",
            "calibration": "four captured real Spark pre-final-norm activations",
            "validation": "validation_report.json; four top-1 matches, SNR >40 dB; component only",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"bundle": str(args.output_dir), "files": files}, indent=2))


if __name__ == "__main__":
    main()
