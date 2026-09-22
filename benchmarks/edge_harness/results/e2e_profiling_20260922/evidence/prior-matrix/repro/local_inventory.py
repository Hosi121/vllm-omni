"""Read-only local artifact/runtime preflight; never claims inference success."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--workspace", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
record = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "platform": platform.platform(),
    "python": sys.executable,
    "evidence": "preflight",
    "packages": {},
    "search_roots": [],
    "artifacts": [],
}
for name in ("vllm", "torch", "transformers", "onnxruntime", "onnxruntime-directml"):
    try:
        record["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        record["packages"][name] = None
roots = [a.workspace / "models", Path.home() / ".cache/huggingface/hub"]
if sys.platform == "win32":
    roots += [Path.home() / "w2/models"]
else:
    roots += [Path("/mnt/c/Users/zhout/w2/models")]
for root in roots:
    record["search_roots"].append({"path": str(root), "exists": root.is_dir()})
    if not root.is_dir():
        continue
    for item in root.iterdir():
        if not item.is_dir() or not any(x in item.name.lower() for x in ("spark", "qwen3", "minicpm", "internvla")):
            continue
        configs = list(item.glob("config.json")) + list(item.glob("snapshots/*/config.json"))
        for config in configs:
            try:
                data = json.loads(config.read_text(encoding="utf-8"))
                weights = list(config.parent.glob("*.safetensors"))
                record["artifacts"].append(
                    {
                        "directory": str(config.parent),
                        "name": item.name,
                        "model_type": data.get("model_type"),
                        "architectures": data.get("architectures"),
                        "weight_shards": len(weights),
                        "weight_bytes": sum(f.stat().st_size for f in weights),
                    }
                )
            except OSError as error:
                record["artifacts"].append({"directory": str(config.parent), "error": str(error)})
try:
    from vllm.platforms import current_platform

    record["vllm_platform"] = current_platform.device_type
    from vllm_omni.config.pipeline_registry import resolve_pipeline_config

    record["registered_pipelines"] = {
        kind: resolve_pipeline_config(kind) is not None for kind in ("spark2_5", "qwen3_tts", "qwen3_5", "minicpmo_4_5")
    }
except Exception as error:
    record["runtime_import_error"] = repr(error)
for command in (
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
    ["adb", "devices", "-l"],
):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        record[command[0]] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        record[command[0]] = {"error": str(error)}
a.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(f"Inventoried {len(record['artifacts'])} artifact directories; report: {a.out}")
