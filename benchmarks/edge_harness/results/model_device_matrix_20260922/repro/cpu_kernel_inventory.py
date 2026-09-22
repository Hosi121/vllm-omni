"""Inspect CPU attention operator availability; this is not a model execution."""

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch
from vllm.platforms import current_platform

current_platform.import_kernels()

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
names = ("cpu_attn_reshape_and_cache", "cpu_attn_get_scheduler_metadata")
record = {
    "platform": platform.platform(),
    "vllm": importlib.metadata.version("vllm"),
    "torch": torch.__version__,
    "extensions": {
        name: getattr(module, "__file__", None) for name, module in sys.modules.items() if name.startswith("vllm._C")
    },
    "cpu_attention_operators": {name: hasattr(torch.ops._C, name) for name in names},
}
a.out.write_text(json.dumps(record, indent=2) + "\n")
print(record["cpu_attention_operators"])
