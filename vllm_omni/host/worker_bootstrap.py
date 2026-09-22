# SPDX-License-Identifier: Apache-2.0
"""Standalone worker bootstrap: record OS identity without importing Omni."""

import json
import os
import runpy
import sys
from pathlib import Path

if os.name == "nt":
    from windows_process import identity

    os.environ["VLLM_OMNI_WORKER_IDENTITY"] = json.dumps(identity())
else:
    os.environ.pop("VLLM_OMNI_WORKER_IDENTITY", None)

worker = sys.argv.pop(1)
sys.argv[0] = worker
sys.path.insert(0, str(Path(worker).resolve().parent))
runpy.run_path(worker, run_name="__main__")
