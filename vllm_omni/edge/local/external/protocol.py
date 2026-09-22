# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the extracted host-copy wire implementation.

Standalone legacy worker scripts load this file directly. The source-tree
fallback loads only the portable package, never the vllm_omni initializer.
"""

import sys
from pathlib import Path

try:
    from omni_stage_contracts.wire import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "omni_stage_contracts":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from omni_stage_contracts.wire import *  # noqa: F403
