#!/usr/bin/env python3
"""Run the MiniCPM-o thinker alone to isolate text quality from speech stages.

This temporarily narrows the registered pipeline in this process. It is a
diagnostic route, not a replacement for the shipped three-stage topology.
"""

from __future__ import annotations

from dataclasses import replace

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE


def main() -> None:
    OMNI_PIPELINES["minicpmo_4_5"] = replace(
        MINICPMO_4_5_PIPELINE,
        stages=(MINICPMO_4_5_PIPELINE.stages[0],),
        duplex_runtime_extension=None,
        duplex_serving_adapter=None,
        duplex_control_enabled=False,
    )
    from examples.offline_inference.minicpmo.end2end import main as run_example
    from examples.offline_inference.minicpmo.end2end import parse_args

    run_example(parse_args())


if __name__ == "__main__":
    main()
