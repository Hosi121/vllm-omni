# SPDX-License-Identifier: Apache-2.0
"""Explicit whole-policy InternVLA CPU and CPU+Radeon deployment topology."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

INTERNVLA_A1_WHOLE_POLICY_PIPELINE = PipelineConfig(
    model_type="internvla_a1_external_whole",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="action",
            execution_type=StageExecutionType.GRAPH,
            final_output=True,
            final_output_type="action",
        ),
    ),
)
