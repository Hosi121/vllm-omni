# SPDX-License-Identifier: Apache-2.0
"""The complete-policy stage must reject malformed observations before dispatch."""

from __future__ import annotations

import io

import numpy as np
import pytest

from vllm_omni.engine.backends.internvla import InternVLAStageClient


def _prompt():
    return {
        **{f"image{i}": np.zeros((1, 2, 3, 224, 224), np.float32) for i in range(3)},
        **{f"mask{i}": np.ones((1,), np.bool_) for i in range(3)},
        "state": np.zeros((1, 32), np.float32),
        "noise": np.zeros((1, 50, 32), np.float32),
        "task": "Place the marker pen in its holder.",
        "observation_timestamp_ns": 1_000_000_000,
    }


def test_action_observation_round_trip():
    raw = InternVLAStageClient._encode_prompt("request-1", _prompt())
    assert len(raw) < 8 << 20
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        assert str(data["request_id"].item()) == "request-1"
        assert data["image0"].shape == (1, 2, 3, 224, 224)
        assert data["mask0"].dtype == np.bool_
        assert int(data["observation_timestamp_ns"].item()) == 1_000_000_000


@pytest.mark.parametrize("field,value", [
    ("image0", np.full((1, 2, 3, 224, 224), np.nan, np.float32)),
    ("image1", np.full((1, 2, 3, 224, 224), 1.1, np.float32)),
    ("mask2", np.ones((1,), np.uint8)),
    ("state", np.zeros((1, 31), np.float32)),
    ("noise", np.full((1, 50, 32), np.inf, np.float32)),
    ("task", "x" * 4097),
    ("observation_timestamp_ns", -1),
])
def test_invalid_observation_rejected(field, value):
    prompt = _prompt()
    prompt[field] = value
    with pytest.raises(ValueError):
        InternVLAStageClient._encode_prompt("request-1", prompt)


def test_extra_observation_field_rejected():
    prompt = _prompt()
    prompt["undocumented"] = 1
    with pytest.raises(ValueError):
        InternVLAStageClient._encode_prompt("request-1", prompt)
