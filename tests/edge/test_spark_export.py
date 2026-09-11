# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the Spark-X2.5 decode-step exporter.

The point of the export is the hybrid cache layout, so that is what these
check: sliding layers stay at a fixed 512 entries however long the context
gets, full layers grow, and the window actually rolls.
"""

import pytest
import torch

from vllm_omni.edge.spark_export import (
    FULL,
    SLIDING,
    SparkDecodeStep,
    SparkStepConfig,
    config_from_spark,
    example_inputs,
    input_names,
    kv_cache_bytes,
    output_names,
)

HF_CONFIG = {
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 256,
    "sliding_window": 8,
    "rms_norm_eps": 1e-6,
    "layer_types": [SLIDING, SLIDING, SLIDING, FULL],
    "rope_parameters": {
        "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
        "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
    },
}


def _cfg(**kw) -> SparkStepConfig:
    return config_from_spark(HF_CONFIG, **kw)


@pytest.mark.core_model
@pytest.mark.cpu
def test_sliding_cache_is_capped_but_full_cache_grows():
    cfg = _cfg()
    assert cfg.cache_len(SLIDING, 4) == 4  # shorter than the window
    assert cfg.cache_len(SLIDING, 1024) == 8  # capped at sliding_window
    assert cfg.cache_len(SLIDING, 1_000_000) == 8
    assert cfg.cache_len(FULL, 1024) == 1024


@pytest.mark.core_model
@pytest.mark.cpu
def test_rotary_dim_differs_by_layer_type():
    cfg = _cfg()
    assert cfg.rotary_dim(SLIDING) == 16  # all head dims
    assert cfg.rotary_dim(FULL) == 4  # first quarter only


@pytest.mark.core_model
@pytest.mark.cpu
def test_kv_bytes_beats_uniform_attention_and_the_gap_grows():
    cfg = _cfg()
    near = kv_cache_bytes(cfg, 16)
    far = kv_cache_bytes(cfg, 4096)
    assert near["hybrid"] < near["uniform"]
    assert far["hybrid"] < far["uniform"]
    # The saving is a growing fraction, not a constant one.
    assert far["saved"] / far["uniform"] > near["saved"] / near["uniform"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_example_inputs_match_the_declared_signature():
    cfg = _cfg()
    args = example_inputs(cfg, context=1024)
    names = input_names(cfg)
    assert len(args) == len(names)
    caches = args[7:]
    assert len(caches) == 2 * cfg.num_layers
    for i, layer_type in enumerate(cfg.layer_types):
        expected = cfg.cache_len(layer_type, 1024)
        assert caches[2 * i].shape == (1, cfg.num_key_value_heads, expected, cfg.head_dim)
    assert len(output_names(cfg)) == 1 + 2 * cfg.num_layers


@pytest.mark.core_model
@pytest.mark.cpu
def test_sliding_layer_rolls_the_window_and_full_layer_appends():
    cfg = _cfg()
    step = SparkDecodeStep(cfg).eval()
    context = 1024
    args = example_inputs(cfg, context)
    with torch.no_grad():
        out = step(*args)
    logits, caches = out[0], out[1:]
    assert logits.shape == (1, cfg.vocab_size)
    for i, layer_type in enumerate(cfg.layer_types):
        k_new = caches[2 * i]
        if layer_type == SLIDING:
            # A sliding layer returns the whole rolled window, same size in.
            assert k_new.shape[2] == cfg.cache_len(SLIDING, context)
        else:
            # A full layer returns only the new entry for the runtime to append.
            assert k_new.shape[2] == 1


@pytest.mark.core_model
@pytest.mark.cpu
def test_sliding_window_drops_the_oldest_entry():
    """The rolled window must equal the old cache shifted by one, with the new
    key appended -- that is what makes a masked pad slot safe to prepend."""
    cfg = _cfg(layer_slice=slice(0, 1))
    step = SparkDecodeStep(cfg).eval()
    args = list(example_inputs(cfg, context=1024))
    k_cache = args[7]
    with torch.no_grad():
        out = step(*args)
    k_new = out[1]
    assert torch.allclose(k_new[:, :, :-1], k_cache[:, :, 1:])


@pytest.mark.core_model
@pytest.mark.cpu
def test_layer_slice_selects_one_layer_type():
    types = HF_CONFIG["layer_types"]
    sliding = _cfg(layer_slice=slice(0, 1))
    assert sliding.layer_types == (SLIDING,)
    full = _cfg(layer_slice=slice(types.index(FULL), types.index(FULL) + 1))
    assert full.layer_types == (FULL,)
    assert full.first_layer == types.index(FULL)


@pytest.mark.core_model
@pytest.mark.cpu
def test_unknown_layer_type_is_rejected():
    bad = dict(HF_CONFIG, layer_types=[SLIDING, "linear_attention", SLIDING, FULL])
    with pytest.raises(ValueError, match="unknown layer types"):
        config_from_spark(bad)
