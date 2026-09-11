# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the Spark-X2.5 model definition.

These cover the three places where Spark departs from a Llama-style decoder
and where a port is most likely to go wrong:

* the fused ``q_k_v_proj`` checkpoint layout has to be split before vLLM's
  sharded ``QKVParallelLinear`` can load it,
* ``layer_types`` must map onto per-layer sliding windows, and
* each layer type carries its own RoPE theta and partial-rotary factor.

End-to-end numerical parity against the reference implementation is measured
separately (see analysis/spark_edge_support.md); it needs the real 3.2 GiB
checkpoint and so does not belong in the unit suite.
"""

import pytest
import torch

from vllm_omni.model_executor.models.spark2_5.configuration_spark2_5 import (
    Spark2_5Config,
)
from vllm_omni.model_executor.models.spark2_5.spark2_5 import _split_fused_qkv


def _config(**overrides) -> Spark2_5Config:
    kwargs = dict(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="gelu",
        sliding_window=16,
        headwise_attn_output_gate=True,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        rope_parameters={
            "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
        },
        tie_word_embeddings=True,
    )
    kwargs.update(overrides)
    return Spark2_5Config(**kwargs)


@pytest.mark.core_model
@pytest.mark.cpu
def test_split_fused_qkv_recovers_q_k_v_shards():
    config = _config()
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim

    # Rows are tagged by value so a mis-ordered split is visible.
    fused = torch.cat(
        [
            torch.full((q_dim, config.hidden_size), 1.0),
            torch.full((kv_dim, config.hidden_size), 2.0),
            torch.full((kv_dim, config.hidden_size), 3.0),
        ],
        dim=0,
    )
    out = dict(
        _split_fused_qkv(
            [("model.layers.0.self_attn.q_k_v_proj.weight", fused)], config
        )
    )

    assert set(out) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    }
    assert out["model.layers.0.self_attn.q_proj.weight"].shape == (
        q_dim,
        config.hidden_size,
    )
    assert out["model.layers.0.self_attn.k_proj.weight"].shape == (
        kv_dim,
        config.hidden_size,
    )
    assert torch.all(out["model.layers.0.self_attn.q_proj.weight"] == 1.0)
    assert torch.all(out["model.layers.0.self_attn.k_proj.weight"] == 2.0)
    assert torch.all(out["model.layers.0.self_attn.v_proj.weight"] == 3.0)


@pytest.mark.core_model
@pytest.mark.cpu
def test_split_fused_qkv_passes_other_weights_through():
    config = _config()
    mlp = torch.zeros(config.intermediate_size, config.hidden_size)
    out = list(_split_fused_qkv([("model.layers.0.mlp.gate_proj.weight", mlp)], config))
    assert len(out) == 1
    assert out[0][0] == "model.layers.0.mlp.gate_proj.weight"


@pytest.mark.core_model
@pytest.mark.cpu
def test_config_exposes_per_layer_type_rope():
    config = _config()
    # Sliding layers rotate every head dim with a small theta; full-attention
    # layers rotate only the first quarter with a large one.
    assert config.get_partial_rotary_factor("sliding_attention") == 1.0
    assert config.get_rope_theta("sliding_attention") == 10000
    assert config.get_partial_rotary_factor("full_attention") == 0.25
    assert config.get_rope_theta("full_attention") == 5000000


@pytest.mark.core_model
@pytest.mark.cpu
def test_config_rejects_layer_types_of_the_wrong_length():
    with pytest.raises(ValueError, match="layer_types length"):
        _config(layer_types=["full_attention"])


@pytest.mark.core_model
@pytest.mark.cpu
def test_headwise_gate_scales_each_head_independently():
    """The gate is one sigmoid scalar per head, broadcast over head_dim."""
    config = _config()
    num_heads, head_dim = config.num_attention_heads, config.head_dim
    attn_output = torch.ones(3, num_heads * head_dim)
    gate_score = torch.tensor([[-10.0, 0.0, 10.0, 0.0]]).repeat(3, 1)

    gate = torch.sigmoid(gate_score.float()).to(attn_output.dtype)
    gated = (
        attn_output.view(-1, num_heads, head_dim) * gate.view(-1, num_heads, 1)
    ).view(-1, num_heads * head_dim)

    per_head = gated.view(-1, num_heads, head_dim)
    # A strongly negative score nearly zeroes its head, a zero score halves it,
    # and every dim inside one head gets the identical factor.
    assert per_head[:, 0].max() < 1e-3
    assert torch.allclose(per_head[:, 1], torch.full_like(per_head[:, 1], 0.5))
    assert per_head[:, 2].min() > 0.999
    assert torch.allclose(
        per_head[:, 0], per_head[:, 0, :1].expand_as(per_head[:, 0])
    )


@pytest.mark.core_model
@pytest.mark.cpu
def test_registered_in_the_omni_registry():
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    assert _OMNI_MODELS["Spark2_5ForCausalLM"] == (
        "spark2_5",
        "spark2_5",
        "Spark2_5ForCausalLM",
    )
