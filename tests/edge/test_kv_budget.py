# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-sized KV budgets, including hybrid attention."""

import pytest

from vllm_omni.edge.kv_budget import dtype_bytes, kv_budget_for

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _spark(**kw) -> _Cfg:
    """Spark-X2.5-1.7B: 28 layers, 3 sliding (512) to 1 full, 2 KV heads, 256."""
    layer_types = ["sliding_attention"] * 3 + ["full_attention"]
    base = dict(
        num_hidden_layers=28, num_attention_heads=8, num_key_value_heads=2,
        head_dim=256, hidden_size=2048, sliding_window=512,
        layer_types=layer_types * 7,
    )
    base.update(kw)
    return _Cfg(**base)


def test_flat_bytes_per_token_matches_the_hand_calculation():
    # 28 layers x 2 KV heads x 256 head_dim x 2 (K and V) x 2 bytes = 56 KiB.
    b = kv_budget_for(_spark(), max_model_len=2048)
    assert b.bytes_per_token_flat == 28 * 2 * 256 * 2 * 2
    assert b.bytes_per_token_flat == 56 * 1024


def test_hybrid_attention_is_charged_for_its_real_cache():
    b = kv_budget_for(_spark(), max_model_len=32768)
    assert (b.full_layers, b.sliding_layers, b.window) == (7, 21, 512)
    assert b.token_layers_hybrid == 7 * 32768 + 21 * 512
    assert b.token_layers_flat == 28 * 32768
    # The recommended budget stays flat, because vLLM's CPU accounting is flat.
    assert b.bytes_total > b.bytes_total_hybrid
    assert b.saving_vs_flat == pytest.approx(917504 / 240128, rel=1e-3)


def test_a_short_context_never_pays_for_a_longer_window():
    """A 512-window layer at a 128-token context caches 128, not 512."""
    b = kv_budget_for(_spark(), max_model_len=128)
    assert b.token_layers_hybrid == 28 * 128


def test_budget_scales_with_sequences_and_dtype():
    one = kv_budget_for(_spark(), max_model_len=2048, max_num_seqs=1)
    four = kv_budget_for(_spark(), max_model_len=2048, max_num_seqs=4)
    assert four.bytes_total == 4 * one.bytes_total
    fp8 = kv_budget_for(_spark(), max_model_len=2048, kv_dtype="fp8")
    assert fp8.bytes_total * 2 == one.bytes_total


def test_edge_sized_budget_is_far_under_the_server_default():
    """The measured point: 2048 tokens needs ~112 MiB, not the 4 GiB default."""
    b = kv_budget_for(_spark(), max_model_len=2048)
    mib = b.bytes_total / 2**20
    assert 130 < mib < 150, mib          # flat 56 KiB/token x 2048 x 1.25
    assert b.bytes_total < 4 * 2**30 / 25  # vs the 4 GiB default
    assert b.bytes_total_hybrid / 2**20 < 70


def test_plain_model_without_layer_types_uses_a_flat_budget():
    cfg = _Cfg(num_hidden_layers=32, num_attention_heads=32,
               num_key_value_heads=8, hidden_size=4096)
    b = kv_budget_for(cfg, max_model_len=4096)
    assert b.sliding_layers == 0 and b.window is None
    assert b.token_layers_hybrid == b.token_layers_flat
    assert b.saving_vs_flat == 1.0
    assert b.bytes_total == b.bytes_total_hybrid
    # head_dim inferred from hidden_size / heads.
    assert b.bytes_per_token_flat == 32 * 8 * 128 * 2 * 2


def test_sliding_window_without_layer_types_is_not_assumed():
    """A config with a window but no per-layer map might apply it everywhere or
    nowhere; charging the smaller cache would risk preemption."""
    cfg = _Cfg(num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=8,
               head_dim=64, sliding_window=256)
    b = kv_budget_for(cfg, max_model_len=4096)
    assert b.sliding_layers == 0
    assert b.token_layers_hybrid == b.token_layers_flat


def test_unreadable_config_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="num_hidden_layers"):
        kv_budget_for(_Cfg(), max_model_len=2048)
    with pytest.raises(ValueError, match="head_dim"):
        kv_budget_for(_Cfg(num_hidden_layers=4), max_model_len=2048)


def test_dtype_bytes_covers_the_kv_dtypes_vllm_accepts():
    assert dtype_bytes("bfloat16") == 2
    assert dtype_bytes("float32") == 4
    assert dtype_bytes("fp8") == 1
    assert dtype_bytes("something_unknown") == 2  # conservative default


def test_summary_names_the_shape_it_priced():
    s = kv_budget_for(_spark(), max_model_len=32768).summary()
    assert "flat" in s and "21 sliding@512" in s and "MiB" in s
