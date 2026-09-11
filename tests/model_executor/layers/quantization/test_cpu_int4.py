# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the cpu_int4 W4A16 method."""

import pytest
import torch

from vllm_omni.model_executor.layers.quantization.cpu_int4 import (
    CPUInt4Config,
    _to_tinygemm,
    dequantize,
    pack_nibbles,
    quantize_groupwise,
    to_split_halves,
    unpack_nibbles,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _weights(n=64, k=256, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, k)


def test_nibble_pack_round_trips():
    q = torch.randint(0, 16, (32, 128), dtype=torch.int32)
    assert torch.equal(unpack_nibbles(pack_nibbles(q)), q)


def test_quantized_weights_stay_on_the_four_bit_grid():
    w = _weights()
    packed, scale, zero = quantize_groupwise(w, 64)
    codes = unpack_nibbles(packed)
    assert codes.min() >= 0 and codes.max() <= 15
    assert packed.dtype == torch.uint8
    assert packed.shape == (64, 128)
    assert scale.shape == zero.shape == (64, 4)


def test_group_extremes_are_reproduced_exactly():
    """min/max grid: the two ends of each group must dequantize back to
    themselves, which is what makes the offset form worth its extra bytes."""
    w = _weights(n=16, k=128)
    group = 64
    packed, scale, zero = quantize_groupwise(w, group)
    codes = unpack_nibbles(packed).float().reshape(16, 128 // group, group)
    deq = (codes - 8.0) * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    wg = w.reshape(16, 128 // group, group)
    torch.testing.assert_close(deq.amax(-1), wg.amax(-1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(deq.amin(-1), wg.amin(-1), rtol=1e-5, atol=1e-5)


def test_split_halves_is_a_relabelling_of_the_same_codes():
    w = _weights()
    packed, _, _ = quantize_groupwise(w, 64)
    halves = to_split_halves(packed)
    codes = unpack_nibbles(packed)
    k = codes.shape[1]
    assert torch.equal((halves & 0x0F).to(torch.int32), codes[:, : k // 2])
    assert torch.equal((halves >> 4).to(torch.int32), codes[:, k // 2 :])


@pytest.mark.parametrize("group", [32, 64, 128])
def test_dequantize_path_agrees_with_the_tinygemm_kernel(group):
    """The two GEMM paths must be the same function of the weights.

    Prefill takes the dequantize-then-AMX path and decode the tinygemm one;
    a mismatch here would show up as a model that generates fine but answers
    differently depending on prompt length, which no throughput benchmark
    would catch.
    """
    w = _weights(n=64, k=256)
    packed, scale, zero = quantize_groupwise(w, group)
    inner, sz = _to_tinygemm(packed, scale, zero)
    halves = to_split_halves(packed)
    sc, ze = scale.to(torch.bfloat16), zero.to(torch.bfloat16)

    x = torch.randn(8, 256, dtype=torch.bfloat16)
    via_kernel = torch.ops.aten._weight_int4pack_mm_for_cpu(x, inner, group, sz)
    via_dequant = torch.nn.functional.linear(x, dequantize(halves, sc, ze, group))

    # Scale-free: both sides accumulate in bf16, so the gap is bf16 rounding
    # on outputs whose magnitude grows with K, not a layout disagreement.
    a, b = via_kernel.float(), via_dequant.float()
    snr = 20 * torch.log10(a.norm() / (a - b).norm())
    assert snr > 40.0, f'paths disagree beyond bf16 rounding: {snr:.1f} dB'



def test_four_bit_error_is_small_enough_to_be_useful():
    w = _weights(n=256, k=1024)
    packed, scale, zero = quantize_groupwise(w, 64)
    codes = unpack_nibbles(packed).float().reshape(256, 1024 // 64, 64)
    deq = ((codes - 8.0) * scale.unsqueeze(-1) + zero.unsqueeze(-1)).reshape(256, 1024)
    snr = 20 * torch.log10(w.norm() / (w - deq).norm())
    assert snr > 19.0, snr


def test_config_round_trips_and_reports_bf16_only():
    cfg = CPUInt4Config.from_config(
        {"quant_method": "cpu_int4", "group_size": 128, "prefill_dequant": False}
    )
    assert cfg.group_size == 128
    assert cfg.prefill_dequant is False
    assert cfg.get_supported_act_dtypes() == [torch.bfloat16]
    assert cfg.get_name() == "cpu_int4"


def test_registered_into_vllms_quantization_registry():
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )

    import vllm_omni.model_executor.layers.quantization  # noqa: F401

    assert "cpu_int4" in QUANTIZATION_METHODS
    assert get_quantization_config("cpu_int4") is CPUInt4Config


def test_ignore_list_falls_back_to_full_precision():
    cfg = CPUInt4Config(group_size=64, ignore=["lm_head"])
    from vllm.model_executor.layers.linear import (
        ReplicatedLinear,
        UnquantizedLinearMethod,
    )

    layer = ReplicatedLinear.__new__(ReplicatedLinear)
    assert isinstance(cfg.get_quant_method(layer, "lm_head"), UnquantizedLinearMethod)
    assert not isinstance(
        cfg.get_quant_method(layer, "model.layers.0.mlp.down_proj"),
        UnquantizedLinearMethod,
    )
