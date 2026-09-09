"""KV-cached incremental residual decoding of the Qwen3 code predictor equals the re-prefill forward (CPU)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorBaseModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny_config():
    return SimpleNamespace(
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=96,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_code_groups=16,
        rope_theta=10000.0,
        attention_bias=False,
    )


@torch.inference_mode()
def test_forward_cached_matches_full_prefix():
    torch.manual_seed(0)
    model = CodePredictorBaseModel(_tiny_config()).eval()
    for p in model.parameters():
        p.normal_(0, 0.2)
    bsz, max_seq, hidden = 2, 17, 64
    x = torch.randn(bsz, max_seq, hidden)
    pos = torch.arange(max_seq).unsqueeze(0).expand(bsz, -1)
    full = model(x, pos)  # [B, 17, H], causal over the whole prefix

    caches = model.new_kv_caches(bsz, max_seq, x.device, x.dtype)
    out_prefill = model.forward_cached(x[:, :2], pos[:, :2], caches, 0)
    assert torch.allclose(out_prefill, full[:, :2], atol=1e-4, rtol=1e-4)
    for step in range(2, max_seq):
        out_step = model.forward_cached(x[:, step : step + 1], pos[:, step : step + 1], caches, step)
        assert torch.allclose(out_step[:, 0], full[:, step], atol=1e-4, rtol=1e-4), step
    # caches hold exactly the prefix K/V that a fresh full pass would produce
    assert caches[0][0].shape == (bsz, 2, max_seq, 16)


@torch.inference_mode()
def test_forward_cached_rejects_multi_token_append():
    model = CodePredictorBaseModel(_tiny_config()).eval()
    x = torch.randn(1, 2, 64)
    pos = torch.arange(2).unsqueeze(0)
    caches = model.new_kv_caches(1, 17, x.device, x.dtype)
    model.forward_cached(x, pos, caches, 0)
    with pytest.raises(ValueError):
        model.forward_cached(x, pos + 2, caches, 2)


@torch.inference_mode()
def test_forward_static_step_matches_full_prefix():
    torch.manual_seed(1)
    model = CodePredictorBaseModel(_tiny_config()).eval()
    for p in model.parameters():
        p.normal_(0, 0.2)
    bsz, max_seq = 2, 17
    x = torch.randn(bsz, max_seq, 64)
    pos = torch.arange(max_seq).unsqueeze(0).expand(bsz, -1)
    full = model(x, pos)
    caches = model.new_kv_caches(bsz, max_seq, x.device, x.dtype)
    model.forward_cached(x[:, :2], pos[:, :2], caches, 0)  # eager 2-token prefill
    for step in range(2, max_seq):
        pos_t = torch.tensor([step])
        mask = (torch.arange(max_seq) <= step).view(1, 1, 1, max_seq).expand(bsz, 1, 1, max_seq)
        out = model.forward_static_step(x[:, step : step + 1], pos[:, step : step + 1], caches, pos_t, mask)
        assert torch.allclose(out[:, 0], full[:, step], atol=1e-4, rtol=1e-4), step
