"""Regression for text requests sharing the Omni preprocess embedding buffer."""

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)


def test_text_preprocess_replaces_uninitialized_embedding_buffer() -> None:
    ids = torch.tensor([4, 7])
    scratch = torch.zeros((2, 3))
    expected = torch.tensor([[4.0, 4.0, 4.0], [7.0, 7.0, 7.0]])
    model = SimpleNamespace(model_stage="llm", get_input_embeddings=lambda input_ids: input_ids[:, None].expand(-1, 3).float())

    _, embeds, _ = MiniCPMO45OmniForConditionalGeneration.preprocess(
        model,
        ids,
        input_embeds=scratch,
        _omni_input_embeds_precomputed=False,
    )

    torch.testing.assert_close(embeds, expected)


def test_text_preprocess_preserves_precomputed_multimodal_embeddings() -> None:
    ids = torch.tensor([4, 7])
    prepared = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    model = SimpleNamespace(
        model_stage="llm",
        get_input_embeddings=lambda _input_ids: (_ for _ in ()).throw(AssertionError("must reuse embeddings")),
    )

    _, embeds, _ = MiniCPMO45OmniForConditionalGeneration.preprocess(
        model,
        ids,
        input_embeds=prepared,
        _omni_input_embeds_precomputed=True,
    )

    assert embeds is prepared
