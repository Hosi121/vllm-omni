"""MiniCPM-o processor contract across vLLM 0.28 CPU and 0.29 CUDA."""

from types import SimpleNamespace

import torch
from transformers.feature_extraction_utils import BatchFeature

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMMultiModalProcessor,
)


class _EmptyItems:
    def get_all_counts(self):
        return {}

    def select(self, _keys):
        return self


def _processor():
    tokenizer = SimpleNamespace(encode=lambda prompt: [len(prompt)])
    return SimpleNamespace(
        _get_hf_mm_data=lambda _items: ({}, {}),
        process_mm_inputs=lambda _data, _kwargs: {},
        info=SimpleNamespace(get_tokenizer=lambda: tokenizer),
        dummy_inputs=SimpleNamespace(get_dummy_text=lambda _counts: "dummy"),
    )


def test_vllm_028_contract_tokenizes_prompt_without_hf_multimodal_call():
    result = MiniCPMO45OmniLLMMultiModalProcessor._apply_hf_processor_main(
        _processor(),
        _EmptyItems(),
        {},
        prompt="(<audio>./</audio>)",
        tokenization_kwargs={},
        enable_hf_prompt_update=False,
    )

    prompt_ids, mm_data, updated = result
    assert prompt_ids == [len("(<audio>./</audio>)")]
    assert isinstance(mm_data, BatchFeature) and not mm_data
    assert updated is False


def test_vllm_029_contract_returns_batch_feature():
    result = MiniCPMO45OmniLLMMultiModalProcessor._apply_hf_processor_main(
        _processor(), _EmptyItems(), {}
    )

    assert isinstance(result, BatchFeature)
    torch.testing.assert_close(result["input_ids"], torch.tensor([[5]]))
