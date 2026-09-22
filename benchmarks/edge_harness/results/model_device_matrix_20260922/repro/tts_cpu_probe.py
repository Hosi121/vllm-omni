"""Real-weight two-frame CPU TTS smoke probe, not full speech qualification."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    record = {"model": a.model, "scope": "two codec frames only; no speech-quality claim", "start": time.time()}
    engine = None
    try:
        import torch
        from transformers import AutoTokenizer
        from vllm.platforms import current_platform

        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
        from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder

        assert current_platform.device_type == "cpu", "CPU probe must not silently use CUDA"
        config = Qwen3TTSConfig.from_pretrained(a.model)
        tok = AutoTokenizer.from_pretrained(a.model)
        info = {
            "task_type": ["CustomVoice"],
            "text": ["A short test."],
            "language": ["English"],
            "speaker": ["Vivian"],
            "max_new_tokens": [2],
        }
        length = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
            additional_information=info,
            task_type="CustomVoice",
            tokenize_prompt=lambda x: tok(x, padding=False)["input_ids"],
            codec_language_id=config.talker_config.codec_language_id,
            spk_is_dialect=config.talker_config.spk_is_dialect,
        )
        engine = Omni(model=a.model, stage_init_timeout=120, parallel_stage_init=False)
        params = engine.default_sampling_params_list
        params[0].max_tokens = 2
        outputs = engine.generate({"prompt_token_ids": [0] * length, "additional_information": info}, params)
        chunks = []
        for output in outputs:
            if output.final_output_type == "audio":
                audio = output.outputs[0].multimodal_output.get("audio")
                chunks.extend(audio if isinstance(audio, list) else [audio])
        values = torch.cat([x.float().cpu().reshape(-1) for x in chunks if x is not None])
        assert values.numel() > 0 and torch.isfinite(values).all() and torch.any(values != 0)
        record.update(status="completed", samples=values.numel(), finite=True, nonzero=True)
    except Exception as error:
        record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if engine is not None:
            engine.close()
        record["end"] = time.time()
        a.out.write_text(json.dumps(record, indent=2) + "\n")
    print(record["status"], flush=True)


if __name__ == "__main__":
    main()
