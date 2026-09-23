# Native Windows CPU Qwen3-TTS standalone text-to-WAV evidence

This is a **scoped complete text-to-WAV request outside Omni**, not an Omni streaming or full speech-quality qualification. The CPU-only model placement was asserted on the Ryzen AI 9 HX 370 laptop running Windows 11 build 26200. The installed native vLLM extension still lacks its CPU attention operators, so this experiment used the official Qwen PyTorch wrapper as a candidate whole-model stage backend.

The local checkpoint is [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tree/85e237c12c027371202489a0ec509ded67b5e4b5) at revision `85e237c12c027371202489a0ec509ded67b5e4b5`. The talker `model.safetensors` is 1,811,626,576 bytes, SHA256 `bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`; `speech_tokenizer/model.safetensors` is 682,293,092 bytes, SHA256 `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`. Both match the remote revision's LFS hashes. Weights stay in the local model cache, not Git. The [official `qwen-tts` wrapper](https://github.com/QwenLM/Qwen3-TTS) was version 0.1.1. The process used Python 3.12.10, PyTorch 2.13.0+cu130 with CUDA masked, BF16, SDPA, eight CPU threads, `Ryan`/English, seed 42, `max_new_tokens=64`, sampling enabled and `non_streaming_mode=True`. All model parameters were checked on CPU.

| Run | Input and output | Timing and observed result | Raw record |
|---|---|---|---|
| First sentence | “Hello from the local computer.” → 4.56 s, 24 kHz mono PCM16 | One request took 14.95 s; waveform RMS 0.0610, not silent. Whisper tiny.en transcribed “Hello from the local compute”, WER 0.20. | [report](qwen_tts_cpu_20260923.json), [WAV](qwen_tts_cpu_20260923.wav), [log](qwen_tts_cpu_20260923.log), [ASR report](qwen_tts_cpu_asr_20260923.json), [ASR log](qwen_tts_cpu_asr_20260923.log) |
| Serial profile, same first sentence | 1 warmup excluded, 20 measured requests in one loaded process; each produced the same 4.56 s nonzero waveform | Nearest-rank complete-generation p50 13.94 s, p95 14.68 s; RTF approximately 3.06/3.22. No concurrency or streaming measurement. | [all request timings](qwen_tts_cpu_profile20_20260923.json), [WAV](qwen_tts_cpu_profile20_20260923.wav), [log](qwen_tts_cpu_profile20_20260923.log), [memory samples](qwen_tts_cpu_profile20_memory_samples_20260923.json) |
| Second sentence | “The blue car is parked beside the library.” → 3.44 s, 24 kHz mono PCM16 | One request took 10.55 s; waveform RMS 0.0476. Whisper tiny.en returned the exact sentence, WER 0.00. | [report](qwen_tts_cpu_second_20260923.json), [WAV](qwen_tts_cpu_second_20260923.wav), [log](qwen_tts_cpu_second_20260923.log), [ASR report](qwen_tts_cpu_second_asr_20260923.json), [ASR log](qwen_tts_cpu_second_asr_20260923.log) |

The memory trace covers only 120 samples over 32.07 s at roughly 0.25 s intervals during the serial profile. Maximum observed Python process working set was 3,202,650,112 bytes and private bytes 5,336,956,928. It is **not** the loading peak, a full-run maximum, or a shared-RAM/iGPU admission measurement. No device power mode or thermal history was captured. The ASR proxy used locally pinned [Whisper tiny.en](https://huggingface.co/openai/whisper-tiny.en/tree/87c7102498dcde7456f24cfd30239ca606ed9063) revision `87c7102498dcde7456f24cfd30239ca606ed9063`, weights SHA256 `db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad`. Neither two transcriptions nor nonzero audio establishes perceptual quality, speaker fidelity, or general intelligibility.

The native Omni environment has Transformers 5.14.1. Importing `qwen-tts` 0.1.1 with that version [failed](qwen_tts_native_transformers5_import_failure_20260923.log) at `check_model_inputs()`. The successful process overlaid the wrapper's pinned Transformers 4.57.3 and compatible dependencies through an isolated `PYTHONPATH` directory, leaving the Omni environment unchanged. The missing SoX executable emitted a warning; it did not prevent these WAV outputs. This dependency split must be resolved when implementing an Omni stage binding.

Reproduce in PowerShell after downloading the exact checkpoint and installing `qwen-tts==0.1.1` and its pinned dependencies into isolated local environments:

```powershell
$root = '\\wsl.localhost\Ubuntu\home\zhout\project\edge_infer\vllm-omni-edge'
$env:PYTHONPATH = 'C:\Users\zhout\w2\qwentts_pins'
$env:CUDA_VISIBLE_DEVICES = ''
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '8'
& 'C:\Users\zhout\w2\omni029venv\Scripts\python.exe' "$root\benchmarks\edge_harness\experiments\probe_qwen_tts_native_cpu.py" `
  --model-dir 'C:\Users\zhout\w2\models\Qwen3-TTS-0.6B-85e237c' `
  --output-wav 'C:\Users\zhout\w2\qwen_tts_cpu_profile20_20260923.wav' `
  --output-report 'C:\Users\zhout\w2\qwen_tts_cpu_profile20_20260923.json' `
  --warmups 1 --repeats 20
```

The ASR check is reproduced with [probe_tts_asr.py](../../../../experiments/probe_tts_asr.py), `--wav` pointing to a generated WAV, `--model-dir` pointing to the pinned Whisper cache, `--model-revision 87c7102498dcde7456f24cfd30239ca606ed9063`, and `--reference-text` matching the chosen sentence. This record does not advance the mobile/NPU or combined-device TTS cells.
