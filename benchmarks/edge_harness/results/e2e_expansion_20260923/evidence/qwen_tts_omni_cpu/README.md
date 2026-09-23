# Native Windows CPU Qwen3-TTS through Omni

This is a **scoped complete-request text-to-WAV pass**, not a playable streaming or general speech-quality qualification. On the Ryzen AI 9 HX 370 laptop with Windows 11 build 26200, `AsyncOmni` and `StageRuntime` ran the pinned Qwen3-TTS 0.6B CustomVoice checkpoint through one `external.qwen_tts.cpu.v1` whole-session stage. The stage owns a bounded local worker, request epoch and terminal audio event; it reserves 10 GiB of host RAM before loading and releases the reservation on cancellation or shutdown. The model and speech tokenizer remain in the worker, with CUDA masked. No NPU, Radeon or RTX execution is claimed.

The checkpoint is [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tree/85e237c12c027371202489a0ec509ded67b5e4b5) revision `85e237c12c027371202489a0ec509ded67b5e4b5`. The talker weight SHA256 is `bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb` (1,811,626,576 bytes); the speech tokenizer weight SHA256 is `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258` (682,293,092 bytes). The worker verifies both, interpreter SHA256 `0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`, and the loaded Python package versions before accepting a request. It uses Python 3.12.10, PyTorch 2.13.0+cu130 on CPU, `qwen-tts` 0.1.1, an isolated Transformers 4.57.3 overlay, BF16/SDPA, eight CPU threads, `Ryan`/English, seed 42, `max_new_tokens=64`, and non-streaming generation. The Omni interpreter retains Transformers 5.14.1. The [standalone baseline](../qwen_tts_native_cpu/README.md) used the same checkpoint and generation settings.

| Check | Result | Raw evidence |
|---|---|---|
| Two named sentences | “Hello from the local computer.” produced 109,440 frames (4.56 s); “The blue car is parked beside the library.” produced 82,560 frames (3.44 s), both at 24 kHz mono PCM16. Their PCM SHA256 values exactly match the standalone outputs: `c8a49b5c73efd642a1eb04be973d6aefdc8ed9cfd18b34ff67690b8464906bed` and `4edc7a62f749b1fde5ea39bedbc15a35f291ab1db8d24d7b7eb3df7270885e16`. | [profile report](omni_qwen_tts_cpu_profile20_20260923.json), [worker log](omni_qwen_tts_cpu_profile20_20260923.log) |
| Resident serial profile | One warmup excluded; 20/20 measured complete requests returned the identical first-sentence PCM. Nearest-rank request-wall p50/p95 were **15.32/15.60 s** for 4.56 s of audio (RTF 3.36/3.42). These times include the Omni stage handoff and exclude startup; they are not first-audio latency. | [all requests and 1,042 memory samples](omni_qwen_tts_cpu_profile20_20260923.json), [driver log](omni_qwen_tts_cpu_profile20_20260923.driver.log), [WAV](omni_qwen_tts_cpu_profile20_20260923.wav) |
| Public API | `AsyncOmni.generate` returned one terminal 24 kHz audio event with 109,440 frames and the same PCM SHA256 in 15.87 s after 13.83 s initialization. | [public API report](omni_qwen_tts_cpu_public_20260923.json), [driver log](omni_qwen_tts_cpu_public_20260923.driver.log), [worker log](omni_qwen_tts_cpu_public_20260923.log) |
| Admission and cancellation | A deliberate 4 GiB demand was refused as `ResourceUnavailable` before worker launch, with zero remaining reservation. An in-flight request was cancelled: no stale output, worker exited, and the 10 GiB reservation returned to zero. | [rejection report](omni_qwen_tts_cpu_reject_20260923.json), [rejection log](omni_qwen_tts_cpu_reject_20260923.driver.log), [profile report](omni_qwen_tts_cpu_profile20_20260923.json) |
| Independent transcription proxy | Pinned Whisper tiny.en transcribed the Omni first-sentence WAV as “Hello from the local compute”, WER 0.20, the same as the standalone WAV. This does not establish perceptual quality or speaker fidelity. | [ASR report](omni_qwen_tts_cpu_asr_20260923.json), [ASR log](omni_qwen_tts_cpu_asr_20260923.log) |

The loaded launcher-plus-child process tree reported 1,856,651,264 bytes RSS and 4,673,892,352 bytes private memory. The sampled maximum during the profile was 3,255,246,848 bytes RSS and 5,392,134,144 bytes private memory, below the declared 10 GiB reservation. Process-tree accounting matters here: the virtualenv launcher by itself has a small working set, while its descendant holds the model. These 0.25 s samples do **not** establish loading peak, unique physical RAM use, pagefile peak, concurrency admission or a 30-minute thermal/power result. The worker returns only a complete WAV, so playable incremental audio, post-cancel restart, a broad voice-quality suite and sustained real-time performance remain open.

Reproduce from the repository root in PowerShell with the exact checkpoint and isolated overlay already installed locally:

```powershell
$root = '\\wsl.localhost\Ubuntu\home\zhout\project\edge_infer\vllm-omni-edge'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $root
$env:VLLM_ENABLE_V1_MULTIPROCESSING = '0'
$python = 'C:\Users\zhout\w2\omni029venv\Scripts\python.exe'
$common = @(
  '--python-bin', $python,
  '--python-sha256', '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14',
  '--model-dir', 'C:\Users\zhout\w2\models\Qwen3-TTS-0.6B-85e237c',
  '--overlay-dir', 'C:\Users\zhout\w2\qwentts_pins',
  '--talker-sha256', 'bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb',
  '--tokenizer-sha256', '836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258'
)
& $python "$root\benchmarks\edge_harness\experiments\probe_omni_qwen_tts_cpu.py" @common `
  --server-log 'C:\Users\zhout\w2\omni_qwen_tts_cpu_profile20_20260923.log' `
  --output-report 'C:\Users\zhout\w2\omni_qwen_tts_cpu_profile20_20260923.json' `
  --output-wav 'C:\Users\zhout\w2\omni_qwen_tts_cpu_profile20_20260923.wav' `
  --warmups 1 --repeats 20 --abort-check
& $python "$root\benchmarks\edge_harness\experiments\probe_omni_qwen_tts_cpu_entrypoint.py" @common `
  --server-log 'C:\Users\zhout\w2\omni_qwen_tts_cpu_public_20260923.log' `
  --output-report 'C:\Users\zhout\w2\omni_qwen_tts_cpu_public_20260923.json'
```

The matrix cell advances only the named native Windows CPU × Qwen3-TTS text-to-WAV route. It does not advance the NPU, mobile, joint-accelerator or other model rows.
