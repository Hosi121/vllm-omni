# WSL CPU Qwen3-TTS complete requests through Omni

This is a scoped text-to-complete-WAV pass on Ryzen AI 9 HX 370 under Ubuntu
26.04/WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`. The public
`AsyncOmni` API and `StageRuntime` used the existing
`external.qwen_tts.cpu.v1` whole-session stage, with a resident isolated
PyTorch worker. The worker's model parameters and output are CPU-only BF16/SDPA,
eight threads, `Ryan`/English, seed 42 and `max_new_tokens=64`. CUDA is masked;
no Radeon, RTX or NPU execution is claimed. The Omni controller uses installed
vLLM 0.29.0; the worker uses PyTorch 2.13.0+cu130 on CPU, `qwen-tts` 0.1.1 and
Transformers 4.57.3. Its exact [overlay manifest](overlay_manifest.json) and
[optional-kernels CPU shim](kernels/__init__.py) are retained. The local SoX
Python package emitted a missing SoX CLI warning; the named text-to-WAV path
completed, while SoX-dependent audio-input paths were not tested.

The checkpoint is Qwen3-TTS-12Hz-0.6B-CustomVoice revision
`85e237c12c027371202489a0ec509ded67b5e4b5`, with talker SHA256
`bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`
and speech-tokenizer SHA256
`836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`.
The stage checked both, the declared interpreter hash and the optional shim
hash before accepting requests. The [profile report](profile20.json) records
the worker's package versions, actual CPU placement and loaded memory.

| Check | WSL CPU observation | Evidence |
|---|---|---|
| Two named complete requests | “Hello from the local computer.” returned 82,560 frames (3.44 s); “The blue car is parked beside the library.” returned 80,640 frames (3.36 s), both 24 kHz mono WAVs. | [report](profile20.json), [worker log](profile20_worker.log) |
| Serial profile | One warmup excluded; **20/20** measured first-sentence requests returned the same PCM SHA256. Nearest-rank complete-request wall p50/p95 **8.27/8.47 s** for 3.44 s audio (RTF 2.40/2.46). These are resident-worker times after 19.82 s stage startup, not first-audio latency. | [all timings and memory samples](profile20.json), [first measured WAV](profile20.wav) |
| Independent speech proxy | Pinned Whisper tiny.en transcribed the first measured WAV as “Hello from the local computer!”, WER 0 against the stated sentence. This covers one output, not perceptual or speaker quality. | [ASR report](profile20_asr.json) |
| Public API | `AsyncOmni.generate` returned one terminal audio event with the same 82,560-frame PCM hash as the measured stage run. Startup was 20.29 s; the one public request took 12.10 s. | [public report](public_report.json), [worker log](public_worker.log) |
| Admission and cancellation | A 10 GiB host-RAM reservation from a declared 16 GiB pool passed; 4 GiB was refused before worker launch. In-flight cancellation left no stale output, exited the worker and returned the ledger to zero. | [refusal report](low_budget_report.json), [profile report](profile20.json) |

The two WSL WAV hashes and durations **differ from the same-checkpoint native
Windows CPU outputs** recorded [here](../qwen_tts_omni_cpu/README.md), despite
matching declared BF16 settings and seed. The WSL profile's own 20 repeated
outputs are identical. The cause of the cross-OS difference is not established,
so Windows PCM parity is not claimed. The first WSL output passed the named ASR
proxy, but broader speech quality and numerical agreement remain open.

A [post-portability-change native Windows regression](windows_regression_report.json)
repeated the two named requests and one measured request with the original
Windows pins: both standalone PCM parity checks still passed, in-flight
cancellation drained the worker, and the ledger returned to zero. Its
[worker log](windows_regression_worker.log) and [WAV](windows_regression.wav)
are retained. This verifies the launch-path change did not alter the already
qualified Windows fixture; it does not explain the WSL waveform difference.

The 0.25 s profile samples observed at most 3.39 GB summed process-tree RSS;
Linux's `psutil` private-memory fallback used RSS, so that column is not a
separate private-memory measurement. The samples do not prove loading peak,
unique physical RAM, pagefile peak or concurrent admission. No power mode or
thermal history was captured. The worker returns complete WAVs only: playable
streaming, post-cancel restart, larger prompts and sustained real-time use are
unqualified; measured RTF remains above one.

The [first failed attempt](attempt1_report.json) resolved the Linux virtualenv
`python` symlink to a bare interpreter, which lost the venv packages. The
adapter now preserves the declared launch path while hashing its target. The
[second failed attempt](attempt2_report.json) produced both complete WAVs but
hit the probe's Windows-only PCM assertion; `--allow-platform-variation` now
records that difference explicitly. The [original logs](attempt1_driver.log),
[worker error](attempt1_worker.log), [second driver](attempt2_driver.log) and
[second worker](attempt2_worker.log) remain available.

Reproduce with
[`probe_omni_qwen_tts_cpu.py`](../../../../experiments/probe_omni_qwen_tts_cpu.py)
and [`probe_omni_qwen_tts_cpu_entrypoint.py`](../../../../experiments/probe_omni_qwen_tts_cpu_entrypoint.py)
using the checkpoint and hashes above, the recorded Python interpreter hash
`021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7`,
the pinned overlay packages in the manifest, and a copy of the shim at
`<overlay>/kernels/__init__.py`. Pass its SHA256
`e155388b1045b573082534132778828a71203522f95a22680f8245362f1e06ee`
as `--overlay-marker-sha256`. The serial profile used
`--warmups 1 --repeats 20 --abort-check --allow-platform-variation`; the
public check used `--expected-frames 82560` and PCM SHA256
`66ca76a468807381c208b21ba458e5005510c752efa89323ee909749047ab1db`.
