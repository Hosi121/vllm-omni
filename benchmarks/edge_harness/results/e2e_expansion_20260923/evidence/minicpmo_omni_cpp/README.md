# MiniCPM-o GGUF through an Omni whole-session stage

These are two scoped, complete **audio+image → text+speech** requests through
`StageRuntime`/`StagePool` and `external.minicpmo.gguf.v1`, on the native
Windows 11 build 26200 HX370 laptop. They are not incremental audio streams or
full vLLM model execution. The C++ worker owns the model session and speech
state for each request; Omni verifies artifacts and placement, reserves shared
RAM, controls the request and cancellation, and emits a terminal audio event.

The Q4_K_M language plus nine F16 GGUF companions are the exact ten files and
revision `db25077c33951fe163b42986fba0132e279872a2` pinned in the
[CPU](cpu_report.json) and [Radeon](radeon_report.json) reports. The external
C++ source is `tc-mb/llama.cpp-omni` at
`5202b7b2f4d11f50b9f996161e7a2f8b8571b890`, with the
[recorded patch](llama_cpp_omni_bounded.patch) adding portable Windows
completion and a 96-token generation bound. Rebuild the CPU and Vulkan CLI
with the [recorded cross-build configuration](../minicpmo_cpp_radeon890m/README.md);
these new binary SHA256 values are pinned in the JSON reports. Native Windows Python used the
installed vLLM 0.29 environment with this Omni checkout. The input is the
[same 16 kHz spoken question and 448×448 red-square JPEG](../minicpmo_cpp_radeon890m_prompted/README.md)
used by the standalone parity run; the source speech was independently
transcribed as “What color is the square in the image?”.

| Placement | Complete-request wall time | Output | Sampled worker RSS peak | Raw report |
|---|---:|---|---:|---|
| Native Windows CPU only | 24.34 s | “The square in the image is red.”; [2.24 s WAV](cpu_output.wav) | 13.11 GB | [JSON](cpu_report.json) |
| Radeon 890M Vulkan0 language+vision, CPU TTS/Token2Wav/vocoder | 18.27 s | Same text; [2.60 s WAV](radeon_output.wav) | 15.57 GB | [JSON](radeon_report.json) |

Each has **one cold request, zero warmups, concurrency one**. The C++ CLI uses
2,048 context tokens, `--max-new-tokens 96`, 180 s bounded Token2Wav wait,
and a 240 s complete-request timeout. The stage reserves 24 GiB from a
declared 30 GiB shared-RAM capacity, including 8.87 GB weights, 12 GiB
headroom/state/workspace allowance, and 64 MiB bounded temporary output.
It checks mono PCM16 16 kHz input, 448×448 JPEG, output text and complete
24 kHz nonzero WAV chunks. Both logs show index-1 audio and vision prefill and
`Audio generation completed.` The [CPU logs](logs_cpu/) show zero GPU layers,
CPU vision and CPU Token2Wav. The [Radeon logs](logs_radeon/) show Radeon 890M
selected by `GGML_VK_VISIBLE_DEVICES=1`, 37/37 language layers and vision on
Vulkan0, 0/21 TTS layers on GPU, and CPU vocoder. Thus the Radeon route is
CPU+iGPU joint execution, not all-GPU or NPU execution.

Each run emitted one terminal audio event, reported output/input hashes, then
cancelled a new in-flight request: no stale output appeared, and the resource
ledger returned to zero. An intentionally insufficient 8 GiB CPU reservation
was [refused before worker launch](low_budget_report.json), with no remaining
ledger owner. The worker logs, reports, generated WAVs and full C++ patch are
retained here. The upstream optional WAV merger is not used; the stage verifies
and assembles contiguous output chunks after the completion marker.

The standalone outputs for this fixture were independently transcribed with
Whisper tiny.en at WER 0, an intelligibility proxy; these new Omni WAVs were
checked for nonzero complete PCM but not independently ASR-scored. One case
does not establish quality across real images or speech, loading peak,
repeatability, playable streaming, cancellation restart, concurrent admission,
or sustained power and thermals. No power condition was controlled.

Reproduce with
[`probe_omni_minicpmo_cpp.py`](../../../../experiments/probe_omni_minicpmo_cpp.py)
using `--placement cpu` or `--placement radeon-hybrid`, the paths and hashes in
the JSON reports, the referenced model revision and the named audio/JPEG
fixture. Add `--abort-check` for cancellation and
`--reserve-gib 8 --expect-admission-refusal` for the budget refusal. The
Windows Python process needs `PYTHONUTF8=1` and `PYTHONPATH` pointing to this
Omni checkout.
