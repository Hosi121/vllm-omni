# E2E check and profiling

All 60 pairings have a disposition. The timing tables below remain the 2026-09-22 measurements. The configuration matrix and affected per-cell reasons incorporate the [2026-09-23 scoped recovery runs](../../../e2e_recovery_20260923/README.md) and [WSL CPU/native Windows expansion](../../../e2e_expansion_20260923/README.md). The expansion adds separate 20-request serial MiniCPM-o text-to-speech profiles on WSL CPU and native Windows GPU, a Spark output-head component profile on the HX370 AMD NPU, complete Spark text-request profiles through standalone llama.cpp and a bounded Omni whole-session stage on native Windows CPU and Radeon 890M, native Windows CPU Qwen3-TTS text-to-WAV profiles through both standalone PyTorch and a bounded Omni whole-session stage, and a hybrid CPU+Radeon 890M Qwen3-TTS profile likewise repeated through a bounded Omni whole-session stage. A pinned GGUF MiniCPM-o route adds combined image+audio to text+speech runs on WSL CPU, native Windows CPU and native Windows Radeon 890M+CPU. All three now pass a bounded complete request through an Omni graph stage; the WSL CPU path additionally passed one warmup plus 20 serial measured requests. These are scoped synthetic cases, not full release qualification.

The Radeon 890M InternVLA cell now has a [Cosmos image-encoder component profile and a separate CPU+Radeon complete synthetic action-policy profile](../../../e2e_expansion_20260923/evidence/internvla_cosmos_radeon890m/README.md). It remains short of robot-task E2E qualification because real observations, reference actions, units/timestamps, complete admission and lifecycle are unverified.

## Configuration × model

The WSL CPU Qwen3-TTS cell also has a separate bounded Omni whole-session
text-to-complete-WAV profile with one warmup, 20 measured requests, a public
audio request, pinned ASR proxy, memory refusal and cancellation evidence.
It does not replace the earlier short-stream finding or establish playable
streaming.

NOT E2E means no complete device/model workload was verified. “Scoped E2E” identifies only the named input/output path; “standalone” means the whole model ran outside Omni. The Spark GGUF Omni stage uses a complete-request control path; it has no incremental token stream or restart-after-cancel contract yet. “Synthetic policy pass” does not establish a real robot-task result. See the per-cell reasons below.

| Device | Spark-X2.5 | Qwen3-TTS 0.6B CustomVoice | Qwen3.8-27B | MiniCPM-o 4.5 | InternVLA-A1 |
|---|---|---|---|---|---|
| PC HX370: CPU / WSL | profile completed | scoped E2E: Omni CPU text → complete WAV | NOT E2E | scoped E2E: Omni GGUF combined image+audio → text+speech | synthetic policy pass |
| PC HX370: CPU / native Windows | scoped E2E: Omni Spark text | scoped E2E: Omni CPU text → complete WAV | NOT E2E | scoped E2E: Omni GGUF image+audio → text+speech | synthetic policy pass |
| PC HX370 + RTX 5090 Laptop / WSL | profile completed | profile completed | scoped E2E: text + image | scoped E2E: text/image/audio → speech | synthetic policy pass |
| PC HX370 + RTX 5090 Laptop / Windows | profile completed | profile completed | NOT E2E | scoped E2E: text/image/audio → speech | synthetic policy pass |
| PC HX370 + Radeon 890M / Windows worker | scoped E2E: Omni Spark text | scoped E2E: Omni Radeon talker/codec + CPU predictor | NOT E2E | scoped E2E: Omni Radeon+CPU image+audio → text+speech | synthetic policy pass: CPU policy + Radeon Cosmos |
| PC HX370 + AMD NPU / Windows worker | NOT E2E (output-head component passed) | NOT E2E | NOT E2E | NOT E2E | NOT E2E |
| PC CPU+iGPU+NPU +/- discrete GPU, joint model execution | NOT E2E | NOT E2E | NOT E2E | NOT E2E | NOT E2E |
| PC Snapdragon X Elite CRD / AI Hub | NOT E2E | NOT E2E | NOT E2E | NOT E2E | NOT E2E |
| Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy | NOT E2E | NOT E2E | NOT E2E | NOT E2E | NOT E2E |
| Mobile Galaxy S24 / Snapdragon 8 Gen 3 | NOT E2E | NOT E2E | NOT E2E | NOT E2E | NOT E2E |
| Embedded SA8775P ADP / AI Hub | NOT E2E | NOT E2E | NOT E2E | NOT E2E | NOT E2E |
| Embedded RB3 Gen 2 / QCS6490 / AI Hub | NOT E2E | NOT E2E | NOT E2E | NOT E2E | NOT E2E |

## Per-cell reasons and next steps

| Device | Model | Current profiling disposition | Reason / remaining work | Next step |
|---|---|---|---|---|
| PC HX370: CPU / WSL | Spark-X2.5 | PROFILE_COMPLETED | not fully qualified: reference-quality and remaining reliability/performance gates must be reviewed | Qualify a matching vLLM 0.29 CPU build and token parity separately. |
| PC HX370: CPU / WSL | Qwen3-TTS 0.6B CustomVoice | SCOPED_E2E_OMNI_COMPLETE_REQUEST | The earlier vLLM 0.28 CPU path passed four short streams but had four simulated playback underruns in one measured request; its wheel remains mismatched with Omni 0.29. Separately, the pinned BF16 0.6B CustomVoice checkpoint ran two named text-to-complete-WAV requests through an Omni 0.29 whole-session stage with an isolated CPU worker on WSL. One warmup plus 20 serial measured requests returned identical 3.44 s audio; nearest-rank complete-request wall p50/p95 was 8.27/8.47 s (RTF 2.40/2.46), excluding 19.82 s startup. Pinned Whisper tiny.en transcribed the first measured WAV exactly (WER 0, one-output intelligibility proxy). Public `AsyncOmni` audio output passed; the stage verified weights, interpreter and CPU placement, reserved 10 GiB host RAM, refused 4 GiB before launch, and drained in-flight cancellation with zero remaining reservation. WSL PCM differs from the same-checkpoint native Windows outputs despite the declared settings; its cause is unqualified. Sampled process-tree RSS reached 3.39 GB, not a loading peak. This worker delivers only a complete WAV, not a playable stream. | Qualify cross-OS numerical/quality difference, playable streaming without underruns, post-cancel restart, loading peak, concurrency and sustained power/thermal behavior; [WSL Omni raw evidence](../../../e2e_expansion_20260923/evidence/qwen_tts_omni_wsl_cpu/README.md), [earlier stream evidence](../../../e2e_recovery_20260923/README.md). |
| PC HX370: CPU / WSL | Qwen3.8-27B | NOT_E2E_PROFILEABLE | No compatible CPU artifact/deployment for this model was found in the checked model roots. Available Qwen 27B FP8/NVFP4 CUDA artifacts do not establish a CPU route. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC HX370: CPU / WSL | MiniCPM-o 4.5 | SCOPED_E2E_OMNI_COMPLETE_REQUEST | The BF16/vLLM 0.28 three-stage path previously passed separate text, synthetic red-square image and synthetic 440 Hz audio inputs to text+speech; its 20-request serial **text-only** profile was p50/p95 21.86/22.97 s with swap and shutdown warnings. A separate pinned Q4_K_M/F16 GGUF worker now runs a **combined spoken question+image → text+speech** request through a bounded Omni StageRuntime/StagePool graph stage on WSL CPU, with installed vLLM 0.29 for the controller and CPU-only C++ model execution. All 20 measured requests after one warmup answered “The square in the image is red.”, emitted complete nonzero speech and a terminal audio event; nearest-rank complete-request wall p50/p95 was 22.67/24.83 s, speech duration 1.60–2.92 s, and maximum sampled worker RSS 13.70 GB. The first measured WAV independently transcribed exactly with pinned Whisper tiny.en (WER 0, one-output intelligibility proxy). Ten weight hashes and CPU placement were verified; 24 GiB shared RAM was reserved, 8 GiB was refused before launch, and in-flight cancellation left no stale output or reservation. Host RAM/swap samples covered only part of the profile. One synthetic fixture and variable speech PCM do not establish general quality, playable streaming, loading peak or sustained performance. | Validate real image/audio suites, speech alignment and quality, video input, concurrent admission, post-cancel restart, loading peak and sustained thermal behavior; [GGUF Omni raw evidence](../../../e2e_expansion_20260923/evidence/minicpmo_omni_wsl_cpu/README.md), [BF16/vLLM CPU evidence](../../../e2e_expansion_20260923/README.md), [earlier standalone GGUF run](../../../e2e_expansion_20260923/evidence/minicpmo_cpp_cpu/README.md). |
| PC HX370: CPU / WSL | InternVLA-A1 | SYNTHETIC_POLICY_PASS | The Place_Markpen checkpoint strict-loaded in CPU-only PyTorch and produced finite `[1, 50, 32]` actions from zero observations/noise in one 3.25 s forward. Compared with the same CUDA checkpoint/input, relative L2 error was 1.30% and cosine similarity 0.999917; no task-quality tolerance is established. | Validate real observations against reference actions, units/timestamps, repeated latency and memory before robot-task qualification; [CPU evidence](../../../e2e_expansion_20260923/README.md). |
| PC HX370: CPU / native Windows | Spark-X2.5 | SCOPED_E2E_OMNI_COMPLETE_REQUEST | The pinned Spark-X2.5-1.7B Q4_K_M GGUF ran in native Windows llama.cpp CPU (`-dev none -ngl 0`) as a whole-session Omni StageRuntime/StagePool backend. `Paris` and the correct `1511` inventory answer passed; one warmup plus 20 serial complete requests were 20/20 correct, nearest-rank wall p50/p95 6.93/7.05 s. The public AsyncOmni short request passed. The stage verifies artifact hashes and CPU placement (0/29 layers offloaded), reserves 4 GiB host RAM, emits a terminal text event and releases reservation after in-flight cancellation. The separate prior standalone profile was 6.65/6.86 s. No vLLM CPU attention repair, incremental stream, restart-after-cancel, loading peak or sustained power/thermal gate is claimed. | Validate broader text/token quality, streaming and cancellation recovery, memory peak and sustained load; [Omni raw evidence](../../../e2e_expansion_20260923/evidence/spark_omni_llamacpp/README.md), [standalone evidence](../../../e2e_expansion_20260923/evidence/spark_radeon890m_llamacpp/README.md). |
| PC HX370: CPU / native Windows | Qwen3-TTS 0.6B CustomVoice | SCOPED_E2E_OMNI_COMPLETE_REQUEST | The pinned 0.6B CustomVoice BF16/SDPA checkpoint ran wholly on CPU through an isolated Qwen wrapper worker owned by one bounded Omni StageRuntime stage. Two named 24 kHz WAVs matched the earlier standalone PCM SHA256 values exactly. One warmup + 20 serial complete Omni requests for 4.56 s of audio had 20/20 identical PCM and nearest-rank wall p50/p95 15.32/15.60 s (RTF 3.36/3.42); independent Whisper tiny.en ASR gave WER 0.20 on the first Omni WAV, an intelligibility proxy only. A public AsyncOmni audio request passed. The stage verified pinned weights/interpreter and CPU placement, reserved 10 GiB host RAM, rejected an insufficient 4 GiB demand before worker launch, and after in-flight cancellation emitted no stale output and returned the ledger to zero. Sampled process-tree private memory peaked at 5.39 GB, not a loading peak. Native vLLM CPU attention operators remain absent; playable streaming, post-cancel restart, broader speech quality, loading peak and sustained power/thermal behavior remain unverified. | Validate playable streaming and restart/recovery, broader voice quality, peak memory, concurrency and sustained behavior; [Omni raw evidence](../../../e2e_expansion_20260923/evidence/qwen_tts_omni_cpu/README.md), [standalone baseline](../../../e2e_expansion_20260923/evidence/qwen_tts_native_cpu/README.md). |
| PC HX370: CPU / native Windows | Qwen3.8-27B | NOT_E2E_PROFILEABLE | The installed native Windows vLLM extension lacks cpu_attn_reshape_and_cache and cpu_attn_get_scheduler_metadata. The separate ORT CPU graph worker is not a whole-model text/TTS adapter; a qualified CPU build/binding is missing. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC HX370: CPU / native Windows | MiniCPM-o 4.5 | SCOPED_E2E_OMNI_COMPLETE_REQUEST | The pinned Q4_K_M/F16 GGUF set ran one combined synthetic JPEG+spoken-question → text+speech request through a bounded Omni whole-session graph stage on native Windows CPU. It answered “The square in the image is red.” and produced 2.24 s of 24 kHz speech in 24.34 s complete-request wall time (one cold sample, zero warmups); sampled worker RSS peaked at 13.11 GB. The stage verified ten weight hashes and the CLI hash, observed CPU placement, reserved 24 GiB shared RAM, and emitted a terminal audio event. An 8 GiB reservation was refused before worker launch; in-flight cancellation left no stale output and released the ledger. The prior standalone prompted run took 24.66 s and independently transcribed output with WER 0. This is a whole-request C++ worker under Omni control, not native vLLM execution or playable streaming; one synthetic quality case is not general qualification. | Validate real image/audio suites, playable streaming, post-cancel restart, loading peak, repeated latency and sustained thermal behavior. [Omni raw evidence](../../../e2e_expansion_20260923/evidence/minicpmo_omni_cpp/README.md), [standalone prompted evidence](../../../e2e_expansion_20260923/evidence/minicpmo_cpp_windows_cpu_prompted/README.md), [failed first run](../../../e2e_expansion_20260923/evidence/minicpmo_cpp_windows_cpu_attempt1/README.md). |
| PC HX370: CPU / native Windows | InternVLA-A1 | SYNTHETIC_POLICY_PASS | The Place_Markpen checkpoint strict-loaded from a UNC path and the direct CPU policy produced finite `[1, 50, 32]` actions on synthetic zero observations/noise. Policy/output CPU placement was checked; one forward took 3.36 s. Relative L2 difference from WSL CPU was 1.08% (no task tolerance). This direct policy route does not require native vLLM CPU attention operators. | Validate real observations/reference actions, units/timestamps, memory and repeated latency before task qualification; [native evidence](../../../e2e_expansion_20260923/README.md). |
| PC HX370 + RTX 5090 Laptop / WSL | Spark-X2.5 | PROFILE_COMPLETED | not fully qualified: reference-quality and remaining reliability/performance gates must be reviewed | Broaden only with a named workload/device/precision qualification. |
| PC HX370 + RTX 5090 Laptop / WSL | Qwen3-TTS 0.6B CustomVoice | PROFILE_COMPLETED | not fully qualified: reference-quality and remaining reliability/performance gates must be reviewed | Resolve and measure playback stalls; run quality, interruption and thermal tests. |
| PC HX370 + RTX 5090 Laptop / WSL | Qwen3.8-27B | SCOPED_E2E | A new one-stage Omni binding ran the real NVFP4 checkpoint: one 16-token text request and one synthetic red-image request (answer `Red`) at an explicit 512-token context with host offload. This is not the full multimodal, long-context or quality workload. | Qualify image/video suites, longer contexts, latency distributions and sustained power; [recovery evidence](../../../e2e_recovery_20260923/README.md). |
| PC HX370 + RTX 5090 Laptop / WSL | MiniCPM-o 4.5 | SCOPED_E2E | The verified checkpoint completed three-stage text, one synthetic image and one synthetic audio input to text+speech under separate constrained plans with 5 GiB host offload. The image response correctly described a 448×448 red square on white and produced a nonzero 5.2 s WAV in 17.49 s; a 2 s/440 Hz tone was described as “A loud whistling sound” with a nonzero 2.16 s WAV in 7.78 s. The original 0.58 thinker GPU budget failed image/KV admission; 0.61 admitted the same context and image. Image shutdown force-killed stage 0, and both multimodal runs reported shared-memory cleanup warnings. General image/audio quality and speech meaning remain unverified. | Validate real image/audio suites, video input, speech intelligibility, interruption, repeated latency, clean shutdown and sustained power; [text evidence](../../../e2e_recovery_20260923/README.md), [image/audio evidence](../../../e2e_expansion_20260923/README.md). |
| PC HX370 + RTX 5090 Laptop / WSL | InternVLA-A1 | SYNTHETIC_POLICY_PASS | Strict real-weight Place_Markpen load and one policy forward on zero observations/noise produced finite `[1, 50, 32]` actions in 1.222 s. Synthetic input and one timing sample do not establish robot-task E2E or action semantics. | Run real observation/reference-action cases and validate units, timestamps and deadline behavior; [raw result](../../../e2e_recovery_20260923/evidence/internvla/internvla_finetune_synthetic.json). |
| PC HX370 + RTX 5090 Laptop / Windows | Spark-X2.5 | PROFILE_COMPLETED | not fully qualified: reference-quality and remaining reliability/performance gates must be reviewed | Broaden only with a named workload/device/precision qualification. |
| PC HX370 + RTX 5090 Laptop / Windows | Qwen3-TTS 0.6B CustomVoice | PROFILE_COMPLETED | not fully qualified: reference-quality and remaining reliability/performance gates must be reviewed | Resolve and measure playback stalls; run quality, interruption and thermal tests. |
| PC HX370 + RTX 5090 Laptop / Windows | Qwen3.8-27B | NOT_E2E_PROFILEABLE | The Omni pipeline binding now exists and passed scoped text/image cases in WSL, but no native Windows Omni run has verified the NVFP4 checkpoint, memory admission or modality behavior. | Run the same checkpoint and explicit context on native Windows, then broaden quality and profiling; [WSL-only evidence](../../../e2e_recovery_20260923/README.md). |
| PC HX370 + RTX 5090 Laptop / Windows | MiniCPM-o 4.5 | SCOPED_E2E | The verified checkpoint completed native Windows three-stage BF16 text, one synthetic image, and one synthetic audio input to text+speech in separate constrained plans with 5 GiB thinker host offload and explicit 2,048/512-token limits. Text and all 32 thinker token IDs matched WSL CUDA; a 5.2 s WAV was nonzero. The red-square image produced the same description and 27 thinker token IDs as WSL plus a nonzero 5.2 s WAV in 17.15 s. The 2 s/440 Hz tone produced the same whistling description and nine IDs as WSL plus a nonzero 2.16 s WAV in 8.26 s. The initial text run needed 900/600 s startup limits after a 300 s timeout; a separate 1-warmup/20-measured serial text run had request wall p50 18.38 s/p95 18.73 s, with sampled host available RAM as low as 0.78 GB and pagefile use up to 11.17 GB. These synthetic inputs do not establish general image/audio or speech quality. | Check real image/audio suites, video input, speech intelligibility/alignment, concurrency, interruption, loading-peak admission and sustained thermal/power behavior before broader qualification; [native evidence](../../../e2e_expansion_20260923/README.md). |
| PC HX370 + RTX 5090 Laptop / Windows | InternVLA-A1 | SYNTHETIC_POLICY_PASS | The Place_Markpen checkpoint strict-loaded from a UNC path and produced finite `[1, 50, 32]` actions on synthetic zero observations/noise. Policy/output CUDA placement was checked; one native forward took 1.12 s, with 6.68 GB peak allocated. Relative L2 difference from WSL CUDA was 0.83% (no task tolerance). | Validate real observations/reference actions, units/timestamps, memory and repeated latency before task qualification; [native evidence](../../../e2e_expansion_20260923/README.md). |
| PC HX370 + Radeon 890M / Windows worker | Spark-X2.5 | SCOPED_E2E_OMNI_COMPLETE_REQUEST | The same pinned Spark-X2.5-1.7B Q4_K_M GGUF completed text prefill/decode through a bounded Omni whole-session stage with llama.cpp Vulkan1. The log identifies Radeon 890M and 29/29 layers offloaded there, rather than RTX 5090; an intentionally wrong device name was refused. `Paris` and `1511` passed; one warmup plus 20 serial complete requests were 20/20 correct, nearest-rank wall p50/p95 2.10/2.14 s. A public AsyncOmni request passed; an oversized context was rejected with a later correct request; in-flight cancellation produced no stale output and released the ledger. The prior standalone run was 2.18/2.20 s. No streaming, restart-after-cancel, GPU-memory peak or sustained power/thermal check is claimed. | Qualify public configuration, broader text/token quality, streaming and cancellation recovery, shared-RAM/GPU peak and sustained use; [Omni raw evidence](../../../e2e_expansion_20260923/evidence/spark_omni_llamacpp/README.md), [standalone evidence](../../../e2e_expansion_20260923/evidence/spark_radeon890m_llamacpp/README.md). |
| PC HX370 + Radeon 890M / Windows worker | Qwen3-TTS 0.6B CustomVoice | SCOPED_E2E_OMNI_COMPLETE_REQUEST_HYBRID | The pinned Q8_0 CustomVoice talker, F16 codec and punctuation GGUFs completed two named text-to-WAV requests through an Omni whole-session CrispASR stage on native Windows. The runtime log identifies Radeon 890M Vulkan0 for talker/codec and the 432 MB code predictor pinned to CPU FP32; this is not all-Radeon or NPU execution. Both PCM hashes exactly match the earlier standalone outputs. Whisper tiny.en transcribed the first Omni WAV and both standalone WAVs exactly (WER 0 as an intelligibility proxy). One warmup plus 20 serial complete Omni requests for 2.56 s of audio had identical PCM and nearest-rank wall p50/p95 3.298/3.336 s (RTF 1.29/1.30). A public AsyncOmni audio request passed; cancellation drained the worker without stale output or remaining reservation. The earlier standalone profile was 3.310/3.345 s. The default Vulkan configuration previously fell back to CPU, all-Vulkan ran away to a 360-frame cap, and DirectML failed waveform quality. No playable stream, post-cancel restart, loading peak, broad speech quality or sustained power/thermal run is claimed. | Validate playable streaming, restart/recovery, broader voice quality, shared-RAM/GPU peak, concurrency and sustained real-time behavior before wider qualification; [Omni raw evidence](../../../e2e_expansion_20260923/evidence/qwen_tts_omni_hybrid/README.md), [standalone evidence](../../../e2e_expansion_20260923/evidence/qwen_tts_radeon890m_joint/README.md), [failed DirectML route](../../../e2e_expansion_20260923/evidence/qwen_tts_radeon890m_directml/README.md). |
| PC HX370 + Radeon 890M / Windows worker | Qwen3.8-27B | NOT_E2E_PROFILEABLE | FP32 448x448 vision tower: DirectML 85/113 nodes; normalized max error 9.095e-06 versus torch CPU. Historical component path, not full 27B or current graph-stage image-to-text integration. | Recover the hashed tower bundle and qualify its current Omni graph handoff with the language stage. |
| PC HX370 + Radeon 890M / Windows worker | MiniCPM-o 4.5 | SCOPED_E2E_OMNI_COMPLETE_REQUEST_HYBRID | The same pinned GGUF set ran one combined synthetic spoken-question+JPEG → text+speech request through a bounded Omni whole-session graph stage. Logs verify `GGML_VK_VISIBLE_DEVICES=1`, Radeon 890M Vulkan0 language (37/37 layers) and vision, with CPU TTS/Token2Wav/vocoder. It answered “The square in the image is red.” and produced 2.60 s of 24 kHz speech in 18.27 s complete-request wall time (one cold sample, zero warmups); sampled worker RSS peaked at 15.57 GB. The stage verified artifact and binary hashes, reserved 24 GiB shared RAM, emitted a terminal audio event, and in-flight cancellation left no stale output or reservation. The prior standalone prompted run took 18.63 s and its output transcribed with WER 0. This is CPU+iGPU joint execution, not all-Radeon or NPU execution; neither run qualifies general quality or sustained performance. | Validate real multimodal quality, playable streaming, post-cancel restart, loading peak, repeated/concurrent latency and sustained thermal behavior. [Omni raw evidence](../../../e2e_expansion_20260923/evidence/minicpmo_omni_cpp/README.md), [standalone prompted evidence](../../../e2e_expansion_20260923/evidence/minicpmo_cpp_radeon890m_prompted/README.md). |
| PC HX370 + Radeon 890M / Windows worker | InternVLA-A1 | SYNTHETIC_POLICY_PASS_HYBRID; NOT_TASK_E2E | The real Place_Markpen checkpoint strict-loaded in WSL CPU Omni diffusion. Its Cosmos encoder ran as a pinned FP32 fixed-six-frame export in Omni's external DirectML worker; the selected device was verified as Radeon 890M, and the remainder of the BF16 policy/action loop stayed on CPU. One warmup plus 20 alternating identical synthetic zero-observation/noise requests produced finite `[1,50,32]` actions. Complete-request wall p50/p95 was 2.952/3.322 s hybrid versus 3.178/3.266 s CPU; 16/20 paired calls were faster, but the hybrid tail regressed. Action relative L2 difference was 0.9007% and cosine 0.999959, without a task tolerance. DirectML BF16 aborted, so FP32 encoder is an explicit precision change. Worker startup+graph load took 3.34 s, excluded from warm timing. This proves a scoped synthetic CPU+iGPU policy path, not real-observation quality, action units/time, complete shared-RAM admission, cancellation or sustained benefit. | Validate real observation/reference-action quality and metadata, integrate an admitted bounded graph-stage plan with cancellation/restart, and resolve latency-tail and startup amortization before deployment. [Raw hybrid and component evidence](../../../e2e_expansion_20260923/evidence/internvla_cosmos_radeon890m/README.md). |
| PC HX370 + AMD NPU / Windows worker | Spark-X2.5 | COMPONENT_PASS; NOT_E2E_PROFILEABLE | The real 1.7B output head (RMS norm and full 131,072-logit projection) ran through Omni `external.graph.v1` with VitisAI: one fused NPU partition and seven CPU nodes per request. The A16W8 graph matched top-1 on four captured activations (full-logit SNR 55.43–56.81 dB). One warmup plus 20 serial repeated-input component requests gave p50/p95 10.21/12.44 ms, versus 17.97/20.29 ms for the original FP32 output head on the same Omni CPU stage. NPU startup was 18.21 s versus 4.48 s for FP32 CPU, and NPU worker peak RSS was 3.52 GiB versus 2.06 GiB. No full prefill/decode/KV/sampling, real handoff, power, or sustained run was verified. | Measure an actual vLLM-to-output-head handoff and whole-generation cost, state/cancellation, memory admission and quality before considering a split plan; [component evidence](../../../e2e_expansion_20260923/evidence/spark_amd_npu_output_head/README.md). |
| PC HX370 + AMD NPU / Windows worker | Qwen3-TTS 0.6B CustomVoice | NOT_E2E_PROFILEABLE | No placement- and quality-verified whole-model graph bundle/stateful adapter for this model on this AMD route. Small backend graphs do not establish model support. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC HX370 + AMD NPU / Windows worker | Qwen3.8-27B | NOT_E2E_PROFILEABLE | The tested A16W8 whole-vision-tower export failed NPU placement. This conclusion is specific to that export, not all future Qwen/NPU combinations. | Validate a supported export or smaller useful component, including same-checkpoint numerical quality. |
| PC HX370 + AMD NPU / Windows worker | MiniCPM-o 4.5 | NOT_E2E_PROFILEABLE | No placement- and quality-verified whole-model graph bundle/stateful adapter for this model on this AMD route. Small backend graphs do not establish model support. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC HX370 + AMD NPU / Windows worker | InternVLA-A1 | NOT_E2E_PROFILEABLE | No placement- and quality-verified whole-model graph bundle/stateful adapter for this model on this AMD route. Small backend graphs do not establish model support. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC CPU+iGPU+NPU +/- discrete GPU, joint model execution | Spark-X2.5 | NOT_E2E_PROFILEABLE | No qualifying same-model multi-accelerator deployment/artifact plan or overlap/recovery evidence. Individual routes are not a joint pipeline. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC CPU+iGPU+NPU +/- discrete GPU, joint model execution | Qwen3-TTS 0.6B CustomVoice | NOT_E2E_PROFILEABLE | No qualifying same-model multi-accelerator deployment/artifact plan or overlap/recovery evidence. Individual routes are not a joint pipeline. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC CPU+iGPU+NPU +/- discrete GPU, joint model execution | Qwen3.8-27B | NOT_E2E_PROFILEABLE | No qualifying same-model multi-accelerator deployment/artifact plan or overlap/recovery evidence. Individual routes are not a joint pipeline. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC CPU+iGPU+NPU +/- discrete GPU, joint model execution | MiniCPM-o 4.5 | NOT_E2E_PROFILEABLE | No qualifying same-model multi-accelerator deployment/artifact plan or overlap/recovery evidence. Individual routes are not a joint pipeline. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC CPU+iGPU+NPU +/- discrete GPU, joint model execution | InternVLA-A1 | NOT_E2E_PROFILEABLE | No qualifying same-model multi-accelerator deployment/artifact plan or overlap/recovery evidence. Individual routes are not a joint pipeline. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC Snapdragon X Elite CRD / AI Hub | Spark-X2.5 | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC Snapdragon X Elite CRD / AI Hub | Qwen3-TTS 0.6B CustomVoice | NOT_E2E_PROFILEABLE | Historical QNN talker profile re-fetched successfully; no numerical/full-pipeline gate. Historical TFLite vocoder target was refused on this Windows ARM device. | Validate target-specific numerical quality, complete local execution, memory and sustained behavior. |
| PC Snapdragon X Elite CRD / AI Hub | Qwen3.8-27B | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC Snapdragon X Elite CRD / AI Hub | MiniCPM-o 4.5 | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| PC Snapdragon X Elite CRD / AI Hub | InternVLA-A1 | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy | Spark-X2.5 | NOT_E2E_PROFILEABLE | Fresh 2026-09-22 W8A16 full-attention component inference/profile passed; full1024 denotes one cache bucket. No full prefill/decode/KV/ring/sampling loop. | Implement and validate local 128-token generation, 512/1024 transitions, cancellation and resident memory. |
| Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy | Qwen3-TTS 0.6B CustomVoice | NOT_E2E_PROFILEABLE | Fresh FP16 predictor inference passed; its one output exactly matches the same compiled artifact/dataset reference. Historical decoder parity exists. No complete co-resident NPU/GPU stream. | Validate the device-local talker/predictor/vocoder pipeline, output quality, recovery and 30-minute thermal behavior. |
| Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy | Qwen3.8-27B | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy | MiniCPM-o 4.5 | NOT_E2E_PROFILEABLE | Fresh FP16 speech-head inference passed; all 41 outputs exactly match the same artifact/dataset reference. Historical FP16 numerical parity passed; tested W8A16 top-1 parity failed. Thinker/encoders/vocoder are not qualified together. | Supply a qualified complete mobile checkpoint/export and validate the full multimodal/stateful pipeline. |
| Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy | InternVLA-A1 | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Mobile Galaxy S24 / Snapdragon 8 Gen 3 | Spark-X2.5 | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Mobile Galaxy S24 / Snapdragon 8 Gen 3 | Qwen3-TTS 0.6B CustomVoice | NOT_E2E_PROFILEABLE | Historical FP16 decoder parity has SNR 39.13 dB. Tested calibrated INT8 decoder parity failed with negative SNR. Neither result is a full local TTS stream. | Validate the complete local pipeline and a quality-passing artifact set. |
| Mobile Galaxy S24 / Snapdragon 8 Gen 3 | Qwen3.8-27B | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Mobile Galaxy S24 / Snapdragon 8 Gen 3 | MiniCPM-o 4.5 | NOT_E2E_PROFILEABLE | Historical FP16 speech-head job re-fetched successfully; SNR 49.22 dB/top-1 match in the recorded parity test. W8A16 top-1 failed. Not complete MiniCPM-o. | Qualify thinker/encoders/vocoder and stateful local integration. |
| Mobile Galaxy S24 / Snapdragon 8 Gen 3 | InternVLA-A1 | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Embedded SA8775P ADP / AI Hub | Spark-X2.5 | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Embedded SA8775P ADP / AI Hub | Qwen3-TTS 0.6B CustomVoice | NOT_E2E_PROFILEABLE | Historical NPU talker and GPU vocoder profiles re-fetched successfully; no complete pipeline, parity or co-residency qualification. | Validate target-specific numerical quality, complete local execution, memory and sustained behavior. |
| Embedded SA8775P ADP / AI Hub | Qwen3.8-27B | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Embedded SA8775P ADP / AI Hub | MiniCPM-o 4.5 | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Embedded SA8775P ADP / AI Hub | InternVLA-A1 | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Embedded RB3 Gen 2 / QCS6490 / AI Hub | Spark-X2.5 | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Embedded RB3 Gen 2 / QCS6490 / AI Hub | Qwen3-TTS 0.6B CustomVoice | NOT_E2E_PROFILEABLE | Historical GPU vocoder profile re-fetched successfully. The attempted NPU talker build was rejected for floating-point inputs. No full model pipeline. | Validate target-specific numerical quality, complete local execution, memory and sustained behavior. |
| Embedded RB3 Gen 2 / QCS6490 / AI Hub | Qwen3.8-27B | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Embedded RB3 Gen 2 / QCS6490 / AI Hub | MiniCPM-o 4.5 | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |
| Embedded RB3 Gen 2 / QCS6490 / AI Hub | InternVLA-A1 | NOT_E2E_PROFILEABLE | No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device. | Supply and validate the missing matching artifact/backend, then run the complete specified workload. |

## Measured request timing

Nearest-rank p50/p95; warmups and sustained requests excluded. Values are observations under recorded power/cache conditions, not latency targets. Spark CPU uses 1.7B INT8; CUDA uses 4B BF16, so these are not same-model acceleration comparisons.

### PC HX370: CPU / WSL — Spark-X2.5

Status: completed. Full timing protocol complete: True. Startup: 20.767 s. Sustained run: 1805.768 s.

| Length | Concurrent requests | n | Metric | p50 | p95 |
|---|---:|---:|---|---:|---:|
| long | 1 | 20 | ttft_s | 3.377 | 3.668 |
| long | 1 | 20 | wall_s | 11.838 | 15.504 |
| long | 1 | 20 | decode_tok_per_s | 14.923 | 15.631 |
| long | 2 | 20 | ttft_s | 3.630 | 6.942 |
| long | 2 | 20 | wall_s | 15.120 | 17.792 |
| long | 2 | 20 | decode_tok_per_s | 11.509 | 15.800 |
| long | 4 | 20 | ttft_s | 8.034 | 14.609 |
| long | 4 | 20 | wall_s | 22.493 | 26.072 |
| long | 4 | 20 | decode_tok_per_s | 8.815 | 15.010 |
| medium | 1 | 20 | ttft_s | 0.762 | 0.801 |
| medium | 1 | 20 | wall_s | 8.830 | 9.446 |
| medium | 1 | 20 | decode_tok_per_s | 15.663 | 16.069 |
| medium | 2 | 20 | ttft_s | 0.824 | 1.584 |
| medium | 2 | 20 | wall_s | 9.409 | 10.202 |
| medium | 2 | 20 | decode_tok_per_s | 15.142 | 16.564 |
| medium | 4 | 20 | ttft_s | 1.618 | 3.173 |
| medium | 4 | 20 | wall_s | 11.504 | 11.861 |
| medium | 4 | 20 | decode_tok_per_s | 13.201 | 15.387 |
| short | 1 | 20 | ttft_s | 0.148 | 0.175 |
| short | 1 | 20 | wall_s | 8.234 | 9.761 |
| short | 1 | 20 | decode_tok_per_s | 15.704 | 16.269 |
| short | 2 | 20 | ttft_s | 0.232 | 0.301 |
| short | 2 | 20 | wall_s | 7.996 | 10.014 |
| short | 2 | 20 | decode_tok_per_s | 16.237 | 16.804 |
| short | 4 | 20 | ttft_s | 0.447 | 0.571 |
| short | 4 | 20 | wall_s | 8.661 | 10.261 |
| short | 4 | 20 | decode_tok_per_s | 15.451 | 16.508 |

### PC HX370 + RTX 5090 Laptop / WSL — Spark-X2.5

Status: completed. Full timing protocol complete: True. Startup: 23.693 s. Sustained run: 1802.602 s.

| Length | Concurrent requests | n | Metric | p50 | p95 |
|---|---:|---:|---|---:|---:|
| long | 1 | 20 | ttft_s | 0.363 | 0.386 |
| long | 1 | 20 | wall_s | 2.987 | 3.163 |
| long | 1 | 20 | decode_tok_per_s | 48.210 | 50.041 |
| long | 2 | 20 | ttft_s | 0.457 | 0.828 |
| long | 2 | 20 | wall_s | 3.566 | 3.982 |
| long | 2 | 20 | decode_tok_per_s | 41.427 | 45.763 |
| long | 4 | 20 | ttft_s | 0.848 | 1.430 |
| long | 4 | 20 | wall_s | 4.175 | 4.354 |
| long | 4 | 20 | decode_tok_per_s | 38.124 | 45.331 |
| medium | 1 | 20 | ttft_s | 0.090 | 0.103 |
| medium | 1 | 20 | wall_s | 2.810 | 2.987 |
| medium | 1 | 20 | decode_tok_per_s | 46.545 | 48.369 |
| medium | 2 | 20 | ttft_s | 0.097 | 0.192 |
| medium | 2 | 20 | wall_s | 2.847 | 3.007 |
| medium | 2 | 20 | decode_tok_per_s | 46.499 | 48.199 |
| medium | 4 | 20 | ttft_s | 0.185 | 0.369 |
| medium | 4 | 20 | wall_s | 3.031 | 3.065 |
| medium | 4 | 20 | decode_tok_per_s | 44.901 | 47.283 |
| short | 1 | 20 | ttft_s | 0.035 | 0.045 |
| short | 1 | 20 | wall_s | 2.552 | 2.703 |
| short | 1 | 20 | decode_tok_per_s | 50.203 | 51.126 |
| short | 2 | 20 | ttft_s | 0.047 | 0.064 |
| short | 2 | 20 | wall_s | 2.576 | 2.604 |
| short | 2 | 20 | decode_tok_per_s | 50.216 | 50.462 |
| short | 4 | 20 | ttft_s | 0.065 | 0.069 |
| short | 4 | 20 | wall_s | 2.655 | 2.675 |
| short | 4 | 20 | decode_tok_per_s | 48.944 | 49.154 |

### PC HX370 + RTX 5090 Laptop / WSL — Qwen3-TTS 0.6B CustomVoice

Status: completed. Full timing protocol complete: True. Startup: 345.168 s. Sustained run: 1801.183 s.

| Length | Concurrent requests | n | Metric | p50 | p95 |
|---|---:|---:|---|---:|---:|
| long | 1 | 20 | ttfa_ms | 68.518 | 100.566 |
| long | 1 | 20 | rtf_total | 0.176 | 0.183 |
| long | 1 | 20 | stall_at_ttfa_ms | 0.000 | 0.000 |
| long | 2 | 20 | ttfa_ms | 79.379 | 138.556 |
| long | 2 | 20 | rtf_total | 0.215 | 0.217 |
| long | 2 | 20 | stall_at_ttfa_ms | 0.000 | 0.000 |
| long | 4 | 20 | ttfa_ms | 117.755 | 144.011 |
| long | 4 | 20 | rtf_total | 0.259 | 0.274 |
| long | 4 | 20 | stall_at_ttfa_ms | 0.000 | 0.000 |
| medium | 1 | 20 | ttfa_ms | 73.956 | 111.604 |
| medium | 1 | 20 | rtf_total | 0.175 | 0.182 |
| medium | 1 | 20 | stall_at_ttfa_ms | 0.000 | 0.000 |
| medium | 2 | 20 | ttfa_ms | 82.497 | 120.043 |
| medium | 2 | 20 | rtf_total | 0.219 | 0.224 |
| medium | 2 | 20 | stall_at_ttfa_ms | 0.000 | 0.000 |
| medium | 4 | 20 | ttfa_ms | 127.905 | 160.158 |
| medium | 4 | 20 | rtf_total | 0.274 | 0.285 |
| medium | 4 | 20 | stall_at_ttfa_ms | 0.000 | 0.000 |
| short | 1 | 20 | ttfa_ms | 67.583 | 91.833 |
| short | 1 | 20 | rtf_total | 0.203 | 0.215 |
| short | 1 | 20 | stall_at_ttfa_ms | 0.000 | 0.000 |
| short | 2 | 20 | ttfa_ms | 95.470 | 120.562 |
| short | 2 | 20 | rtf_total | 0.246 | 0.255 |
| short | 2 | 20 | stall_at_ttfa_ms | 0.000 | 0.000 |
| short | 4 | 20 | ttfa_ms | 118.914 | 145.742 |
| short | 4 | 20 | rtf_total | 0.302 | 0.323 |
| short | 4 | 20 | stall_at_ttfa_ms | 0.000 | 0.000 |

### PC HX370 + RTX 5090 Laptop / Windows — Spark-X2.5

Status: completed. Full timing protocol complete: True. Startup: 95.155 s. Sustained run: 1803.249 s.

| Length | Concurrent requests | n | Metric | p50 | p95 |
|---|---:|---:|---|---:|---:|
| long | 1 | 20 | ttft_s | 0.363 | 0.415 |
| long | 1 | 20 | wall_s | 3.045 | 3.986 |
| long | 1 | 20 | decode_tok_per_s | 47.550 | 48.902 |
| long | 2 | 20 | ttft_s | 0.412 | 0.738 |
| long | 2 | 20 | wall_s | 3.383 | 3.497 |
| long | 2 | 20 | decode_tok_per_s | 44.115 | 48.104 |
| long | 4 | 20 | ttft_s | 0.751 | 1.375 |
| long | 4 | 20 | wall_s | 4.009 | 4.211 |
| long | 4 | 20 | decode_tok_per_s | 39.814 | 46.459 |
| medium | 1 | 20 | ttft_s | 0.100 | 0.129 |
| medium | 1 | 20 | wall_s | 2.827 | 3.537 |
| medium | 1 | 20 | decode_tok_per_s | 44.656 | 48.311 |
| medium | 2 | 20 | ttft_s | 0.122 | 0.250 |
| medium | 2 | 20 | wall_s | 3.595 | 4.379 |
| medium | 2 | 20 | decode_tok_per_s | 36.433 | 46.294 |
| medium | 4 | 20 | ttft_s | 0.189 | 0.367 |
| medium | 4 | 20 | wall_s | 3.034 | 3.138 |
| medium | 4 | 20 | decode_tok_per_s | 44.989 | 47.472 |
| short | 1 | 20 | ttft_s | 0.052 | 0.097 |
| short | 1 | 20 | wall_s | 2.677 | 3.256 |
| short | 1 | 20 | decode_tok_per_s | 48.084 | 50.892 |
| short | 2 | 20 | ttft_s | 0.058 | 0.145 |
| short | 2 | 20 | wall_s | 2.652 | 2.816 |
| short | 2 | 20 | decode_tok_per_s | 48.831 | 49.710 |
| short | 4 | 20 | ttft_s | 0.068 | 0.106 |
| short | 4 | 20 | wall_s | 2.642 | 2.710 |
| short | 4 | 20 | decode_tok_per_s | 49.173 | 50.890 |

### PC HX370 + RTX 5090 Laptop / Windows — Qwen3-TTS 0.6B CustomVoice

Status: completed. Full timing protocol complete: True. Startup: 79.792 s. Sustained run: 1804.298 s.

| Length | Concurrent requests | n | Metric | p50 | p95 |
|---|---:|---:|---|---:|---:|
| long | 1 | 20 | ttfa_ms | 98.490 | 311.112 |
| long | 1 | 20 | rtf_total | 0.175 | 0.192 |
| long | 1 | 20 | stall_at_ttfa_ms | 0.000 | 118.271 |
| long | 2 | 20 | ttfa_ms | 131.331 | 344.353 |
| long | 2 | 20 | rtf_total | 0.207 | 0.219 |
| long | 2 | 20 | stall_at_ttfa_ms | 0.000 | 105.641 |
| long | 4 | 20 | ttfa_ms | 228.525 | 299.547 |
| long | 4 | 20 | rtf_total | 0.256 | 0.263 |
| long | 4 | 20 | stall_at_ttfa_ms | 0.000 | 118.746 |
| medium | 1 | 20 | ttfa_ms | 111.521 | 302.522 |
| medium | 1 | 20 | rtf_total | 0.178 | 0.210 |
| medium | 1 | 20 | stall_at_ttfa_ms | 0.000 | 40.773 |
| medium | 2 | 20 | ttfa_ms | 128.019 | 255.811 |
| medium | 2 | 20 | rtf_total | 0.224 | 0.246 |
| medium | 2 | 20 | stall_at_ttfa_ms | 0.000 | 77.119 |
| medium | 4 | 20 | ttfa_ms | 158.285 | 265.148 |
| medium | 4 | 20 | rtf_total | 0.281 | 0.296 |
| medium | 4 | 20 | stall_at_ttfa_ms | 73.005 | 75.003 |
| short | 1 | 20 | ttfa_ms | 103.066 | 206.838 |
| short | 1 | 20 | rtf_total | 0.203 | 0.260 |
| short | 1 | 20 | stall_at_ttfa_ms | 0.000 | 44.172 |
| short | 2 | 20 | ttfa_ms | 123.335 | 137.089 |
| short | 2 | 20 | rtf_total | 0.258 | 0.292 |
| short | 2 | 20 | stall_at_ttfa_ms | 0.000 | 134.419 |
| short | 4 | 20 | ttfa_ms | 161.902 | 299.999 |
| short | 4 | 20 | rtf_total | 0.291 | 0.327 |
| short | 4 | 20 | stall_at_ttfa_ms | 0.000 | 120.313 |

