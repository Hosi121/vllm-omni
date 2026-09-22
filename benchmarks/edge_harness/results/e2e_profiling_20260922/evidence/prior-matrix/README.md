# Mobile / PC × model qualification audit

2026-09-22: **60 named pairings checked**, not 60 successful executions.

PASS = specified full workload passed; PARTIAL = functional output with acceptance gaps; COMPONENT = one model component only; PROFILE_ONLY = component timing without quality; FAILED = attempted execution/startup failed; REJECTED = specific artifact refused; BLOCKED = a named prerequisite is missing. Historical evidence is marked in each finding.

| Device / execution configuration | Spark-X2.5 | Qwen3-TTS 0.6B CustomVoice | Qwen3.8-27B | MiniCPM-o 4.5 | InternVLA-A1 |
|---|---|---|---|---|---|
| PC HX370: CPU / WSL | [PASS](#pc_cpu_wsl-spark) | [FAILED](#pc_cpu_wsl-tts) | [BLOCKED](#pc_cpu_wsl-qwen27) | [BLOCKED](#pc_cpu_wsl-minicpm) | [BLOCKED](#pc_cpu_wsl-vla) |
| PC HX370: CPU / native Windows | [BLOCKED](#pc_cpu_windows-spark) | [BLOCKED](#pc_cpu_windows-tts) | [BLOCKED](#pc_cpu_windows-qwen27) | [BLOCKED](#pc_cpu_windows-minicpm) | [BLOCKED](#pc_cpu_windows-vla) |
| PC HX370 + RTX 5090 Laptop / WSL | [PASS](#pc_cuda_wsl-spark) | [PARTIAL](#pc_cuda_wsl-tts) | [FAILED](#pc_cuda_wsl-qwen27) | [BLOCKED](#pc_cuda_wsl-minicpm) | [BLOCKED](#pc_cuda_wsl-vla) |
| PC HX370 + RTX 5090 Laptop / Windows | [PASS](#pc_cuda_windows-spark) | [PARTIAL](#pc_cuda_windows-tts) | [FAILED](#pc_cuda_windows-qwen27) | [BLOCKED](#pc_cuda_windows-minicpm) | [BLOCKED](#pc_cuda_windows-vla) |
| PC HX370 + Radeon 890M / Windows worker | [BLOCKED](#pc_igpu-spark) | [BLOCKED](#pc_igpu-tts) | [COMPONENT](#pc_igpu-qwen27) | [BLOCKED](#pc_igpu-minicpm) | [BLOCKED](#pc_igpu-vla) |
| PC HX370 + AMD NPU / Windows worker | [BLOCKED](#pc_npu-spark) | [BLOCKED](#pc_npu-tts) | [REJECTED](#pc_npu-qwen27) | [BLOCKED](#pc_npu-minicpm) | [BLOCKED](#pc_npu-vla) |
| PC CPU+iGPU+NPU +/- discrete GPU, joint model execution | [BLOCKED](#pc_joint-spark) | [BLOCKED](#pc_joint-tts) | [BLOCKED](#pc_joint-qwen27) | [BLOCKED](#pc_joint-minicpm) | [BLOCKED](#pc_joint-vla) |
| PC Snapdragon X Elite CRD / AI Hub | [BLOCKED](#pc_xelite-spark) | [PROFILE_ONLY](#pc_xelite-tts) | [BLOCKED](#pc_xelite-qwen27) | [BLOCKED](#pc_xelite-minicpm) | [BLOCKED](#pc_xelite-vla) |
| Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy | [COMPONENT](#mobile_s25-spark) | [COMPONENT](#mobile_s25-tts) | [BLOCKED](#mobile_s25-qwen27) | [COMPONENT](#mobile_s25-minicpm) | [BLOCKED](#mobile_s25-vla) |
| Mobile Galaxy S24 / Snapdragon 8 Gen 3 | [BLOCKED](#mobile_s24-spark) | [COMPONENT](#mobile_s24-tts) | [BLOCKED](#mobile_s24-qwen27) | [COMPONENT](#mobile_s24-minicpm) | [BLOCKED](#mobile_s24-vla) |
| Embedded SA8775P ADP / AI Hub | [BLOCKED](#embedded_sa8775p-spark) | [PROFILE_ONLY](#embedded_sa8775p-tts) | [BLOCKED](#embedded_sa8775p-qwen27) | [BLOCKED](#embedded_sa8775p-minicpm) | [BLOCKED](#embedded_sa8775p-vla) |
| Embedded RB3 Gen 2 / QCS6490 / AI Hub | [BLOCKED](#embedded_rb3-spark) | [PROFILE_ONLY](#embedded_rb3-tts) | [BLOCKED](#embedded_rb3-qwen27) | [BLOCKED](#embedded_rb3-minicpm) | [BLOCKED](#embedded_rb3-vla) |

## Scope and reproduction

PC rows are execution configurations on the HX370/890M/AMD NPU/RTX 5090 Laptop machine. X Elite is a separate Windows ARM device accessed through AI Hub. The two embedded kits are not phones. Other SKUs are outside this finite audit, not implied supported.

The checked roots contain MiniCPM5-2B, which must not substitute for MiniCPM-o 4.5. Missing complete MiniCPM-o and InternVLA artifacts block those local runs. ADB reports no attached phone; Hub component jobs cannot prove a device-local controller or sustained co-residency.

The CPU venv is vLLM 0.28; the CUDA/native-Windows venvs are 0.29. The CPU TTS tuple failure is specific to the mixed-version environment. Preserve failures instead of describing these as universal model limitations.

Raw selected reports are in `evidence/`. `matrix.json` retains every source path/hash, cell finding and next step. Fresh probe scripts are in `repro/`; see their `--help` and the workspace experiment README for exact environments. No model weights or API credentials are included.

## Per-pair results

<a id="pc_cpu_wsl-spark"></a>
### PC HX370: CPU / WSL × Spark-X2.5

**PASS — P**. Fresh Spark 1.7B INT8 acceptance: 12 prompts x 128 tokens; clean cancellation. Current Omni branch with the existing vLLM 0.28 CPU wheel; version mismatch warning retained, not a claim about a 0.29 CPU wheel.

Evidence: [spark-cpu](evidence/spark-cpu.json), [spark-cpu-log](evidence/spark-cpu-log.log).

Next: Qualify a matching vLLM 0.29 CPU build and token parity separately.

<a id="pc_cpu_wsl-tts"></a>
### PC HX370: CPU / WSL × Qwen3-TTS 0.6B CustomVoice

**FAILED — execution**. Fresh real-weight two-frame probe loaded both stages but generation failed in sanitize_min_tokens_stop_ids: expected four tuple fields, got three. Existing CPU wheel is vLLM 0.28 versus Omni 0.29. No audio pass.

Evidence: [tts-cpu](evidence/tts-cpu.json), [tts-cpu-log](evidence/tts-cpu-log.log).

Next: Use a matching CPU runtime or explicitly validated compatibility fix; repeat real audio, then full streaming gates.

<a id="pc_cpu_wsl-qwen27"></a>
### PC HX370: CPU / WSL × Qwen3.8-27B

**BLOCKED — none**. No compatible CPU artifact/deployment for this model was found in the checked model roots. Available Qwen 27B FP8/NVFP4 CUDA artifacts do not establish a CPU route.

Evidence: [host-wsl](evidence/host-wsl.json).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_cpu_wsl-minicpm"></a>
### PC HX370: CPU / WSL × MiniCPM-o 4.5

**BLOCKED — none**. No complete matching checkpoint found in the inventoried workspace, HF cache and native model roots. MiniCPM5-2B is not MiniCPM-o 4.5. Runtime registration alone is not a model run.

Evidence: [host-wsl](evidence/host-wsl.json), [host-windows](evidence/host-windows.json).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_cpu_wsl-vla"></a>
### PC HX370: CPU / WSL × InternVLA-A1

**BLOCKED — none**. No complete matching checkpoint found in the inventoried workspace, HF cache and native model roots. MiniCPM5-2B is not MiniCPM-o 4.5. Runtime registration alone is not a model run.

Evidence: [host-wsl](evidence/host-wsl.json), [host-windows](evidence/host-windows.json).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_cpu_windows-spark"></a>
### PC HX370: CPU / native Windows × Spark-X2.5

**BLOCKED — none**. The installed native Windows vLLM extension lacks cpu_attn_reshape_and_cache and cpu_attn_get_scheduler_metadata. The separate ORT CPU graph worker is not a whole-model text/TTS adapter; a qualified CPU build/binding is missing.

Evidence: [host-windows](evidence/host-windows.json), [windows-cpu-kernels](evidence/windows-cpu-kernels.json), [backend-factory](evidence/backend-factory.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_cpu_windows-tts"></a>
### PC HX370: CPU / native Windows × Qwen3-TTS 0.6B CustomVoice

**BLOCKED — none**. The installed native Windows vLLM extension lacks cpu_attn_reshape_and_cache and cpu_attn_get_scheduler_metadata. The separate ORT CPU graph worker is not a whole-model text/TTS adapter; a qualified CPU build/binding is missing.

Evidence: [host-windows](evidence/host-windows.json), [windows-cpu-kernels](evidence/windows-cpu-kernels.json), [backend-factory](evidence/backend-factory.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_cpu_windows-qwen27"></a>
### PC HX370: CPU / native Windows × Qwen3.8-27B

**BLOCKED — none**. The installed native Windows vLLM extension lacks cpu_attn_reshape_and_cache and cpu_attn_get_scheduler_metadata. The separate ORT CPU graph worker is not a whole-model text/TTS adapter; a qualified CPU build/binding is missing.

Evidence: [host-windows](evidence/host-windows.json), [windows-cpu-kernels](evidence/windows-cpu-kernels.json), [backend-factory](evidence/backend-factory.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_cpu_windows-minicpm"></a>
### PC HX370: CPU / native Windows × MiniCPM-o 4.5

**BLOCKED — none**. No complete matching checkpoint found in the inventoried workspace, HF cache and native model roots. MiniCPM5-2B is not MiniCPM-o 4.5. Runtime registration alone is not a model run.

Evidence: [host-wsl](evidence/host-wsl.json), [host-windows](evidence/host-windows.json).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_cpu_windows-vla"></a>
### PC HX370: CPU / native Windows × InternVLA-A1

**BLOCKED — none**. No complete matching checkpoint found in the inventoried workspace, HF cache and native model roots. MiniCPM5-2B is not MiniCPM-o 4.5. Runtime registration alone is not a model run.

Evidence: [host-wsl](evidence/host-wsl.json), [host-windows](evidence/host-windows.json).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_cuda_wsl-spark"></a>
### PC HX370 + RTX 5090 Laptop / WSL × Spark-X2.5

**PASS — P**. Spark 4B BF16: 11 M0 tests passed on this branch for text, state, cancellation and memory; scoped to the recorded workload.

Evidence: [spark-cuda-wsl](evidence/spark-cuda-wsl.log).

Next: Broaden only with a named workload/device/precision qualification.

<a id="pc_cuda_wsl-tts"></a>
### PC HX370 + RTX 5090 Laptop / WSL × Qwen3-TTS 0.6B CustomVoice

**PARTIAL — P-limited**. Real-weight two-stage audio completes. One measured utterance has playback stalls; speech quality, long streaming, interruption/recovery and thermal gates remain open.

Evidence: [tts-cuda-wsl](evidence/tts-cuda-wsl.json).

Next: Resolve and measure playback stalls; run quality, interruption and thermal tests.

<a id="pc_cuda_wsl-qwen27"></a>
### PC HX370 + RTX 5090 Laptop / WSL × Qwen3.8-27B

**FAILED — startup**. Fresh public Omni startup rejects the real NVFP4 checkpoint: no registered Omni pipeline. Historical bare-vLLM short-text success does not qualify this integration or multimodal inference.

Evidence: [qwen-wsl](evidence/qwen-wsl.json), [qwen-text-historical](evidence/qwen-text-historical.json).

Next: Add/validate an Omni pipeline binding; rerun text, image, context and artifact-quality tests.

<a id="pc_cuda_wsl-minicpm"></a>
### PC HX370 + RTX 5090 Laptop / WSL × MiniCPM-o 4.5

**BLOCKED — none**. No complete matching checkpoint found in the inventoried workspace, HF cache and native model roots. MiniCPM5-2B is not MiniCPM-o 4.5. Runtime registration alone is not a model run.

Evidence: [host-wsl](evidence/host-wsl.json), [host-windows](evidence/host-windows.json).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_cuda_wsl-vla"></a>
### PC HX370 + RTX 5090 Laptop / WSL × InternVLA-A1

**BLOCKED — none**. No complete matching checkpoint found in the inventoried workspace, HF cache and native model roots. MiniCPM5-2B is not MiniCPM-o 4.5. Runtime registration alone is not a model run.

Evidence: [host-wsl](evidence/host-wsl.json), [host-windows](evidence/host-windows.json).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_cuda_windows-spark"></a>
### PC HX370 + RTX 5090 Laptop / Windows × Spark-X2.5

**PASS — P**. Spark 4B BF16: 11 M0 tests passed on this branch for text, state, cancellation and memory; scoped to the recorded workload.

Evidence: [spark-cuda-windows](evidence/spark-cuda-windows.log).

Next: Broaden only with a named workload/device/precision qualification.

<a id="pc_cuda_windows-tts"></a>
### PC HX370 + RTX 5090 Laptop / Windows × Qwen3-TTS 0.6B CustomVoice

**PARTIAL — P-limited**. Real-weight two-stage audio completes. One measured utterance has playback stalls; speech quality, long streaming, interruption/recovery and thermal gates remain open.

Evidence: [tts-cuda-windows](evidence/tts-cuda-windows.json).

Next: Resolve and measure playback stalls; run quality, interruption and thermal tests.

<a id="pc_cuda_windows-qwen27"></a>
### PC HX370 + RTX 5090 Laptop / Windows × Qwen3.8-27B

**FAILED — startup**. Fresh public Omni startup rejects the real NVFP4 checkpoint: no registered Omni pipeline. Historical bare-vLLM short-text success does not qualify this integration or multimodal inference.

Evidence: [qwen-windows](evidence/qwen-windows.json), [qwen-text-historical](evidence/qwen-text-historical.json).

Next: Add/validate an Omni pipeline binding; rerun text, image, context and artifact-quality tests.

<a id="pc_cuda_windows-minicpm"></a>
### PC HX370 + RTX 5090 Laptop / Windows × MiniCPM-o 4.5

**BLOCKED — none**. No complete matching checkpoint found in the inventoried workspace, HF cache and native model roots. MiniCPM5-2B is not MiniCPM-o 4.5. Runtime registration alone is not a model run.

Evidence: [host-wsl](evidence/host-wsl.json), [host-windows](evidence/host-windows.json).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_cuda_windows-vla"></a>
### PC HX370 + RTX 5090 Laptop / Windows × InternVLA-A1

**BLOCKED — none**. No complete matching checkpoint found in the inventoried workspace, HF cache and native model roots. MiniCPM5-2B is not MiniCPM-o 4.5. Runtime registration alone is not a model run.

Evidence: [host-wsl](evidence/host-wsl.json), [host-windows](evidence/host-windows.json).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_igpu-spark"></a>
### PC HX370 + Radeon 890M / Windows worker × Spark-X2.5

**BLOCKED — none**. No placement- and quality-verified whole-model graph bundle/stateful adapter for this model on this AMD route. Small backend graphs do not establish model support.

Evidence: [backend-dml](evidence/backend-dml.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_igpu-tts"></a>
### PC HX370 + Radeon 890M / Windows worker × Qwen3-TTS 0.6B CustomVoice

**BLOCKED — none**. No placement- and quality-verified whole-model graph bundle/stateful adapter for this model on this AMD route. Small backend graphs do not establish model support.

Evidence: [backend-dml](evidence/backend-dml.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_igpu-qwen27"></a>
### PC HX370 + Radeon 890M / Windows worker × Qwen3.8-27B

**COMPONENT — C-historical**. FP32 448x448 vision tower: DirectML 85/113 nodes; normalized max error 9.095e-06 versus torch CPU. Historical component path, not full 27B or current graph-stage image-to-text integration.

Evidence: [qwen-vision-dml](evidence/qwen-vision-dml.json).

Next: Recover the hashed tower bundle and qualify its current Omni graph handoff with the language stage.

<a id="pc_igpu-minicpm"></a>
### PC HX370 + Radeon 890M / Windows worker × MiniCPM-o 4.5

**BLOCKED — none**. No placement- and quality-verified whole-model graph bundle/stateful adapter for this model on this AMD route. Small backend graphs do not establish model support.

Evidence: [backend-dml](evidence/backend-dml.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_igpu-vla"></a>
### PC HX370 + Radeon 890M / Windows worker × InternVLA-A1

**BLOCKED — none**. No placement- and quality-verified whole-model graph bundle/stateful adapter for this model on this AMD route. Small backend graphs do not establish model support.

Evidence: [backend-dml](evidence/backend-dml.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_npu-spark"></a>
### PC HX370 + AMD NPU / Windows worker × Spark-X2.5

**BLOCKED — none**. No placement- and quality-verified whole-model graph bundle/stateful adapter for this model on this AMD route. Small backend graphs do not establish model support.

Evidence: [backend-npu](evidence/backend-npu.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_npu-tts"></a>
### PC HX370 + AMD NPU / Windows worker × Qwen3-TTS 0.6B CustomVoice

**BLOCKED — none**. No placement- and quality-verified whole-model graph bundle/stateful adapter for this model on this AMD route. Small backend graphs do not establish model support.

Evidence: [backend-npu](evidence/backend-npu.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_npu-qwen27"></a>
### PC HX370 + AMD NPU / Windows worker × Qwen3.8-27B

**REJECTED — artifact**. The tested A16W8 whole-vision-tower export failed NPU placement. This conclusion is specific to that export, not all future Qwen/NPU combinations.

Evidence: [qwen-vision-npu](evidence/qwen-vision-npu.json).

Next: Validate a supported export or smaller useful component, including same-checkpoint numerical quality.

<a id="pc_npu-minicpm"></a>
### PC HX370 + AMD NPU / Windows worker × MiniCPM-o 4.5

**BLOCKED — none**. No placement- and quality-verified whole-model graph bundle/stateful adapter for this model on this AMD route. Small backend graphs do not establish model support.

Evidence: [backend-npu](evidence/backend-npu.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_npu-vla"></a>
### PC HX370 + AMD NPU / Windows worker × InternVLA-A1

**BLOCKED — none**. No placement- and quality-verified whole-model graph bundle/stateful adapter for this model on this AMD route. Small backend graphs do not establish model support.

Evidence: [backend-npu](evidence/backend-npu.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_joint-spark"></a>
### PC CPU+iGPU+NPU +/- discrete GPU, joint model execution × Spark-X2.5

**BLOCKED — none**. No qualifying same-model multi-accelerator deployment/artifact plan or overlap/recovery evidence. Individual routes are not a joint pipeline.

Evidence: [backend-dml](evidence/backend-dml.json), [backend-npu](evidence/backend-npu.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_joint-tts"></a>
### PC CPU+iGPU+NPU +/- discrete GPU, joint model execution × Qwen3-TTS 0.6B CustomVoice

**BLOCKED — none**. No qualifying same-model multi-accelerator deployment/artifact plan or overlap/recovery evidence. Individual routes are not a joint pipeline.

Evidence: [backend-dml](evidence/backend-dml.json), [backend-npu](evidence/backend-npu.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_joint-qwen27"></a>
### PC CPU+iGPU+NPU +/- discrete GPU, joint model execution × Qwen3.8-27B

**BLOCKED — none**. No qualifying same-model multi-accelerator deployment/artifact plan or overlap/recovery evidence. Individual routes are not a joint pipeline.

Evidence: [backend-dml](evidence/backend-dml.json), [backend-npu](evidence/backend-npu.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_joint-minicpm"></a>
### PC CPU+iGPU+NPU +/- discrete GPU, joint model execution × MiniCPM-o 4.5

**BLOCKED — none**. No qualifying same-model multi-accelerator deployment/artifact plan or overlap/recovery evidence. Individual routes are not a joint pipeline.

Evidence: [backend-dml](evidence/backend-dml.json), [backend-npu](evidence/backend-npu.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_joint-vla"></a>
### PC CPU+iGPU+NPU +/- discrete GPU, joint model execution × InternVLA-A1

**BLOCKED — none**. No qualifying same-model multi-accelerator deployment/artifact plan or overlap/recovery evidence. Individual routes are not a joint pipeline.

Evidence: [backend-dml](evidence/backend-dml.json), [backend-npu](evidence/backend-npu.json), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_xelite-spark"></a>
### PC Snapdragon X Elite CRD / AI Hub × Spark-X2.5

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_xelite-tts"></a>
### PC Snapdragon X Elite CRD / AI Hub × Qwen3-TTS 0.6B CustomVoice

**PROFILE_ONLY — C-timing-only**. Historical QNN talker profile re-fetched successfully; no numerical/full-pipeline gate. Historical TFLite vocoder target was refused on this Windows ARM device.

Evidence: [tts-xelite](evidence/tts-xelite.json), [tts-xelite-vocoder-rejected](evidence/tts-xelite-vocoder-rejected.json), [tts-other-devices](evidence/tts-other-devices.json), [tts-other-device-code](evidence/tts-other-device-code.py).

Next: Validate target-specific numerical quality, complete local execution, memory and sustained behavior.

<a id="pc_xelite-qwen27"></a>
### PC Snapdragon X Elite CRD / AI Hub × Qwen3.8-27B

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_xelite-minicpm"></a>
### PC Snapdragon X Elite CRD / AI Hub × MiniCPM-o 4.5

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="pc_xelite-vla"></a>
### PC Snapdragon X Elite CRD / AI Hub × InternVLA-A1

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="mobile_s25-spark"></a>
### Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy × Spark-X2.5

**COMPONENT — C**. Fresh 2026-09-22 W8A16 full-attention component inference/profile passed; full1024 denotes one cache bucket. No full prefill/decode/KV/ring/sampling loop.

Evidence: [spark-s25](evidence/spark-s25.json), [spark-s25-parity](evidence/spark-s25-parity.json).

Next: Implement and validate local 128-token generation, 512/1024 transitions, cancellation and resident memory.

<a id="mobile_s25-tts"></a>
### Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy × Qwen3-TTS 0.6B CustomVoice

**COMPONENT — C**. Fresh FP16 predictor inference passed; its one output exactly matches the same compiled artifact/dataset reference. Historical decoder parity exists. No complete co-resident NPU/GPU stream.

Evidence: [tts-s25](evidence/tts-s25.json), [hub-repeatability](evidence/hub-repeatability.json), [tts-mobile-parity](evidence/tts-mobile-parity.json).

Next: Validate the device-local talker/predictor/vocoder pipeline, output quality, recovery and 30-minute thermal behavior.

<a id="mobile_s25-qwen27"></a>
### Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy × Qwen3.8-27B

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="mobile_s25-minicpm"></a>
### Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy × MiniCPM-o 4.5

**COMPONENT — C**. Fresh FP16 speech-head inference passed; all 41 outputs exactly match the same artifact/dataset reference. Historical FP16 numerical parity passed; tested W8A16 top-1 parity failed. Thinker/encoders/vocoder are not qualified together.

Evidence: [minicpm-s25](evidence/minicpm-s25.json), [hub-repeatability](evidence/hub-repeatability.json), [minicpm-mobile-parity](evidence/minicpm-mobile-parity.json).

Next: Supply a qualified complete mobile checkpoint/export and validate the full multimodal/stateful pipeline.

<a id="mobile_s25-vla"></a>
### Mobile Galaxy S25 / Snapdragon 8 Elite for Galaxy × InternVLA-A1

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="mobile_s24-spark"></a>
### Mobile Galaxy S24 / Snapdragon 8 Gen 3 × Spark-X2.5

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="mobile_s24-tts"></a>
### Mobile Galaxy S24 / Snapdragon 8 Gen 3 × Qwen3-TTS 0.6B CustomVoice

**COMPONENT — C-historical**. Historical FP16 decoder parity has SNR 39.13 dB. Tested calibrated INT8 decoder parity failed with negative SNR. Neither result is a full local TTS stream.

Evidence: [tts-mobile-parity](evidence/tts-mobile-parity.json).

Next: Validate the complete local pipeline and a quality-passing artifact set.

<a id="mobile_s24-qwen27"></a>
### Mobile Galaxy S24 / Snapdragon 8 Gen 3 × Qwen3.8-27B

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="mobile_s24-minicpm"></a>
### Mobile Galaxy S24 / Snapdragon 8 Gen 3 × MiniCPM-o 4.5

**COMPONENT — C-historical**. Historical FP16 speech-head job re-fetched successfully; SNR 49.22 dB/top-1 match in the recorded parity test. W8A16 top-1 failed. Not complete MiniCPM-o.

Evidence: [minicpm-s24](evidence/minicpm-s24.json), [minicpm-mobile-parity](evidence/minicpm-mobile-parity.json).

Next: Qualify thinker/encoders/vocoder and stateful local integration.

<a id="mobile_s24-vla"></a>
### Mobile Galaxy S24 / Snapdragon 8 Gen 3 × InternVLA-A1

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="embedded_sa8775p-spark"></a>
### Embedded SA8775P ADP / AI Hub × Spark-X2.5

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="embedded_sa8775p-tts"></a>
### Embedded SA8775P ADP / AI Hub × Qwen3-TTS 0.6B CustomVoice

**PROFILE_ONLY — C-timing-only**. Historical NPU talker and GPU vocoder profiles re-fetched successfully; no complete pipeline, parity or co-residency qualification.

Evidence: [tts-sa8775p-npu](evidence/tts-sa8775p-npu.json), [tts-sa8775p-gpu](evidence/tts-sa8775p-gpu.json), [tts-other-devices](evidence/tts-other-devices.json), [tts-other-device-code](evidence/tts-other-device-code.py).

Next: Validate target-specific numerical quality, complete local execution, memory and sustained behavior.

<a id="embedded_sa8775p-qwen27"></a>
### Embedded SA8775P ADP / AI Hub × Qwen3.8-27B

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="embedded_sa8775p-minicpm"></a>
### Embedded SA8775P ADP / AI Hub × MiniCPM-o 4.5

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="embedded_sa8775p-vla"></a>
### Embedded SA8775P ADP / AI Hub × InternVLA-A1

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="embedded_rb3-spark"></a>
### Embedded RB3 Gen 2 / QCS6490 / AI Hub × Spark-X2.5

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="embedded_rb3-tts"></a>
### Embedded RB3 Gen 2 / QCS6490 / AI Hub × Qwen3-TTS 0.6B CustomVoice

**PROFILE_ONLY — C-timing-only**. Historical GPU vocoder profile re-fetched successfully. The attempted NPU talker build was rejected for floating-point inputs. No full model pipeline.

Evidence: [tts-rb3-gpu](evidence/tts-rb3-gpu.json), [tts-rb3-talker-rejected](evidence/tts-rb3-talker-rejected.json), [tts-other-devices](evidence/tts-other-devices.json), [tts-other-device-code](evidence/tts-other-device-code.py).

Next: Validate target-specific numerical quality, complete local execution, memory and sustained behavior.

<a id="embedded_rb3-qwen27"></a>
### Embedded RB3 Gen 2 / QCS6490 / AI Hub × Qwen3.8-27B

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="embedded_rb3-minicpm"></a>
### Embedded RB3 Gen 2 / QCS6490 / AI Hub × MiniCPM-o 4.5

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.

<a id="embedded_rb3-vla"></a>
### Embedded RB3 Gen 2 / QCS6490 / AI Hub × InternVLA-A1

**BLOCKED — none**. No complete qualified target artifact and device-local stateful Omni backend. Hub jobs provide component execution, not an on-device generation/streaming controller; ADB has no attached device.

Evidence: [hub-inventory](evidence/hub-inventory.json), [host-wsl](evidence/host-wsl.json), [backend-factory](evidence/backend-factory.py), [graph-contract](evidence/graph-contract.py).

Next: Supply and validate the missing matching artifact/backend, then run the complete specified workload.
