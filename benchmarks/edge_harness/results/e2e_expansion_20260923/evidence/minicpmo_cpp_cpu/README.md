# MiniCPM-o 4.5 GGUF whole-chain CPU feasibility run

This is a **standalone scoped complete-path** audio+image-to-text+speech run on the local Ryzen AI 9 HX 370 under Ubuntu 26.04/WSL2 (`6.18.33.2-microsoft-standard-WSL2`). It is an additional backend feasibility result for the already-scoped WSL CPU × MiniCPM-o cell, not an Omni integration, a native Windows result, a Radeon/NPU result, or a semantic-quality pass. The generated response to a synthetic 448×448 red square and 2 s/440 Hz tone was generic rather than a description of those inputs.

The complete ten-file artifact set came from [openbmb/MiniCPM-o-4_5-gguf](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf/tree/db25077c33951fe163b42986fba0132e279872a2) revision `db25077c33951fe163b42986fba0132e279872a2`: Q4_K_M language model, F16 vision/audio/TTS/projector and five Token2Wav GGUF files. [The report](minicpm_omni_cpu_full_report.json) records the byte count and SHA256 of each. The C++ source is [tc-mb/llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni/tree/5202b7b2f4d11f50b9f996161e7a2f8b8571b890) at `5202b7b2f4d11f50b9f996161e7a2f8b8571b890`, built locally with GCC 15.2, CMake 4.2.3, `GGML_CUDA=OFF`, `GGML_VULKAN=OFF`, and [this small CLI patch](llama_cpp_omni_cpu.patch) adding an explicit CPU Token2Wav selector and a portable wait. The binary SHA256 is in the report. The current branch has a CLI target but no `llama-omni-server` CMake target, so this result remains outside the Omni StageRuntime.

| Observation | Evidence |
|---|---|
| The runtime logged `GPU layers: 0`, `vision using CPU backend`, and CPU Token2Wav/vocoder placement. Prefill index 0 was the system/reference-audio prompt; index 1 actually encoded one JPEG vision chunk (64 tokens) and the 16 kHz user audio (20 positions). | [raw runtime and `/usr/bin/time` log](minicpm_omni_cpu_full.log), [input JPEG](fixture_0001.jpg), [input WAV](fixture_0001.wav) |
| The decoder returned nonempty [text](response_text.txt), TTS generated audio tokens, and the vocoder wrote ten sequential, nonempty mono 24 kHz PCM16 [chunks](output/round_000/tts_wav/). The bounded verifier joined their original PCM into a [9.24 s WAV](minicpm_omni_cpu_full_merged.wav), 221,760 frames, PCM SHA256 `372c4c7b453849eb1843b0477ce74d5db04a46cacc663ef01d02698f34a1a158`, RMS 2,452.44 in PCM16 units. | [verification report](minicpm_omni_cpu_full_report.json), [raw chunks and timing](output/round_000/tts_wav/) |
| Pinned `openai/whisper-tiny` on CPU transcribed the generated Mandarin WAV with 9.30% CJK/alphanumeric character error against the model's own text after OpenCC traditional-to-simplified normalization; unnormalized error was 32.56%. This is an intelligibility proxy, not a human listening score or input-grounded correctness. | [ASR report](speech_asr_proxy.json), [raw ASR log](speech_asr_proxy.log), [probe](../../../../experiments/probe_tts_asr_cjk.py) |
| The one cold process completed in 40.70 s; `/usr/bin/time` reported 13,369,604 KiB maximum RSS. The Token2Wav calls in the raw log were roughly 2 s per 1 s output after the first call, so this CPU route was slower than real time. This is one observation, not a p50/p95 or sustained profile. | [raw log](minicpm_omni_cpu_full.log) |

The C++ runtime's optional merger searched for `tts_output_chunk_N.wav` while its vocoder wrote `wav_N.wav`, and it attempted merging before all vocoder work ended. The raw log preserves its “no valid WAV files to merge” message. [The verifier](../../../../experiments/verify_minicpm_cpp_output.py) checks the completion flag, contiguous names, format, duration bound, nonzero samples, nonempty text and logged CPU placement before assembling the actual chunks. It does not repair the runtime's live streaming behavior. The text response did not establish input-grounded quality; the quantized language artifact must be compared against the same task/reference before release qualification. Playable streaming, request cancellation, loading peak, 20-request latency and 30-minute power/thermal behavior remain unverified.

Reproduce from a local checkout of the pinned C++ commit after applying the recorded patch and downloading the exact GGUF revision:

```bash
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-cpu --target llama-omni-cli -j 8
OMP_NUM_THREADS=8 timeout 900 /usr/bin/time -v ./build-cpu/bin/llama-omni-cli \
  -m /path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  -ngl 0 -c 2048 --omni --t2w-device cpu \
  --test /path/to/fixture_ 2
```

The fixture must have both `fixture_0000.wav` for index 0 and `fixture_0001.wav` plus `fixture_0001.jpg` for the user observation. The recorded run reused the same tone at indices 0 and 1; the image is the JPEG conversion of the previous [red-square fixture](../minicpmo_wsl_image/red_square.png). The weights and local build remain outside Git.
