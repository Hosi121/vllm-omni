# MiniCPM-o 4.5 on native Windows Radeon 890M + CPU

The Ryzen AI 9 HX 370/Radeon 890M completed a **standalone scoped full request** with the pinned ten-file Q4_K_M/F16 [MiniCPM-o GGUF set](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf/tree/db25077c33951fe163b42986fba0132e279872a2). One synthetic red-square JPEG plus 2 s/440 Hz tone traversed vision, audio, language, TTS and vocoder to text and speech. A second [spoken visual question](../minicpmo_cpp_radeon890m_prompted/README.md) returned the correct answer, “The square in the image is red.” The corresponding [native Windows CPU request](../minicpmo_cpp_windows_cpu_prompted/README.md) produced identical language token IDs. This is a CPU+iGPU hybrid, outside Omni StageRuntime/StagePool; it is not an all-Radeon or AMD NPU result.

The C++ source is [tc-mb/llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni/tree/5202b7b2f4d11f50b9f996161e7a2f8b8571b890) at `5202b7b2f4d11f50b9f996161e7a2f8b8571b890`, with the [CPU selection, Windows and bounded-completion patch](../minicpmo_cpp_cpu/llama_cpp_omni_cpu.patch). It was cross-built from Ubuntu with MinGW-w64 GCC 13-posix, x86-64-v3, Vulkan headers 1.4.341, glslc 2026.1-1, CMake/Ninja Release, and `GGML_CUDA=OFF`, `GGML_VULKAN=ON`, `GGML_OPENMP=OFF`. The initial build failed because pointing the Windows compiler at all of `/usr/include` picked up Linux `stdint.h`; [the failed log](build_failed_linux_include.log) is retained. Restricting the include path to Vulkan headers alone succeeded; see [configure](configure.log) and [build](build_success.log) logs. The Windows Vulkan loader was linked from `vulkan-1.dll` with the native driver. The exact executable and all ten Windows GGUF hashes are in the [report](full_report.json); weights and binaries remain outside Git.

The host was Windows 11 build 26200. [Vulkan enumeration](vulkaninfo_summary.log) lists RTX 5090 as GPU0 and Radeon 890M (driver `25.20.18.06 (LLPC)`) as GPU1. The run set `GGML_VK_VISIBLE_DEVICES=1`, making Radeon the only visible Vulkan device. The [runtime log](full.combined.log) then named `Vulkan0 (AMD Radeon(TM) 890M Graphics)`, offloaded all 37/37 language layers to it, placed vision on Vulkan0, and recorded CPU Token2Wav/vocoder. TTS weights used 0/21 offloaded layers, although a Vulkan compute buffer was logged; the audio encoder was CPU. The log is the placement evidence, not just requested flags.

| One cold combined tone+image request | Observation |
|---|---|
| Input path | Index 0 supplied reference/system audio; index 1 encoded the [JPEG](fixture_0001.jpg) and [16 kHz tone](fixture_0001.wav). The verifier checked both in the log. |
| Output | Nonempty [text](response_text.txt) and a completion marker; 42 contiguous, nonzero mono 24 kHz PCM16 [WAV chunks](round_000/tts_wav/) totaling 41.96 s. The [merged WAV](full_merged.wav) preserves their PCM exactly. The generic response to a tone and square without an explicit question is not a grounded-quality result. |
| Time and memory | 113.43 s cold process wall; 15.57 GB sampled maximum RSS and 15.45 GB sampled maximum private bytes at 100 ms intervals. These are one-run observations, not p50/p95, guaranteed loading peaks or sustained thermal results. [Process profile](full.profile.json). |
| Speech proxy | Local pinned `openai/whisper-tiny` on CPU transcribed the Mandarin output with 8.79% character error against the model's own text after OpenCC script normalization. This does not measure whether the text answered the input. [ASR report](speech_asr_proxy.json). |

The verifier checked the pinned artifacts, exact logged placement, index-1 image+audio prefill, completion marker, contiguous WAV sequence, audio format/bounds and nonempty text. The upstream optional WAV merger still looks for the wrong chunk names and runs before vocoder completion; [the verifier](../../../../experiments/verify_minicpm_cpp_output.py) joined original PCM afterward. The correctly prompted result provides a stronger one-case quality check, but real image/audio suites, video, playable streaming, Omni admission/state/cancellation, repeated latency, shared-RAM/GPU loading peak and sustained power/thermal behavior remain open.

Reproduce after applying the patch and obtaining the pinned GGUF set. Cross-configure with `-DCMAKE_SYSTEM_NAME=Windows -DCMAKE_SYSTEM_PROCESSOR=x86_64`, MinGW-w64 `gcc-posix/g++-posix`, `-DCMAKE_C_FLAGS=-march=x86-64-v3 -DCMAKE_CXX_FLAGS=-march=x86-64-v3`, `-DGGML_VULKAN=ON`, `-DVulkan_INCLUDE_DIR=/path/to/isolated/vulkan-headers`, `-DVulkan_LIBRARY=/path/to/vulkan-1.dll` and `-DVulkan_GLSLC_EXECUTABLE=/usr/bin/glslc`. Build `llama-omni-cli`, then on native Windows:

```powershell
$env:GGML_VK_VISIBLE_DEVICES = '1'
llama-omni-cli.exe -m C:\path\to\MiniCPM-o-4_5-Q4_K_M.gguf `
  -ngl 99 -c 2048 --omni --t2w-device cpu --t2w-wait-seconds 900 `
  --ref-audio C:\path\to\default_ref_audio.wav --test C:\path\to\fixture_ 2
```
