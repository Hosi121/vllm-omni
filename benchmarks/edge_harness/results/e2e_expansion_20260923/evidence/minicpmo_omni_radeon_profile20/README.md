# Native Windows Radeon MiniCPM-o: repeated Omni complete requests

**Twenty scoped audio+image → text+speech requests passed; performance regressed under this memory condition.** The same pinned MiniCPM-o 4.5 GGUF bundle and bounded `llama-omni-cli.exe` from the [earlier one-request Omni pass](../minicpmo_omni_cpp/README.md) ran through `StageRuntime`/`StagePool` on the HX370 laptop, native Windows 11 build 26200 with Radeon 890M driver 32.0.22018.6001, Python 3.12.10, installed vLLM 0.29 and this Omni checkout. Every request launches a fresh C++ model session; the timings include its weight load, combined input processing, text generation, Token2Wav, and complete 24 kHz WAV assembly. They are not resident-model inference latency or playable streaming.

The exact model revision `db25077c33951fe163b42986fba0132e279872a2`, ten GGUF artifact hashes, patched CLI SHA256 `66a512386aafeb708f49a7394a9054d94ec777c362395c8917a3d198aef1b523`, input hashes and 2,048-token/96-output-token bounds are in [profile20.json](profile20.json). The requested placement was `radeon-hybrid`: `GGML_VK_VISIBLE_DEVICES=1` selected `AMD Radeon(TM) 890M Graphics` Vulkan0 for language and vision, with CPU TTS/Token2Wav/vocoder. All **21 successful** worker logs (one warmup and 20 measured) identify the Radeon adapter, 37/37 language layers offloaded, and CPU Token2Wav. The [full logs](worker_logs/) are preserved with line endings and trailing whitespace normalized for text review; the original Windows files remain under the paths recorded in the JSON report. The 22nd log is the deliberately aborted request. This is CPU+iGPU joint execution, not NPU or all-GPU execution.

The fixture is the same 16 kHz spoken question “What color is the square in the image?” plus a 448×448 synthetic red-square JPEG used by the earlier run. After one excluded warmup, all 20 measured requests answered “The square in the image is red.” and emitted a finite, nonzero complete WAV with a terminal audio event. Speech duration ranged **2.08–3.00 s** and the 20 PCM hashes differed; this is expected nondeterminism, not text-answer disagreement. The [first measured WAV](profile20.wav) transcribed exactly as “The square in the image is red.” with pinned local Whisper tiny.en revision `87c7102498dcde7456f24cfd30239ca606ed9063`, WER 0 for **one output only** ([ASR report](profile20_first_asr.json)). This is an intelligibility proxy, not broad speech or image-grounding quality.

| Condition | Complete-request wall | Startup outside requests | Memory evidence |
|---|---:|---:|---|
| One warmup + 20 serial measured requests | Nearest-rank p50/p95 **87.27/91.45 s**; range 69.91–97.90 s | 51.92 s | 21 GiB shared-RAM reservation and capacity; 23.80 GB OS-available before load; max sampled worker RSS 15.30 GB |
| Separate cold rerun after profile | One **88.01 s** request | 47.63 s | Same 21 GiB budget; 23.99 GB OS-available before load; sampled worker RSS 15.30 GB ([report](cold_after_profile.json), [log](cold_after_worker_logs/)) |
| Earlier cold Omni smoke | One **18.27 s** request | 6.83 s | 24 GiB reservation under a declared 30 GiB ceiling; [earlier record](../minicpmo_omni_cpp/radeon_report.json) |

These are different runs, not a paired speed comparison. The new repeated profile and cold rerun both passed functionally but were roughly five times slower than the older single cold request. Each of the 21 successful new worker logs contains two `PrefetchVirtualMemory failed` warnings, while the earlier single-request log contains none. Representative new language-model load times were about 25–27 s versus 6.07 s in the earlier log; the first measured new request took 89.67 s wall, including speech generation. Those observations are consistent with memory pressure or changed file-cache/paging conditions, **but do not identify a root cause**. Windows free RAM was only about 22.17 GiB at admission for a 21 GiB controller ceiling. Power/thermal state, full OS commit/pagefile time series, shared-iGPU allocation and a matched-condition CPU control were not captured. Do not reuse the older 18.27 s result as expected warmed performance or claim the Radeon path is faster than CPU under this condition.

The stage verified artifacts and placement, admitted the explicit shared-RAM demand, and the final in-flight abort produced no stale output. Both abort and shutdown left zero reserved/quarantined bytes in the [profile report](profile20.json). The probe now refuses a declared capacity above OS-available RAM before loading. Sampled worker RSS does not establish the weight-loading peak or bound shared GPU allocation. No post-cancel restart, concurrency, real image/audio suite, playable streaming or sustained power/thermal gate was run.

Reproduce from the repository root in PowerShell with the pinned local artifacts:

```powershell
$root = '\\wsl.localhost\Ubuntu\home\zhout\project\edge_infer'
$repo = Join-Path $root 'vllm-omni-edge'
$fixture = Join-Path $repo 'benchmarks\edge_harness\results\e2e_expansion_20260923\evidence\minicpmo_cpp_radeon890m_prompted'
$evidence = Join-Path $repo 'benchmarks\edge_harness\results\e2e_expansion_20260923\evidence\minicpmo_omni_radeon_profile20'
$py = 'C:\Users\zhout\w2\omni029venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $repo
$env:VLLM_ENABLE_V1_MULTIPROCESSING = '0'
& $py (Join-Path $repo 'benchmarks\edge_harness\experiments\probe_omni_minicpmo_cpp.py') `
  --model-dir (Join-Path $root 'models\MiniCPM-o-4_5-gguf') `
  --cli-bin 'C:\Users\zhout\w2\minicpmo_omni_bin_radeon\llama-omni-cli.exe' `
  --cli-sha256 '66a512386aafeb708f49a7394a9054d94ec777c362395c8917a3d198aef1b523' `
  --reference-wav (Join-Path $fixture 'fixture_0000.wav') `
  --audio-wav (Join-Path $fixture 'fixture_0001.wav') `
  --image-jpeg (Join-Path $fixture 'fixture_0001.jpg') `
  --artifact-report (Join-Path $fixture 'full_report.json') `
  --work-root 'C:\Users\zhout\w2\minicpmo_omni_profile20_work_radeon' `
  --log-root 'C:\Users\zhout\w2\minicpmo_omni_profile20_logs_radeon' `
  --placement radeon-hybrid --reserve-gib 21 --capacity-gib 21 `
  --warmups 1 --repeats 20 --abort-check `
  --output-report (Join-Path $evidence 'profile20.json') `
  --output-wav (Join-Path $evidence 'profile20.wav')
```

Next experiment: capture Windows available/commit/pagefile and GPU shared-memory time series across a matched CPU/Radeon pair, then isolate UNC weight I/O from model execution. Keep the full input/output quality and task checks separate from performance diagnosis.
