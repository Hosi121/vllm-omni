# WSL CPU MiniCPM-o GGUF complete-request Omni stage

This record exercises the same pinned MiniCPM-o 4.5 Q4_K_M/F16 GGUF set and
spoken red-square input as the [native Windows CPU/Radeon Omni runs](../minicpmo_omni_cpp/README.md),
but on the HX370's Ubuntu 26.04 WSL2 CPU route. The Linux C++ CLI was rebuilt
from `tc-mb/llama.cpp-omni` commit
`5202b7b2f4d11f50b9f996161e7a2f8b8571b890` with the
[recorded bounded-generation patch](../minicpmo_omni_cpp/llama_cpp_omni_bounded.patch).
Its SHA256 is `5c41298e2a234f76f10342e8b3b3aee7c93f77ea246c121f259ea89c682a6e42`;
the Release build has `GGML_CUDA=OFF`, `GGML_VULKAN=OFF`, `GGML_OPENMP=ON`.
The installed Python environment has vLLM 0.29.0 and PyTorch 2.13.0+cu130,
but the external GGUF process is CPU-only. The runtime uses the current Omni
source checkout via an explicit `PYTHONPATH`.

The [first complete-request report](report.json) and [worker logs](single_logs/)
verify all ten GGUF hashes, 2,048 context tokens, a 96-token generation limit,
CPU placement (`-ngl 0`, CPU vision and Token2Wav), index-1 audio and image
prefill, and the audio completion marker. The spoken question asks the color
of the synthetic 448×448 red square. The stage answered “The square in the
image is red.” and returned a complete, nonzero [2.52 s, 24 kHz WAV](output.wav)
in 25.77 s complete-request wall time. This was one cold sample, not a latency
distribution. The worker's maximum 100 ms sampled RSS was 13.70 GB; this is
not a proven loading peak. A 24 GiB host-RAM reservation from a declared
30 GiB WSL capacity includes weights, workspace/state and headroom. A subsequent
in-flight cancellation returned no stale output and left the ledger at zero.

The separate [serial profile](profile20.json) excluded one warmup and measured
20 complete requests at concurrency one. **All 20** answered the same correct
sentence, returned contiguous nonzero 24 kHz speech and one terminal audio
event, and passed the input/output checks. Nearest-rank complete-request wall
p50/p95 was **22.67/24.83 s**, with a 19.15–30.87 s range. Speech duration
varied from 1.60 to 2.92 s, and all 20 PCM hashes differed; this run did not
establish deterministic speech or alignment across outputs. The
[first measured WAV](profile20_first.wav) was independently transcribed by
pinned Whisper tiny.en as the exact sentence (WER 0) in the
[ASR report](profile20_first_asr.json), an intelligibility proxy for that one
output only. The [22 worker logs](profile20_logs/) retain the warmup,
20 measured requests and in-flight cancellation. The stage returned no stale
output after cancellation and the ledger again ended at zero.

Maximum per-request 100 ms sampled worker RSS across the measured requests was
13.70 GB. The separate [0.5 s host samples](host_memory_samples.jsonl)
started after the first few requests and saw available WSL RAM no lower than
22.55 GB and swap use between 0.994 and 1.060 GB. These samples cover only
part of the run and do not establish a loading peak or unique model RAM;
summed process-tree RSS can double-count shared pages. The [8 GiB admission
test](low_budget_report.json) was refused before worker launch and left no
reservation. No power mode or thermal history was captured.

The [first launch failed](driver_wrong_source.log) before model loading: the
Python environment selected an older editable Omni checkout without this
stage. Setting `PYTHONPATH` to the current checkout fixed the setup; the
[successful driver log](driver.log) is retained. The single synthetic visual
question does not qualify real image/audio understanding, speech alignment,
playable streaming, video input, post-cancel restart, concurrency, loading
peak or sustained power/thermal behavior. No power condition was controlled.

Reproduce with
[`probe_omni_minicpmo_cpp.py`](../../../../experiments/probe_omni_minicpmo_cpp.py)
using `--placement cpu`, the model revision and ten hashes in the JSON report,
and the fixture linked from the Windows evidence above. The CLI build uses the
patch and source commit named here. Use `--abort-check` to verify cancellation.
For the serial profile, add `--warmups 1 --repeats 20`; the
[host-memory sampler](../../../../experiments/sample_host_memory.py) accepts
the profile driver's PID and writes JSONL snapshots while it runs.
