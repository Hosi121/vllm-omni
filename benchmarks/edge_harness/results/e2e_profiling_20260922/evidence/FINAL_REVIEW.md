# End-to-end check and profiling review — 2026-09-22

All 60 device/model pairs have a disposition. Five current desktop execution paths were selected for profiling; the other 55 retain concrete complete-pipeline blockers. This is a check of the current implementation, not a claim that all planned backends or release gates are implemented.

The [full matrix](summary/README.md) contains all cells and next steps. [Requirement audit](requirements-audit.json), [raw performance](summary/matrix.json), [resource/trace analysis](resource-summary.json), and [stage counters](attribution-summary.json) retain the evidence and limits.

## Executed timing protocol

Twenty measured requests per short/medium/long × concurrency 1/2/4 group; separate warmups, a 30-minute medium/concurrency-1 sustained phase, and separate trace/fault diagnostics. Nearest-rank percentiles describe measured requests. Existing disk/JIT caches were retained.

| Path | Protocol complete | Normal requests checked | Constructor startup s | Sustained s | Medium c1 latency p50 / p95 | Medium c1 rate p50 |
|---|---|---:|---:|---:|---|---|
| pc_cpu_wsl / Spark-X2.5 | True | 405 | 20.767 | 1805.768 | 0.762 / 0.801 s | 15.663 token/s |
| pc_cuda_wsl / Spark-X2.5 | True | 839 | 23.693 | 1802.602 | 0.090 / 0.103 s | 46.545 token/s |
| pc_cuda_wsl / Qwen3-TTS 0.6B CustomVoice | True | 1472 | 345.168 | 1801.183 | 73.956 / 111.604 ms | 0.175 RTF |
| pc_cuda_windows / Spark-X2.5 | True | 708 | 95.155 | 1803.249 | 0.100 / 0.129 s | 44.656 token/s |
| pc_cuda_windows / Qwen3-TTS 0.6B CustomVoice | True | 1290 | 79.792 | 1804.298 | 111.521 / 302.522 ms | 0.178 RTF |

Constructor startup excludes earlier imports, artifact hashing and plan construction. Resource analysis also retains inclusive host-launch-to-first-request/output timing. Native Windows Spark reads its checkpoint through the WSL UNC path; native Windows TTS uses a local Windows copy. Startup comparisons therefore include filesystem and preparation differences. Spark CPU uses 1.7B INT8; CUDA Spark uses 4B BF16. These are not same-model speedup comparisons. TTS is the pinned 0.6B CustomVoice checkpoint, English/Vivian. Other voices, languages and TTS modes are unqualified.

## Reliability and output checks

| Path | Recorded normal-output invariants | Targeted probe status | Cancellation / failure observation |
|---|---|---|---|
| pc_cpu_wsl / spark | True | completed | queue saturation=True; same tokens after drain=True; oversized refusal=True; post-cancel events=0; worker error=True, 5.907 s; harness cleanup processes=0 |
| pc_cuda_wsl / spark | True | completed | queue saturation=True; same tokens after drain=True; oversized refusal=True; post-cancel events=0; worker error=True, 4.160 s; harness cleanup processes=0 |
| pc_cuda_wsl / tts | True | completed | abort ack=0.002 s; late PCM samples=0; worker error=True, 4.107 s; harness cleanup processes=0 |
| pc_cuda_windows / spark | True | completed | queue saturation=True; same tokens after drain=True; oversized refusal=True; post-cancel events=0; worker error=True, 0.315 s; harness cleanup processes=0 |
| pc_cuda_windows / tts | True | completed | abort ack=0.003 s; late PCM samples=0; worker error=True, 0.269 s; harness cleanup processes=0 |

A completed probe means it ran. Inspect each outcome. Oversized admission is not an induced OOM; the TTS slow-consumer probe does not saturate every internal buffer. Targeted checks do not establish perceptual/reference quality or all possible state transitions.

## Findings requiring follow-up

- pc_cpu_wsl Spark: 35 measured concurrency-2/4 requests produced a token sequence absent from the corresponding concurrency-1 set. This is a reproducibility/reference gap; timing evidence alone does not identify numerical or state-related causes.
- pc_cuda_wsl Spark: 32 measured concurrency-2/4 requests produced a token sequence absent from the corresponding concurrency-1 set. This is a reproducibility/reference gap; timing evidence alone does not identify numerical or state-related causes.
- pc_cuda_wsl TTS: 1/180 measured requests had a simulated post-startup playback deficit. Raw chunk timings/underruns are retained. Actual sound-device playback, speech tail/reference agreement, ASR and perceptual/speaker quality remain unqualified.
- pc_cuda_wsl TTS sustained: 0/1269 requests had RTF above 1; 0 had simulated playback deficits. Requests submitted in the final 60 seconds: n=42, RTF p50/p95=0.180/0.185. This tail window is separate from the full-run and first/last-five-minute distributions.
- pc_cuda_windows Spark: 40 measured concurrency-2/4 requests produced a token sequence absent from the corresponding concurrency-1 set. This is a reproducibility/reference gap; timing evidence alone does not identify numerical or state-related causes.
- pc_cuda_windows TTS: 49/180 measured requests had a simulated post-startup playback deficit. Raw chunk timings/underruns are retained. Actual sound-device playback, speech tail/reference agreement, ASR and perceptual/speaker quality remain unqualified.
- pc_cuda_windows TTS sustained: 21/1087 requests had RTF above 1; 98 had simulated playback deficits. Requests submitted in the final 60 seconds: n=5, RTF p50/p95=1.316/1.337. This tail window is separate from the full-run and first/last-five-minute distributions.
- WSL TTS startup briefly coincided with only 24,514,560 bytes of available Windows RAM. The retained minimum sample includes other processes. Native-stage migration into the shared memory ledger and loading/workspace headroom still need work; sampled peaks are lower bounds, not allocation accounting.
- The normal WSL TTS baseline required forced vocoder shutdown after its grace period. The isolated worker-failure probe later retired owned workers; that does not erase the baseline shutdown finding.
- The original TTS profiler used the same rank-0 filename for both stages. The diagnostic workaround uses separate public-API stage prefixes. Original collided traces and their logs are preserved under trace-collision-1; production profiler naming remains a repair item.
- Native Windows TTS slowed substantially at the end of sustained execution while GPU telemetry reported P5 and an 810 MHz memory clock. AC/Performance conditions were still recorded. A subsequent 18:21 driver snapshot reports software power capping active, a 33 W current ceiling versus 95 W default, and inactive GPU thermal slowdown flags ([raw snapshot](gpu-low-clock-diagnostic.txt)). This spot check does not establish when or why the platform changed the limit. The clock transition and latency change are not proof of a thermal cause. Investigate power-state/clock behavior before claiming sustained real-time speech.
- CPU TTS has a vLLM/Omni compatibility blocker; native Windows CPU lacks the needed attention operators. Pin and qualify matching backend builds before claiming those cells.
- Qwen3.8-27B lacks the current public Omni pipeline registration; historical bare-vLLM text and vision-component results do not qualify its complete multimedia route.
- Matching full MiniCPM-o 4.5 and InternVLA-A1 checkpoints/stage artifacts are absent from checked roots. Obtain pinned artifacts, establish desktop reference outputs, then repeat the appropriate modality/action suites.
- AMD 890M evidence remains a vision component; the tested NPU whole-vision artifacts were rejected. Complete downstream adapters and compatible compiled artifacts are missing. Joint CPU/iGPU/NPU/GPU execution has no complete co-resident serial/overlap comparison.
- Mobile/embedded and X Elite targets lack device-local application access and integrated stateful model pipelines. AI Hub components do not supply the prefill/decode/KV/sampling loop, playback, co-residency or sustained device thermals. Do not deploy through AI Hub.

## Measurement limits

Windows physical RAM includes WSL; WSL quota, process RSS/PSS and VRAM are distinct scopes and must not be added. Power/temperature/clock samples cover the whole NVIDIA GPU, including other applications. CPU package temperature and CPU/NPU/whole-device energy are unavailable in these environments. CPU effective-frequency trends were not captured; the separate WMI check exposes nominal/aggregate counters, not a validated per-core frequency trace. No model-exclusive energy or causal thermal-throttling claim is made.

Text delivery intervals can combine tokens into output updates. Effective prefill includes queue, host and first-output costs. Torch category duration sums overlap; interval unions are not additive across categories. Sequential profiler stop/export can extend trace windows with idle time. TTS trace overhead comparisons retain speech-length variation. StepStats counters use host timing with synchronization disabled, overlapping scopes and per-process reservoirs; they are not pooled stage GPU latencies. Baseline-versus-counter ratios also include different observed power/clock conditions after the GPU limit change; they are not isolated instrumentation overhead. Engine-core step counters include idle/no-work iterations. The existing orchestrator-dispatch hook is only in the optional event-driven loop, which these runs did not enable; no dispatch latency is inferred from its absent counter.

Failures and corrected retries remain archived. The Windows TTS report-write failure interrupted its first sustained phase; that attempt is not a sustained pass. No universal model/device release qualification follows from a completed performance protocol.
