# E2E checks and profiling — 2026-09-22

This file is a chronological experiment log, including intermediate states.
The final review is generated as `FINAL_REVIEW.md` after all serial jobs exit;
use it together with `summary/README.md` for the final dispositions.

User scope: finish checking and profiling the accepted mobile/PC × model plan;
where the current implementation does not support a complete pipeline, record
the reason for later repair. Do not convert component results into E2E claims.

## Current experiment

The prior 60-cell audit is the input backlog. Current read-only preflights in
`preflight-windows.json` and `preflight-wsl-cpu.json` use real filesystem paths
(the older Windows audit included a PowerShell provider-prefixed workspace
path). They retain missing/broken artifact links explicitly. No Android device
is attached; no complete MiniCPM-o 4.5 or InternVLA-A1 checkpoint is available
in the checked roots. MiniCPM5-2B is a different model and is not substituted.

Five currently runnable complete request paths are being measured serially:

1. Spark 1.7B INT8, WSL CPU, existing vLLM 0.28 CPU build.
2. Spark 4B BF16, WSL CUDA, vLLM 0.29.
3. Spark 4B BF16, native Windows CUDA, vLLM 0.29+cu134.
4. Qwen3-TTS 0.6B CustomVoice, WSL CUDA, edge deployment profile.
5. The same pinned Qwen3-TTS checkpoint, native Windows CUDA.

The CPU run started first. `run_remaining.py` waits for its actual live WSL
process to exit and its host collector to finalize, then runs the four GPU
benchmarks serially. Inspect actual process/session state before restarting
anything. `queue.json` is written once the initial wait finishes.

Each run measures three length bands × concurrency 1/2/4 × 20 requests, excluding
warmups, followed by 1800 seconds of continuous medium-length requests. Spark
forces 128 output tokens for timing; this is not recommended application
sampling. Actual prompt token counts are archived. TTS retains terminal audio
separately from nonterminal chunks and records hashes/finite checks. Warmup PCM
chunks are archived individually as float32 bytes, with rate/order in JSON.
Terminal incremental/cumulative semantics and perceptual quality remain separate
qualification gates. Playback stalls are simulated from arrivals, not measured
through an audio output device.

Update: inspection of `resolve_sampling_params_list`, `drain_delta_payload` and
the output processor confirms DELTA payload draining. The TTS recorder now
resolves and checks the final stage's actual DELTA output setting, includes
terminal audio once in the primary complete playback metrics, and keeps
prefix-only metrics and raw terminal bytes separately. Waveform/reference
quality and correct flush boundaries still require their own evidence.

Each `<run>-host` directory contains the exact command, combined log, process
exit status and one-second Windows RAM/whole-NVIDIA-device telemetry. AC status
is sampled. GPU power is not whole-system/CPU/NPU energy. Background desktop
activity is not disabled. Brief read-only inventory and test-runner development
occurred during the initial CPU sweep; results are observational and carry
these conditions, not isolated laboratory SLA claims. The initial CPU collector
could not decode the localized power-scheme label; AC and battery telemetry
remain available. The collector's decoding was corrected for subsequent runs.

Model startup uses existing disk/JIT caches. No cold-disk startup or component
timing sum is reported as E2E latency. Memory peaks sampled by the engine are
lower bounds; Windows RAM and WSL RAM must not be added.

## Scripts

Source lives in `vllm-omni-edge/benchmarks/edge_harness/`:

- `profile_local_text.py`: existing Omni local text API, raw events, requests,
  lengths/concurrency, sustained run, slow consumer and cancellation/recovery.
- `profile_local_tts.py`: public AsyncOmni, pinned real weights, raw chunk
  arrivals, finite checks, terminal samples, sustained run and abort/recovery.
- `profile_host_command.py`: Windows host telemetry around one explicit command.
- `summarize_e2e_profiles.py`: checks all 60 pairings and raw-evidence hashes;
  joins the five profiling runs; unavailable E2E paths keep their exact reasons.
- `test_summarize_e2e_profiles.py`: nearest-rank percentile, per-configuration
  coverage, warmup exclusion and thermal-duration gate tests (3 passed).
- `profile_trace_diagnostic.py`: three untraced / three traced / three untraced
  diagnostic requests using existing Omni profiler RPCs. Executed separately
  after all sweep/thermal runs by `run_diagnostics.py`; its supervisor verifies
  the live predecessor process identity before waiting. Three samples are
  diagnostic overhead observations, not latency percentiles.
- `analyze_profile_resources.py`: whole-GPU telemetry/energy integration and
  trace category/event summaries. Concurrent event durations are not added
  and called wall time; long telemetry gaps are excluded from energy
  integration. Two resource-accounting tests pass (5 analyzer tests total).

CPU sensor probe results: Windows ACPI query returns access denied; WSL exposes
no hwmon/thermal-zone temperature inputs. CPU package temperature/energy cannot
be claimed. GPU temperature/power and sustained CPU throughput remain measured.
The Windows power-scheme query confirms Performance mode; raw probe saved.

The completed CPU sweep exposes a numerical/reproducibility qualification gap:
all 20 long-prompt concurrency-1 requests share one output sequence, while
concurrency 2 has two sequences (10 each) and concurrency 4 has four sequences
(5 each). Inputs all contain 1928 tokens and force 128 output tokens. First
differences from the c1 sequence appear at output indices 1–4 for differing
variants. Raw examples are in `spark-cpu-concurrency-output-variants.json`.
Without captured reference logits, this does not establish state contamination
or its cause. Do not promote successful timing completion into batch-invariant
quality acceptance. Snapshot sequence/epoch/queue-bound checks found no stream
invariant failures; final cancellation checks still run after the sustained phase.

Generate/update the current report from the implementation repository:

```bash
../.venvs/omni-cpu/bin/python benchmarks/edge_harness/summarize_e2e_profiles.py \
  --matrix benchmarks/edge_harness/results/model_device_matrix_20260922/matrix.json \
  --runs ../analysis/experiments/e2e_profiling_20260922 \
  --out ../analysis/experiments/e2e_profiling_20260922/summary
```

## Completion requirements still open

Progress at 2026-09-22 15:05 local time:

- CPU sweep completed all nine groups × 20 measured requests, plus warmups,
  1805.77 seconds sustained generation, slow-consumer and after-cancel requests
  (405 recorded normal requests). Cancellation acknowledged backend abort,
  producer stopped, zero in-flight requests, and no delivered stale events.
  Original CPU workers exited. CPU queue-saturation backpressure still needs
  a stronger delay probe: the original 30 ms consumer delay can be faster than
  CPU production, so a bounded queue alone does not prove backpressure engaged.
- CPU diagnostic completed, with 3 before / 3 traced / 3 after requests and a
  74,474,545-byte compressed trace. Traced median request time was 1.232× before;
  this excludes profiler start/stop/export cost. Trace postprocessing reached
  roughly 27 GiB worker RSS under the 30.9 GiB WSL quota. It completed before
  any manual intervention; shutdown used a forced process-manager cleanup.
  The original full instrumentation script is archived as
  `profile_trace_diagnostic_full.py`. Future diagnostics use 32 text output
  tokens and disable shape/memory instrumentation; comparisons remain paired
  within the same diagnostic configuration, separate from 128-token baselines.
- Initial GPU launches failed before model execution: unquoted inherited Windows
  PATH expansion in WSL, and native Python's GBK default decoding a Torch template.
  They are harness failures, not hardware/model failures. Preserved under
  `launcher-failures-1/`. Retried commands use a minimal Linux PATH and
  `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8`. The host wrapper now propagates child
  exit status rather than always returning zero.
- `run_retry.py` serializes corrected sweeps then diagnostics. Its live status
  is `retry-status.json`; `queue.json` records the current sweep supervisor.
  WSL CUDA Spark has loaded and is generating requests successfully. Do not
  restart live runs based only on an observation timeout.

Wait for and inspect all five real runs, diagnose runner failures if any, then
derive p50/p95 by exact configuration from raw samples. Inspect actual device
placement, state/recovery results, TTS final-tail correctness, memory and
thermal trends. Run separate diagnostic profiling where the runtime supports
it, keeping traced latency separate and quantifying overhead. Recheck all
unavailable pairings against current artifacts/runtime evidence and record
specific repair actions. Publish a reproducible evidence bundle and update
the fork only after the final matrix and hashes verify. Do not mark the goal
complete while benchmark processes or required checks remain outstanding.

### Profiling supervision update — 15:28 local

The completed CPU run has 405 request records with no recorded sequence,
epoch, completion or queue-bound violations. Batch-dependent token differences
remain a separate qualification gap. The summarizer now includes recorded
output invariants and separate trace/reliability reports without treating probe
completion as a pass. Ten focused analyzer tests pass.

`run_reliability.py` waits for the corrected sweep/trace supervisor, then runs
three isolated Spark checks. `run_tts_reliability.py` waits for that supervisor
and runs WSL/native Windows TTS interruption, recovery, slow-consumer and owned
vocoder-failure probes. Fault injection selects process handles belonging to
the probe's own engine and confirms process ancestry. Forced cleanup, if needed,
is retained as a finding. TTS slow-consumer tests do not claim to saturate the
internal 64 MiB transport buffer.

`run_final_analysis.py` waits for both reliability sequences before parsing
large traces, keeping that CPU/memory overhead outside timing runs. Its queue
record is `analysis-queue.json`. Final publication still requires review of the
actual results, finalized requirement dispositions, reproducible evidence and
hash checks; none of these supervisors automatically push changes.

### Profiling update — 15:47 local

WSL CUDA Spark completed 839 normal request records, including all 180 measured
requests and 1802.60 seconds sustained generation. Recorded sequence/epoch/
completion/queue checks pass. Cancellation acknowledged backend abort, stopped
the producer, left zero in flight and delivered no remaining stale events.
Short/medium/long concurrency-1 TTFT p50 values are 0.0352/0.0901/0.3633 s;
decode p50 values are 50.20/46.55/48.21 tokens/s (4B BF16, forced 128 outputs).
These are not same-model CPU/GPU comparisons.

Native Windows Spark's second launcher attempt failed before model execution:
`VLLM_CUDART_SO_PATH` was omitted. The benchmark common helper now reuses the
acceptance environment's torch DLL and short `~/c29` cache directory; native
Omni import passed. Before trace diagnostics, `run_diagnostics.py` will archive
this specific failed attempt under `launcher-failures-2` and retry its full
baseline after other sweeps finish. This keeps GPU workloads serial. Future
native TTS/trace/reliability launches inherit the same helper via their shared
benchmark import. No runtime wheel was changed.

WSL TTS loaded both real-weight stages and initialized in about 345 seconds,
including graph capture under substantial GPU/host memory pressure. It completed
the short concurrency-1 group and continues. Preserve this startup behavior;
do not substitute the earlier 40-second diagnostic initialization measurement.

### Profiling update — 16:23 local

WSL TTS completed all 180 measured requests and 1801.18 seconds sustained
operation: 1472 normal request records total. Recorded output invariants pass.
Concurrency-1 short/medium/long TTFA p50 values are 67.58/73.96/68.52 ms; p95
values are 91.83/111.60/100.57 ms. Corresponding RTF p50 values are
0.2033/0.1748/0.1758. One of the 180 measured requests (medium, concurrency 4)
has a 0.66094 ms simulated playback deficit. Actual sound-device playback and
perceptual/reference speech quality remain unqualified. All 223 archived
warmup PCM chunks match their hashes and sample counts; see
`tts-cuda-wsl-warmup-audio-integrity.json`.

Important qualification gaps remain: startup took 345.17 s, whole-GPU memory
peaked around 23.3 GiB, and whole-Windows-host available RAM briefly reached
24,514,560 bytes (about 23.4 MiB). The raw minimum sample is retained in
`tts-cuda-wsl-memory-headroom.json`; it includes other processes and is not
exclusive model attribution. Normal shutdown force-killed the remaining
vocoder stage after its grace interval; both stage PIDs and the controller were
subsequently confirmed absent. Abort acknowledgement took 35.3 ms and the next
request completed; late-event fencing still needs the queued reliability probe.

Native Windows TTS has passed the corrected import and reached stage startup.
The Windows Spark baseline retry follows that run before diagnostics. All runs
remain serialized. The harness now passes the repository-pinned Ruff 0.14.10
checks and 12 focused analyzer tests. See the new `PROFILING.md` in the harness
for portable per-run commands; supervisor scripts here are historical local
orchestration records with specific process IDs, not generic rerun entrypoints.

### Reporting failure and serial retry — 16:40 local

The first native Windows TTS sweep completed all 180 measured requests. It had
26 simulated underruns (positive durations 8.05–46.87 ms), and all 219 archived
warmup chunks passed integrity checks. Its sustained phase then stopped at 245
normal request records because `Path.replace(report.tmp, report.json)` raised
Windows access denied. This is a reporting failure, not a model-execution
failure, and the interrupted phase is not a 30-minute sustained pass.

A native Windows reproducer in `report-write-lock-probe/` confirms that a reader
without FILE_SHARE_DELETE blocks replacement on this workspace UNC path. The
shared report writer now defers running snapshots under this contention (raw
request JSONL remains authoritative), records the permission error, and requires
initial/final snapshots to persist with a bounded retry. Two regression tests
and the real Windows lock probe pass. Profile and trace CLIs now return a
nonzero exit code for failed reports. The earlier host exit code 0 must not be
mistaken for a successful TTS profile.

Read live reports using `inspect_live.py` with WSL Python. Avoid Windows
Get-Content/native readers on live report files: the currently running native
Spark process loaded the earlier writer before the fix. It has completed its
180-request sweep and is in sustained generation.

`run_windows_tts_retry.py` (queue: `windows-tts-retry-queue.json`) waits for the
existing sweep/trace/reliability/analysis chain to exit, then preserves the
interrupted native TTS run under `reporting-failures-1/` and performs a complete
fresh baseline. It reuses a completed, hash-verified paired trace if available,
otherwise reruns it; then performs native TTS reliability checks and regenerates
the final analyses. This additional supervisor must finish before publication
or goal completion. All inference and heavy trace analysis remain serial.

### Additional stage attribution

`run_attribution.py` waits for `run_windows_tts_retry.py` to finish, then runs
one medium-input/concurrency-1 TTS diagnostic per OS using the existing
`VLLM_OMNI_STEP_STATS_DIR` hooks, 20 measured requests and no sustained phase.
These directories are explicitly separate from the five full baselines.
`VLLM_OMNI_STEP_STATS_SYNC=0` keeps their counters as host-side timing;
Torch traces provide separate accelerator evidence. Warmup, slow-consumer,
abort and recovery calls also contribute to the per-process counters. Nested
counter durations must not be added into E2E latency. The existing counter
percentiles use rounded sample indices/a capped reservoir and are not pooled
across processes; forced shutdown may lose the final unflushed counter tail.

The attribution report compares 20 matched input/concurrency requests against
the uninstrumented baseline, retaining speech-length metrics and the caveat
that separate-process cache/thermal history and stochastic speech preclude a
pure causal overhead interpretation. Final task completion also requires
`attribution-queue.json` to be terminal and its actual process to have exited.

### Final diagnostic and archive preparation

`run_cpu_trace_refresh.py` waits for the attribution supervisor, preserves the
earlier full CPU diagnostic under `spark-cpu-wsl-trace-full`, and runs the same
lightweight 32-token diagnostic used for CUDA. The original full CPU trace did
not record profiler start/stop RPC durations; this separate run fills that
measurement gap without changing any baseline. Its queue must also be terminal
before final review/publication.

The summarizer now reports pooled within-request delivery intervals and
effective prefill throughput. Text timestamps are output-update timestamps and
can represent coalesced tokens; they are not per-token kernel measurements.
Intervals never cross request boundaries. Audio intervals exclude empty chunks
and retain nonempty terminal audio. Fifteen focused analyzer/report-writer tests
pass after adding this check.

The evidence archive utility preserves original file bytes and hashes. Files
larger than 50 MiB that are not already gzip are reversibly compressed with an
explicit original-to-stored mapping in the manifest. In particular, native
Windows Spark produced a roughly 194 MiB uncompressed trace, which cannot be
committed directly to GitHub. Restore the archive before replaying original
report-relative trace paths; unchanged original reports retain original hashes.

### Reliability results — 17:36 local

All three Spark probes reached the four-chunk queue capacity, preserved the
baseline token sequence after draining, rejected stale post-cancel handles and
oversized admission plans, and returned no late events after cancellation.
Terminating an owned model worker produced an explicit request error in
5.907 s (CPU WSL), 4.160 s (CUDA WSL), and 0.315 s (native Windows CUDA).
No owned process required harness cleanup. These target specific scenarios;
they do not prove every internal queue, reference-quality gate or memory-pressure
case. The oversized-plan check is admission rejection, not destructive OOM.

WSL TTS's slow consumer completed. Abort after the first nonempty finite audio
chunk acknowledged in 2.438 ms, delivered only an empty terminal event after
acknowledgement, and the following request completed with finite PCM. Killing
the owned vocoder produced `OmniEngineDeadError` after 4.107 s. No owned process
required harness cleanup. This does not qualify perceptual quality or saturation
of the internal 64 MiB buffer. See the `*-reliability/report.json` records.

### TTS trace filename collision found during evidence review

The first paired TTS diagnostics each produced only one trace. Both stage logs
explicitly exported to the same `trace_rank0.json`: stage 1 overwrote stage 0.
These records measure paired RPC/request overhead but preserve only the vocoder
trace, so they cannot establish full two-stage attribution. The runtime's
profiler artifact naming needs a stage identifier for general multi-stage use.

The diagnostic harness works around this through the existing public API:
`start_profile(profile_prefix="e2e-stage0", stages=[0])` and a distinct
`e2e-stage1` prefix for stage 1, with both active for the traced requests.
It requires an artifact for each stage. Background gzip is disabled for new
diagnostics; archive compression occurs after every writer has exited, avoiding
hashing a file still being compressed.

`run_tts_trace_refresh.py` waits for the final CPU diagnostic, preserves the
earlier TTS trace/host directories under `trace-collision-1/`, repeats both OS
diagnostics, then regenerates analyses. Its actual process and
`tts-trace-refresh-queue.json` must be terminal before publication. Baseline
timings are unchanged. Windows Spark's trace and both retained vocoder traces
contain actual CUDA kernel, memcpy and runtime events; file existence alone
was not used to infer accelerator capture.

### Native TTS retry and final audit preparation — 18:05 local

The fresh native Windows TTS run completed its full 180-request measured sweep
and is in sustained generation. This sweep has 49 simulated playback deficits,
positive durations 1.130–136.304 ms; its largest measured RTF is 0.3525. Thus
RTF below one does not imply the proposed zero-underrun gate passed. The earlier
interrupted attempt's 26/180 count remains a separate record, not pooled with
this rerun. Constructor startup is 79.792 s for the fresh run.

The resource analyzer now retains inclusive host-launch-to-first-request/output
timing, which includes imports, plan construction and artifact hashing before
the constructor. For example, native Spark's constructor took 95.155 s, while
host launch to first request was 175.590 s. Its checkpoint was accessed over the
WSL UNC path; native TTS uses a local Windows checkpoint copy. These are recorded
conditions, not an isolated OS startup comparison.

Sixteen focused tests now pass, including a check that missing desktop MiniCPM-o
and InternVLA checkpoints remain the primary machine-readable blocker. A single
Windows WMI frequency inventory is recorded separately; it is not a sustained
CPU effective-frequency trace. `render_final_review.py` and
`finalize_requirements.py` require terminal diagnostic queues before resolving
the final review and all 23 requirement groups. The evidence secret scan and
archive verification must run after all writers exit.

### All five baselines complete — 18:17 local

Native Windows TTS completed 1290 normal requests and 1804.298 seconds sustained
generation. All recorded normal-output invariants pass, and all 218 archived
warmup PCM chunks match their hashes and lengths. Normal shutdown retired both
stages without a forced-kill log. Its separate fault probe completed: abort
acknowledged in 3.488 ms with no late PCM, the following request completed, and
vocoder termination produced an explicit error in 0.269 s. No owned worker
required harness cleanup.

Sustained real-time performance failed despite functional completion. Of 1087
sustained requests, 21 had RTF above one and 98 had simulated playback deficits.
The first five minutes had RTF p50 0.1775 (n=208); requests submitted in the final
60 seconds had RTF p50/p95 1.3163/1.3370 (n=5), TTFA p50 346.34 ms and simulated
playback deficit p50 3674.54 ms. The full-run RTF p95 is only 0.2065, illustrating
why aggregate request percentiles alone hide this late slowdown. GPU telemetry
in the last minute reports an 810 MHz memory clock throughout, P5 in raw samples,
temperature p50 57 C and whole-GPU power p50 33.62 W. These are correlated
observations, not proof of a thermal cause. AC/Performance conditions remained
recorded. See `tts-cuda-windows-sustained-tail-check.json` and the raw telemetry.

The resource analyzer now includes the whole sustained request distribution,
RTF-above-one/deficit counts and a final-60-second window, alongside the existing
first/last-five-minute windows. Stage attribution and refreshed traces continue
serially after these completed baselines.

At 18:21, `nvidia-smi -q -d PERFORMANCE,POWER,CLOCK,TEMPERATURE` reported
software power capping active, a current GPU ceiling of 33 W versus a 95 W
default, P5 and an 810 MHz memory clock. GPU hardware/software thermal slowdown
flags were inactive at that instant; cumulative reason counters are not scoped
to this run. The raw snapshot is `gpu-low-clock-diagnostic.txt`. This later spot
check does not prove when or why the platform reduced its limit. No power-policy
setting was changed by this profiling task. Baseline-versus-stage-counter ratios
therefore include observed power/clock differences and cannot isolate profiler
overhead. Paired trace-before/during/after requests remain separately recorded.

### Diagnostic deadline correction — 18:36 local

The WSL stage-counter run reached engine readiness after 842.48 s, then its
900-second launcher deadline expired after one warmup and two measured requests.
The timeout/controller/owned stage PIDs were confirmed absent. This is an
insufficient diagnostic time allowance under the observed conditions, not a
claim that the model cannot execute. The raw last snapshot remains `running`;
the host exit code 124 is authoritative. The summarizer now resolves such stale
snapshots as failed/interrupted without editing the original data; 17 tests pass.

Only the still-waiting TTS trace supervisor was replaced, before it launched any
model, to extend its future WSL deadline to 1800 s. Its prior queue record is
`tts-trace-refresh-supervisor-superseded.json`. `run_attribution_retry.py` waits
for those traces, preserves the interrupted WSL counter run and original summary
under `attribution-timeouts-1/`, and repeats the same 20-request diagnostic with
an 1800-second allowance. It retains the completed native counter run and
regenerates summaries. Final review/publication also requires this retry queue
and actual supervisor process to be terminal. No ongoing inference was restarted
merely because an observation wait expired.

The native counter run completed all 20 measured requests (RTF p50 1.4450 in
the reduced-power conditions). Both stage counter files are present. Engine
step counts include idle/no-work iterations; the default orchestrator loop has
no dispatch counter hook, so an absent counter is not reported as zero latency.
The lightweight CPU trace also completed: paired request medians were
1.8122/1.9962/1.8516 seconds (before/traced/after), with profiler stop/export RPC
21.5903 seconds. The earlier full CPU trace remains under `spark-cpu-wsl-trace-full`.
## Corrected two-stage trace results

Both TTS trace refreshes completed with separate `e2e-stage0` and `e2e-stage1`
artifacts. WSL constructor startup was 937.495 s; before/traced/after request
medians were 5.0944/4.7426/4.3724 s and stop/export RPC was 15.5334 s. Native
Windows startup was 56.8888 s; corresponding medians were
4.2874/4.4784/4.5978 s and stop/export RPC was 18.0903 s. These are the separate
reduced-power diagnostic conditions, not replacements for baseline timings or
isolated OS/instrumentation effects. Speech lengths may differ.

The resource analyzer found actual CUDA kernel, memcpy and runtime events in
both stages on both OS paths. Stage-0/stage-1 kernel event counts were
196,938/10,977 for WSL and 54,380/10,977 for native Windows. Event counts are
trace evidence, not throughput or interchangeable work counts across builds.
Nested event sums and per-stage interval unions must not be added into E2E time.
The old collided traces remain under `trace-collision-1/`.

The WSL stage-counter retry began at 19:00 after the trace supervisor and its
analysis exited. It preserves the first timed-out attempt, retains the completed
native counter run, and repeats the unchanged 20-request medium/c1 workload.
## Terminal review and publication checks

The WSL counter retry completed all 20 medium/c1 measured requests with exit 0,
constructor startup 917.27 s and RTF p50 1.4879 under the recorded reduced-power
conditions. Normal shutdown completed; both known stage processes and the
supervisor were confirmed absent. The native counter run also has 20 measured
requests and both stage counter files. Their timings are not an isolated
instrumentation-overhead estimate.

`FINAL_REVIEW.md` is the canonical human review. All 23 groups in
`requirements-audit.json` now have terminal dispositions, including explicit
implementation, reference-quality and sensor gaps. `completion-integrity.json`
verifies all 60 cells, the 900 measured baseline requests, and original hashes
for the final trace/counter artifacts. The verifier normalizes Windows path
separators when reading native reports on Linux; raw reports are unchanged.
All five baselines completed at least 1800 seconds of sustained generation.

The 245-file text evidence scan had one reviewed false positive: the Qualcomm
helper selects a client using a token read from stdin/configuration; the flagged
assignment contains no string literal. The scan reports locations, never values,
and remains a heuristic rather than proof of absence. See `secret-scan.json`.
