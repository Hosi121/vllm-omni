# Local PC/mobile end-to-end support and profiling roadmap

This plan covers the [60 named device/model pairings](../../benchmarks/edge_harness/results/model_device_matrix_20260922/README.md).
Each pairing must finish with either a qualified local pipeline or a documented
feasibility/implementation blocker. It does not promise that every model fits
every device. The [current profiling review](../../benchmarks/edge_harness/results/e2e_profiling_20260922/README.md)
records measured execution separately from release qualification.

The [2026-09-23 recovery run](../../benchmarks/edge_harness/results/e2e_recovery_20260923/README.md)
adds a functional CPU TTS stream with playback underruns, a scoped RTX
Qwen3.8 text/image pass, verified MiniCPM-o and InternVLA weights, a MiniCPM-o
three-stage text-to-speech functional pass after correcting request embeddings,
and an InternVLA real-weight synthetic action forward. Model-quality and device gates
remain open; the 2026-09-22 audit remains the historical baseline.

The [WSL CPU expansion](../../benchmarks/edge_harness/results/e2e_expansion_20260923/README.md)
adds constrained three-stage MiniCPM-o text-to-speech requests and a CPU-only
InternVLA synthetic policy forward. A subsequent 20-request serial MiniCPM-o
profile measured p50 21.86 s/p95 22.97 s request wall time after one warmup, with
swap use under the 30.91 GiB WSL limit. It does not qualify speech quality,
concurrency, streaming, loading peak or sustained behavior. A separate native
Windows RTX 5090 Laptop MiniCPM-o three-stage text-to-speech request passed
with explicit 900/600 s startup limits after the default 300 s overall timeout
had expired. A later native Windows 20-request serial run measured p50 18.38 s
and p95 18.73 s, but sampled host available RAM fell to 0.78 GB and pagefile
use reached 11.17 GB. Native Windows InternVLA direct-policy runs separately
passed on CPU and RTX 5090 Laptop with synthetic inputs; AMD and mobile cells
do not inherit these passes.

## Architecture and prerequisites

Keep Omni's PipelineConfig, StageRuntime, StageClient, orchestration and admission
as the control plane. Keep vLLM's model execution, caches and internal batching.
Other runtimes implement stage backends. Model adapters own preprocessing,
sampling, modality alignment, state semantics and output interpretation.

The portable boundary is request/events, tensor or buffer contracts, opaque
state handles, device capabilities, artifact manifests and execution plans.
Persistent weights, KV/ring/recurrent state and workspaces stay with the backend.
Prefer one backend per autoregressive session; split coarse stages only after
measuring a complete-chain benefit.

Android may use a small native controller derived from Omni's stage, session and
streaming semantics, constrained by the same contract tests. A full Python/vLLM
service port is not a prerequisite. AI Hub remains a validation facility rather
than a deployment dependency.

Before advancing individual cells:

1. Pin installed runtime builds and checkpoint revisions. Repair the CPU
   vLLM/Omni mismatch and native Windows CPU attention-operator gap independently.
2. The complete MiniCPM-o 4.5 and InternVLA-A1 source checkpoints are now
   SHA-256 verified. Record export, quantization, calibration, compiler, shape
   buckets, state layout and quality evidence before target-specific claims.
3. Integrate native and external stages into the shared admission ledger.
   CPU/iGPU/NPU share physical RAM; Windows availability and WSL quota are
   separate constraints on that RAM. Budget loading peaks, weights, state,
   activations, workspace, transfers and safety margin. Discrete VRAM is separate.
4. Require bounded queues, consumer acknowledgement, cancellation propagation,
   sequence/epoch fencing, state retirement and actual placement reporting.
   No silent precision/model/context changes or CPU fallback labelled as NPU.
5. Record exact SKU/RAM, OS, drivers, installed packages, power state and device
   access. A local mobile application needs full on-device execution access;
   AI Hub component jobs alone do not establish that access or an E2E pipeline.

Implementation ownership follows the existing source boundaries:
[portable contracts](../../packages/omni-stage-contracts/omni_stage_contracts),
[host routes and process lifetime](../../vllm_omni/host),
[StageClient](../../vllm_omni/engine/stage_client.py) and
[backend adapters](../../vllm_omni/engine/backends),
[StageRuntime](../../vllm_omni/engine/stage_runtime.py), and the
[resource ledger](../../vllm_omni/engine/resource_ledger.py).
The current `GraphStageClient` represents one bounded, non-preemptible graph
call. Persistent mobile sessions require an explicitly negotiated backend
capability and lifecycle implementation; a state-handle type alone is insufficient.

## Work packages and completion gates

| Order | Work package | Implementation | Gate before wider rollout |
|---|---|---|---|
| Foundation | Runtime, contracts and accounting | Close the prerequisites above; reuse the existing qualification runners | Reproducible launch or explicit pre-load refusal, correct state ownership and placement, auditable memory budget |
| M1 | S25 Spark | Device-local Omni-compatible controller; embedding, prefill, continuous decode, ring/full KV, head and sampler; resident state | Reference checks, at least 128 output tokens, 512-window/1024-bucket transitions, cancellation, resident memory and sustained profile |
| M2 | Qwen3-TTS | Resolve desktop playback and shutdown findings and CPU compatibility; then mobile talker/predictor plus GPU vocoder | Complete PCM and tail, history/chunk correctness, interruption/recovery, quality checks, no post-startup underruns and RTF below 1 in the declared workload |
| M3 | AMD stages | Connect compatible real encoders or other useful coarse stages through current Omni workers; start from the existing 890M vision evidence | Actual node/device placement, numerical and task quality, complete downstream output, shared-memory and handoff costs; reject unhelpful splits |
| M4a | MiniCPM-o 4.5 | Three-stage desktop text+WAV produced coherent text and nonzero WAV on RTX 5090 Laptop under WSL and native Windows, and WSL CPU, using separate constrained plans. The CPU plan needed vLLM 0.28/0.29 processor compatibility and used swap; 20 serial CPU requests measured p50 21.86 s/p95 22.97 s. Native Windows needed longer startup limits; 20 serial requests measured p50 18.38 s/p95 18.73 s with host RAM/pagefile pressure. Add encoder and persistent-state reference before mobile artifacts | Speech intelligibility/alignment, loading-peak admission, concurrency, separate text, image and audio-understanding suites, then combined streaming and interruption |
| M4b | InternVLA-A1 | Real weights, observation preprocessing, shared prefix state, iterative action head and versioned action metadata; strict-load synthetic policy forwards now pass on WSL and native Windows, each on CPU and RTX 5090 Laptop | Reference action agreement on real observations, units/order/timestamps, observation age, stale-input handling and application-specific deadline reporting |
| Independent | Qwen3.8-27B | Public Omni pipeline binding; text, then image/video and longer contexts; same-model CPU/iGPU artifact and explicit offload evaluation | Modality-specific quality, actual loading/runtime memory and measured performance; each precision/context receives its own qualification |

The initial M0 desktop text acceptance remains scoped to its tested checkpoint,
precision and workload. New concurrency, state-boundary or quality evidence is
required before broadening that acceptance. Current batch-dependent greedy
outputs need reference analysis; timing alone cannot identify their cause.

VLA qualification here ends at action data. Physical robot control, collision
avoidance and actuator safety belong to the external controller.

## Hardware routes

Every row below covers all five model families through the work packages above.
Specific artifact/backend feasibility is a gate, not assumed from the route.

| Execution configuration | Route to qualify | Main additional gate |
|---|---|---|
| CPU only, WSL | Compatible vLLM CPU; a model-specific external CPU backend where required | ISA, complete model operations, artifact format and total RAM |
| CPU only, native Windows | Native CPU worker implementing the same contracts | Executable attention/state operations and matching model adapter; CUDA success does not cover this row |
| CPU + NVIDIA, WSL | Complete-model reference using vLLM/Omni CUDA | Full modalities and quality, stable power conditions, loading and sustained memory |
| CPU + NVIDIA, native Windows | Native installation of the same declared pipeline | IPC/process lifetime, filesystem/startup scope, streaming and shutdown validated separately |
| CPU + Radeon 890M | Compatible whole-model backend or CPU model plus coarse iGPU stages | Full downstream model result and transfers, not only an encoder result |
| CPU + AMD NPU | CPU model plus compiled NPU stages | Accepted artifact, actual NPU placement, quality and net E2E contribution |
| CPU+iGPU+NPU, with/without NVIDIA | Select stage placements after establishing individual routes | Co-resident memory/power; identical-workload single-backend versus split-serial versus split-overlapped comparison |
| Snapdragon X Elite PC | Windows ARM controller and target-specific CPU/GPU/QNN stages | Full application access, Windows ARM artifacts and RAM; Android binaries are not interchangeable |
| Galaxy S25 | Spark first, TTS second; memory-qualified MiniCPM-o/VLA afterward | Complete device-local state/streaming loop and sustained measurements |
| Galaxy S24 | Rebuild and revalidate the selected mobile pipeline | Independent SoC/runtime/quality qualification; S25 results do not transfer |
| SA8775P ADP | Target-local controller and compatible compiled stages | Complete application access, parity, memory and sustained co-residency |
| RB3 Gen 2 / QCS6490 | Supported integer NPU artifacts or an explicitly chosen CPU/GPU route | Resolve rejected artifact constraints, then full-model quality and performance |

Expand the joint-execution row into separate deployments with and without
NVIDIA when implementing it. Likewise, checkpoint size, precision, RAM SKU and
OS variants require separate execution plans and results beneath each overview
cell; a pass for one variant does not qualify the entire model or device family.

For Qwen27, nominal four-bit language weights alone are about 12.6 GiB;
the actual artifact, remaining weights, state, workspace and OS require more.
Gate small-memory targets before export/integration investment. Record an
artifact-specific capacity rejection rather than substituting a smaller model.
Treat unenumerated mobile/embedded SKUs as new qualification rows.

## Profiling protocol for every executable cell

Use [the profiling harness](../../benchmarks/edge_harness/PROFILING.md) and keep
raw samples, outputs, configuration, logs and traces. Record the actual installed
runtime as well as the source revision. Run competing workloads serially on
the same physical machine.

| Dimension | Required measurements |
|---|---|
| Startup | Compilation, process/import/preparation, constructor/load, first output and warm restart separately; explicit cache conditions |
| Workload | Short/medium/long inputs, concurrency 1 then 2/4 where admitted, at least 20 measured requests per group; separate warmup and state-boundary cases |
| Text | TTFT, delivery/token intervals, decode rate, complete request time; effective prefill separated from isolated device compute |
| Speech | First playable PCM, RTF, chunk arrival gaps, playback deficit count/duration, tail/flush and interruption latency |
| Multimodal/VLA | Preprocessing, encoders, downstream completion, modality alignment, observation-to-action age and deadline misses |
| Memory | Loading/runtime peaks, state growth with context/concurrency, workspace and transfers; physical RAM, WSL quota and VRAM kept distinct |
| Placement | Actual executed devices and fallback; copies, conversions, synchronization, stage queues and handoff time |
| Sustained behavior | At least 30 minutes; power mode, clocks, temperature, memory and latency trends; energy only within available sensor scope |
| Reliability | Slow consumers, cancellation, repeated sessions, memory admission and owned-worker failure; no stale/duplicate output and verified retirement |
| Instrumentation | Separate matched before/traced/after diagnostics; profiler RPC/export costs; retain uninstrumented timing baseline |

Report p50/p95 with sample counts and measurement boundaries. Preserve failures
and retries. Component sums are not E2E measurements, and whole-GPU power is
not CPU/NPU or whole-device energy. Low-clock/power-limited measurements remain
valid for those observed conditions; they do not establish optimized performance.

For heterogeneous candidates, hold inputs, artifact quality, context, concurrency
and power conditions fixed. Keep a split only if the full chain improves latency,
capacity or energy after transfers, synchronization, initialization and contention.

## Per-cell completion and repair record

Progress each cell through artifact readiness, correctness, complete local
execution, profiling and qualification. Functional and performance qualification
are separate fields. A final record contains:

- Exact device/OS and artifact/backend identities; declared modalities and limits.
- Reproducible command and explicit execution plan, including actual placement.
- Quality, state/streaming and memory-admission outcomes with raw evidence.
- Performance distributions, sustained behavior and sensor limitations.
- On failure: failing layer, configuration, observed reason, evidence and the
  smallest next repair/experiment. Distinguish missing prerequisites from an
  attempted run that failed, a rejected artifact and an unqualified result.

Text latency and VLA deadlines require workload-specific product targets.
Do not invent universal thresholds. A completed timing protocol alone is not
a release pass. The current 60-cell matrix and its profiling follow-up are the
starting backlog; subsequent repairs should replace only the affected cells
after repeating their complete gates.
