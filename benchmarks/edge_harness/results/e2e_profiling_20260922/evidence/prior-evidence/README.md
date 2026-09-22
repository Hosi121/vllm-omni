# Omni abstraction implementation and qualification — 2026-09-22

Implementation: `vllm-omni-edge`, branch `codex/omni-edge-abstraction`, based on
`00510ce38bbb96ef2e4ce462618d06ab142fae7c`. Implementation committed as
`723c5502` and pushed to the user's
[vLLM-Omni fork branch](https://github.com/tzhouam/vllm-omni/tree/codex/omni-edge-abstraction).
The working tree is clean. The pre-commit wheel/source identities below remain
the original validation snapshot; the committed runtime matches it, with the
hardware-support and validation table added to the guide before publication.
The original dirty `vllm-omni` and clean `vllm-omni-029` trees were preserved.
See the [backend guide](../../../vllm-omni-edge/docs/design/omni_edge_backends.md)
and [completion plan](../../plan_omni_edge_completion_20260922.md).

## Implemented boundary

- Standalone `omni_stage_contracts`: version/features, tensors, requests/events,
  opaque state, physical topology and hashed artifacts; compatible legacy imports.
- Explicit host routes, worker bootstrap, Windows process identity and verified
  cross-domain retirement. Host-local initialization locks work from UNC checkouts.
- `external.graph.v1` driven by real `PipelineConfig`, `StageRuntime`, `StagePool`,
  orchestrator and public `Omni`/`AsyncOmni`; native execution/KV/batching stay vLLM.
- Graph-containing plans reserve all declared memory pools before loading.
  CPU/iGPU/NPU shared RAM, discrete VRAM and WSL quota are separate constraints.
  Unverified release is quarantined.
- One admitted graph request, bounded copied tensors, consumer ACK,
  epoch/generation fencing and worker retirement for non-preemptible cancellation.
- Native Windows SHM uses host-local locks, exact payload framing and retained
  producer handles until a generation-specific consumer ACK. Retention is capped
  at 64 MiB / 256 buffers; cancellation/close releases unconsumed mappings.

This implements stateless, fixed-bucket graphs. Full native admission migration,
stateful mobile generation, general fan-out and all model/device qualification
are unfinished. G1/G2/G3 remain open at their full accepted scope.

## Evidence matrix

D = detection, B = backend execution, C = model component, P = complete specified
pipeline, E = estimate. Synthetic topology and test doubles are contract tests.

| Route | Raw record | Proven scope |
|---|---|---|
| WSL Spark 4B BF16 CUDA | [wsl-m0-cuda-1.log](wsl-m0-cuda-1.log), 11 passed | P for the M0 test workload: 12 prompts × 128 tokens, context/state/cancel and memory checks. Not a new reference-token parity benchmark. |
| Native Windows Spark 4B BF16 CUDA | [windows-m0-cuda-6.log](windows-m0-cuda-6.log), 11 passed | P for the same M0 acceptance workload on this native Windows environment; includes UNC checkpoint/worktree paths. |
| WSL Qwen3-TTS 0.6B, real weights | [wsl-tts-6-real.log](wsl-tts-6-real.log), 1 passed | Two-stage edge-profile smoke test only; helper caps generation at two codec frames, so this does not close streaming/M2. |
| Native Windows Qwen3-TTS 0.6B | [tts-stream-windows-5.log](tts-stream-windows-5.log) | Public AsyncOmni two-stage real-weight streaming completes with audio and clean shutdown. Playback stalls remain; not full M2 acceptance. |
| WSL ORT CPU Add | [graph-cpu-wsl-2.json](graph-cpu-wsl-2.json) | B, public Omni, 20 requests, zero retained reservation after shutdown. |
| Native Windows ORT CPU Add | [graph-cpu-windows-2.json](graph-cpu-windows-2.json) | B, native controller/worker, 20 requests. Not CPU text support. |
| WSL → Windows DirectML Add | [graph-dml-crossdomain.json](graph-dml-crossdomain.json) | B, actual DML nodes, 20 requests. Not the historical 890M vision component. |
| WSL → Windows AMD NPU A16W8 | [graph-npu-crossdomain-2.json](graph-npu-crossdomain-2.json) | B, VitisAI=1 / CPU=2 nodes explicitly allowed; verified retirement. |
| Same A16W8 artifact on CPU | [graph-npu-cpu-reference.json](graph-npu-cpu-reference.json) | Backend reference, not original FP checkpoint quality. |
| Installed Omni wheel outside checkout | [installed-wheel-graph.json](installed-wheel-graph.json) | Public API + real ORT CPU, 20 requests. |
| Rebuilt Omni wheel after transport fixes | [installed-wheel-bounded.json](installed-wheel-bounded.json) | Same public API test from `/tmp`, actual worker under `/tmp/omni-platform-bounded-check`, exact outputs and zero retained reservation. Earlier builds remain separately recorded. |
| Qualcomm S25 component | [inference](aihub-s25-infer.json), [profile](aihub-s25-profile.json) | C, existing Spark attention component; not full Spark or an Android Omni controller. |

[Numerical checks](numerical-checks.json): CPU/Windows/DML Add outputs are exact.
NPU versus the same quantized CPU graph: maximum absolute error
`4.458427429199219e-05`, normalized maximum `3.111236521857185e-05`, below the
preselected `0.002` bound. These probes do not validate model quality.

Raw timings are functional-check samples, not controlled performance comparisons.
Some initialization overlapped independent CPU checks. Power policy and thermal
equilibrium were not controlled. No speedup or sustained power claim is made.

## Desktop TTS streaming acceptance remains open

The real-weight edge-profile smoke test passes, and a longer Chinese utterance
runs through `AsyncOmni` with the existing 2→4→8→16→25-frame ramp and 72-frame
left context. [Raw streaming record](tts-stream/Qwen3-TTS-12Hz-0.6B-CustomVoice/profile_edge/20260922-120641_abstraction.json)
and [log](tts-stream.log) preserve one warmup and **one measured request**, concurrency
one, same pinned checkpoint, BF16, RTX 5090 Laptop, native vLLM sampler.

The measured request emitted six nonterminal chunks (6.4 seconds of audio), then
a terminal tail. First audio was about 411 ms, but playback starting then would
stall about 4.42 seconds in total. Thus functional streaming works while the
continuous-playback acceptance criterion is **not met**. This is a diagnostic
single-request result, not a qualified latency percentile or performance claim.
Saved WAVs contain the streamed prefix only; the existing benchmark does not
append the terminal tail. No listening/ASR quality or interruption/recovery gate
has been passed. Peak GPU usage in this tool is explicitly whole-GPU, not exclusive
model attribution; power/thermal conditions were not fixed.

Native Windows now also completes the same two-stage stream after the SHM fixes.
[Raw Windows record](tts-stream-windows/Qwen3-TTS-0.6B-85e237c/profile_edge/20260922-123626_abstraction.json)
has one warmup and one measured request using the byte-identical native checkpoint
bundle. The measured request produced six nonterminal chunks / 6.4 seconds of
audio, first audio about 387 ms, and about 4.08 seconds of playback stall when
starting at first audio. Wall time was 11.51 seconds. These are diagnostic samples,
not a Windows-versus-WSL performance comparison. Both worker processes retired
normally. [Audio checks](tts-audio-checks.json) verify finite, nonzero 24 kHz saved
prefixes; they do not establish listening quality or terminal-tail correctness.

Reproduce from the worktree, using the CUDA_HOME/PATH of the inventoried venv:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  ../.venvs/omni-cuda-029/bin/python benchmarks/tts/stream_latency_bench.py \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --query-type CustomVoice \
  --txt-prompts ../analysis/experiments/omni_abstraction_20260922/tts-prompts.txt \
  --deploy-profile edge --warmup 1 --repeat 1 --save-wav \
  --output-dir ../analysis/experiments/omni_abstraction_20260922/tts-stream
```

Next desktop experiment: attribute the native-sampler/per-frame synchronization
and transport cost under the same input before changing stage placement. Do not
split onto another accelerator just because it is available.

## Qualcomm AI Hub Workbench

The user authorized this test facility. The supplied token was passed to the SDK
in memory via stdin and is not stored in the repository or runtime configuration.
AI Hub remains outside the local deployment path.

- [Inventory](aihub-inventory.json): SDK 0.55.0, 78 device/OS entries, including
  S25 Android 15, Snapdragon X Elite/X2 Elite Windows 11, SA8775P and RB3. Listing
  is D, not qualification of every listed device.
- Existing compile [jgdd1ljrg](https://workbench.aihub.qualcomm.com/jobs/jgdd1ljrg/),
  target `mnowl6xpq`. The historical label `full1024` means **one full-attention
  component with a 1024 cache bucket**, not the complete model.
- Fresh inference [jp0mvkj6g](https://workbench.aihub.qualcomm.com/jobs/jp0mvkj6g/)
  and NPU profile [jpyo7nx75](https://workbench.aihub.qualcomm.com/jobs/jpyo7nx75/)
  succeeded on Samsung Galaxy S25, Android 15, Snapdragon 8 Elite for Galaxy.
- Inference reused dataset `d7m83gjy2` from `j57er3q9p`, with the same compiled
  W8A16 artifact. [Repeatability](aihub-repeatability.json): all three outputs
  finite and exactly equal. This does not establish original-checkpoint quality.
- The profile preserves raw `all_inference_times`, load/memory fields, input
  shapes/types, options, graph detail and Hub version. `estimated_inference_time`
  is not relabeled a median or full-model latency.

`qualcomm_workbench.py` inventories, submits jobs for explicit existing model IDs,
and collects results. It uses a configured SDK profile or `--token-stdin`.
Example with an already configured profile, from the workspace root:

```bash
.venvs/aihub/bin/python analysis/experiments/omni_abstraction_20260922/qualcomm_workbench.py \
  collect --job jpyo7nx75 --out /tmp/s25-profile.json
```

No ADB device was connected. M1 still requires local prefill, 128-token generation,
KV/ring/sampling, 512/1024 transitions, cancel/reset and full resident memory.
M2 still requires local NPU/GPU co-residency, streaming recovery and 30-minute
thermal behavior. Separate Hub jobs do not establish these properties.

## Reproduction and environment

Final focused regression: [WSL](wsl-regression-shm.log) **415 passed, 1 skipped, 20
deselected**; [native Windows](windows-regression-shm.log) **78 passed, 2
skipped** (the controller lacks ONNX for two tests; actual ORT runs use separate
worker environments). The final Windows buffer/cancellation/retained-key follow-up
also passes [three tests](windows-shm-bounded.log). Ruff and diff whitespace checks pass. Earlier snapshots
and failed test-fixture setup attempts remain alongside the final logs.

[WSL inventory](wsl-inventory.json), [Windows inventory](windows-inventory.json),
[CPU](windows-cpu.json), [display](windows-display.json) and
[source baseline](source-baseline.txt) record actual imports/packages and hardware.
GPU: RTX 5090 **Laptop**, 24463 MiB, Windows driver 610.71. Desktop/server numbers
are not interchangeable. Commands below run from `vllm-omni-edge`:

```bash
../.venvs/omni-cuda-029/bin/python -m pytest -q -m 'not omni and not local_model' \
  tests/engine/test_graph_backend.py tests/edge/local \
  tests/config/test_vllm_native_pipelines.py tests/engine/test_async_omni_engine_stage_init.py \
  tests/engine/test_parallel_stage_init.py tests/engine/test_stage_runtime_launch_concurrency.py \
  tests/engine/test_stage_pool_collective_rpc.py tests/engine/test_stage_metadata.py \
  tests/entrypoints/test_omni_entrypoints.py tests/entrypoints/test_async_omni.py \
  tests/worker/test_forward_kwargs_filtering.py

../.venvs/omni-cuda-029/bin/python -m pytest -q tests/edge/local/test_e2e_local_text.py
```

Generate hashed probe bundles using `prepare_probes.py` from the workspace root.
Use the [public graph runner](../../../vllm-omni-edge/examples/edge/run_graph_stage.py)
and [route examples](../../../vllm-omni-edge/examples/edge/README.md). Reports retain
route, artifact identity, placement and reservation. Workers:
`.venvs/omni-cuda-029/bin/python` (WSL),
`C:\Users\zhout\npu-ep\Scripts\python.exe` (Windows CPU/VitisAI), and
`C:\Users\zhout\dml-ep\Scripts\python.exe` (DirectML).

Windows controller: `C:\Users\zhout\w2\omni029venv\Scripts\python.exe`;
`PYTHONPATH` is the ordinary UNC worktree path, `PYTHONUTF8=1`, and
`VLLM_CUDART_SO_PATH` points to that environment's `torch\lib\cudart64_13.dll`.
Usage statistics were disabled. No vendor wheel was modified here.

Package source/wheel identities are in [release-identities.json](release-identities.json).
The tested main wheel and standalone contracts wheel/sdist are preserved in
[`packages/`](packages/) in this evidence directory.
Build commands: `uv build --wheel --out-dir /tmp/omni-platform-shm-wheels` and
`uv build packages/omni-stage-contracts --out-dir /tmp/omni-contracts-final-wheels`.
The contracts sdist also installs in a fresh, dependency-free environment and
imports under `python -I` without NumPy, torch, vLLM or Omni. For the final main
wheel, install with `uv pip install --no-deps --target /tmp/omni-platform-bounded-check
/tmp/omni-platform-shm-wheels/*.whl`, then run the graph example from `/tmp`
with `PYTHONPATH=/tmp/omni-platform-bounded-check` using the existing vLLM environment.

## Retained failures

- Initial NPU execution passed but shutdown quarantined memory: socket closure
  preceded native identity verification. `graph-npu-crossdomain.json` retains it;
  the corrected `-2` run verifies release.
- Windows M0 `-2` failed in default worker-path discovery; planning passes after
  the fix. `-3`/`-4` were stopped while stalled. `-5` traced a blocking `/tmp`
  open through UNC, then exited with native access violation. The host-local
  path/creation fix is covered by lock and model reruns.
- TTS `-1` lacked complete artifacts. Pinned CustomVoice 0.6B revision
  `85e237c12c027371202489a0ec509ded67b5e4b5` was then downloaded. `-2`/`-3` lacked
  CUDA compiler/ninja paths; `-4` lacked the C++ frontend needed by FlashInfer
  sampling; `-5` exposed CUDA wrapper introspection. These tests defaulted to
  **dummy weights** and are not model-quality evidence. The introspection fix
  is tested with the actual vLLM wrapper. Real-weight tests require
  `--run-level advanced_model`; this environment explicitly selects the native
  vLLM sampler with `VLLM_USE_FLASHINFER_SAMPLER=0`.
- Native Windows `windows-tts-real-1.log` stopped in test-fixture setup because
  an unrelated diffusion test helper imports missing `diffusers`. Direct public
  benchmark `tts-stream-windows-1.log` then reported an incomplete snapshot:
  Windows could not follow the Linux cache symlinks. A native copy of the same
  pinned checkpoint was made without overwriting an existing bundle; all 13 files
  were SHA256-compared with the source. [Bundle provenance](tts-native-bundle.json).
- `tts-stream-windows-2.log` loaded the real talker weights, then failed publishing
  the compiler cache at a 261-character Windows path (`WinError 3`). A separate
  rerun explicitly sets `VLLM_CACHE_ROOT=C:\Users\zhout\c29`, shortening that
  target to 253 characters. This uses the runtime's cache setting, with no global
  Windows policy change and no precision/model substitution.
- `tts-stream-windows-3.log` passed stage compilation and initialized both
  stages with the short cache path, then stalled on the first request. The
  benchmark created `asyncio.run` before importing Omni and installing Windows
  selector policy. Its entry point now explicitly installs the existing policy
  before creating the loop, as the local-text CLI already does. The original
  attempt was retired with its two worker processes before retrying.
- `tts-stream-windows-4.log` still stalled with a verified selector loop.
  [Talker stack](windows-tts-talker-stack.txt) located the blocked file lock:
  `/dev/shm` resolved through the UNC checkout. Paths now select the actual host
  directory. Windows named mappings also needed retained producer handles and
  payload-length framing; closing the only handle immediately can destroy them.
  The bounded generation/ACK implementation and permanent-lock-error handling
  pass native cross-process and cancellation tests. The `-5` stream then completed.
- `forward-regression.log` used a nonexistent path; `-2` passed. Metadata fixture
  setup failures are retained; the corrected metadata/lock suite passed 95 tests
  in `metadata-lock-regression.log`.

Three early graph reports have ephemeral localhost handshake tokens redacted;
measurements remain unchanged. Historical vision/CPU/mobile/VLA results are not
promoted to proof of this implementation. Remaining required rows stay open.
