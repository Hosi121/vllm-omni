# InternVLA-A1 admitted Omni complete-policy stage: WSL CPU and CPU+Radeon 890M

**Depth P for the specified synthetic action-policy request only; not robot-task E2E.** The real Place_Markpen checkpoint is strict-loaded by the existing Omni `InternVLAA1Pipeline`. An explicit single graph stage wraps the complete policy. Omni `StageRuntime`/`StagePool` own shared-RAM admission, one outstanding request, terminal action output, acknowledgement, cancellation and worker lifetime. The policy remains on WSL CPU. The optional hybrid route sends only the six-frame Cosmos encoder call through Omni's existing external DirectML worker on Radeon 890M, then returns its latent to the CPU policy. It requires a pinned FP32 graph because the tested DirectML BF16 route aborted; CPU-only Cosmos remains BF16. There is no silent placement or precision fallback.

The host was Ryzen AI 9 HX 370, Windows 11 build 26200, AMD Radeon 890M driver 32.0.22018.6001, and WSL with 30.91 GiB RAM limit. The policy worker used PyTorch 2.13.0+cpu and installed vLLM 0.28.0 with current Omni 0.29 source (version mismatch warning). The external WSL GPU-PV worker used PyTorch 2.4.1+cu121 with torch-directml. The model revision was `0a557c6a503e6545b369bff2f34c76a410bbd190`; model, Cosmos, processor, interpreter and graph SHA256 values are in both profile JSON files. The external worker selected `AMD Radeon(TM) 890M Graphics`, and its graph output was observed on DirectML before readback. The report's `fraction_on_target=1.0` describes the one output tensor, **not internal operator placement**. No device utilization trace was captured.

The fixed input has three camera histories, each normalized float32 `[1,2,3,224,224]` with simple colored squares, three valid masks, zero `[1,32]` state, a fixed text task, and zero `[1,50,32]` diffusion noise. The stage rejects other shapes/dtypes, nonfinite inputs and an insufficient reservation before dispatch. Both routes returned finite float32 `[1,50,32]` actions, terminal `action` events with sequence and epoch, and observation/generation timestamps. The checkpoint says `action_mode=delta`; physical units, joint order and per-action step time are unverified and reported as such (`control_ready=false`). The generated timestamp is a software completion time, not a robot actuation deadline.

| Route | Warmup | Measured serial requests | Complete-request wall p50/p95, nearest rank | Startup | Loaded process-tree RSS |
|---|---:|---:|---:|---:|---:|
| Omni CPU | 1 | 20/20, same action hash | 3.175/3.219 s | 40.03 s | 9.28 GB |
| Omni CPU + Radeon Cosmos | 1 | 20/20, same action hash | 3.120/3.197 s | 43.36 s | 9.94 GB |

The warm median difference is 55.7 ms in favor of the hybrid and its p95 difference is 22.3 ms. These are separate serial runs on one pinned synthetic fixture, not a paired randomized comparison or a sustained speedup claim. The hybrid took 3.33 s longer to start. Relative to this CPU output, the hybrid action chunk differs by max absolute `0.07546`, relative L2 `2.305%`, cosine `0.999737`. No task-quality tolerance exists. This comparison is **not** the earlier zero-observation direct hybrid experiment, which had different input/runtime and different timings. Neither run measured power, thermal behavior, loading peak, shared GPU allocation or concurrency. The loaded process-tree RSS sample is not a peak-memory bound.

The declared WSL host-RAM capacity was 30 GiB and each successful stage reserved 16 GiB for weights, policy/KV-like state, activation/workspace, transfers and headroom. [An 8 GiB CPU reservation](refusal_cpu.json) was refused before model-worker launch; the ledger ended at zero. The [CPU profile](cpu_profile20.json) and [hybrid profile](radeon_profile20.json) each ended with zero reserved/quarantined bytes. A [public `AsyncOmni.generate` hybrid request](public_radeon.json) returned one terminal action output. The [in-flight hybrid abort](radeon_abort.json) produced no stale output and released the reservation. The [first hybrid smoke](radeon_smoke.json) produced correct actions but left the ledger quarantined when a DirectML child became a transient zombie at shutdown. The worker-tree retirement fix was verified by [smoke rerun](radeon_smoke2.json), abort and both profiles; the failed record is retained rather than being presented as a pass. Post-cancel restart in one process remains unverified.

Raw [CPU](cpu_profile20_driver.log), [hybrid](radeon_profile20_driver.log), [abort](radeon_abort_driver.log) and [public entrypoint](public_radeon_driver.log) driver logs and the matching worker logs are retained beside these reports. First measured action arrays are [CPU](cpu_profile20_actions.npy) and [hybrid](radeon_profile20_actions.npy). The graph is 129,872,312 bytes and SHA256 `9111b3841893acaaa11cf45715d246da3ac9cdd03c7b33fb3c7c20fe38d74704`; it is stored outside Git at `/home/zhout/project/edge_infer/models/InternVLA-cosmos-encoder-batch6.pt2`. Recreate it with [`export_internvla_cosmos.py`](../../../../experiments/export_internvla_cosmos.py); its export report is in the [earlier Cosmos evidence](../internvla_cosmos_radeon890m/export_batch6.json).

A public-probe rerun without this checkout on `PYTHONPATH` failed at import before model loading; its [setup failure log](public_missing_pythonpath_driver.log) is retained. With `PYTHONPATH` set explicitly, the [public rerun](public_radeon.json) passed after the final worker-property checks.

Reproduce from this checkout after supplying the pinned model, processor, Cosmos and graph artifacts. The `cpu` route omits `--graph-file`; change `--placement` to `radeon-cosmos` and add it for the hybrid route. This example records one warmup plus 20 measured requests under the same declared capacity and demand:

```bash
ROOT=/home/zhout/project/edge_infer
EVIDENCE=benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/internvla_omni_policy
PYTHONPATH="$PWD" "$ROOT/.venvs/omni-cpu/bin/python" benchmarks/edge_harness/experiments/probe_omni_internvla_policy.py \
  --model-dir "$ROOT/models/InternVLA-A1-3B-FT-Place_Markpen" \
  --processor-dir "$ROOT/models/Qwen3-VL-2B-Instruct-processor" \
  --cosmos-dir "$ROOT/models/Cosmos-Tokenizer-CI8x8" \
  --python-bin "$ROOT/.venvs/omni-cpu/bin/python" \
  --placement radeon-cosmos \
  --graph-file "$ROOT/models/InternVLA-cosmos-encoder-batch6.pt2" \
  --capacity-gib 30 --reserve-gib 16 --warmups 1 --repeats 20 \
  --log-file "$EVIDENCE/radeon_profile20_worker.log" \
  --output-report "$EVIDENCE/radeon_profile20.json" \
  --output-actions "$EVIDENCE/radeon_profile20_actions.npy" \
  --reference-actions "$EVIDENCE/cpu_profile20_actions.npy"
```

Next gate: real Place_Markpen observation/reference-action pairs and task tolerance, verified physical action units/order/step time and deadlines, then loading peak, concurrent admission, post-cancel restart and sustained tail/power/thermal measurements. The CPU route remains the default until the hybrid route wins that complete gate.
