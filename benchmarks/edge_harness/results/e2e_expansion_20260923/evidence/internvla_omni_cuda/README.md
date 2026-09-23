# WSL RTX 5090 Laptop: admitted InternVLA Omni action stage

**Scoped synthetic policy path passed; robot-task E2E remains unqualified.** The real `InternVLA-A1-3B-FT-Place_Markpen` checkpoint (revision `0a557c6a503e6545b369bff2f34c76a410bbd190`) strict-loaded through the existing Omni `InternVLAA1Pipeline` in a bounded whole-policy graph stage. Both the BF16 policy and BF16 Cosmos encoder ran on the NVIDIA GeForce RTX 5090 **Laptop** GPU; action output was copied to a terminal float32 `[1,50,32]` host buffer. This is an explicit `cuda` plan, not a CPU fallback or a GPU component summed into a pipeline estimate.

The host was Ryzen AI 9 HX 370 under Ubuntu 26.04/WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`, with Windows NVIDIA driver 610.71 and 24,463 MiB GPU memory. The worker used Python 3.12, PyTorch 2.13.0+cu130 and installed vLLM 0.29.0 with this Omni checkout. Checkpoint, processor, Cosmos and interpreter SHA256 values, actual device name, precision and state metadata are in the [20-request report](final_profile20.json). The input was three patterned normalized synthetic two-frame camera histories, valid masks, zero `[1,32]` state, fixed text task and zero `[1,50,32]` diffusion noise. It was **not** a real robot observation or a reference-action quality case.

After one warmup, **20/20 serial complete policy requests** returned one identical finite action hash. Nearest-rank complete-request wall p50/p95 was **0.382/0.399 s** (range 0.367–0.408 s), excluding **43.82 s** controller/worker startup; each request remained in the same loaded worker. The separate [public `AsyncOmni.generate` request](final_public.json) passed with one terminal action event; its first-request wall was 0.965 s after 45.69 s startup. These numbers are specific to this fixed synthetic input, one concurrency slot and current power/system state. The earlier direct CUDA zero-observation forward is a different workload and measurement boundary.

The stage reserved **16 GiB host RAM plus 16 GiB `cuda:0` VRAM** under explicit 28/21 GiB ceilings, each checked against OS/GPU availability before loading. The real policy reported 7.80 GB CUDA reserved after load; the maximum PyTorch CUDA peak reserved across measured requests was **7.86 GB**, below the VRAM reservation. Loaded worker process-tree RSS was 8.62 GB, which is not a loading-peak bound. An [8 GiB VRAM demand](refusal.json) was rejected before worker launch because pinned policy/Cosmos weights plus explicit workspace/headroom exceeded the reservation; no refusal worker log was created. The [in-flight abort](final_profile20.json) emitted no stale action and left zero host/VRAM reservation or quarantine; shutdown did likewise.

Against the separate patterned WSL CPU Omni action array, CUDA's first measured chunk had maximum absolute difference **0.01890**, relative L2 **2.006%** and cosine **0.999809**. No task tolerance has been defined, so this is numerical characterization, not task equivalence. Separate [CPU](cpu_regression.json) and [CPU+Radeon](radeon_regression.json) one-request checks passed after the worker response added CUDA peak metadata (zero on both), preserving those routes' response contract. The [native Windows CUDA Omni route](../internvla_omni_windows_cuda/README.md) was subsequently profiled separately under its own host-memory condition; its timing and action output must not be substituted for this WSL run.

The action metadata explicitly reports `action_mode=delta`, unverified units and joint order, unknown step time, and `control_ready=false`. Physical action quality, deadlines, real observation/reference pairs, post-cancel restart, concurrent admission, weight-loading peak, full-device GPU memory, sustained tail latency, power and thermal behavior remain open. The [driver](final_profile20_driver.log) and [worker](final_profile20_worker.log) logs, first measured [action array](final_profile20_actions.npy), [public logs](final_public_driver.log), [refusal log](refusal_driver.log) and [CPU regression log](cpu_regression_driver.log) are retained.

Reproduce from this checkout with the pinned local artifacts:

```bash
ROOT=/home/zhout/project/edge_infer
EVIDENCE=benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/internvla_omni_cuda
PYTHONPATH="$PWD" "$ROOT/.venvs/omni-cuda-029/bin/python" benchmarks/edge_harness/experiments/probe_omni_internvla_policy.py \
  --model-dir "$ROOT/models/InternVLA-A1-3B-FT-Place_Markpen" \
  --processor-dir "$ROOT/models/Qwen3-VL-2B-Instruct-processor" \
  --cosmos-dir "$ROOT/models/Cosmos-Tokenizer-CI8x8" \
  --python-bin "$ROOT/.venvs/omni-cuda-029/bin/python" --placement cuda \
  --capacity-gib 28 --reserve-gib 16 --vram-capacity-gib 21 --vram-reserve-gib 16 \
  --warmups 1 --repeats 20 --abort-check \
  --log-file "$EVIDENCE/final_profile20_worker.log" \
  --output-report "$EVIDENCE/final_profile20.json" \
  --output-actions "$EVIDENCE/final_profile20_actions.npy" \
  --reference-actions benchmarks/edge_harness/results/e2e_expansion_20260923/evidence/internvla_omni_policy/cpu_profile20_actions.npy
```
