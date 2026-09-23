# Native Windows RTX 5090 Laptop: admitted InternVLA Omni action stage

**Scoped synthetic policy path passed; robot-task E2E remains unqualified.** The real `InternVLA-A1-3B-FT-Place_Markpen` checkpoint (revision `0a557c6a503e6545b369bff2f34c76a410bbd190`) strict-loaded from the WSL UNC model path into the existing Omni `InternVLAA1Pipeline`. A bounded whole-policy graph stage kept the BF16 policy and Cosmos encoder on the NVIDIA GeForce RTX 5090 **Laptop** GPU under native Windows 11 build 26200. The output was one terminal float32 `[1,50,32]` host action chunk per request. Worker properties and runtime checks verified CUDA placement, exact model/interpreter artifact hashes, and absence of the external DirectML route.

The host was Ryzen AI 9 HX 370 with NVIDIA driver 610.71. The worker used native Python 3.12.10, PyTorch 2.13.0+cu130, installed vLLM 0.29.0 and this Omni checkout. The fixture was three normalized patterned synthetic two-frame camera histories, valid masks, zero `[1,32]` state, the fixed Place_Markpen task and zero `[1,50,32]` diffusion noise. Neither this fixture nor the resulting actions establish physical task quality.

After one warmup, **20/20 serial complete policy requests** produced the same finite action hash. Nearest-rank complete-request wall p50/p95 was **0.746/0.806 s** (range 0.698–0.806 s), excluding **147.77 s** controller/worker startup. Each measured request used the same resident worker. A separate [public `AsyncOmni.generate` request](public.json) passed with the same action hash and one terminal event; its first-request wall was 1.862 s after 155.16 s startup. These timing samples are not a comparison with the earlier direct zero-observation CUDA forward or the separately measured WSL CUDA profile.

The declared budget was **16 GiB host RAM plus 16 GiB `cuda:0` VRAM** under 18/21 GiB ceilings. At the start of the [20-request profile](profile20.json), Windows had 21.58 GB available host RAM and CUDA reported 24.28 GB free. The worker reported 7.80 GB PyTorch CUDA reserved after load and a maximum **7.86 GB** peak reserved across measured requests. Sampled loaded process-tree RSS was 4.05 GB; it is not the host loading peak, and PyTorch's CUDA reservation is not a full-device VRAM trace. An [8 GiB VRAM demand](refusal.json) was rejected before worker launch because pinned weights plus declared workspace/headroom exceeded the reservation. In-flight cancellation produced no stale action and released both ledgers; shutdown left zero reservation or quarantine.

The first measured action chunk differed from the separate native Windows CPU Omni action array by maximum absolute **0.07049** and relative L2 **2.340%**; see the exact values in the [profile report](profile20.json). Against the separately profiled [WSL CUDA action array](wsl_cuda_parity.json), relative L2 was **1.555%** and cosine 0.999886. No task tolerance has been established, so neither comparison proves task equivalence. Metadata explicitly says `action_mode=delta`, unverified physical units and joint order, unknown step time, and `control_ready=false`.

The [driver](profile20_driver.log), [worker](profile20_worker.log), first measured [actions](profile20_actions.npy), [public logs](public_driver.log) and [refusal log](refusal_driver.log) are retained. The [preflight](preflight.json) documents available host/GPU memory before the Windows attempt; availability was sampled again by each probe. Real robot observations/reference actions, deadlines, post-cancel restart, concurrent admission, loading peak, sustained tail latency, power and thermals remain open.

Reproduce in PowerShell with the pinned local artifacts:

```powershell
$root = '\\wsl.localhost\Ubuntu\home\zhout\project\edge_infer'
$repo = Join-Path $root 'vllm-omni-edge'
$evidence = Join-Path $repo 'benchmarks\edge_harness\results\e2e_expansion_20260923\evidence\internvla_omni_windows_cuda'
$py = 'C:\Users\zhout\w2\omni029venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $repo
$env:VLLM_ENABLE_V1_MULTIPROCESSING = '0'
& $py (Join-Path $repo 'benchmarks\edge_harness\experiments\probe_omni_internvla_policy.py') `
  --model-dir (Join-Path $root 'models\InternVLA-A1-3B-FT-Place_Markpen') `
  --processor-dir (Join-Path $root 'models\Qwen3-VL-2B-Instruct-processor') `
  --cosmos-dir (Join-Path $root 'models\Cosmos-Tokenizer-CI8x8') `
  --python-bin $py --placement cuda `
  --capacity-gib 18 --reserve-gib 16 --vram-capacity-gib 21 --vram-reserve-gib 16 `
  --warmups 1 --repeats 20 --abort-check `
  --log-file (Join-Path $evidence 'profile20_worker.log') `
  --output-report (Join-Path $evidence 'profile20.json') `
  --output-actions (Join-Path $evidence 'profile20_actions.npy') `
  --reference-actions (Join-Path $repo 'benchmarks\edge_harness\results\e2e_expansion_20260923\evidence\internvla_omni_windows_cpu\cpu_profile20_actions.npy')
```
