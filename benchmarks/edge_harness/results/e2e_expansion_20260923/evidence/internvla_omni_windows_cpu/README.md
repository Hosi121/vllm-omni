# InternVLA-A1 admitted Omni whole-policy stage on native Windows CPU

**Scoped synthetic action-policy pass, not robot-task E2E.** The real
Place_Markpen checkpoint (revision
`0a557c6a503e6545b369bff2f34c76a410bbd190`) strict-loaded in the
existing Omni InternVLA diffusion policy. One `StageRuntime`/`StagePool` graph
stage owns an isolated whole-policy worker, bounded one-request capacity,
terminal action event, shared-RAM reservation and cancellation. The model
parameters and output actions were checked on CPU; the CUDA-capable wheel does
not imply GPU execution. This is the native Windows counterpart of the
[WSL CPU/Radeon Omni policy record](../internvla_omni_policy/README.md).

The host was Ryzen AI 9 HX 370 running Windows 11 build 26200. The controller
and worker used Python 3.12.10, PyTorch 2.13.0+cu130 on CPU, vLLM 0.29.0 and
the current Omni source. The real checkpoint, processor and Cosmos files were
read over the local WSL UNC path. Their SHA256 values, interpreter hash and
worker properties are in the [20-request report](cpu_profile20.json). The
fixture comprises three normalized float32 `[1,2,3,224,224]` patterned camera
histories, valid masks, zero `[1,32]` state, the task text “Place the marker
pen in its holder.” and zero `[1,50,32]` diffusion noise. Each response is a
finite float32 `[1,50,32]` action chunk with an epoch/sequence-tagged terminal
event and observation/generation timestamps. The checkpoint declares
`action_mode=delta`; physical units, joint order and per-action step time are
unverified and `control_ready=false`.

| Check | Result | Evidence |
|---|---|---|
| Resident serial profile | One warmup excluded; 20/20 requests had the same action hash. Nearest-rank complete-request wall p50/p95 **4.878/4.902 s**, excluding **135.60 s** startup. Loaded worker-tree RSS was 9.00 GB, not loading peak. | [report](cpu_profile20.json), [driver](cpu_profile20_driver.log), [worker](cpu_profile20_worker.log), [actions](cpu_profile20_actions.npy) |
| WSL CPU numerical comparison | Same pinned patterned fixture: max absolute action difference `0.06504`, relative L2 `2.1795%`, cosine `0.999764`. This is numerical comparison without a task tolerance. | [profile report](cpu_profile20.json), [WSL reference](../internvla_omni_policy/cpu_profile20_actions.npy) |
| Public entrypoint | `AsyncOmni.generate` returned one terminal `action` event with the same action hash and finite shape `[1,50,32]`. Startup **134.61 s**, request wall **4.979 s**; OS-available RAM before launch was 23.67 GB. | [report](public.json), [driver](public_driver.log), [worker](public_worker.log) |
| Admission and abort | A 16 GiB stage demand/capacity ran within the Windows available-RAM check. An 8 GiB demand was refused before model-worker launch. In-flight abort produced no stale output; the ledger had zero reserved and quarantined bytes after abort and shutdown. | [profile report](cpu_profile20.json), [refusal](refusal.json), [refusal log](refusal_driver.log) |

The [first smoke](cpu_smoke.json) used a 24 GiB declared capacity when Windows
reported about 20 GiB free, so it is retained as exploratory evidence only;
the measured run uses a conservative 16 GiB capacity and demand. A first
[public smoke](public_overdeclared_capacity.json) also used a 30 GiB ceiling;
its valid action does not qualify memory admission. The public probe now
checks that its declared ceiling fits OS-available RAM before loading, and
the corrected [public rerun](public.json) passed. The probe's declared memory
ceiling is a controller admission budget, not a measured loading peak or a
claim that Windows could admit another stage concurrently. No real
Place_Markpen observations or reference actions, robot-task tolerance,
post-cancel restart, concurrency or sustained power/thermal behavior were
verified.

Reproduce from the repository root in PowerShell, using the pinned files and
native environment already present on this host:

```powershell
$root = '\\wsl.localhost\Ubuntu\home\zhout\project\edge_infer'
$repo = Join-Path $root 'vllm-omni-edge'
$evidence = Join-Path $repo 'benchmarks\edge_harness\results\e2e_expansion_20260923\evidence\internvla_omni_windows_cpu'
$py = 'C:\Users\zhout\w2\omni029venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $repo
$env:VLLM_ENABLE_V1_MULTIPROCESSING = '0'
$common = @(
  '--model-dir', (Join-Path $root 'models\InternVLA-A1-3B-FT-Place_Markpen'),
  '--processor-dir', (Join-Path $root 'models\Qwen3-VL-2B-Instruct-processor'),
  '--cosmos-dir', (Join-Path $root 'models\Cosmos-Tokenizer-CI8x8'),
  '--python-bin', $py, '--placement', 'cpu',
  '--capacity-gib', '16', '--reserve-gib', '16'
)
& $py (Join-Path $repo 'benchmarks\edge_harness\experiments\probe_omni_internvla_policy.py') @common `
  --warmups 1 --repeats 20 --abort-check `
  --reference-actions (Join-Path $repo 'benchmarks\edge_harness\results\e2e_expansion_20260923\evidence\internvla_omni_policy\cpu_profile20_actions.npy') `
  --log-file (Join-Path $evidence 'cpu_profile20_worker.log') `
  --output-report (Join-Path $evidence 'cpu_profile20.json') `
  --output-actions (Join-Path $evidence 'cpu_profile20_actions.npy')
& $py (Join-Path $repo 'benchmarks\edge_harness\experiments\probe_omni_internvla_entrypoint.py') @common `
  --log-file (Join-Path $evidence 'public_worker.log') `
  --output-report (Join-Path $evidence 'public.json')
```

Next gate: real observation/reference-action agreement and physical action
metadata, then loading peak, post-cancel restart, concurrent admission and
sustained deadline/power/thermal behavior.
