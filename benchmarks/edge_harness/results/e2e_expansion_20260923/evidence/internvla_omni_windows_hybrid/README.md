# InternVLA-A1 admitted Omni CPU+Radeon policy on native Windows

**Scoped synthetic complete-policy pass, not robot-task E2E.** The real
Place_Markpen checkpoint (revision
`0a557c6a503e6545b369bff2f34c76a410bbd190`) strict-loaded in Omni's
InternVLA diffusion policy on native Windows CPU. Its fixed-six-frame Cosmos
encoder call ran as a pinned FP32 `.pt2` graph in Omni's existing external
torch-directml worker, with the selected output device checked as
`AMD Radeon(TM) 890M Graphics`. The Qwen3-VL/flow action policy, its state and
the final action tensor stayed on CPU. The stage owns one bounded request,
explicit shared-RAM admission, terminal action event and in-flight
cancellation. The original Cosmos encoder remains BF16 in the CPU baseline;
FP32 is an explicit precision change because the tested DirectML BF16 route
aborted. The graph report's `fraction_on_target=1.0` counts its one output
tensor, **not internal per-operator placement**.

The host was Ryzen AI 9 HX 370, native Windows 11 build 26200, Radeon 890M
driver 32.0.22018.6001. The policy used Python 3.12.10, PyTorch
2.13.0+cu130 with CUDA masked, vLLM 0.29.0 and current Omni source. The
separate native DirectML worker used PyTorch 2.4.1+cpu and torch-directml
0.2.5.dev240914. The model, processor, Cosmos files, interpreter and graph
SHA256 values are in the [profile report](profile20.json). The graph itself is
129,872,312 bytes, SHA256
`9111b3841893acaaa11cf45715d246da3ac9cdd03c7b33fb3c7c20fe38d74704`,
stored outside Git at
`/home/zhout/project/edge_infer/models/InternVLA-cosmos-encoder-batch6.pt2`.
The [export report](../internvla_cosmos_radeon890m/export_batch6.json) and
[`export_internvla_cosmos.py`](../../../../experiments/export_internvla_cosmos.py)
permit regeneration. No model or graph was silently substituted.

The pinned request has three normalized float32 `[1,2,3,224,224]` patterned
camera histories, valid masks, zero `[1,32]` state, task text “Place the marker
pen in its holder.” and zero `[1,50,32]` diffusion noise. It returns finite
float32 `[1,50,32]` actions with a terminal epoch/sequence-tagged event and
observation/generation timestamps. The checkpoint declares `action_mode=delta`;
physical units, joint order and step time remain unverified and the output
reports `control_ready=false`.

| Check | Result | Raw record |
|---|---|---|
| Resident serial profile | One warmup excluded; 20/20 measured actions had the same hash. Nearest-rank complete-request wall p50/p95 **4.575/4.725 s**, excluding **140.31 s** startup. Loaded policy-plus-DirectML process-tree RSS was 11.69 GB, not a loading peak. | [report](profile20.json), [driver](profile20_driver.log), [worker](profile20_worker.log), [actions](profile20_actions.npy) |
| Native Windows CPU comparison | Separate [CPU baseline](../internvla_omni_windows_cpu/README.md) was 4.878/4.902 s p50/p95, so this warmed serial hybrid run was 0.303/0.178 s lower at those ranks; startup was 4.72 s longer. Action max absolute difference `0.09171`, relative L2 `2.4391%`, cosine `0.999705`. These are separate runs on one synthetic fixture, not paired randomized measurements or a task-quality pass. | [hybrid profile](profile20.json), [CPU profile](../internvla_omni_windows_cpu/cpu_profile20.json) |
| Public API | `AsyncOmni.generate` returned one finite terminal `action` event with the same action hash. Its startup was 170.50 s and request wall 4.374 s; OS-available RAM before launch was 22.96 GB. | [report](public.json), [driver](public_driver.log), [worker](public_worker.log) |
| Admission and cancellation | A declared 16 GiB host-RAM capacity/demand admitted the profile; an 8 GiB demand was refused before worker launch. In-flight abort produced no stale output; the reservation and quarantine counts were zero after abort and shutdown. | [profile](profile20.json), [refusal](refusal.json), [refusal log](refusal_driver.log) |

The [first complete-action smoke](smoke.json) passed but is one cold sample,
not a latency distribution. Native Windows uses the same action contract as
the [WSL CPU+Radeon path](../internvla_omni_policy/README.md), but the two
runtime/OS profiles are separate experiments. No real Place_Markpen
observation/reference-action tolerance, physical control metadata,
post-cancel restart, concurrent admission, loading peak or sustained
power/thermal behavior was verified. The small warm latency difference and
precision change do not yet justify making the hybrid placement a default.

Reproduce from the repository root in PowerShell with the pinned local files:

```powershell
$root = '\\wsl.localhost\Ubuntu\home\zhout\project\edge_infer'
$repo = Join-Path $root 'vllm-omni-edge'
$evidence = Join-Path $repo 'benchmarks\edge_harness\results\e2e_expansion_20260923\evidence\internvla_omni_windows_hybrid'
$py = 'C:\Users\zhout\w2\omni029venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $repo
$env:VLLM_ENABLE_V1_MULTIPROCESSING = '0'
$env:VLLM_OMNI_EXTERNAL_PYTHON_TORCH_DML = 'C:\Users\zhout\w2\qwen_dml_venv\Scripts\python.exe'
$common = @(
  '--model-dir', (Join-Path $root 'models\InternVLA-A1-3B-FT-Place_Markpen'),
  '--processor-dir', (Join-Path $root 'models\Qwen3-VL-2B-Instruct-processor'),
  '--cosmos-dir', (Join-Path $root 'models\Cosmos-Tokenizer-CI8x8'),
  '--graph-file', (Join-Path $root 'models\InternVLA-cosmos-encoder-batch6.pt2'),
  '--python-bin', $py, '--placement', 'radeon-cosmos',
  '--capacity-gib', '16', '--reserve-gib', '16'
)
& $py (Join-Path $repo 'benchmarks\edge_harness\experiments\probe_omni_internvla_policy.py') @common `
  --warmups 1 --repeats 20 --abort-check `
  --reference-actions (Join-Path $repo 'benchmarks\edge_harness\results\e2e_expansion_20260923\evidence\internvla_omni_windows_cpu\cpu_profile20_actions.npy') `
  --log-file (Join-Path $evidence 'profile20_worker.log') `
  --output-report (Join-Path $evidence 'profile20.json') `
  --output-actions (Join-Path $evidence 'profile20_actions.npy')
& $py (Join-Path $repo 'benchmarks\edge_harness\experiments\probe_omni_internvla_entrypoint.py') @common `
  --log-file (Join-Path $evidence 'public_worker.log') `
  --output-report (Join-Path $evidence 'public.json')
```

Next gate: real observations/reference actions and a task tolerance, physical
action metadata and deadlines, then post-cancel restart, peak shared memory,
concurrent load and sustained tail/power/thermal measurement.
