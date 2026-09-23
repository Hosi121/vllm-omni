# InternVLA Cosmos encoder and synthetic hybrid policy on Radeon 890M

The first experiment below is a **component (C) pass**. The unchanged Omni `model_cosmos.py` and `cosmos_ci_torch.py` loaded the pinned Cosmos-Tokenizer-CI8x8 encoder and executed one patterned `[1,3,256,256]` float32 image in `[-1,1]` on CPU and DirectML Radeon 890M. The pattern is synthetic; this component result alone covers no camera, language, proprioception, diffusion action loop, unit/timestamp or robot-task quality. A later section records a separate complete synthetic policy path.

The host was a Ryzen AI 9 HX 370 on native Windows 11 build 26200, AMD Radeon 890M driver 32.0.22018.6001 (2026-08-12), PyTorch 2.4.1+cpu and torch-directml 0.2.5.dev240914. The separate DirectML environment is not the installed Omni/vLLM 0.29 policy environment. Explicit DirectML device index 1 resolved to `AMD Radeon(TM) 890M Graphics`; all encoder parameters and the pre-readback output were checked on `privateuseone`. The encoder checkpoint SHA256 is `520a3b878e72ee872faccb1211b5d622d194c0fdd177f2f6df097586b8fa98e5`. Source SHA256 values are in [profile20.json](profile20.json). The 88 persistent encoder state tensors all had checkpoint keys. Two nonpersistent patcher buffers were also present in the checkpoint and ignored by the existing model loader; the wavelet buffer differs by at most `7.55e-05` because the checkpoint uses BF16 while the source constructs FP32. The same source-generated buffer was used for CPU and DirectML.

One warmup followed by 20 serial measured calls produced `[1,16,32,32]` outputs. The complete per-call timing includes host-to-DirectML upload, encoder forward, DirectML-to-host readback/synchronization and float32 conversion. Nearest-rank wall p50/p95 were **26.56/27.49 ms** on Radeon and **156.49/158.71 ms** on CPU. The DirectML result versus this CPU source had max absolute error `1.168e-05`, relative L2 error `4.732e-07`, and cosine similarity `0.99999994`. Load plus weight transfer took 0.71 s after CPU construction. These are one-shape, warmed component observations, without external device-utilization trace, process isolation, peak shared-RAM, concurrent load, power or sustained thermal measurement. They cannot be added to prior whole-policy times to claim a hybrid speedup; the full handoff, co-residency, policy output and action quality must be measured together.

The [first failed report](first_report.json) and [log](first_driver.log) preserve a DirectML `torch.inference_mode()` version-counter failure in the first experiment. Using `torch.no_grad()` in the probe resolved that runtime incompatibility without changing model operations. The final [profile report](profile20.json) records checkpoint-key coverage, the two nonpersistent buffers and the successful run. The [single-call pass](second_report.json) preceded the 20-request measurement.

Reproduce in PowerShell from this checkout, with `PYTHONUTF8=1` and the recorded DirectML environment:

```powershell
$root = '\\wsl.localhost\Ubuntu\home\zhout\project\edge_infer'
$env:PYTHONUTF8 = '1'
& 'C:\Users\zhout\w2\qwen_dml_venv\Scripts\python.exe' `
  "$root\vllm-omni-edge\benchmarks\edge_harness\experiments\probe_internvla_cosmos_directml.py" `
  --encoder "$root\models\Cosmos-Tokenizer-CI8x8\encoder.safetensors" `
  --output "$root\vllm-omni-edge\benchmarks\edge_harness\results\e2e_expansion_20260923\evidence\internvla_cosmos_radeon890m\profile20.json" `
  --warmups 1 --repeats 20 --dml-index 1
```

## Fixed six-frame component and complete synthetic policy

The policy's three-camera × two-history fixture sends **six** frames through
Cosmos in one call. A separate [six-frame component profile](profile_batch6_20.json)
used one warmup plus 20 serial patterned calls: Radeon p50/p95
133.82/144.88 ms including upload/readback, CPU FP32 835.17/851.48 ms,
relative L2 error `5.14e-07`. This is a more relevant shape than the first
single-image component test, but still not a whole-policy measurement.

DirectML in the tested PyTorch 2.4 worker **aborts on BF16**; the
[fatal log](bf16_batch6_probe.log) is retained. The existing InternVLA policy
uses BF16 Cosmos on CPU. Rather than silently cast it, the hybrid experiment
uses an explicitly FP32 fixed-shape [`.pt2` export](export_batch6.json) of the
same checkpoint, then converts the returned latent to the policy's BF16.
`torch.export(..., strict=False)` produced a CPU output identical to the
unchanged source on the export sample. The 129,872,312-byte graph is stored
outside Git at the report's artifact path; its SHA256 is
`9111b3841893acaaa11cf45715d246da3ac9cdd03c7b33fb3c7c20fe38d74704`.
The experiment refuses a hash, shape or dtype mismatch. Recreate it with
[`export_internvla_cosmos.py`](../../../../experiments/export_internvla_cosmos.py).

The [hybrid policy probe](../../../../experiments/probe_internvla_hybrid_actions.py)
loaded the real Place_Markpen checkpoint strictly in the WSL CPU Omni
diffusion pipeline, kept the Qwen3-VL/flow action policy on CPU and used
Omni's existing external `torch-dml` worker for only the Cosmos encoder.
The worker reported the selected adapter as `AMD Radeon(TM) 890M Graphics`;
its output remained on DirectML before readback. Its graph placement report
counts one **output tensor**, not every internal operation; no per-operator
attribution was captured. The controller used PyTorch 2.13.0+cpu, vLLM
0.28.0 and current Omni 0.29 source (a version mismatch warning remains);
the worker used PyTorch 2.4.1+cu121 over WSL GPU-PV. The underlying host was
the same Windows 11 build 26200 HX370/Radeon with driver 32.0.22018.6001.
Both policies received identical synthetic zero camera/state observations,
empty task and zero diffusion noise; output was a finite `[1,50,32]` action
tensor. This is a **synthetic complete policy path**, not a real robot-task
E2E result. The checkpoint's `train_config.json` says `action_mode=delta`,
but physical units, joint order, observation timestamps, action horizon and
reference action quality were not validated.

The [paired 20-request result](hybrid_profile20.json) alternated CPU-only and
CPU+Radeon complete policy calls after one warmup pair, with one loaded policy
and persistent worker. Nearest-rank complete-request wall p50/p95 was
**3.178/3.266 s** on CPU and **2.952/3.322 s** on the hybrid. The paired
median improvement was 0.207 s (16/20 hybrid calls faster), but the hybrid
p95 regressed because several DirectML calls took about 0.5 s instead of
about 0.15 s. The largest action max-absolute difference was `0.00715`,
relative L2 `0.9007%`, cosine `0.999959`; the difference was repeatable on
this one fixture. No task tolerance is established, and the FP32/BF16 change
needs real-observation reference actions before deployment. Policy load was
32.96 s, worker start+graph load 3.34 s; these costs are excluded from the
request percentiles. Parent RSS after profiling was 9.30 GB and worker peak
RSS 0.91 GB; these are not shared-iGPU allocation or loading-peak bounds.
Power and thermal state were not recorded.

The [first hybrid smoke](hybrid_smoke.json) and the later
[selected-adapter check](hybrid_selected_adapter.json) are retained. The
candidate still lacks an Omni graph-stage deployment plan with combined
shared-RAM admission, cancellation/restart validation, observation/action
metadata and real input quality. A single cold request loses the worker
startup cost, and the latency tail currently prevents a general speedup
claim. Keep CPU as the qualified baseline until those gates pass.

Recreate the graph with the recorded Windows PyTorch 2.4 environment. Use the
freshly generated report alongside its graph, since serialization may produce
a different byte hash on another run:

```powershell
$root = '\\wsl.localhost\Ubuntu\home\zhout\project\edge_infer'
$env:PYTHONUTF8 = '1'
& 'C:\Users\zhout\w2\qwen_dml_venv\Scripts\python.exe' `
  "$root\vllm-omni-edge\benchmarks\edge_harness\experiments\export_internvla_cosmos.py" `
  --encoder "$root\models\Cosmos-Tokenizer-CI8x8\encoder.safetensors" `
  --output "$root\models\InternVLA-cosmos-encoder-batch6.pt2" `
  --report "$root\models\InternVLA-cosmos-encoder-batch6.export.json" --batch-size 6
```

Reproduce the complete synthetic policy probe from this repository root on
WSL, using the matching local export report:

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  ../.venvs/omni-cpu/bin/python benchmarks/edge_harness/experiments/probe_internvla_hybrid_actions.py \
  --model-dir ../models/InternVLA-A1-3B-FT-Place_Markpen \
  --cosmos-dir ../models/Cosmos-Tokenizer-CI8x8 \
  --processor-dir ../models/Qwen3-VL-2B-Instruct-processor \
  --graph ../models/InternVLA-cosmos-encoder-batch6.pt2 \
  --export-report ../models/InternVLA-cosmos-encoder-batch6.export.json \
  --output /tmp/internvla-hybrid-profile.json --warmups 1 --repeats 20
```
