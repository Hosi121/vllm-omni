# InternVLA Cosmos image-encoder component on Radeon 890M

This is a **component (C) pass**, not an InternVLA action-policy (P) pass. The unchanged Omni `model_cosmos.py` and `cosmos_ci_torch.py` loaded the pinned Cosmos-Tokenizer-CI8x8 encoder and executed one patterned `[1,3,256,256]` float32 image in `[-1,1]` on CPU and DirectML Radeon 890M. The pattern is synthetic; no camera, language, proprioception, diffusion action loop, unit/timestamp or robot-task quality is covered.

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
