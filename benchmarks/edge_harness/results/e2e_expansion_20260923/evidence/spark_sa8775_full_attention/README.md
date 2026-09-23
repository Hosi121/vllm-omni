# Spark-X2.5 attention layer on SA8775P ADP

**One fixed-shape attention layer from the real Spark-X2.5-1.7B checkpoint executed on the hosted SA8775P ADP NPU (Android 14). This is component evidence, not complete Spark generation or an embedded Omni deployment.**

The [Galaxy S24 provenance record](../spark_s24_full_attention/README.md) pins the FP32 source ONNX, W8A16 QDQ ONNX, calibration and exact input fixture. Its QNN DLC compile job `jpe7z8l15` produced target model `mq9y70k0n`; the same DLC and fixture dataset `d7m83gjy2` were reused here without recompilation. [Inference submission](inference_submission.json) job `jp1nj7j2g` and [profile submission](profile_submission.json) job `jgdd383eg` both requested `--compute_unit npu` for exact device **SA8775P ADP**. The completed [inference](inference_report.json) and [profile](profile_report.json) reports pin service build, Android version and chipset.

The [three outputs](device_output.npz) were finite FP32 hidden/new-key/new-value tensors at a 1,024-token cache shape. They were bitwise identical to the earlier S24/S25 NPU outputs on the same fixture. The [audit](audit_report.json) measured hidden/key/value relative L2 of **1.351%/0.377%/0.650%** against original FP32 ONNX CPU and **0.555%/0.0065%/0.0126%** against QDQ ONNX CPU. No next-token or task-quality tolerance has been established.

All 73 execution-detail rows in the profile were attributed to NPU. The 100 component samples had nearest-rank p50/p95 **1.837/2.079 ms** (minimum 1.749 ms); the service's `estimated_inference_time` was that minimum, not the median. Reported component inference peak memory was 90,877,952 bytes. These hosted component measurements exclude prefill, 27 other decoder layers, sampling, handoff, loading and a resident KV/ring/controller. They are not paired speed comparisons with other devices or OS builds.

The [audit script](../../../../experiments/audit_spark_qualcomm_attention.py) reproduces the provenance and numerical checks using pinned source artifact paths in the S24 record. The [collector](../../../../experiments/collect_spark_aihub_component.py) retrieves these existing jobs via the local client configuration; neither script contains credentials.

**Matrix status: component NPU inference/profiled; NOT E2E.** Full qualification needs a complete device-resident Spark artifact and Omni backend with prefill/decode/KV-ring/sampling, state, token quality, cancellation and memory admission, followed by sustained power and thermal measurement.
