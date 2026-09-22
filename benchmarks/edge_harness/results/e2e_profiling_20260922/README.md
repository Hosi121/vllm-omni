# PC/mobile E2E check and profiling — 2026-09-22

**60 device/model pairs checked; five current desktop paths profiled; 55 retain
explicit complete-pipeline blockers.** The five paths completed 900 measured
requests in total and at least 30 minutes of sustained generation each.
Completed performance protocols do not establish full release qualification.

- [Final findings and measured results](evidence/FINAL_REVIEW.md)
- [Complete configuration × model matrix and repair steps](evidence/summary/README.md)
- [Implementation roadmap for all configurations and models](../../../../docs/design/omni_edge_e2e_roadmap.md)
- [Requirement-by-requirement audit](evidence/requirements-audit.json)
- [Raw performance summary](evidence/summary/matrix.json)
- [Memory, startup, power, sustained trends and trace analysis](evidence/resource-summary.json)
- [Separate TTS stage counters](evidence/attribution-summary.json)
- [Completion and artifact-integrity checks](evidence/completion-integrity.json)
- [Chronological experiments, failures and retries](evidence/README.md)
- [Reproduction harness and measurement boundaries](../../PROFILING.md)

## Main findings

Spark text ran on WSL CPU and on CUDA in WSL/native Windows. CPU uses 1.7B INT8;
CUDA uses 4B BF16, so these are not same-model speedup comparisons. Batch-dependent
greedy outputs remain a reference/reproducibility investigation item.

Qwen3-TTS 0.6B CustomVoice ran both stages on CUDA in WSL/native Windows.
The measured groups had simulated playback deficits in 1/180 WSL requests and
49/180 native Windows requests. Windows sustained generation slowed late in the
run: the final-minute RTF p50 was 1.316, with five requests in that window. A later
driver snapshot reported a 33 W GPU power ceiling; the evidence does not establish
when or why it changed. Actual sound-device playback and speech reference quality
remain unqualified. WSL loading headroom and normal vocoder shutdown also need work.

The other cells retain specific runtime, artifact, registration, memory-feasibility
or device-access gaps. AI Hub component evidence is not promoted to complete
mobile generation, playback or sustained co-residency. No missing latency is zero,
and a rejected export is not a permanent rejection of the model/device family.

## Verify and reproduce

From the repository root:

```bash
python benchmarks/edge_harness/archive_profiles.py verify \
  benchmarks/edge_harness/results/e2e_profiling_20260922

python benchmarks/edge_harness/archive_profiles.py restore \
  benchmarks/edge_harness/results/e2e_profiling_20260922 /absolute/path/to/new-run-root

python benchmarks/edge_harness/summarize_e2e_profiles.py \
  --matrix /absolute/path/to/new-run-root/prior-matrix/matrix.json \
  --runs /absolute/path/to/new-run-root --out /absolute/path/to/new-summary

python benchmarks/edge_harness/analyze_profile_resources.py \
  --runs /absolute/path/to/new-run-root --out /absolute/path/to/new-resource-summary.json
```

[manifest.json](manifest.json) records original and stored hashes/sizes. Large
uncompressed files are stored as `.archive.gz`; restore before replaying original
relative trace paths. Raw reports retain their original platform paths and
measurement metadata. The portable harness is documented separately; historical
supervisor scripts contain this run's local paths and process identities.

Evidence bytes are protected from Git newline conversion by local attributes.
Hash verification proves archive integrity, not numerical/model correctness.
The harness's 17 focused tests and pinned Ruff 0.14.10 checks passed. All trace and
counter diagnostics finished before archiving; failures and interrupted attempts
remain available with their original configurations.
