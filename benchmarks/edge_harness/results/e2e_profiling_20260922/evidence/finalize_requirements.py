"""Resolve each requested requirement against terminal evidence and explicit gaps."""
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
for name in ('windows-tts-retry', 'attribution', 'cpu-trace-refresh', 'tts-trace-refresh', 'attribution-retry'):
    assert json.loads((HERE / (name + '-queue.json')).read_text())['status'] == 'exited', name
matrix = json.loads((HERE / 'summary/matrix.json').read_text())
assert len(matrix['cases']) == 60
runs = [c for c in matrix['cases'] if c['profiling']['status'] != 'not_run']
assert len(runs) == 5
assert all(c['profiling'].get('profile_protocol_complete') for c in runs), 'Inspect incomplete timing protocols first'
assert all(c.get('diagnostics', {}).get('trace', {}).get('status') == 'completed' for c in runs)
assert all(c.get('diagnostics', {}).get('reliability', {}).get('status') in ('completed', 'failed') for c in runs)
resources = json.loads((HERE / 'resource-summary.json').read_text())
assert resources['traces'] and all('error' not in r for r in resources['traces'])
audit = json.loads((HERE / 'requirements-audit.json').read_text())
updates = {
    'S3': ('checked_with_limits', 'Exact local hardware, runtime packages, memory, OS, driver and AC/Performance conditions are recorded. Remote component targets lack full application access and complete device build/RAM/power inventories. CPU package temperature and whole-device/CPU/NPU energy are unavailable.', ['preflight-windows.json', 'preflight-wsl-cpu.json', 'resource-summary.json', 'prior-matrix/matrix.json']),
    'S5': ('checked_with_blockers', 'Desktop real-model cancellation/recovery and worker-failure probes completed; inspect individual outcomes. Spark reached queue capacity and rejected stale state. Stateful mobile sessions and complete device-local controllers remain absent; TTS internal buffer saturation and universal shared-ledger release are not qualified.', ['summary/matrix.json', 'prior-evidence/README.md']),
    'S6': ('completed_local_checks', 'Reusable text/TTS/host/trace/reliability runners and matrix/resource analyzers are implemented. All 60 pairs have evidence-backed dispositions; raw requests, warmup PCM, counters, traces and failed attempts are retained. Publication integrity is verified separately by the archive manifest.', ['summary/matrix.json', 'FINAL_REVIEW.md']),
    'M2': ('checked_with_gaps', 'Both CUDA OS paths completed all nine measured groups and 30-minute sustained generation. Simulated playback deficits, model-specific speech/reference quality, forced normal shutdown and memory headroom remain findings. Mobile TTS and CPU compatibility remain blocked; see exact per-run counts and probe results.', ['FINAL_REVIEW.md', 'summary/matrix.json', 'resource-summary.json']),
    'COVERAGE': ('completed_check', 'All 12 configurations x five models checked: five runnable current desktop paths profiled, 55 explicitly not E2E-profileable with reasons and next actions. Component and historical results are retained without promotion to complete pipelines.', ['summary/matrix.json', 'prior-matrix/matrix.json']),
    'P1': ('measured_with_limits', 'Constructor startup, host-launch/import interval, first warmup and runtime-reported load/compile lines are retained separately. Existing disk/JIT caches were not purged; isolated cold compilation, cold-disk startup and exhaustive warm-restart distributions are not qualified.', ['resource-summary.json']),
    'P2': ('completed_measurement', 'All five paths contain at least 20 measured requests in every short/medium/long x concurrency 1/2/4 group. Raw requests and nearest-rank p50/p95 are retained; warmup/sustained records are excluded from these percentiles.', ['summary/matrix.json']),
    'P3': ('measured_with_quality_gaps', 'Text TTFT/decode/completion, effective prefill and delivery-spacing proxies; speech TTFA/RTF/chunk gaps/simulated underruns and interruption are measured. Text updates can coalesce tokens. Speech perceptual/tail/reference quality and independent text numerical parity are unqualified. Missing multimodal/VLA pipelines block modality/action/deadline measurements.', ['summary/matrix.json', 'FINAL_REVIEW.md']),
    'P4': ('measured_with_accounting_gaps', 'Windows host RAM, WSL quota/process measurements and whole-GPU VRAM sampled during startup and runtime. Peaks are lower bounds; per-allocation weights/state/workspace/transfer attribution and complete native multi-pool admission remain unimplemented. WSL TTS briefly coincided with only 24.5 MB Windows RAM available; see raw sample.', ['resource-summary.json', 'tts-cuda-wsl-memory-headroom.json', 'prior-evidence/README.md']),
    'P5': ('measured_with_limits', 'Actual CPU/CUDA runtime evidence, CUDA kernels/copies/runtime calls, and separate TTS stage counters retained. Stage-specific trace prefixes avoid the observed rank-0 filename collision. Counters are host timers with overlapping scopes, not pooled GPU latency. No unmeasured joint-device execution, fallback-free blanket claim or component-sum E2E speedup.', ['resource-summary.json', 'attribution-summary.json', 'summary/matrix.json']),
    'P6': ('completed_measurement_with_sensor_gaps', 'All five paths completed at least 1800 seconds of medium-input concurrency-1 generation. Whole-NVIDIA-GPU clocks/power/temperature and host RAM sampled; first/last five-minute trends retained. CPU temperature and CPU/NPU/whole-device energy unavailable. GPU energy excludes telemetry gaps over five seconds and includes other processes.', ['resource-summary.json', 'cpu-sensor-availability-windows.json', 'cpu-sensor-availability-wsl.json']),
    'P7': ('checked_with_limits', 'Repeated sessions, real slow consumers, cancellation/recovery, stale-state fencing, oversized text admission and owned-worker failures checked. Inspect per-probe flags/errors/cleanup. TTS full buffer saturation, destructive OOM recovery, every internal queue and exhaustive state-boundary cases are unqualified.', ['summary/matrix.json', 'FINAL_REVIEW.md']),
    'P8': ('completed_diagnostics_with_limits', 'Separate three-before/three-traced/three-after diagnostics with profiler RPC times for each runnable path. Original CPU full trace and overwritten TTS traces preserved; final TTS traces use distinct stage prefixes. Tracing/export overhead is separate from baseline; stochastic speech length and sequential export idle time preclude a pure causal universal overhead ratio.', ['summary/matrix.json', 'resource-summary.json', 'trace-collision-1/tts-cuda-wsl-trace-host/command.log']),
    'RELEASE': ('not_fully_qualified', 'Checks/profiles are complete for the executable current paths; full universal release gates are not met. Missing artifacts/backends/device access, quality suites, memory accounting and playback guarantees remain an explicit user repair backlog. The fork publication and archive hashes establish delivery/integrity, not model qualification.', ['FINAL_REVIEW.md', 'summary/matrix.json', 'resource-summary.json']),
}
for requirement in audit['requirements']:
    if requirement['id'] in updates:
        status, finding, evidence = updates[requirement['id']]
        requirement.update(status=status, finding=finding, evidence=evidence)
    assert 'pending' not in requirement['status'] and requirement['status'] != 'in_progress'
    for source in requirement['evidence']:
        assert (HERE / source).is_file(), source
audit['generated_unix'] = time.time()
audit['disposition'] = 'Implementation and unavailable qualification gates are recorded for later user repair; performance completion is separate from release qualification.'
(HERE / 'requirements-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(f"Resolved {len(audit['requirements'])} requirement groups")
