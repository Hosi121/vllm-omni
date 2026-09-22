"""Render a review of terminal evidence; does not promote profiling into qualification."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
queues = ('windows-tts-retry', 'attribution', 'cpu-trace-refresh', 'tts-trace-refresh', 'attribution-retry')
for name in queues:
    assert json.loads((HERE / (name + '-queue.json')).read_text())['status'] == 'exited', name
matrix = json.loads((HERE / 'summary/matrix.json').read_text())
resources = json.loads((HERE / 'resource-summary.json').read_text())
assert len(matrix['cases']) == 60
profiled = [c for c in matrix['cases'] if c['profiling']['status'] != 'not_run']
assert len(profiled) == 5

def num(x):
    return 'unavailable' if x is None else f'{x:.3f}'

lines = [
    '# End-to-end check and profiling review — 2026-09-22', '',
    'All 60 device/model pairs have a disposition. Five current desktop execution paths were selected '
    'for profiling; the other 55 retain concrete complete-pipeline blockers. This is a check of the '
    'current implementation, not a claim that all planned backends or release gates are implemented.', '',
    'The [full matrix](summary/README.md) contains all cells and next steps. '
    '[Requirement audit](requirements-audit.json), [raw performance](summary/matrix.json), '
    '[resource/trace analysis](resource-summary.json), and [stage counters](attribution-summary.json) '
    'retain the evidence and limits.', '',
    '## Executed timing protocol', '',
    'Twenty measured requests per short/medium/long × concurrency 1/2/4 group; separate warmups, '
    'a 30-minute medium/concurrency-1 sustained phase, and separate trace/fault diagnostics. '
    'Nearest-rank percentiles describe measured requests. Existing disk/JIT caches were retained.', '',
    '| Path | Protocol complete | Normal requests checked | Constructor startup s | Sustained s | Medium c1 latency p50 / p95 | Medium c1 rate p50 |',
    '|---|---|---:|---:|---:|---|---|',
]
for case in profiled:
    p = case['profiling']
    name = case['device'] + ' / ' + matrix['models'][case['model']]
    groups = [g for g in p.get('measurements', []) if g['length_band'] == 'medium' and g['concurrency'] == 1]
    group = groups[0]['metrics'] if groups else {}
    latency_key, rate_key, latency_unit, rate_unit = ('ttfa_ms', 'rtf_total', 'ms', 'RTF') if case['model'] == 'tts' else ('ttft_s', 'decode_tok_per_s', 's', 'token/s')
    latency, rate = group.get(latency_key, {}), group.get(rate_key, {})
    lines.append(f"| {name} | {p.get('profile_protocol_complete', False)} | {p.get('recorded_output_checks', {}).get('requests_checked', 0)} | {num(p.get('startup_s'))} | {num(p.get('sustained_wall_s'))} | {num(latency.get('p50'))} / {num(latency.get('p95'))} {latency_unit} | {num(rate.get('p50'))} {rate_unit} |")
lines += ['', 'Constructor startup excludes earlier imports, artifact hashing and plan construction. '
          'Resource analysis also retains inclusive host-launch-to-first-request/output timing. '
          'Native Windows Spark reads its checkpoint through the WSL UNC path; native Windows TTS uses a local Windows copy. '
          'Startup comparisons therefore include filesystem and preparation differences. '
          'Spark CPU uses 1.7B INT8; CUDA Spark uses 4B BF16. These are not same-model speedup comparisons. '
          'TTS is the pinned 0.6B CustomVoice checkpoint, English/Vivian. Other voices, languages and TTS modes are unqualified.', '',
          '## Reliability and output checks', '',
          '| Path | Recorded normal-output invariants | Targeted probe status | Cancellation / failure observation |',
          '|---|---|---|---|']
for case in profiled:
    p, d = case['profiling'], case.get('diagnostics', {}).get('reliability', {})
    if case['model'] == 'spark':
        cancel, crash = d.get('cancel', {}), d.get('crash_injection', {})
        detail = f"queue saturation={d.get('saturation_reached')}; same tokens after drain={d.get('same_tokens_after_saturation')}; oversized refusal={d.get('oversized_plan_refused')}; post-cancel events={len(cancel.get('late_events', []))}; worker error={crash.get('explicit_error_reported')}, {num(crash.get('observation_s'))} s"
    else:
        abort, crash = d.get('abort', {}), d.get('crash', {})
        detail = f"abort ack={num(abort.get('ack_s'))} s; late PCM samples={abort.get('late_audio_samples', 'inspect terminal outcome')}; worker error={crash.get('explicit_error_reported')}, {num(crash.get('observation_s'))} s"
    cleanup = d.get('owned_processes_requiring_harness_cleanup', [])
    detail += f"; harness cleanup processes={len(cleanup)}"
    if d.get('error'):
        detail += '; ' + d['error'].replace('|', ';')
    lines.append(f"| {case['device']} / {case['model']} | {p.get('recorded_output_checks', {}).get('recorded_invariants_pass', False)} | {d.get('status')} | {detail} |")
lines += ['', 'A completed probe means it ran. Inspect each outcome. Oversized admission is not an induced OOM; '
          'the TTS slow-consumer probe does not saturate every internal buffer. Targeted checks do not establish '
          'perceptual/reference quality or all possible state transitions.', '', '## Findings requiring follow-up', '']
for case in profiled:
    p = case['profiling']
    if case['model'] == 'tts':
        groups = p.get('measurements', [])
        n = sum(g['n'] for g in groups)
        underruns = sum(g['requests_with_playback_stalls'] for g in groups)
        lines.append(f"- {case['device']} TTS: {underruns}/{n} measured requests had a simulated post-startup playback deficit. Raw chunk timings/underruns are retained. Actual sound-device playback, speech tail/reference agreement, ASR and perceptual/speaker quality remain unqualified.")
        resource = resources['telemetry'].get('tts-cuda-windows' if case['device'].endswith('windows') else 'tts-cuda-wsl', {})
        sustained = resource.get('sustained_request_summary', {})
        tail = resource.get('sustained_request_trends', {}).get('last_60s', {})
        tail_rtf = tail.get('metrics', {}).get('rtf_total', {})
        lines.append(f"- {case['device']} TTS sustained: {sustained.get('rtf_above_one_requests', 'unavailable')}/{sustained.get('n', 'unavailable')} requests had RTF above 1; {sustained.get('simulated_playback_deficit_requests', 'unavailable')} had simulated playback deficits. Requests submitted in the final 60 seconds: n={tail.get('n', 'unavailable')}, RTF p50/p95={num(tail_rtf.get('p50'))}/{num(tail_rtf.get('p95'))}. This tail window is separate from the full-run and first/last-five-minute distributions.")
    else:
        differing = sum(g.get('requests_with_sequences_not_seen_at_concurrency_1') or 0 for g in p.get('measurements', []) if g['concurrency'] != 1)
        lines.append(f"- {case['device']} Spark: {differing} measured concurrency-2/4 requests produced a token sequence absent from the corresponding concurrency-1 set. This is a reproducibility/reference gap; timing evidence alone does not identify numerical or state-related causes.")
lines += [
    '- WSL TTS startup briefly coincided with only 24,514,560 bytes of available Windows RAM. '
    'The retained minimum sample includes other processes. Native-stage migration into the shared memory '
    'ledger and loading/workspace headroom still need work; sampled peaks are lower bounds, not allocation accounting.',
    '- The normal WSL TTS baseline required forced vocoder shutdown after its grace period. '
    'The isolated worker-failure probe later retired owned workers; that does not erase the baseline shutdown finding.',
    '- The original TTS profiler used the same rank-0 filename for both stages. '
    'The diagnostic workaround uses separate public-API stage prefixes. Original collided traces and '
    'their logs are preserved under trace-collision-1; production profiler naming remains a repair item.',
    '- Native Windows TTS slowed substantially at the end of sustained execution while GPU telemetry '
    'reported P5 and an 810 MHz memory clock. AC/Performance conditions were still recorded. '
    'A subsequent 18:21 driver snapshot reports software power capping active, a 33 W current ceiling '
    'versus 95 W default, and inactive GPU thermal slowdown flags '
    '([raw snapshot](gpu-low-clock-diagnostic.txt)). This spot check does not establish when or why '
    'the platform changed the limit. The clock transition and latency change are not proof of a thermal cause. '
    'Investigate power-state/clock behavior before claiming sustained real-time speech.',
    '- CPU TTS has a vLLM/Omni compatibility blocker; native Windows CPU lacks the needed attention operators. '
    'Pin and qualify matching backend builds before claiming those cells.',
    '- Qwen3.8-27B lacks the current public Omni pipeline registration; historical bare-vLLM text and '
    'vision-component results do not qualify its complete multimedia route.',
    '- Matching full MiniCPM-o 4.5 and InternVLA-A1 checkpoints/stage artifacts are absent from checked roots. '
    'Obtain pinned artifacts, establish desktop reference outputs, then repeat the appropriate modality/action suites.',
    '- AMD 890M evidence remains a vision component; the tested NPU whole-vision artifacts were rejected. '
    'Complete downstream adapters and compatible compiled artifacts are missing. Joint CPU/iGPU/NPU/GPU '
    'execution has no complete co-resident serial/overlap comparison.',
    '- Mobile/embedded and X Elite targets lack device-local application access and integrated stateful '
    'model pipelines. AI Hub components do not supply the prefill/decode/KV/sampling loop, '
    'playback, co-residency or sustained device thermals. Do not deploy through AI Hub.',
    '', '## Measurement limits', '',
    'Windows physical RAM includes WSL; WSL quota, process RSS/PSS and VRAM are distinct scopes and '
    'must not be added. Power/temperature/clock samples cover the whole NVIDIA GPU, including other '
    'applications. CPU package temperature and CPU/NPU/whole-device energy are unavailable in these '
    'environments. CPU effective-frequency trends were not captured; the separate WMI check exposes '
    'nominal/aggregate counters, not a validated per-core frequency trace. No model-exclusive energy '
    'or causal thermal-throttling claim is made.', '',
    'Text delivery intervals can combine tokens into output updates. Effective prefill includes queue, '
    'host and first-output costs. Torch category duration sums overlap; interval unions are not '
    'additive across categories. Sequential profiler stop/export can extend trace windows with idle '
    'time. TTS trace overhead comparisons retain speech-length variation. StepStats counters use '
    'host timing with synchronization disabled, overlapping scopes and per-process reservoirs; '
    'they are not pooled stage GPU latencies. Baseline-versus-counter ratios also include different '
    'observed power/clock conditions after the GPU limit change; they are not isolated instrumentation overhead. '
    'Engine-core step counters include idle/no-work iterations. The existing orchestrator-dispatch hook '
    'is only in the optional event-driven loop, which these runs did not enable; no dispatch latency '
    'is inferred from its absent counter.', '',
    'Failures and corrected retries remain archived. The Windows TTS report-write failure interrupted '
    'its first sustained phase; that attempt is not a sustained pass. No universal model/device '
    'release qualification follows from a completed performance protocol.',
]
(HERE / 'FINAL_REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('Rendered review of 60 pair dispositions and five profiling paths')
