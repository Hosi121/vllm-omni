"""Repeat only the interrupted WSL counter run with its observed startup budget."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] / 'vllm-omni-edge'
WSL = '/home/zhout/project/edge_infer'
sys.path.insert(0, str(REPO / 'benchmarks/edge_harness'))
from summarize_e2e_profiles import execution_report, read_lines, stats

prior = psutil.Process(58756)
assert 'run_tts_trace_refresh.py' in ' '.join(prior.cmdline())
identity = prior.create_time()
record = {'status': 'waiting', 'pid': os.getpid(), 'predecessor_pid': prior.pid,
          'predecessor_created': identity, 'reason': '842 s startup consumed the first diagnostic deadline; retain exact workload and allow 1800 s.'}
def persist():
    (HERE / 'attribution-retry-queue.json').write_text(json.dumps(record, indent=2))
persist()
while prior.is_running() and prior.create_time() == identity:
    time.sleep(3)
assert json.loads((HERE / 'tts-trace-refresh-queue.json').read_text())['status'] == 'exited'
name = 'tts-cuda-wsl-attribution'
assert json.loads((HERE / (name + '-host') / 'status.json').read_text())['returncode'] == 124
archive = HERE / 'attribution-timeouts-1'
archive.mkdir(exist_ok=False)
for folder in (name, name + '-host', name + '-step-stats'):
    source, target = HERE / folder, archive / folder
    assert source.resolve().is_relative_to(HERE.resolve())
    assert target.resolve().is_relative_to(HERE.resolve()) and not target.exists()
    if source.exists():
        source.rename(target)
shutil.copyfile(HERE / 'attribution-summary.json', archive / 'attribution-summary.json')
env = os.environ.copy()
env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
cuda = WSL + '/.venvs/omni-cuda-029/lib/python3.12/site-packages/nvidia/cu13'
model = '/home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5'
shell = (f'cd {WSL}/vllm-omni-edge && PYTHONPATH=. HF_HUB_OFFLINE=1 '
         f'VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WSL2_ENABLE_PIN_MEMORY=1 CUDA_HOME={cuda} '
         f'VLLM_OMNI_STEP_STATS_DIR={WSL}/analysis/experiments/e2e_profiling_20260922/{name}-step-stats '
         f'VLLM_OMNI_STEP_STATS_SYNC=0 PATH={cuda}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib '
         f'timeout --kill-after=30 1800 ../.venvs/omni-cuda-029/bin/python benchmarks/edge_harness/profile_local_tts.py '
         f'--model {model} --out ../analysis/experiments/e2e_profiling_20260922/{name} '
         f'--length-band medium --concurrency 1 --repeats 20 --sustained-seconds 0')
command = [sys.executable, str(REPO / 'benchmarks/edge_harness/profile_host_command.py'),
           '--out', str(HERE / (name + '-host')), '--cwd', str(REPO),
           '--', 'wsl', '-d', 'Ubuntu', '--', 'bash', '-lc', shell]
record.update(status='running', command=command, start=time.time())
persist()
done = subprocess.run(command, env=env)
report = execution_report(HERE / name)
record.update(returncode=done.returncode, benchmark_status=report['status'], error=report.get('error'))
summary = json.loads((HERE / 'attribution-summary.json').read_text())
item = next(x for x in summary['runs'] if x['run'] == name)
item.update(status=report['status'], error=report.get('error'), counter_files=[])
item['comparison_caveat'] = 'Separate processes, cache/thermal history, observed GPU power-limit/clock changes and stochastic speech length; ratios do not isolate causal instrumentation overhead.'
for path in sorted((HERE / (name + '-step-stats')).glob('*.json')):
    raw = path.read_bytes()
    item['counter_files'].append({'path': path.relative_to(HERE).as_posix(),
                                  'sha256': hashlib.sha256(raw).hexdigest(), 'record': json.loads(raw)})
for label, folder in [('baseline', HERE / 'tts-cuda-wsl'), ('instrumented', HERE / name)]:
    rows = [r for r in read_lines(folder / 'requests.jsonl') if r.get('phase') == 'measured'
            and r.get('length_band') == 'medium' and r.get('concurrency') == 1]
    item['measurements'][label] = {'n': len(rows), 'metrics': {
        k: stats([r.get(k) for r in rows]) for k in ('ttfa_ms', 'rtf_total', 'total_wall_s', 'streamed_audio_s', 'stall_at_ttfa_ms')}}
for metric in ('ttfa_ms', 'rtf_total', 'total_wall_s'):
    before = item['measurements']['baseline']['metrics'][metric]['p50']
    after = item['measurements']['instrumented']['metrics'][metric]['p50']
    item['p50_ratios'][metric] = after / before if before and after is not None else None
(HERE / 'attribution-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
record['status'] = 'running_analysis'
persist()
for script, args in [
    ('summarize_e2e_profiles.py', ['--matrix', str(HERE / 'prior-matrix/matrix.json'), '--runs', str(HERE), '--out', str(HERE / 'summary')]),
    ('analyze_profile_resources.py', ['--runs', str(HERE), '--out', str(HERE / 'resource-summary.json')])]:
    result = subprocess.run([sys.executable, str(REPO / 'benchmarks/edge_harness' / script), *args], env=env)
    record[script] = result.returncode
record.update(status='exited', end=time.time())
persist()
