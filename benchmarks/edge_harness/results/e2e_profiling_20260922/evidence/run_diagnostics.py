"""Wait for the verified sweep supervisor, then serialize separate profiler runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

p = argparse.ArgumentParser()
p.add_argument('--wait-pid', type=int, required=True)
a = p.parse_args()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPO = ROOT/'vllm-omni-edge'
WSL = '/home/zhout/project/edge_infer'
try:
    live = psutil.Process(a.wait_pid)
except psutil.NoSuchProcess:
    live = None
if live is not None:
    assert 'run_remaining.py' in ' '.join(live.cmdline()), 'Unexpected predecessor process'
identity = live.create_time() if live is not None else None
manifest = {'status': 'waiting', 'pid': os.getpid(), 'predecessor_pid': a.wait_pid,
            'predecessor_created': identity, 'start': time.time(), 'jobs': []}
def persist():
    (HERE/'diagnostic-queue.json').write_text(json.dumps(manifest, indent=2))
persist()
while live is not None and live.is_running() and live.create_time() == identity:
    time.sleep(3)
queue = json.loads((HERE/'queue.json').read_text())
assert queue['status'] == 'exited', 'Sweep supervisor exited without a terminal queue record'
assert queue['pid'] == a.wait_pid, 'Terminal queue belongs to a different supervisor'
manifest['status'] = 'running'
env = os.environ.copy()
env.update(PYTHONPATH=str(REPO), HF_HUB_OFFLINE='1', VLLM_USE_FLASHINFER_SAMPLER='0',
           PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
           VLLM_WSL2_ENABLE_PIN_MEMORY='1', CUDA_HOME=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4')
env['PATH'] = env['CUDA_HOME']+r'\bin;'+env['PATH']
# The prior native launcher omitted the already-established cudart setting.
# Retry only this identified pre-model failure, after all other sweeps exit.
native = next(job for job in queue['jobs'] if job['name'] == 'spark-cuda-windows')
native_log = HERE/'spark-cuda-windows-host/command.log'
if native.get('returncode') == 1 and native_log.is_file() and 'libcudart is not loaded' in native_log.read_text(encoding='utf-8'):
    archive = HERE/'launcher-failures-2'
    archive.mkdir(exist_ok=False)
    (archive/'queue.json').write_text(json.dumps(queue, indent=2))
    for suffix in ('', '-host'):
        source = HERE/('spark-cuda-windows'+suffix)
        destination = archive/source.name
        assert source.resolve().is_relative_to(HERE.resolve())
        assert destination.resolve().is_relative_to(HERE.resolve()) and not destination.exists()
        if source.exists():
            source.rename(destination)
    manifest['baseline_retry'] = {'name': native['name'], 'status': 'running',
                                  'reason': 'Missing VLLM_CUDART_SO_PATH in native launcher', 'start': time.time()}
    persist()
    result = subprocess.run([sys.executable, str(REPO/'benchmarks/edge_harness/profile_host_command.py'),
                             '--out', str(HERE/'spark-cuda-windows-host'), '--cwd', str(REPO),
                             '--', *native['command']], env=env)
    path = HERE/'spark-cuda-windows/report.json'
    report = json.loads(path.read_text()) if path.exists() else {}
    native.update(status='exited', returncode=result.returncode,
                  start=manifest['baseline_retry']['start'], end=time.time(),
                  benchmark_status=report.get('status','missing_report'), error=report.get('error'),
                  retry_reason='Corrected native CUDA runtime DLL setting; prior attempt archived')
    queue.update(end=time.time(), note='Includes serial native Spark launcher correction before traces.')
    (HERE/'queue.json').write_text(json.dumps(queue, indent=2))
    manifest['baseline_retry'].update(status='exited', returncode=result.returncode,
                                      benchmark_status=report.get('status','missing_report'), end=time.time())
    persist()
cuda = WSL+'/.venvs/omni-cuda-029/lib/python3.12/site-packages/nvidia/cu13'
names = ['spark-cpu-wsl']+[job['name'] for job in queue['jobs']]
for base in names:
    name = base+'-trace'
    existing_path = HERE/name/'report.json'
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text())
        if existing.get('status') == 'completed':
            import hashlib
            assert all(hashlib.sha256((HERE/name/t['path']).read_bytes()).hexdigest() == t['sha256']
                       for t in existing.get('traces', []))
            assert existing.get('traces'), 'Completed diagnostic has no trace artifacts'
            manifest['jobs'].append({'name': name, 'status': 'exited', 'benchmark_status': 'completed',
                                     'reused_verified_report': str(existing_path)})
            persist()
            continue
    kind = 'spark' if base.startswith('spark') else 'tts'
    if base.endswith('wsl'):
        cpu = 'cpu' in base
        venv = 'omni-cpu' if cpu else 'omni-cuda-029'
        model = '../models/Spark-X2.5-1.7B-int8' if cpu else '../models/Spark-X2.5-4B'
        if kind == 'tts':
            model = '/home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5'
        device_env = 'VLLM_CPU_KVCACHE_SPACE=2 OMP_NUM_THREADS=8' if cpu else f'CUDA_HOME={cuda} PATH={cuda}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib'
        shell = (f'cd {WSL}/vllm-omni-edge && PYTHONPATH=. HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 '
                 f'VLLM_WSL2_ENABLE_PIN_MEMORY=1 {device_env} timeout 1800 ../.venvs/{venv}/bin/python '
                 f'benchmarks/edge_harness/profile_trace_diagnostic.py --kind {kind} --model {model} '
                 f'--out ../analysis/experiments/e2e_profiling_20260922/{name}')
        command = ['wsl', '-d', 'Ubuntu', '--', 'bash', '-lc', shell]
    else:
        model = str(ROOT/'models/Spark-X2.5-4B') if kind == 'spark' else r'C:\Users\zhout\w2\models\Qwen3-TTS-0.6B-85e237c'
        command = [sys.executable, str(REPO/'benchmarks/edge_harness/profile_trace_diagnostic.py'),
                   '--kind', kind, '--model', model, '--out', str(HERE/name)]
    record = {'name': name, 'command': command, 'status': 'running', 'start': time.time()}
    manifest['jobs'].append(record)
    persist()
    call = [sys.executable, str(REPO/'benchmarks/edge_harness/profile_host_command.py'),
            '--out', str(HERE/(name+'-host')), '--cwd', str(REPO), '--', *command]
    completed = subprocess.run(call, env=env)
    report_path = HERE/name/'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    record.update(status='exited', returncode=completed.returncode, end=time.time(),
                  benchmark_status=report.get('status', 'missing_report'), error=report.get('error'))
    persist()
manifest.update(status='exited', end=time.time())
persist()
