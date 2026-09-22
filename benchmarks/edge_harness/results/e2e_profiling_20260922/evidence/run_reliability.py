"""Run isolated fault/backpressure checks only after all timing and trace jobs exit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPO = ROOT/'vllm-omni-edge'
WSL = '/home/zhout/project/edge_infer'
prior = psutil.Process(48364)
assert 'run_retry.py' in ' '.join(prior.cmdline())
identity = prior.create_time()
record = {'status': 'waiting', 'pid': os.getpid(), 'predecessor_pid': prior.pid,
          'predecessor_created': identity, 'jobs': []}
def persist():
    (HERE/'reliability-queue.json').write_text(json.dumps(record, indent=2))
persist()
while prior.is_running() and prior.create_time() == identity:
    time.sleep(3)
assert json.loads((HERE/'retry-status.json').read_text())['status'] == 'exited'
record['status'] = 'running'
env = os.environ.copy()
env.update(PYTHONPATH=str(REPO), HF_HUB_OFFLINE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
           VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_WSL2_ENABLE_PIN_MEMORY='1',
           CUDA_HOME=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4')
env['PATH'] = env['CUDA_HOME']+r'\bin;'+env['PATH']
cuda = WSL+'/.venvs/omni-cuda-029/lib/python3.12/site-packages/nvidia/cu13'
for base in ('spark-cpu-wsl', 'spark-cuda-wsl', 'spark-cuda-windows'):
    baseline = json.loads((HERE/base/'report.json').read_text())
    if baseline.get('status') != 'completed':
        record['jobs'].append({'name': base+'-reliability', 'status': 'not_run',
                               'reason': 'Baseline did not complete', 'baseline_error': baseline.get('error')})
        persist()
        continue
    name = base+'-reliability'
    if base.endswith('wsl'):
        cpu = 'cpu' in base
        venv = 'omni-cpu' if cpu else 'omni-cuda-029'
        model = '../models/Spark-X2.5-1.7B-int8' if cpu else '../models/Spark-X2.5-4B'
        device_env = 'VLLM_CPU_KVCACHE_SPACE=2 OMP_NUM_THREADS=8' if cpu else f'CUDA_HOME={cuda} PATH={cuda}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib'
        shell = (f'cd {WSL}/vllm-omni-edge && PYTHONPATH=. HF_HUB_OFFLINE=1 '
                 f'VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WSL2_ENABLE_PIN_MEMORY=1 {device_env} '
                 f'timeout 600 ../.venvs/{venv}/bin/python benchmarks/edge_harness/check_text_reliability.py '
                 f'--model {model} --out ../analysis/experiments/e2e_profiling_20260922/{name}')
        command = ['wsl', '-d', 'Ubuntu', '--', 'bash', '-lc', shell]
    else:
        command = [sys.executable, str(REPO/'benchmarks/edge_harness/check_text_reliability.py'),
                   '--model', str(ROOT/'models/Spark-X2.5-4B'), '--out', str(HERE/name)]
    job = {'name': name, 'status': 'running', 'start': time.time(), 'command': command}
    record['jobs'].append(job)
    persist()
    completed = subprocess.run([sys.executable, str(REPO/'benchmarks/edge_harness/profile_host_command.py'),
                                '--out', str(HERE/(name+'-host')), '--cwd', str(REPO), '--', *command], env=env)
    report_path = HERE/name/'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    job.update(status='exited', returncode=completed.returncode, end=time.time(),
               benchmark_status=report.get('status', 'missing_report'), error=report.get('error'))
    persist()
record.update(status='exited', end=time.time())
persist()
