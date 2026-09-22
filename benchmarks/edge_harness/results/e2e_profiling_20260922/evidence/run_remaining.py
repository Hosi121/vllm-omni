"""Serialize the four GPU benchmark runs after the current CPU benchmark exits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPO = ROOT / 'vllm-omni-edge'
WSL = '/home/zhout/project/edge_infer'
HOST = REPO / 'benchmarks/edge_harness/profile_host_command.py'
env = os.environ.copy()
env.update(PYTHONPATH=str(REPO), HF_HUB_OFFLINE='1', VLLM_USE_FLASHINFER_SAMPLER='0',
           PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
           VLLM_WSL2_ENABLE_PIN_MEMORY='1', CUDA_HOME=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4')
env['PATH'] = env['CUDA_HOME'] + r'\bin;' + env['PATH']

wait_path = HERE / 'spark-cpu-wsl-host/status.json'
observed = json.loads(wait_path.read_text())
pid = observed['pid']
if observed['status'] == 'running':
    live = psutil.Process(pid)
    assert 'wsl' in live.name().lower(), (pid, live.name())
    created = live.create_time()
    while live.is_running() and live.create_time() == created:
        time.sleep(2)
    # The parent telemetry collector writes its terminal status after child exit.
    for _ in range(10):
        observed = json.loads(wait_path.read_text())
        if observed['status'] == 'exited':
            break
        time.sleep(1)
    assert observed['status'] == 'exited', 'Prior host collector has not finalized'

jobs = []
cuda = WSL + '/.venvs/omni-cuda-029/lib/python3.12/site-packages/nvidia/cu13'
for kind, model in [('spark', '../models/Spark-X2.5-4B'),
                    ('tts', '/home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5')]:
    script = 'profile_local_text.py' if kind == 'spark' else 'profile_local_tts.py'
    name = kind + '-cuda-wsl'
    shell = (f'cd {WSL}/vllm-omni-edge && PYTHONPATH=. HF_HUB_OFFLINE=1 '
             f'VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WSL2_ENABLE_PIN_MEMORY=1 CUDA_HOME={cuda} '
             f'PATH={cuda}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib timeout 10800 ../.venvs/omni-cuda-029/bin/python '
             f'benchmarks/edge_harness/{script} --model {model} '
             f'--out ../analysis/experiments/e2e_profiling_20260922/{name}')
    jobs.append((name, ['wsl', '-d', 'Ubuntu', '--', 'bash', '-lc', shell]))
    name = kind + '-cuda-windows'
    native_model = str(ROOT / 'models/Spark-X2.5-4B') if kind == 'spark' else r'C:\Users\zhout\w2\models\Qwen3-TTS-0.6B-85e237c'
    jobs.append((name, [sys.executable, str(REPO / 'benchmarks/edge_harness' / script),
                        '--model', native_model, '--out', str(HERE / name)]))

manifest = {'status': 'running', 'pid': os.getpid(), 'start': time.time(),
            'jobs': [{'name': n, 'command': c, 'status': 'pending'} for n, c in jobs]}
def persist():
    (HERE / 'queue.json').write_text(json.dumps(manifest, indent=2))
persist()
for record in manifest['jobs']:
    record.update(status='running', start=time.time())
    persist()
    command = [sys.executable, str(HOST), '--out', str(HERE / (record['name']+'-host')),
               '--cwd', str(REPO), '--', *record['command']]
    result = subprocess.run(command, env=env)
    report_path = HERE / record['name'] / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    record.update(status='exited', returncode=result.returncode, end=time.time(),
                  benchmark_status=report.get('status', 'missing_report'), error=report.get('error'))
    persist()
manifest.update(status='exited', end=time.time())
persist()
