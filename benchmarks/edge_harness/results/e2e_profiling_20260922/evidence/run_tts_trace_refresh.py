"""Repeat TTS traces with separate stage prefixes after all other diagnostics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] / 'vllm-omni-edge'
WSL = '/home/zhout/project/edge_infer'
prior = psutil.Process(19252)
assert 'run_cpu_trace_refresh.py' in ' '.join(prior.cmdline())
identity = prior.create_time()
record = {'status': 'waiting', 'pid': os.getpid(), 'predecessor_pid': prior.pid,
          'predecessor_created': identity, 'jobs': []}
def persist():
    (HERE / 'tts-trace-refresh-queue.json').write_text(json.dumps(record, indent=2))
persist()
while prior.is_running() and prior.create_time() == identity:
    time.sleep(3)
assert json.loads((HERE / 'cpu-trace-refresh-queue.json').read_text())['status'] == 'exited'
record['status'] = 'running'
persist()
archive = HERE / 'trace-collision-1'
archive.mkdir(exist_ok=False)
env = os.environ.copy()
env.update(PYTHONPATH=str(REPO), HF_HUB_OFFLINE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
           VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_WSL2_ENABLE_PIN_MEMORY='1',
           CUDA_HOME=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4')
env['PATH'] = env['CUDA_HOME'] + r'\bin;' + env['PATH']
cuda = WSL + '/.venvs/omni-cuda-029/lib/python3.12/site-packages/nvidia/cu13'
for base in ('tts-cuda-wsl', 'tts-cuda-windows'):
    name = base + '-trace'
    for folder in (name, name + '-host'):
        source, target = HERE / folder, archive / folder
        assert source.resolve().is_relative_to(HERE.resolve())
        assert target.resolve().is_relative_to(HERE.resolve()) and not target.exists()
        source.rename(target)
    if base.endswith('wsl'):
        model = '/home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5'
        shell = (f'cd {WSL}/vllm-omni-edge && PYTHONPATH=. HF_HUB_OFFLINE=1 '
                 f'VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WSL2_ENABLE_PIN_MEMORY=1 CUDA_HOME={cuda} '
                 f'PATH={cuda}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib '
                 f'timeout 1800 ../.venvs/omni-cuda-029/bin/python benchmarks/edge_harness/profile_trace_diagnostic.py '
                 f'--kind tts --model {model} --out ../analysis/experiments/e2e_profiling_20260922/{name}')
        command = ['wsl', '-d', 'Ubuntu', '--', 'bash', '-lc', shell]
    else:
        command = [sys.executable, str(REPO / 'benchmarks/edge_harness/profile_trace_diagnostic.py'),
                   '--kind', 'tts', '--model', r'C:\Users\zhout\w2\models\Qwen3-TTS-0.6B-85e237c',
                   '--out', str(HERE / name)]
    job = {'name': name, 'command': command, 'status': 'running', 'start': time.time()}
    record['jobs'].append(job)
    persist()
    done = subprocess.run([sys.executable, str(REPO / 'benchmarks/edge_harness/profile_host_command.py'),
                           '--out', str(HERE / (name + '-host')), '--cwd', str(REPO), '--', *command], env=env)
    path = HERE / name / 'report.json'
    report = json.loads(path.read_text()) if path.exists() else {}
    job.update(status='exited', returncode=done.returncode, benchmark_status=report.get('status', 'missing_report'),
               error=report.get('error'), end=time.time())
    persist()
record['status'] = 'running_analysis'
persist()
for script, args in [
    ('summarize_e2e_profiles.py', ['--matrix', str(HERE / 'prior-matrix/matrix.json'), '--runs', str(HERE), '--out', str(HERE / 'summary')]),
    ('analyze_profile_resources.py', ['--runs', str(HERE), '--out', str(HERE / 'resource-summary.json')])]:
    result = subprocess.run([sys.executable, str(REPO / 'benchmarks/edge_harness' / script), *args], env=env)
    record[script] = result.returncode
record.update(status='exited', end=time.time())
persist()
