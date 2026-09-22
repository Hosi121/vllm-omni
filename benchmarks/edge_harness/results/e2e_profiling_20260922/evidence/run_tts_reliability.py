"""Run TTS reliability after the Spark reliability supervisor has exited."""
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
prior = psutil.Process(38844)
assert 'run_reliability.py' in ' '.join(prior.cmdline())
identity = prior.create_time()
record = {'status':'waiting', 'pid':os.getpid(), 'predecessor_pid':prior.pid,
          'predecessor_created':identity, 'jobs':[]}
def persist():
    (HERE/'tts-reliability-queue.json').write_text(json.dumps(record, indent=2))
persist()
while prior.is_running() and prior.create_time() == identity:
    time.sleep(3)
assert json.loads((HERE/'reliability-queue.json').read_text())['status'] == 'exited'
record['status'] = 'running'
env = os.environ.copy()
env.update(PYTHONPATH=str(REPO), HF_HUB_OFFLINE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
           VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_WSL2_ENABLE_PIN_MEMORY='1',
           CUDA_HOME=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4')
env['PATH'] = env['CUDA_HOME']+r'\bin;'+env['PATH']
cuda = WSL+'/.venvs/omni-cuda-029/lib/python3.12/site-packages/nvidia/cu13'
for base in ('tts-cuda-wsl','tts-cuda-windows'):
    path = HERE/base/'report.json'
    baseline = json.loads(path.read_text()) if path.exists() else {}
    name = base+'-reliability'
    if baseline.get('status') != 'completed':
        record['jobs'].append({'name':name,'status':'not_run','reason':'Baseline did not complete','baseline_error':baseline.get('error')})
        persist()
        continue
    if base.endswith('wsl'):
        model = '/home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5'
        shell = (f'cd {WSL}/vllm-omni-edge && PYTHONPATH=. HF_HUB_OFFLINE=1 '
                 f'VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WSL2_ENABLE_PIN_MEMORY=1 CUDA_HOME={cuda} '
                 f'PATH={cuda}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib '
                 f'timeout 900 ../.venvs/omni-cuda-029/bin/python benchmarks/edge_harness/check_tts_reliability.py '
                 f'--model {model} --out ../analysis/experiments/e2e_profiling_20260922/{name}')
        command = ['wsl','-d','Ubuntu','--','bash','-lc',shell]
    else:
        command = [sys.executable,str(REPO/'benchmarks/edge_harness/check_tts_reliability.py'),
                   '--model',r'C:\Users\zhout\w2\models\Qwen3-TTS-0.6B-85e237c','--out',str(HERE/name)]
    job = {'name':name,'status':'running','start':time.time(),'command':command}
    record['jobs'].append(job)
    persist()
    result = subprocess.run([sys.executable,str(REPO/'benchmarks/edge_harness/profile_host_command.py'),
                             '--out',str(HERE/(name+'-host')),'--cwd',str(REPO),'--',*command],env=env)
    path = HERE/name/'report.json'
    report = json.loads(path.read_text()) if path.exists() else {}
    job.update(status='exited',returncode=result.returncode,end=time.time(),benchmark_status=report.get('status','missing_report'),error=report.get('error'))
    persist()
record.update(status='exited',end=time.time())
persist()
