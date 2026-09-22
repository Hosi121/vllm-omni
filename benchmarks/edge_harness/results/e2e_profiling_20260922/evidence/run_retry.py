"""Preserve failed launch attempts, then retry with fixed PATH and UTF-8 settings."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
prior_pid = 20772
prior = psutil.Process(prior_pid)
assert 'run_diagnostics.py' in ' '.join(prior.cmdline())
identity = prior.create_time()
status = {'status': 'waiting', 'pid': os.getpid(), 'predecessor_pid': prior_pid,
          'reason': 'Initial GPU launch failures: expanded Windows PATH in bash and native Python GBK default.'}
def persist():
    (HERE/'retry-status.json').write_text(json.dumps(status, indent=2))
persist()
while prior.is_running() and prior.create_time() == identity:
    time.sleep(2)
old = json.loads((HERE/'diagnostic-queue.json').read_text())
assert old['status'] == 'exited'
archive = HERE/'launcher-failures-1'
archive.mkdir(exist_ok=False)
names = ['queue.json', 'diagnostic-queue.json']
for base in ('spark-cuda-wsl', 'spark-cuda-windows', 'tts-cuda-wsl', 'tts-cuda-windows'):
    names.extend([base, base+'-host', base+'-trace', base+'-trace-host'])
cpu = HERE/'spark-cpu-wsl-trace/report.json'
if cpu.exists() and json.loads(cpu.read_text()).get('status') != 'completed':
    names.extend(['spark-cpu-wsl-trace', 'spark-cpu-wsl-trace-host'])
for name in names:
    source, destination = HERE/name, archive/name
    assert source.resolve().is_relative_to(HERE.resolve())
    assert destination.resolve().is_relative_to(HERE.resolve()) and not destination.exists()
    if source.exists():
        source.rename(destination)
status.update(status='running_sweeps', archive=str(archive))
persist()
env = os.environ.copy()
env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
result = subprocess.run([sys.executable, str(HERE/'run_remaining.py')], env=env)
status['sweep_returncode'] = result.returncode
queue = json.loads((HERE/'queue.json').read_text())
assert queue['status'] == 'exited'
status['status'] = 'running_diagnostics'
persist()
result = subprocess.run([sys.executable, str(HERE/'run_diagnostics.py'), '--wait-pid', str(queue['pid'])], env=env)
status.update(status='exited', diagnostic_returncode=result.returncode, end=time.time())
persist()
