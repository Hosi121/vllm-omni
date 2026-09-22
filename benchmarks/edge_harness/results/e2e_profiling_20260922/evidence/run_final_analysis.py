"""Wait for all real-model probes before CPU-heavy trace analysis."""
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
prior = psutil.Process(61368)
assert 'run_tts_reliability.py' in ' '.join(prior.cmdline())
identity = prior.create_time()
record = {'status':'waiting','pid':os.getpid(),'predecessor_pid':prior.pid,'predecessor_created':identity,'jobs':[]}
def persist():
    (HERE/'analysis-queue.json').write_text(json.dumps(record,indent=2))
persist()
while prior.is_running() and prior.create_time() == identity:
    time.sleep(3)
assert json.loads((HERE/'tts-reliability-queue.json').read_text())['status'] == 'exited'
record['status'] = 'running'
commands = [
 [sys.executable,str(REPO/'benchmarks/edge_harness/summarize_e2e_profiles.py'),
  '--matrix',str(REPO/'benchmarks/edge_harness/results/model_device_matrix_20260922/matrix.json'),
  '--runs',str(HERE),'--out',str(HERE/'summary')],
 [sys.executable,str(REPO/'benchmarks/edge_harness/analyze_profile_resources.py'),
  '--runs',str(HERE),'--out',str(HERE/'resource-summary.json')]]
for command in commands:
    job = {'command':command,'status':'running','start':time.time()}
    record['jobs'].append(job)
    persist()
    result = subprocess.run(command)
    job.update(status='exited',returncode=result.returncode,end=time.time())
    persist()
record.update(status='exited',end=time.time())
persist()
