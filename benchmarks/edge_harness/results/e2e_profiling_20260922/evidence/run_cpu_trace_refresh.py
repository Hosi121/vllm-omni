"""Final serial lightweight CPU trace; preserve the earlier full diagnostic."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] / 'vllm-omni-edge'
prior = psutil.Process(27528)
assert 'run_attribution.py' in ' '.join(prior.cmdline())
identity = prior.create_time()
record = {'status': 'waiting', 'pid': os.getpid(), 'predecessor_pid': prior.pid,
          'predecessor_created': identity}
def persist():
    (HERE / 'cpu-trace-refresh-queue.json').write_text(json.dumps(record, indent=2))
persist()
while prior.is_running() and prior.create_time() == identity:
    time.sleep(3)
assert json.loads((HERE / 'attribution-queue.json').read_text())['status'] == 'exited'
record['status'] = 'running'
persist()
for old, new in [('spark-cpu-wsl-trace', 'spark-cpu-wsl-trace-full'),
                 ('spark-cpu-wsl-trace-host', 'spark-cpu-wsl-trace-full-host')]:
    source, target = HERE / old, HERE / new
    assert source.resolve().is_relative_to(HERE.resolve())
    assert target.resolve().is_relative_to(HERE.resolve()) and not target.exists()
    source.rename(target)
shell = ('cd /home/zhout/project/edge_infer/vllm-omni-edge && '
         'PYTHONPATH=. HF_HUB_OFFLINE=1 VLLM_CPU_KVCACHE_SPACE=2 OMP_NUM_THREADS=8 '
         'timeout 900 ../.venvs/omni-cpu/bin/python benchmarks/edge_harness/profile_trace_diagnostic.py '
         '--kind spark --model ../models/Spark-X2.5-1.7B-int8 '
         '--out ../analysis/experiments/e2e_profiling_20260922/spark-cpu-wsl-trace')
command = [sys.executable, str(REPO / 'benchmarks/edge_harness/profile_host_command.py'),
           '--out', str(HERE / 'spark-cpu-wsl-trace-host'), '--cwd', str(REPO),
           '--', 'wsl', '-d', 'Ubuntu', '--', 'bash', '-lc', shell]
record.update(command=command, start=time.time())
persist()
done = subprocess.run(command)
record['returncode'] = done.returncode
for script, args in [
    ('summarize_e2e_profiles.py', ['--matrix', str(REPO / 'benchmarks/edge_harness/results/model_device_matrix_20260922/matrix.json'), '--runs', str(HERE), '--out', str(HERE / 'summary')]),
    ('analyze_profile_resources.py', ['--runs', str(HERE), '--out', str(HERE / 'resource-summary.json')])]:
    result = subprocess.run([sys.executable, str(REPO / 'benchmarks/edge_harness' / script), *args])
    record[script] = result.returncode
record.update(status='exited', end=time.time())
persist()
