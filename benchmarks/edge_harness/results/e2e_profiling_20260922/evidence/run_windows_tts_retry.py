"""Retry the native speech profile after the earlier complete serial queue exits."""
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
ROOT = HERE.parents[2]
REPO = ROOT/'vllm-omni-edge'
prior = psutil.Process(61376)
assert 'run_final_analysis.py' in ' '.join(prior.cmdline())
identity = prior.create_time()
record = {'status':'waiting','pid':os.getpid(),'predecessor_pid':prior.pid,
          'predecessor_created':identity,'reason':'Native TTS report replacement hit Windows read-handle sharing contention.','jobs':[]}
def persist():
    (HERE/'windows-tts-retry-queue.json').write_text(json.dumps(record,indent=2))
persist()
while prior.is_running() and prior.create_time() == identity:
    time.sleep(3)
assert json.loads((HERE/'analysis-queue.json').read_text())['status'] == 'exited'
base = 'tts-cuda-windows'
old = json.loads((HERE/base/'report.json').read_text())
assert old['status'] == 'failed' and 'PermissionError' in old['error'], 'Unexpected baseline state; inspect before retry'
archive = HERE/'reporting-failures-1'
archive.mkdir(exist_ok=False)
for name in (base,base+'-host',base+'-warmup-audio-integrity.json'):
    source,destination=HERE/name,archive/name
    assert source.resolve().is_relative_to(HERE.resolve())
    assert destination.resolve().is_relative_to(HERE.resolve()) and not destination.exists()
    if source.exists():
        source.rename(destination)
shutil.copy2(HERE/'queue.json',archive/'queue.json')
env=os.environ.copy()
env.update(PYTHONPATH=str(REPO),HF_HUB_OFFLINE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
           VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_WSL2_ENABLE_PIN_MEMORY='1',
           CUDA_HOME=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4')
env['PATH']=env['CUDA_HOME']+r'\bin;'+env['PATH']
model=r'C:\Users\zhout\w2\models\Qwen3-TTS-0.6B-85e237c'

def run_job(name,script,extra=()):
    command=[sys.executable,str(REPO/'benchmarks/edge_harness'/script),
             '--model',model,'--out',str(HERE/name),*extra]
    job={'name':name,'command':command,'status':'running','start':time.time()}
    record['jobs'].append(job)
    persist()
    result=subprocess.run([sys.executable,str(REPO/'benchmarks/edge_harness/profile_host_command.py'),
                           '--out',str(HERE/(name+'-host')),'--cwd',str(REPO),'--',*command],env=env)
    path=HERE/name/'report.json'
    report=json.loads(path.read_text()) if path.exists() else {}
    job.update(status='exited',returncode=result.returncode,end=time.time(),
               benchmark_status=report.get('status','missing_report'),error=report.get('error'))
    persist()
    return report

record['status']='running_baseline'
result=run_job(base,'profile_local_tts.py')
if result.get('status') == 'completed':
    trace=HERE/(base+'-trace')
    path=trace/'report.json'
    existing=json.loads(path.read_text()) if path.exists() else {}
    valid=existing.get('status')=='completed' and existing.get('traces') and all(
        hashlib.sha256((trace/t['path']).read_bytes()).hexdigest()==t['sha256'] for t in existing['traces'])
    if valid:
        record['jobs'].append({'name':base+'-trace','status':'exited','benchmark_status':'completed',
                               'reused_verified_report':str(path),'reason':'Same checkpoint/runtime/config; valid paired diagnostic from earlier queue.'})
        persist()
    else:
        for name in (base+'-trace',base+'-trace-host'):
            source,destination=HERE/name,archive/name
            assert source.resolve().is_relative_to(HERE.resolve())
            assert destination.resolve().is_relative_to(HERE.resolve()) and not destination.exists()
            if source.exists():
                source.rename(destination)
        record['status']='running_trace'
        run_job(base+'-trace','profile_trace_diagnostic.py',('--kind','tts'))
    record['status']='running_reliability'
    run_job(base+'-reliability','check_tts_reliability.py')
    rows=[json.loads(line) for line in (HERE/base/'requests.jsonl').read_text().splitlines()]
    checked=[]
    for row in rows:
        if row['phase']!='warmup':
            continue
        for chunk in row['chunks_all']:
            path=HERE/base/'audio_chunks'/f"{row['request_id']}-{chunk['idx']}.f32"
            raw=path.read_bytes()
            checked.append({'path':str(path.relative_to(HERE)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                            'expected_sha256':chunk['sha256'],'expected_bytes':chunk['samples']*4})
    failures=[r for r in checked if r['sha256']!=r['expected_sha256'] or r['bytes']!=r['expected_bytes']]
    (HERE/(base+'-warmup-audio-integrity.json')).write_text(json.dumps({
        'created_unix':time.time(),'scope':'Archived warmup chunks including terminal tails; integrity only, not speech quality.',
        'chunks_checked':len(checked),'failures':failures,'files':checked},indent=2)+'\n')
record['status']='running_analysis'
persist()
commands=[
 [sys.executable,str(REPO/'benchmarks/edge_harness/summarize_e2e_profiles.py'),
  '--matrix',str(REPO/'benchmarks/edge_harness/results/model_device_matrix_20260922/matrix.json'),
  '--runs',str(HERE),'--out',str(HERE/'summary')],
 [sys.executable,str(REPO/'benchmarks/edge_harness/analyze_profile_resources.py'),
  '--runs',str(HERE),'--out',str(HERE/'resource-summary.json')]]
for command in commands:
    completed=subprocess.run(command,env=env)
    record['jobs'].append({'name':Path(command[1]).name,'status':'exited','returncode':completed.returncode})
    persist()
record.update(status='exited',end=time.time())
persist()
