"""Collect existing TTS stage counters separately after every baseline/repair exits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REPO=ROOT/'vllm-omni-edge'
WSL='/home/zhout/project/edge_infer'
sys.path.insert(0,str(REPO/'benchmarks/edge_harness'))
from summarize_e2e_profiles import read_lines,stats
prior=psutil.Process(60180)
assert 'run_windows_tts_retry.py' in ' '.join(prior.cmdline())
identity=prior.create_time()
record={'status':'waiting','pid':os.getpid(),'predecessor_pid':prior.pid,'predecessor_created':identity,'jobs':[]}
def persist():
    (HERE/'attribution-queue.json').write_text(json.dumps(record,indent=2))
persist()
while prior.is_running() and prior.create_time()==identity:
    time.sleep(3)
assert json.loads((HERE/'windows-tts-retry-queue.json').read_text())['status']=='exited'
record['status']='running'
env=os.environ.copy()
env.update(PYTHONPATH=str(REPO),HF_HUB_OFFLINE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
           VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_WSL2_ENABLE_PIN_MEMORY='1',
           CUDA_HOME=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4')
env['PATH']=env['CUDA_HOME']+r'\bin;'+env['PATH']
cuda=WSL+'/.venvs/omni-cuda-029/lib/python3.12/site-packages/nvidia/cu13'
results=[]
for base in ('tts-cuda-wsl','tts-cuda-windows'):
    baseline=json.loads((HERE/base/'report.json').read_text())
    name=base+'-attribution'
    if baseline.get('status')!='completed':
        record['jobs'].append({'name':name,'status':'not_run','reason':'Full baseline did not complete'})
        persist()
        continue
    step_dir=HERE/(name+'-step-stats')
    job_env=env.copy()
    job_env['VLLM_OMNI_STEP_STATS_DIR']=str(step_dir)
    job_env['VLLM_OMNI_STEP_STATS_SYNC']='0'
    if base.endswith('wsl'):
        model='/home/zhout/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5'
        shell=(f'cd {WSL}/vllm-omni-edge && PYTHONPATH=. HF_HUB_OFFLINE=1 '
               f'VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WSL2_ENABLE_PIN_MEMORY=1 CUDA_HOME={cuda} '
               f'VLLM_OMNI_STEP_STATS_DIR={WSL}/analysis/experiments/e2e_profiling_20260922/{name}-step-stats '
               f'VLLM_OMNI_STEP_STATS_SYNC=0 PATH={cuda}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib '
               f'timeout 900 ../.venvs/omni-cuda-029/bin/python benchmarks/edge_harness/profile_local_tts.py '
               f'--model {model} --out ../analysis/experiments/e2e_profiling_20260922/{name} '
               f'--length-band medium --concurrency 1 --repeats 20 --sustained-seconds 0')
        command=['wsl','-d','Ubuntu','--','bash','-lc',shell]
    else:
        command=[sys.executable,str(REPO/'benchmarks/edge_harness/profile_local_tts.py'),
                 '--model',r'C:\Users\zhout\w2\models\Qwen3-TTS-0.6B-85e237c','--out',str(HERE/name),
                 '--length-band','medium','--concurrency','1','--repeats','20','--sustained-seconds','0']
    job={'name':name,'status':'running','start':time.time(),'command':command,'step_stats_dir':str(step_dir)}
    record['jobs'].append(job)
    persist()
    done=subprocess.run([sys.executable,str(REPO/'benchmarks/edge_harness/profile_host_command.py'),
                         '--out',str(HERE/(name+'-host')),'--cwd',str(REPO),'--',*command],env=job_env)
    path=HERE/name/'report.json'
    report=json.loads(path.read_text()) if path.exists() else {}
    job.update(status='exited',returncode=done.returncode,end=time.time(),benchmark_status=report.get('status','missing_report'),error=report.get('error'))
    persist()
    item={'run':name,'status':report.get('status','missing_report'),'counter_files':[],
          'counter_scope':'Existing StepStats with accelerator sync disabled; host timers, including warmup/slow-consumer/cancel/recovery. Nested counters overlap and must not be summed.',
          'percentiles':'Per-process native summaries use rounded sample indices and a capped 20000-value reservoir; not pooled percentiles. Periodic flush every 500 adds can omit final unflushed samples after forced shutdown.',
          'comparison_caveat':'Separate processes, cache/thermal history and stochastic speech length; ratios are diagnostic observations, not isolated causal instrumentation overhead.'}
    for path in sorted(step_dir.glob('*.json')):
        raw=path.read_bytes()
        item['counter_files'].append({'path':str(path.relative_to(HERE)),'sha256':hashlib.sha256(raw).hexdigest(),'record':json.loads(raw)})
    item['measurements']={}
    for label,folder in (('baseline',HERE/base),('instrumented',HERE/name)):
        rows=[r for r in read_lines(folder/'requests.jsonl') if r.get('phase')=='measured' and r.get('length_band')=='medium' and r.get('concurrency')==1]
        item['measurements'][label]={'n':len(rows),'metrics':{k:stats([r.get(k) for r in rows]) for k in ('ttfa_ms','rtf_total','total_wall_s','streamed_audio_s','stall_at_ttfa_ms')}}
    item['p50_ratios']={}
    for metric in ('ttfa_ms','rtf_total','total_wall_s'):
        before=item['measurements']['baseline']['metrics'][metric]['p50']
        after=item['measurements']['instrumented']['metrics'][metric]['p50']
        item['p50_ratios'][metric]=after/before if before and after is not None else None
    results.append(item)
    (HERE/'attribution-summary.json').write_text(json.dumps({'runs':results},indent=2)+'\n')
record['status']='running_analysis'
persist()
cmd=[sys.executable,str(REPO/'benchmarks/edge_harness/analyze_profile_resources.py'),
     '--runs',str(HERE),'--out',str(HERE/'resource-summary.json')]
done=subprocess.run(cmd,env=env)
record.update(status='exited',end=time.time(),analysis_returncode=done.returncode)
persist()
