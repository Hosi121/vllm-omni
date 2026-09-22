"""Read live report snapshots from Linux, avoiding Windows deny-delete handles."""
import json
import os
from pathlib import Path
import time
assert os.name != 'nt', 'Run with the WSL Python interpreter.'
root=Path(__file__).resolve().parent
for name in ('spark-cpu-wsl','spark-cuda-wsl','spark-cuda-windows','tts-cuda-wsl','tts-cuda-windows'):
    path=root/name/'report.json'
    if not path.exists():
        continue
    try:
        r=json.loads(path.read_text())
    except json.JSONDecodeError:
        print(json.dumps({'name':name,'status':'snapshot_mid_write'}))
        continue
    row={'name':name,**{key:r.get(key) for key in ('status','requests','error','sustained_wall_s','report_write_permission_errors')}}
    if r.get('status')=='running' and r.get('sustained_start_unix'):
        row['sustained_elapsed_s']=round(time.time()-r['sustained_start_unix'])
    print(json.dumps(row))
for name in ('retry-status','diagnostic-queue','reliability-queue','tts-reliability-queue','analysis-queue','windows-tts-retry-queue','attribution-queue','cpu-trace-refresh-queue','tts-trace-refresh-queue','attribution-retry-queue'):
    path=root/(name+'.json')
    if path.exists():
        try:
            r=json.loads(path.read_text())
            print(json.dumps({'queue':name,'status':r.get('status'),'pid':r.get('pid'),'baseline_retry':r.get('baseline_retry')}))
        except json.JSONDecodeError:
            print(json.dumps({'queue':name,'status':'snapshot_mid_write'}))
