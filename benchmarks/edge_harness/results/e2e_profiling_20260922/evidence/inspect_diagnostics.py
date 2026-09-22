"""Read diagnostic outcomes through Linux without holding Windows report locks."""
import json,os
from pathlib import Path
assert os.name!='nt'
root=Path(__file__).resolve().parent
for path in sorted([*root.glob('*-trace/report.json'),*root.glob('*-reliability/report.json')]):
    try:
        r=json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        continue
    d={'name':path.parent.name,'status':r.get('status'),'error':r.get('error')}
    if path.parent.name.endswith('-trace'):
        d.update(traces=len(r.get('traces',[])),medians=r.get('median_wall_s'),stop_rpc_s=r.get('stop_rpc_s'))
    else:
        for key in ('saturation_reached','same_tokens_after_saturation','stale_handle_rejected','oversized_plan_refused','cleanup_error'):
            if key in r:
                d[key]=r[key]
        crash=r.get('crash_injection',r.get('crash',{}))
        d['crash_error_reported']=crash.get('explicit_error_reported')
        if 'owned_processes_requiring_harness_cleanup' in r:
            d['forced_cleanup_count']=len(r['owned_processes_requiring_harness_cleanup'])
    print(json.dumps(d))
