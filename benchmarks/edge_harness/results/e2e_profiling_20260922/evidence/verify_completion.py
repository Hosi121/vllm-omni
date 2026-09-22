"""Verify terminal sample coverage and diagnostic artifact hashes before publication."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


for queue in ('tts-trace-refresh', 'attribution-retry'):
    assert read(ROOT / (queue + '-queue.json'))['status'] == 'exited', queue
matrix = read(ROOT / 'summary/matrix.json')
assert len(matrix['cases']) == 60
cases = [case for case in matrix['cases'] if case['profiling']['status'] != 'not_run']
assert len(cases) == 5
measured = 0
for case in cases:
    profile = case['profiling']
    assert profile['profile_protocol_complete'], case['id'] if 'id' in case else case['device']
    groups = profile['measurements']
    assert {(group['length_band'], group['concurrency']) for group in groups} == {
        (length, concurrency) for length in ('short', 'medium', 'long') for concurrency in (1, 2, 4)
    }
    assert all(group['n'] == 20 for group in groups)
    measured += sum(group['n'] for group in groups)
assert measured == 900

traces = []
for name in ('spark-cpu-wsl', 'spark-cuda-wsl', 'spark-cuda-windows', 'tts-cuda-wsl', 'tts-cuda-windows'):
    folder = ROOT / (name + '-trace')
    report = read(folder / 'report.json')
    assert report['status'] == 'completed', name
    for item in report['traces']:
        path = folder / item['path'].replace('\\', '/')
        assert path.resolve().is_relative_to(folder.resolve())
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], path
        traces.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': item['sha256']})
    if name.startswith('tts'):
        assert set(report['stage_trace_prefixes']) == {'0', '1'}
        for prefix in report['stage_trace_prefixes'].values():
            assert any(prefix in item['path'] for item in report['traces'])

counters = []
for run in read(ROOT / 'attribution-summary.json')['runs']:
    assert run['status'] == 'completed', run['run']
    assert run['measurements']['instrumented']['n'] == 20, run['run']
    assert len(run['counter_files']) == 2, run['run']
    for record in run['counter_files']:
        path = ROOT / record['path'].replace('\\', '/')
        assert digest(path) == record['sha256'], path
        counters.append({'path': record['path'], 'sha256': record['sha256']})

result = {'status': 'verified', 'cells': 60, 'profiled_paths': 5, 'measured_baseline_requests': measured,
          'trace_files': traces, 'counter_files': counters,
          'scope': 'Terminal sample coverage and source artifact integrity; not model-quality or release qualification.'}
(ROOT / 'completion-integrity.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key not in ('trace_files', 'counter_files')}))
