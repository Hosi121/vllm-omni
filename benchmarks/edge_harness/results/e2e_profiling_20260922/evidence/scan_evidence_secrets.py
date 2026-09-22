"""Pre-publication heuristic scan; report locations only, never matched values."""
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
patterns = {
    'credential_assignment': re.compile(r'''(?i)(?:api[_-]?(?:token|key)|access_token|authorization|bearer|X-Amz-Credential|X-Amz-Signature)["']?\s*[:=]\s*["']?([A-Za-z0-9_./+=-]{16,})'''),
    'credential_prefix': re.compile(r'\b(?:sk-proj-|ghp_|github_pat_)[A-Za-z0-9_-]{16,}'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}
hits, scanned = [], 0
for path in sorted(HERE.rglob('*')):
    if not path.is_file() or path.name == Path(__file__).name:
        continue
    if path.suffix not in ('.py', '.md', '.json', '.jsonl', '.txt', '.log'):
        continue
    # Huge trace files contain generated event records, but are scanned as streams too.
    scanned += 1
    with path.open(encoding='utf-8', errors='replace') as stream:
        for number, line in enumerate(stream, 1):
            for name, pattern in patterns.items():
                if pattern.search(line):
                    hits.append({'path': path.relative_to(HERE).as_posix(), 'line': number, 'pattern': name})
result = {'files_scanned': scanned, 'matches': hits,
          'scope': 'Heuristic credential-pattern scan. Matches need review; absence is not a proof of no secrets. Values are never emitted.'}
(HERE / 'secret-scan.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
