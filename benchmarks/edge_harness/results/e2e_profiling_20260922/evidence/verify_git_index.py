"""Check staged raw evidence bytes against the publication manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('repository', type=Path)
parser.add_argument('bundle', type=Path)
args = parser.parse_args()
repo, bundle = args.repository.resolve(), args.bundle.resolve()
prefix = bundle.relative_to(repo).as_posix()
manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
for record in manifest['files']:
    path = prefix + '/' + record['stored_path']
    proc = subprocess.Popen(['git', '-C', str(repo), 'show', ':' + path], stdout=subprocess.PIPE)
    hasher = hashlib.sha256()
    size = 0
    with proc.stdout as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
            size += len(chunk)
    assert proc.wait() == 0, path
    assert (hasher.hexdigest(), size) == (record['stored_sha256'], record['stored_bytes']), path
print(f"Verified {len(manifest['files'])} staged evidence files against the manifest")
