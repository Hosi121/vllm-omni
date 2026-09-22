"""Read-only current inventory without loading model runtimes during a benchmark."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
r = {'timestamp': time.time(), 'python': sys.executable, 'platform': platform.platform(),
     'packages': {}, 'roots': [], 'artifacts': [], 'source_sha256': {}}
for name in ('vllm', 'torch', 'vllm-omni', 'transformers', 'qai-hub', 'onnxruntime'):
    try:
        r['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        r['packages'][name] = None
roots = [a.root/'models', Path.home()/'.cache/huggingface/hub',
         Path('C:/Users/zhout/w2/models') if sys.platform == 'win32' else Path('/mnt/c/Users/zhout/w2/models')]
for root in roots:
    r['roots'].append({'path': str(root), 'exists': root.is_dir()})
    if not root.is_dir():
        continue
    for item in root.iterdir():
        if not any(s in item.name.lower() for s in ('spark', 'qwen3', 'minicpm', 'internvla')):
            continue
        for cfg in list(item.glob('config.json')) + list(item.glob('snapshots/*/config.json')):
            data = json.loads(cfg.read_text(encoding='utf-8'))
            weights = list(cfg.parent.glob('*.safetensors'))
            r['artifacts'].append({'path': str(cfg.parent), 'model_type': data.get('model_type'),
                                   'weight_shards': len(weights),
                                   'missing_shards': [str(w) for w in weights if not w.is_file()],
                                   'weight_bytes': sum(w.stat().st_size for w in weights if w.is_file())})
for path in (a.root/'vllm-omni-edge/benchmarks/edge_harness').glob('profile_*.py'):
    r['source_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
for cmd in (['git', '-C', str(a.root/'vllm-omni-edge'), 'rev-parse', 'HEAD'], ['adb', 'devices', '-l']):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors='replace', timeout=10)
        r[cmd[0]] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as error:
        r[cmd[0]] = {'error': repr(error)}
a.out.write_text(json.dumps(r, indent=2))
print(f'{len(r["artifacts"])} artifact configurations inventoried: {a.out}')
