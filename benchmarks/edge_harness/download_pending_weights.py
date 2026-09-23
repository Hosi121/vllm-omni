#!/usr/bin/env python3
"""Resume the missing E2E checkpoints and verify every safetensors SHA-256.

Weights and status files stay outside the Git checkout. This helper can wait for
an already-running Hugging Face transfer before resuming it, so an interrupted
interactive session does not leave an unverified partial checkpoint behind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


@dataclass(frozen=True)
class Checkpoint:
    repo: str
    revision: str
    directory: str


CHECKPOINTS = {
    "minicpmo": Checkpoint(
        "openbmb/MiniCPM-o-4_5",
        "503e754207c94da6bb26850b4469f367c9ea3582",
        "MiniCPM-o-4_5",
    ),
    "internvla_base": Checkpoint(
        "InternRobotics/InternVLA-A1-3B",
        "08cf716d413aa8208a34f346cd8824e6ff901205",
        "InternVLA-A1-3B",
    ),
    "internvla_place_markpen": Checkpoint(
        "Jia-Zeng/InternVLA-A1-3B-FineTuned-Place_Markpen",
        "0a557c6a503e6545b369bff2f34c76a410bbd190",
        "InternVLA-A1-3B-FT-Place_Markpen",
    ),
}


def _status(path: Path, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(fields, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _wait_for_existing(pid: int | None, repo: str) -> None:
    if pid is None:
        return
    command_path = Path(f"/proc/{pid}/cmdline")
    while command_path.exists():
        try:
            command = command_path.read_bytes()
        except (FileNotFoundError, PermissionError):
            break
        if repo.encode() not in command:
            break
        time.sleep(30)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(checkpoint: Checkpoint, directory: Path) -> list[dict[str, object]]:
    info = HfApi().model_info(checkpoint.repo, revision=checkpoint.revision, files_metadata=True)
    weights = [sibling for sibling in info.siblings if sibling.rfilename.endswith(".safetensors")]
    if not weights:
        raise RuntimeError(f"No safetensors weight files advertised by {checkpoint.repo}")
    verified = []
    for sibling in weights:
        path = directory / sibling.rfilename
        expected_size = sibling.size
        expected_hash = sibling.lfs.sha256 if sibling.lfs else None
        if expected_size is None or expected_hash is None:
            raise RuntimeError(f"Missing size or SHA-256 metadata for {sibling.rfilename}")
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"Missing or incomplete weight: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {path}: {actual_hash} != {expected_hash}")
        verified.append({"file": sibling.rfilename, "bytes": expected_size, "sha256": actual_hash})
        print(f"Verified {path.name}: {expected_size} bytes", flush=True)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=CHECKPOINTS)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()

    checkpoint = CHECKPOINTS[args.model]
    root = args.model_root.resolve()
    status_path = root / ".weight_download_status" / f"{args.model}.json"
    if args.detach:
        log_path = status_path.with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), args.model, "--model-root", str(root)]
        if args.wait_pid is not None:
            command.extend(["--wait-pid", str(args.wait_pid)])
        with log_path.open("ab") as log:
            process = subprocess.Popen(  # noqa: S603
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                env={**os.environ, "HF_XET_HIGH_PERFORMANCE": "1"},
            )
        print(json.dumps({"pid": process.pid, "log": str(log_path)}))
        return

    directory = root / checkpoint.directory
    _status(status_path, status="waiting", repo=checkpoint.repo, revision=checkpoint.revision)
    try:
        _wait_for_existing(args.wait_pid, checkpoint.repo)
        _status(status_path, status="downloading", repo=checkpoint.repo, revision=checkpoint.revision)
        snapshot_download(
            repo_id=checkpoint.repo,
            revision=checkpoint.revision,
            local_dir=str(directory),
            max_workers=4,
        )
        _status(status_path, status="verifying", repo=checkpoint.repo, revision=checkpoint.revision)
        files = _verify(checkpoint, directory)
        _status(status_path, status="verified", repo=checkpoint.repo, revision=checkpoint.revision, files=files)
    except Exception as exc:
        _status(status_path, status="failed", repo=checkpoint.repo, revision=checkpoint.revision, error=str(exc))
        raise


if __name__ == "__main__":
    main()
