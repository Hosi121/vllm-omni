# SPDX-License-Identifier: Apache-2.0
"""Archive profiling evidence with byte hashes and reversible large-file compression.

Run only after all writers exit. No benchmark result is changed by this utility.
"""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def digest(path, compressed=False):
    hasher = hashlib.sha256()
    size = 0
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = root / record["stored_path"]
        assert path.resolve().is_relative_to(root.resolve()), "Archive path escapes root"
        assert digest(path) == (record["stored_sha256"], record["stored_bytes"]), path
        assert digest(path, record["encoding"] == "gzip") == (record["sha256"], record["bytes"]), path
    return len(manifest["files"])


def archive(source, destination):
    assert source.resolve() != destination.resolve()
    assert not destination.resolve().is_relative_to(source.resolve())
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or "__pycache__" in relative.parts or path.suffix in (".pyc", ".tmp"):
            continue
        assert not path.is_symlink(), f"Unexpected evidence symlink: {relative}"
        original_hash, original_size = digest(path)
        compress = original_size > 50 * 1024 * 1024 and path.suffix != ".gz"
        stored = Path("evidence") / relative
        if compress:
            stored = stored.with_name(stored.name + ".archive.gz")
        target = destination / stored
        target.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            with path.open("rb") as reader, target.open("wb") as writer:
                with gzip.GzipFile(filename="", mode="wb", fileobj=writer, mtime=0) as packed:
                    shutil.copyfileobj(reader, packed, length=1024 * 1024)
        else:
            shutil.copyfile(path, target)
        stored_hash, stored_size = digest(target)
        assert digest(path) == (original_hash, original_size), f"Source changed during archive: {relative}"
        records.append(
            {
                "source_path": relative.as_posix(),
                "stored_path": stored.as_posix(),
                "sha256": original_hash,
                "bytes": original_size,
                "stored_sha256": stored_hash,
                "stored_bytes": stored_size,
                "encoding": "gzip" if compress else "identity",
            }
        )
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "scope": "Raw evidence bytes; hashes establish integrity, not model correctness or qualification.",
                "files": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / ".gitattributes").write_text("evidence/** -text -whitespace\n", encoding="utf-8")
    return verify(destination)


def restore(root, destination):
    verify(root)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        target = destination / record["source_path"]
        assert target.resolve().is_relative_to(destination.resolve()), "Restore path escapes root"
        target.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if record["encoding"] == "gzip" else open
        with opener(root / record["stored_path"], "rb") as reader, target.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        assert digest(target) == (record["sha256"], record["bytes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    create = sub.add_parser("create")
    create.add_argument("source", type=Path)
    create.add_argument("destination", type=Path)
    check = sub.add_parser("verify")
    check.add_argument("archive", type=Path)
    unpack = sub.add_parser("restore")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.operation == "create":
        print(f"Archived and verified {archive(args.source, args.destination)} files")
    elif args.operation == "verify":
        print(f"Verified {verify(args.archive)} files")
    else:
        restore(args.archive, args.destination)
        print(f"Restored evidence to {args.destination}")


if __name__ == "__main__":
    main()
