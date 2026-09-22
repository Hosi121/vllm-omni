# SPDX-License-Identifier: Apache-2.0
"""Check full matrix coverage and archived evidence integrity, not model quality.

Run from any directory with Python's standard library:
    python verify_qualification_matrix.py path/to/matrix.json
"""

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path


def verify(path: Path):
    record = json.loads(path.read_text(encoding="utf-8"))
    pairs = [(case["device"], case["model"]) for case in record["cases"]]
    expected = set(itertools.product(record["devices"], record["models"]))
    assert len(pairs) == len(set(pairs)) and set(pairs) == expected, "missing or duplicate pairings"
    allowed = {"PASS", "PARTIAL", "COMPONENT", "PROFILE_ONLY", "FAILED", "REJECTED", "BLOCKED"}
    for case in record["cases"]:
        assert case["status"] in allowed
        assert all(case.get(field) for field in ("finding", "next_step", "check", "depth", "evidence"))
        assert case["status"] != "PASS" or case["depth"] == "P", "component pass promoted to full workload"
        assert all(ref in record["sources"] for ref in case["evidence"])
    for source in record["sources"].values():
        evidence = (path.parent / source["file"]).resolve()
        assert evidence.is_relative_to(path.parent.resolve()), "evidence escapes bundle"
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == source["sha256"], evidence
    counts = dict(Counter(case["status"] for case in record["cases"]))
    assert counts == record["counts"]
    print(f"Verified {len(pairs)} pairings and {len(record['sources'])} evidence hashes: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    verify(parser.parse_args().matrix)
