"""Same compiled artifact and dataset repeatability, not FP checkpoint quality."""

import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
rows = []
for model in ("tts", "minicpm"):
    reference = json.loads((root / f"{model}-s25-reference.json").read_text())
    current = json.loads((root / f"{model}-s25-repeat.json").read_text())
    assert current["status"] == "SUCCESS"
    assert current["model_id"] == reference["model_id"]
    assert current["input_dataset_id"] == reference["input_dataset_id"]
    with np.load(root / reference["outputs"]) as a, np.load(root / current["outputs"]) as b:
        assert set(a.files) == set(b.files)
        outputs = []
        for name in a.files:
            assert a[name].shape == b[name].shape
            outputs.append(
                {
                    "name": name,
                    "shape": list(b[name].shape),
                    "finite": bool(np.isfinite(b[name]).all()),
                    "exact": bool(np.array_equal(a[name], b[name])),
                    "max_abs_difference": float(np.max(np.abs(a[name].astype(float) - b[name].astype(float)))),
                }
            )
    rows.append(
        {
            "model": model,
            "job": current["job_id"],
            "reference_job": reference["job_id"],
            "model_id": current["model_id"],
            "input_dataset_id": current["input_dataset_id"],
            "scope": "same compiled component repeatability only",
            "outputs": outputs,
        }
    )
assert all(output["finite"] for row in rows for output in row["outputs"])
(root / "hub-repeatability.json").write_text(json.dumps(rows, indent=2) + "\n")
print([(row["model"], len(row["outputs"]), all(o["exact"] for o in row["outputs"])) for row in rows])
