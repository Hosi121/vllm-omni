# SPDX-License-Identifier: Apache-2.0
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from analyze_profile_resources import interval_union, telemetry_summary, trace_summary


class ResourceEvidenceTests(unittest.TestCase):
    def test_compressed_streaming_trace_with_large_event(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.json.gz"
            events = [
                {"ph": "X", "cat": "kernel", "name": "a", "ts": 0, "dur": 10, "args": {"large": "x" * 70000}},
                {"ph": "X", "cat": "kernel", "name": "b", "ts": 5, "dur": 10},
            ]
            with gzip.open(path, "wt") as f:
                json.dump({"traceEvents": events, "metadata": "tail"}, f)
            summary = trace_summary(path)
            self.assertEqual(summary["categories"]["kernel"]["duration_sum_us"], 20)
            self.assertEqual(summary["categories"]["kernel"]["busy_union_us"], 15)

    def test_overlapping_kernels_are_not_added_as_wall_time(self):
        self.assertEqual(interval_union([(0, 10), (3, 5), (6, 12), (20, 25)]), 17)

    def test_energy_units_and_gaps(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "telemetry.jsonl"

            def row(t):
                return {
                    "unix": t,
                    "host_memory": {"available": 1, "used": 2},
                    "gpu_csv": "id, name, 1000, 2000, 60, 100, 900, 500, 90, P0",
                }

            path.write_text("\n".join(json.dumps(row(t)) for t in (0, 1, 10)))
            result = telemetry_summary(path)
            self.assertEqual(result["whole_gpu_energy_j_over_observed_intervals"], 100)
            self.assertEqual(result["energy_observed_duration_s"], 1)
            self.assertEqual(result["gaps_over_5s"], 1)
            result = telemetry_summary(path, start=1)
            self.assertIsNone(result["whole_gpu_energy_j_over_observed_intervals"])


if __name__ == "__main__":
    unittest.main()
