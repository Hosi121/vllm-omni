# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from profile_local_text import save


class ReportWriteTests(unittest.TestCase):
    def test_reader_contention_does_not_sleep_or_abort_running_inference(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "report.json"
            value = {"status": "running", "requests": 0}
            save(path, value)
            value["requests"] = 1
            with (
                patch.object(Path, "replace", side_effect=PermissionError("reader denies delete")),
                patch("profile_local_text.time.sleep", side_effect=AssertionError("must not block inference")),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                save(path, value)
            self.assertEqual(json.loads(path.read_text())["requests"], 0)
            self.assertEqual(value["report_write_permission_errors"], 1)
            value["status"] = "completed"
            save(path, value)
            self.assertEqual(json.loads(path.read_text())["requests"], 1)

    def test_persistent_final_write_failure_remains_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "report.json"
            with (
                patch.object(Path, "replace", side_effect=PermissionError("persistent denial")),
                patch("profile_local_text.time.sleep"),
                self.assertRaises(PermissionError),
            ):
                save(path, {"status": "completed"})
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
