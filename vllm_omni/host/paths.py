# SPDX-License-Identifier: Apache-2.0
"""Host-local paths, independent of the checkout and model location."""

import os
import tempfile


def device_lock_directory() -> str:
    # A rooted POSIX path resolves against the current UNC share on Windows.
    # Opening /tmp from a WSL checkout can block inside the filesystem driver.
    # All same-user controllers/workers use the same native temporary directory.
    return tempfile.gettempdir() if os.name == "nt" else "/tmp"
