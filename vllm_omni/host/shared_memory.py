# SPDX-License-Identifier: Apache-2.0
"""Bounded, single-consumer shared buffers with explicit Windows ownership.

Windows destroys a named mapping when its last handle closes. Keep a producer
handle until a generation-specific acknowledgement arrives from the reader.
The header also records payload length: opening a Windows mapping can report a
page-rounded allocation size. POSIX's existing unlink-based path is unchanged.
"""

from __future__ import annotations

import hashlib
import struct
import threading
import uuid
from multiprocessing import shared_memory
from pathlib import Path

_HEADER = struct.Struct("<8s16sQ")
_MAGIC = b"OMNISHM1"


def _ack_path(name: str, generation: bytes) -> Path:
    from vllm_omni.windows.paths import shm_dir

    identity = hashlib.sha256(name.encode() + generation).hexdigest()
    return Path(shm_dir()) / f"ack_{identity}"


class SharedBufferOwner:
    def __init__(self, max_bytes: int = 64 << 20, max_buffers: int = 256):
        self.max_bytes, self.max_buffers = max_bytes, max_buffers
        self._owned = {}
        self._lock = threading.RLock()
        self._collector = None

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return sum(size for _, _, size in self._owned.values())

    @property
    def retained_names(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._owned)

    def collect(self) -> None:
        with self._lock:
            for name, (_, generation, _) in list(self._owned.items()):
                if _ack_path(name, generation).exists():
                    self.release(name)

    def release(self, name: str) -> None:
        with self._lock:
            item = self._owned.get(name)
            if item is not None:
                segment, generation, _ = item
                segment.close()
                segment.unlink()  # no-op on Windows; useful for contract tests
                self._owned.pop(name)
                _ack_path(name, generation).unlink(missing_ok=True)

    def _collect_loop(self) -> None:
        event = threading.Event()
        while not event.wait(0.05):
            try:
                self.collect()
            except OSError:
                # Keep ownership charged if acknowledgement cleanup fails.
                continue

    def write(self, payload: bytes, name: str | None = None) -> dict:
        with self._lock:
            self.collect()
            size = _HEADER.size + len(payload)
            if len(self._owned) >= self.max_buffers or self.retained_bytes + size > self.max_bytes:
                raise BufferError("Windows shared-buffer ownership limit exceeded")
            generation = uuid.uuid4().bytes
            segment = shared_memory.SharedMemory(create=True, size=size, name=name)
            try:
                segment.buf[: _HEADER.size] = _HEADER.pack(_MAGIC, generation, len(payload))
                segment.buf[_HEADER.size : size] = payload
            except BaseException:
                segment.close()
                segment.unlink()
                raise
            self._owned[segment.name] = (segment, generation, size)
            if self._collector is None:
                self._collector = threading.Thread(target=self._collect_loop, daemon=True, name="omni-shm-owner")
                self._collector.start()
            return {"name": segment.name, "size": len(payload), "generation": generation.hex()}


def read_buffer(meta: dict) -> bytes:
    segment = shared_memory.SharedMemory(name=meta["name"])
    try:
        magic, generation, size = _HEADER.unpack(segment.buf[: _HEADER.size])
        if magic != _MAGIC or size > segment.size - _HEADER.size:
            raise ValueError("invalid Windows shared-buffer header")
        if meta.get("generation", generation.hex()) != generation.hex():
            raise ValueError("stale Windows shared-buffer generation")
        payload = bytes(segment.buf[_HEADER.size : _HEADER.size + size])
    finally:
        segment.close()
    _ack_path(meta["name"], generation).touch()
    return payload


owner = SharedBufferOwner()
