"""Ownership/framing tests also run on POSIX; native connector tests run on Windows."""

import subprocess
import sys
import uuid

import pytest

from vllm_omni.host.shared_memory import SharedBufferOwner, read_buffer


def test_producer_retains_until_a_different_process_reads():
    owner = SharedBufferOwner()
    meta = owner.write(b"payload", f"omni-test-{uuid.uuid4().hex}")
    try:
        assert owner.retained_bytes > 0
        code = (
            "from vllm_omni.host.shared_memory import read_buffer; assert read_buffer(" + repr(meta) + ") == b'payload'"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
        owner.collect()
        assert owner.retained_bytes == 0
    finally:
        owner.release(meta["name"])


def test_bounds_stale_generation_and_name_collision():
    owner = SharedBufferOwner(max_bytes=128, max_buffers=1)
    meta = owner.write(b"one", f"omni-test-{uuid.uuid4().hex}")
    try:
        with pytest.raises(FileExistsError):
            SharedBufferOwner().write(b"collision", meta["name"])
        with pytest.raises(ValueError, match="stale"):
            read_buffer({**meta, "generation": "wrong"})
        with pytest.raises(BufferError):
            owner.write(b"two")
        assert owner.retained_bytes > 0
        assert read_buffer(meta) == b"one"
        owner.collect()
        assert owner.retained_bytes == 0
        with pytest.raises(BufferError):
            owner.write(b"x" * 129)
    finally:
        owner.release(meta["name"])


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows path and mapping lifetime")
def test_native_connector_roundtrip_and_cleanup():
    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
    from vllm_omni.host.shared_memory import owner
    from vllm_omni.windows.paths import shm_dir

    assert not shm_dir().startswith("/dev/shm")
    producer, consumer = SharedMemoryConnector({}), SharedMemoryConnector({})
    key = f"omni-test-{uuid.uuid4().hex}"
    try:
        ok, size, _ = producer.put("0", "1", key, {"data": [1, 2, 3]})
        assert ok and owner.retained_bytes > 0
        assert consumer.get("0", "1", key) == ({"data": [1, 2, 3]}, size)
        owner.collect()
        assert owner.retained_bytes == 0
        assert producer.put("0", "1", key, {"unconsumed": True})[0]
        assert owner.retained_bytes > 0
        producer.cleanup(key)
        assert owner.retained_bytes == 0
        assert consumer.get("0", "1", key) is None
        for index in range(8):
            next_key = f"{key}_{index}"
            assert producer.put("0", "1", next_key, {"index": index})[0]
            assert len(producer._pending_keys) <= 1
            assert consumer.get("0", "1", next_key)[0] == {"index": index}
            owner.collect()
    finally:
        producer.close()
        consumer.close()
