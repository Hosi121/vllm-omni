# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A listening port alone must not make a different test's server ready."""

import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
from filelock import FileLock

from tests.helpers import runtime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_server_does_not_accept_another_process_listener(monkeypatch):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    server = runtime.OmniServer("fake-model", [])
    popen = subprocess.Popen

    # Occupy the chosen port after allocation, while the requested subprocess
    # is still importing. The old TCP-only probe returns ready for this socket.
    with socket.socket() as foreign_listener:

        def launch(*args, **kwargs):
            foreign_listener.bind((server.host, server.port))
            foreign_listener.listen()
            return popen([sys.executable, "-c", "import time; time.sleep(0.2); raise SystemExit(7)"])

        monkeypatch.setattr(runtime.subprocess, "Popen", launch)
        with pytest.raises(RuntimeError, match="exited with code 7"):
            with server:
                pass
    # Failed __enter__ must also release the lease for a later server.
    retry = runtime.OmniServer("retry-model", [], port=server.port)
    try:
        retry._reserve_port()
    finally:
        retry.__exit__(None, None, None)


def test_auto_ports_are_reserved_before_subprocesses_bind(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    # Both constructors see the same free port before either server has bound.
    first = runtime.OmniServer("first-model", [])
    candidate = first.port
    monkeypatch.setattr(runtime, "get_open_port", lambda *args: candidate)
    second = runtime.OmniServer("second-model", [])
    first._reserve_port()
    with socket.socket() as occupied, socket.socket() as probe:
        occupied.bind((first.host, candidate))
        probe.bind((first.host, 0))
        alternative = probe.getsockname()[1]
    monkeypatch.setattr(runtime, "get_open_port", lambda *args: alternative)
    try:
        second._reserve_port()
        assert first.port == candidate
        assert second.port == alternative
        # Check OS-level lock exclusion, not only the Python object's state.
        lock_path = next(tmp_path.glob(f"*-{candidate}.lock"))
        contender = FileLock(lock_path)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\nfrom filelock import FileLock, Timeout\n"
                "try:\n FileLock(sys.argv[1]).acquire(timeout=0)\n"
                "except Timeout:\n sys.exit(23)\n",
                str(lock_path),
            ],
            timeout=10,
            check=False,
        )
        assert result.returncode == 23
        explicit = runtime.OmniServer("explicit-model", [], port=candidate)
        with pytest.raises(RuntimeError, match="already reserved"):
            with explicit:
                pass
    finally:
        first.__exit__(None, None, None)
        second.__exit__(None, None, None)
    with contender.acquire(timeout=0):
        pass


@pytest.mark.parametrize("in_child", [False, True])
def test_server_accepts_its_own_listener(monkeypatch, in_child):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    server = runtime.OmniServer("fake-model", [])
    popen = subprocess.Popen

    def launch(*args, **kwargs):
        code = (
            "import socket,time; s=socket.socket(); "
            f"s.bind(({server.host!r}, {server.port})); s.listen(); time.sleep(30)"
        )
        if in_child:
            code = f"import subprocess,sys; child=subprocess.Popen([sys.executable,'-c',{code!r}]); child.wait()"
        return popen([sys.executable, "-c", code])

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    with server:
        assert server.proc is not None
        assert server.proc.poll() is None
        with socket.create_connection((server.host, server.port), timeout=1):
            pass


@pytest.mark.parametrize("init_timeout, expected_wait", [(None, 1200), (900, 1200), (1800, 2100)])
@pytest.mark.parametrize("port", [None, 18099])
def test_server_fixture_wait_covers_engine_init(monkeypatch, init_timeout, expected_wait, port):
    captured = {}

    class FakeServer:
        def __init__(self, model, serve_args, **kwargs):
            captured.update(kwargs)
            captured["serve_args"] = serve_args

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(runtime, "OmniServer", FakeServer)
    monkeypatch.setattr(runtime, "release_audio_transcriber", lambda: None)
    request = SimpleNamespace(
        param=runtime.OmniServerParams(model="fake-model", port=port, init_timeout=init_timeout),
        node=SimpleNamespace(get_closest_marker=lambda name: None),
    )
    generator = runtime.iter_omni_server(request, "core_model", threading.Lock())
    try:
        next(generator)
        assert captured["startup_timeout"] == expected_wait
        args = captured["serve_args"]
        assert args[args.index("--init-timeout") + 1] == str(init_timeout or 900)
    finally:
        generator.close()


def test_server_keeps_polling_after_default_deadline(monkeypatch):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    server = runtime.OmniServer("fake-model", [], startup_timeout=2100)
    monkeypatch.setattr(server, "_reserve_port", lambda: None)
    monkeypatch.setattr(server, "_owns_listening_port", lambda: True)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(poll=lambda: None))
    # The server becomes ready after the old 1200 s limit, still within 2100 s.
    ticks = iter([0, 1500])
    monkeypatch.setattr(runtime.time, "time", lambda: next(ticks))

    class ReadySocket:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def settimeout(self, _timeout):
            pass

        def connect_ex(self, _address):
            return 0

    monkeypatch.setattr(runtime.socket, "socket", lambda *a: ReadySocket())
    server._start_server()
