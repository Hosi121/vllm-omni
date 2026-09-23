# SPDX-License-Identifier: Apache-2.0
"""Local CPU Qwen3-TTS whole-session stage with isolated dependencies.

The worker alone imports Qwen's PyTorch wrapper and Transformers 4.57.3.
This adapter reuses the bounded complete-WAV request, fencing and ownership
methods from the CrispASR audio client; its model and launch path are separate.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from vllm_omni.engine.resource_ledger import ResourceUnavailable

from .crisp_tts import CrispTTSStageClient, _local_port, _sha256


class QwenTTSCPUStageClient(CrispTTSStageClient):
    def __init__(self, metadata, config: dict, ledger, reservation) -> None:
        for name, value in vars(metadata).items():
            setattr(self, name, value)
        self.stage_type = "graph"  # Omni complete-request control path.
        self._ledger, self._reservation = ledger, reservation
        self._generation = uuid.uuid4().hex
        self._proc: subprocess.Popen | None = None
        self._log_stream = None
        self._closed = False
        self._active = None
        self._task = None
        self._output = None
        self._epoch = 0
        self._memory_pool = str(config.get("memory_pool", "host_ram"))
        self._max_text_bytes = int(config.get("max_text_bytes", 4096))
        self._max_wav_bytes = int(config.get("max_wav_bytes", 4 << 20))
        self._max_audio_s = float(config.get("max_audio_s", 10))
        self._request_timeout_s = float(config.get("request_timeout_s", 180))
        self._voice = "Ryan"
        self._seed = 42
        self._metric_name = "qwen_cpu_tts_wall_s"
        self._port = int(config.get("port") or _local_port())
        try:
            if (
                self._max_text_bytes <= 0 or self._max_wav_bytes <= 0 or self._max_audio_s <= 0
                or self._request_timeout_s <= 0 or not 0 < self._port < 65536
            ):
                raise ValueError("invalid Qwen CPU TTS input, output, timeout or port bound")
            if self._memory_pool not in reservation.demands:
                raise ResourceUnavailable("CPU TTS memory pool absent from stage reservation")
            demand = reservation.demands[self._memory_pool]
            if self._max_wav_bytes > demand:
                raise ResourceUnavailable("CPU TTS output bound exceeds stage reservation")
            # On POSIX, a virtualenv's python is often a symlink to the base
            # interpreter. Launch via the declared path so Python discovers
            # the virtualenv; _sha256 still follows the link for verification.
            self._python = Path(config["python_bin"]).absolute()
            if not self._python.is_file():
                raise FileNotFoundError(f"CPU TTS interpreter absent: {self._python}")
            self._model_dir = Path(config["model_dir"]).resolve(strict=True)
            self._overlay = Path(config["overlay_dir"]).resolve(strict=True)
            self._talker = self._model_dir / "model.safetensors"
            self._tokenizer = self._model_dir / "speech_tokenizer" / "model.safetensors"
            self._worker = Path(__file__).with_name("qwen_tts_cpu_worker.py").resolve(strict=True)
            expected = {
                "python_sha256": (self._python, str(config["python_sha256"]).lower()),
                "talker_sha256": (self._talker, str(config["talker_sha256"]).lower()),
                "tokenizer_sha256": (self._tokenizer, str(config["tokenizer_sha256"]).lower()),
            }
            if not (self._overlay / "transformers" / "__init__.py").is_file():
                raise FileNotFoundError("pinned Transformers overlay is absent")
            overlay_marker_hash = config.get("overlay_marker_sha256")
            if overlay_marker_hash is not None:
                marker = self._overlay / "kernels" / "__init__.py"
                if not marker.is_file() or _sha256(marker) != str(overlay_marker_hash).lower():
                    raise ValueError("CPU TTS optional-kernels overlay differs from declared hash")
            for label, (path, digest) in expected.items():
                if _sha256(path) != digest:
                    raise ValueError(f"CPU TTS {label} differs from the declared artifact hash")
            overhead = int(config["memory_overhead_bytes"])
            weight_bytes = self._talker.stat().st_size + self._tokenizer.stat().st_size
            if overhead <= 0 or weight_bytes + overhead > demand:
                raise ResourceUnavailable("CPU TTS weights plus state/workspace/headroom exceed reservation")
            self._log_path = Path(config["log_file"]).resolve()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_stream = self._log_path.open("wb")
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": str(self._overlay),
                "QWEN_TTS_OVERLAY_DIR": str(self._overlay),
                "CUDA_VISIBLE_DEVICES": "",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": "8",
                "PYTHONUTF8": "1",
            })
            command = [
                str(self._python), "-u", str(self._worker),
                "--model-dir", str(self._model_dir), "--port", str(self._port),
                "--threads", "8",
            ]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._proc = subprocess.Popen(
                command, stdout=self._log_stream, stderr=subprocess.STDOUT,
                env=env, creationflags=flags,
            )
            base = f"http://127.0.0.1:{self._port}"
            deadline = time.monotonic() + float(config.get("start_timeout_s", 180))
            while True:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"CPU TTS worker exited during load; see {self._log_path}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"CPU TTS worker did not become ready; see {self._log_path}")
                time.sleep(0.2)
            with urllib.request.urlopen(base + "/props", timeout=5) as response:
                props = json.load(response)
            if (
                Path(props.get("model_dir", "")).resolve() != self._model_dir
                or props.get("placement") != "cpu"
                or props.get("transformers") != "4.57.3"
                or props.get("qwen_tts") != "0.1.1"
                or props.get("torch") != str(config["expected_torch"])
                or props.get("dtype") != "bfloat16"
                or props.get("sample_rate") != 24000
                or props.get("speaker") != "Ryan"
                or props.get("max_new_tokens") != 64
            ):
                raise RuntimeError("CPU TTS worker environment, model or placement differs from plan")
            import psutil

            root = psutil.Process(self._proc.pid)
            processes = [root, *root.children(recursive=True)]
            self._loaded_process_tree = []
            for process in processes:
                memory = process.memory_full_info()
                self._loaded_process_tree.append({
                    "pid": process.pid,
                    "rss_bytes": int(memory.rss),
                    "private_bytes": int(getattr(memory, "private", memory.rss)),
                })
            self._loaded_rss_bytes = sum(row["rss_bytes"] for row in self._loaded_process_tree)
            self._loaded_private_bytes = sum(row["private_bytes"] for row in self._loaded_process_tree)
            if max(self._loaded_rss_bytes, self._loaded_private_bytes) > demand:
                raise ResourceUnavailable("loaded CPU TTS process tree exceeds stage reservation")
            self._base_url = base
            self.execution_plan = {
                "backend": "external.qwen_tts.cpu.v1",
                "stage_id": self.stage_id,
                "worker_generation": self._generation,
                "worker_pid": self._proc.pid,
                "worker_script_sha256": _sha256(self._worker),
                "artifact_sha256": {label: digest for label, (_, digest) in expected.items()},
                "overlay_marker_sha256": overlay_marker_hash,
                "model_dir": str(self._model_dir),
                "placement": "CPU BF16 SDPA, 8 threads; isolated Transformers 4.57.3",
                "worker_props": props,
                "loaded_rss_bytes": self._loaded_rss_bytes,
                "loaded_private_bytes": self._loaded_private_bytes,
                "loaded_process_tree": self._loaded_process_tree,
                "reserved_bytes": dict(reservation.demands),
                "memory_overhead_bytes": overhead,
                "max_text_bytes": self._max_text_bytes,
                "max_wav_bytes": self._max_wav_bytes,
                "max_audio_s": self._max_audio_s,
                "voice": self._voice,
                "seed": self._seed,
                "request_capacity": 1,
                "stateful_session": "Qwen PyTorch worker-owned; complete WAV per request",
                "evidence": "B",
            }
        except BaseException:
            self.shutdown()
            raise

    def _terminate(self) -> bool:
        import psutil

        children = []
        try:
            if self._proc is not None:
                children = psutil.Process(self._proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            return False
        drained = True
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                drained = False
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                drained = False
        if psutil.wait_procs(alive, timeout=5)[1]:
            drained = False
        return super()._terminate() and drained
