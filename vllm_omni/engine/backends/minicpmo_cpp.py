# SPDX-License-Identifier: Apache-2.0
"""Bounded whole-session MiniCPM-o GGUF stage in Omni's graph control path.

The C++ worker owns weights, KV, TTS and vocoder state for one complete request.
Omni owns artifact verification, shared-memory admission, request identity,
terminal output acknowledgement and process cancellation. This stage has no
incremental playback or state handoff contract.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from omni_stage_contracts import StageEvent, StageRequest
from PIL import Image
from vllm.outputs import CompletionOutput

from vllm_omni.engine.resource_ledger import ResourceUnavailable
from vllm_omni.engine.stage_client import StageClientBase
from vllm_omni.outputs import OmniRequestOutput

_ARTIFACTS = (
    "MiniCPM-o-4_5-Q4_K_M.gguf",
    "audio/MiniCPM-o-4_5-audio-F16.gguf",
    "vision/MiniCPM-o-4_5-vision-F16.gguf",
    "tts/MiniCPM-o-4_5-tts-F16.gguf",
    "tts/MiniCPM-o-4_5-projector-F16.gguf",
    "token2wav-gguf/encoder.gguf",
    "token2wav-gguf/flow_matching.gguf",
    "token2wav-gguf/flow_extra.gguf",
    "token2wav-gguf/hifigan2.gguf",
    "token2wav-gguf/prompt_cache.gguf",
)
_WAV_NAME = re.compile(r"wav_(\d+)\.wav")
_CHUNK_NAME = re.compile(r"chunk_(\d+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_format(data: bytes, *, sample_rate: int, max_frames: int) -> int:
    with wave.open(io.BytesIO(data), "rb") as source:
        fmt = (source.getnchannels(), source.getsampwidth(), source.getframerate())
        frames = source.getnframes()
        if fmt != (1, 2, sample_rate) or not 0 < frames <= max_frames:
            raise ValueError(f"expected bounded mono PCM16 {sample_rate} Hz WAV")
        if len(source.readframes(frames)) != frames * 2:
            raise ValueError("truncated input WAV")
    return frames


def _jpeg_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG":
            raise ValueError("MiniCPM-o image input must be JPEG")
        size = image.size
        image.verify()
    return size


def _tree_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            pass  # The worker may atomically replace an output while we sample.
    return total


def _read_result(work_dir: Path, max_audio_s: float, max_wav_bytes: int,
                 max_text_bytes: int) -> tuple[str, bytes, int]:
    round_dir = work_dir / "tools" / "omni" / "output" / "round_000"
    wav_dir = round_dir / "tts_wav"
    if not (wav_dir / "generation_done.flag").is_file():
        raise RuntimeError("MiniCPM-o audio generation is incomplete")
    chunks = sorted(
        wav_dir.glob("wav_*.wav"),
        key=lambda path: int(_WAV_NAME.fullmatch(path.name).group(1)),
    )
    if not chunks or len(chunks) > 256 or [item.name for item in chunks] != [
        f"wav_{index}.wav" for index in range(len(chunks))
    ]:
        raise RuntimeError("MiniCPM-o WAV chunk sequence has gaps or is unbounded")
    pcm_parts: list[bytes] = []
    frames = 0
    for path in chunks:
        with wave.open(str(path), "rb") as source:
            fmt = (source.getnchannels(), source.getsampwidth(), source.getframerate())
            count = source.getnframes()
            if fmt != (1, 2, 24000) or not 0 < count <= 24000 * 10:
                raise RuntimeError(f"MiniCPM-o invalid WAV chunk: {path.name}")
            pcm = source.readframes(count)
        if len(pcm) != count * 2:
            raise RuntimeError(f"MiniCPM-o truncated WAV chunk: {path.name}")
        frames += count
        if frames > max_audio_s * 24000 or frames * 2 > max_wav_bytes:
            raise ResourceUnavailable("MiniCPM-o speech exceeds admitted output bound")
        pcm_parts.append(pcm)
    pcm = b"".join(pcm_parts)
    if not np.any(np.frombuffer(pcm, dtype="<i2")):
        raise RuntimeError("MiniCPM-o generated silent speech")
    text_paths = sorted(
        (round_dir / "llm_debug").glob("chunk_*/llm_text.txt"),
        key=lambda path: int(_CHUNK_NAME.fullmatch(path.parent.name).group(1)),
    )
    text = "".join(path.read_text(encoding="utf-8") for path in text_paths).strip()
    if not text or len(text.encode("utf-8")) > max_text_bytes:
        raise ResourceUnavailable("MiniCPM-o text is empty or exceeds admitted bound")
    return text, pcm, frames


class MiniCPMOCppStageClient(StageClientBase):
    def __init__(self, metadata, config: dict, ledger, reservation) -> None:
        for name, value in vars(metadata).items():
            setattr(self, name, value)
        self.stage_type = "graph"
        self._ledger, self._reservation = ledger, reservation
        self._generation = uuid.uuid4().hex
        self._closed = False
        self._active: str | None = None
        self._task: asyncio.Task | None = None
        self._output: OmniRequestOutput | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._epoch = 0
        self._config = dict(config)
        self._placement = str(config["placement"])
        self._memory_pool = str(config.get("memory_pool", "host_ram"))
        self._context_tokens = int(config.get("context_tokens", 2048))
        self._max_new_tokens = int(config.get("max_new_tokens", 128))
        self._max_input_bytes = int(config.get("max_input_bytes", 2 << 20))
        self._max_wav_bytes = int(config.get("max_wav_bytes", 2 << 20))
        self._max_text_bytes = int(config.get("max_text_bytes", 16 << 10))
        self._max_work_bytes = int(config.get("max_work_bytes", 64 << 20))
        self._max_input_audio_s = float(config.get("max_input_audio_s", 10))
        self._max_output_audio_s = float(config.get("max_output_audio_s", 20))
        self._request_timeout_s = float(config.get("request_timeout_s", 300))
        self._t2w_wait_seconds = int(config.get("t2w_wait_seconds", 180))
        self._retain_workdir = bool(config.get("retain_workdir", False))
        try:
            if self._placement not in ("cpu", "radeon-hybrid"):
                raise ValueError("MiniCPM-o placement must be cpu or radeon-hybrid")
            if self._placement == "radeon-hybrid" and (
                config.get("ggml_vk_visible_devices") != "1"
                or config.get("expected_gpu_name") != "AMD Radeon(TM) 890M Graphics"
            ):
                raise ValueError("Radeon stage requires the pinned Vulkan device name and selector")
            if not (
                0 < self._max_new_tokens < self._context_tokens
                and 0 < self._max_input_audio_s <= 30
                and 0 < self._max_output_audio_s <= 120
                and 0 < self._max_input_bytes <= 16 << 20
                and 0 < self._max_wav_bytes <= self._max_work_bytes
                and 0 < self._max_text_bytes <= self._max_work_bytes
                and 0 < self._t2w_wait_seconds < self._request_timeout_s <= 3600
            ):
                raise ValueError("invalid MiniCPM-o context, media, time or output bound")
            if self._memory_pool not in reservation.demands:
                raise ResourceUnavailable("MiniCPM-o shared-RAM pool absent from reservation")
            demand = int(reservation.demands[self._memory_pool])
            if psutil.virtual_memory().available < demand:
                raise ResourceUnavailable("available host/shared RAM below MiniCPM-o reservation")
            self._model_dir = Path(config["model_dir"]).resolve(strict=True)
            self._binary = Path(config["cli_bin"]).resolve(strict=True)
            self._reference_wav = Path(config["reference_wav"]).resolve(strict=True)
            expected = dict(config["artifact_sha256"])
            if set(expected) != set(_ARTIFACTS):
                raise ValueError("MiniCPM-o requires the exact ten-artifact manifest")
            self._artifacts = {name: self._model_dir / name for name in _ARTIFACTS}
            for name, path in self._artifacts.items():
                if not path.is_file() or _sha256(path) != str(expected[name]).lower():
                    raise ValueError(f"MiniCPM-o GGUF differs from declared hash: {name}")
            for path, digest in (
                (self._binary, str(config["cli_sha256"]).lower()),
                (self._reference_wav, str(config["reference_sha256"]).lower()),
            ):
                if _sha256(path) != digest:
                    raise ValueError(f"MiniCPM-o executable/reference differs: {path.name}")
            _wav_format(self._reference_wav.read_bytes(), sample_rate=16000,
                        max_frames=16000 * 10)
            overhead = int(config["memory_overhead_bytes"])
            total_weights = sum(path.stat().st_size for path in self._artifacts.values())
            if overhead <= 0 or total_weights + overhead + self._max_work_bytes > demand:
                raise ResourceUnavailable("MiniCPM-o weights, state/workspace, I/O and headroom exceed reservation")
            help_result = subprocess.run(
                [str(self._binary), "--help"], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=10, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if not all(flag in help_result.stdout for flag in (
                b"--max-new-tokens", b"--t2w-device", b"--t2w-wait-seconds"
            )):
                raise ValueError("MiniCPM-o CLI lacks bounded-generation or completion support")
            self._work_root = Path(config["work_root"]).resolve()
            self._log_root = Path(config["log_root"]).resolve()
            self._work_root.mkdir(parents=True, exist_ok=True)
            self._log_root.mkdir(parents=True, exist_ok=True)
            self.execution_plan = {
                "backend": "external.minicpmo.gguf.v1",
                "stage_id": self.stage_id,
                "worker_generation": self._generation,
                "model_revision": str(config["model_revision"]),
                "artifact_sha256": {key: str(value).lower() for key, value in expected.items()},
                "cli_sha256": str(config["cli_sha256"]).lower(),
                "requested_placement": self._placement,
                "ggml_vk_visible_devices": config.get("ggml_vk_visible_devices"),
                "shared_ram_reservation": dict(reservation.demands),
                "weights_bytes": total_weights,
                "memory_overhead_bytes": overhead,
                "context_tokens": self._context_tokens,
                "max_new_tokens": self._max_new_tokens,
                "request_capacity": 1,
                "stateful_session": "C++-owned within one request; discarded after completion",
                "placement_evidence": "pending first request",
            }
        except BaseException:
            self.shutdown()
            raise

    async def add_request_async(self, request_id: str, prompt: Any, params: Any = None) -> None:
        self.check_health()
        if self._active is not None:
            raise ResourceUnavailable("MiniCPM-o stage has an unacknowledged request; capacity is one")
        if not isinstance(prompt, dict):
            raise ValueError("MiniCPM-o request must contain audio_wav and image_jpeg bytes")
        audio, image = prompt.get("audio_wav"), prompt.get("image_jpeg")
        if not isinstance(audio, (bytes, bytearray)) or not isinstance(image, (bytes, bytearray)):
            raise ValueError("MiniCPM-o media must use bounded byte buffers")
        audio, image = bytes(audio), bytes(image)
        if not 0 < len(audio) <= self._max_input_bytes or not 0 < len(image) <= self._max_input_bytes:
            raise ResourceUnavailable("MiniCPM-o input exceeds admitted transfer bound")
        _wav_format(audio, sample_rate=16000,
                    max_frames=int(self._max_input_audio_s * 16000))
        if _jpeg_size(image) != (448, 448):
            raise ValueError("MiniCPM-o v1 stage accepts only a 448x448 JPEG")
        self._epoch += 1
        request = StageRequest(request_id, self.stage_id, self._epoch, self._generation)
        self._active = request_id
        self._task = asyncio.create_task(
            self._run(request, audio, image), name=f"minicpmo-{request_id}"
        )

    async def _run(self, request: StageRequest, audio: bytes, image: bytes) -> None:
        try:
            result = await asyncio.to_thread(self._execute, request, audio, image)
            samples = np.frombuffer(result["pcm"], dtype="<i2").astype(np.float32) / 32768.0
            completion = CompletionOutput(0, result["text"], [], None, None, finish_reason="stop")
            completion.multimodal_output = {"audio": torch.from_numpy(samples), "sr": 24000}
            event = StageEvent(
                request.request_id, self.stage_id, request.epoch, 1, "audio",
                self._generation, terminal=True,
            )
            output = OmniRequestOutput(
                request_id=request.request_id,
                prompt="[local audio+image]",
                stage_id=self.stage_id,
                final_output_type="audio",
                outputs=[completion],
                _custom_output={
                    "stage_event": dataclasses.asdict(event),
                    "audio_metadata": {
                        "sample_rate_hz": 24000,
                        "channels": 1,
                        "pcm_frames": result["frames"],
                        "duration_s": result["frames"] / 24000,
                        "pcm_sha256": hashlib.sha256(result["pcm"]).hexdigest(),
                    },
                    "input_sha256": {
                        "audio": hashlib.sha256(audio).hexdigest(),
                        "image": hashlib.sha256(image).hexdigest(),
                    },
                },
                metrics={
                    "minicpmo_whole_request_wall_s": result["wall_s"],
                    "peak_sampled_rss_bytes": result["peak_rss"],
                    "worker_log": str(result["log_path"]),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            output = OmniRequestOutput.from_error(request.request_id, str(exc))
            output.stage_id = self.stage_id
        if not self._closed and self._epoch == request.epoch:
            loop = asyncio.get_running_loop()
            output._stage_release = lambda: loop.call_soon_threadsafe(
                self.acknowledge, request.request_id, request.epoch, request.worker_generation
            )
            self._output = output

    def _execute(self, request: StageRequest, audio: bytes, image: bytes) -> dict:
        work_dir = Path(tempfile.mkdtemp(prefix=f"omni-minicpmo-{request.epoch}-", dir=self._work_root))
        if not work_dir.resolve().is_relative_to(self._work_root):
            raise RuntimeError("MiniCPM-o work directory escaped its declared root")
        log_path = self._log_root / f"minicpmo-{request.epoch}-{uuid.uuid4().hex}.log"
        started = time.perf_counter()
        peak_rss = 0
        try:
            shutil.copyfile(self._reference_wav, work_dir / "fixture_0000.wav")
            (work_dir / "fixture_0001.wav").write_bytes(audio)
            (work_dir / "fixture_0001.jpg").write_bytes(image)
            command = [
                str(self._binary), "-m", str(self._artifacts[_ARTIFACTS[0]]),
                "-ngl", "0" if self._placement == "cpu" else "99",
                "-c", str(self._context_tokens),
                "--max-new-tokens", str(self._max_new_tokens),
                "--omni", "--t2w-device", "cpu",
                "--t2w-wait-seconds", str(self._t2w_wait_seconds),
                "--ref-audio", str(self._reference_wav),
                "--test", str(work_dir / "fixture_"), "2",
            ]
            env = os.environ.copy()
            env["GGML_VK_VISIBLE_DEVICES"] = (
                "" if self._placement == "cpu" else "1"
            )
            with log_path.open("wb") as log:
                with self._lock:
                    if self._closed or self._epoch != request.epoch:
                        raise RuntimeError("MiniCPM-o request was cancelled before launch")
                    self._proc = subprocess.Popen(
                        command, cwd=work_dir, stdout=log, stderr=subprocess.STDOUT,
                        env=env,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                    proc = self._proc
                process = psutil.Process(proc.pid)
                deadline = time.monotonic() + self._request_timeout_s
                while proc.poll() is None:
                    if time.monotonic() >= deadline:
                        self._kill_process(proc)
                        raise TimeoutError("MiniCPM-o complete request timed out")
                    if log_path.stat().st_size + _tree_bytes(work_dir) > self._max_work_bytes:
                        self._kill_process(proc)
                        raise ResourceUnavailable("MiniCPM-o worker exceeded admitted file/I/O bound")
                    try:
                        peak_rss = max(peak_rss, process.memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                    if peak_rss > self._reservation.demands[self._memory_pool]:
                        self._kill_process(proc)
                        raise ResourceUnavailable("MiniCPM-o worker RSS exceeded shared-RAM reservation")
                    try:
                        proc.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        pass
                if proc.returncode != 0:
                    raise RuntimeError(f"MiniCPM-o CLI exited {proc.returncode}; see {log_path}")
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            if self._placement == "cpu":
                required = (
                    "GPU layers: 0", "vision_ctx: vision using CPU backend",
                    "Token2Wav device: cpu",
                )
            else:
                required = (
                    "GPU layers: 99",
                    "using device Vulkan0 (AMD Radeon(TM) 890M Graphics)",
                    "offloaded 37/37 layers to GPU",
                    "vision_ctx: vision using Vulkan0 backend",
                    "TTS model: loading with n_gpu_layers=0",
                    "Token2Wav: non-CUDA backend, vocoder using CPU",
                )
            if any(item not in log_text for item in required):
                raise RuntimeError("REFUSE_DEVICE_PLACEMENT: MiniCPM-o did not use declared CPU/Radeon route")
            if not all(item in log_text for item in (
                "stream_prefill(index=1): processing user audio:",
                "stream_prefill: prefilled 1 vision chunks",
                "Audio generation completed.",
            )):
                raise RuntimeError("MiniCPM-o complete input/output path not observed")
            text, pcm, frames = _read_result(
                work_dir, self._max_output_audio_s,
                self._max_wav_bytes, self._max_text_bytes,
            )
            self.execution_plan["placement_evidence"] = {
                "observed": self._placement,
                "worker_pid": proc.pid,
                "peak_sampled_rss_bytes": peak_rss,
                "worker_log": str(log_path),
            }
            return {
                "text": text, "pcm": pcm, "frames": frames,
                "peak_rss": peak_rss, "wall_s": time.perf_counter() - started,
                "log_path": log_path,
            }
        finally:
            with self._lock:
                proc = self._proc
            if proc is not None:
                self._kill_process(proc)
                with self._lock:
                    if self._proc is proc:
                        self._proc = None
            if not self._retain_workdir:
                if not work_dir.resolve().is_relative_to(self._work_root):
                    raise RuntimeError("refusing cleanup outside MiniCPM-o work root")
                shutil.rmtree(work_dir)

    @staticmethod
    def _kill_process(proc: subprocess.Popen) -> bool:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except OSError:
            return False
        return proc.poll() is not None

    def get_graph_output_nowait(self) -> OmniRequestOutput | None:
        output, self._output = self._output, None
        return output

    def acknowledge(self, request_id: str, epoch: int, generation: str) -> None:
        if (
            self._active == request_id and epoch == self._epoch
            and generation == self._generation
            and (self._task is None or self._task.done())
        ):
            self._active = None
            self._task = None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if self._active not in request_ids:
            return
        self._epoch += 1
        self._output = None
        with self._lock:
            self._closed = True
            proc = self._proc
        drained = await asyncio.to_thread(self._kill_process, proc) if proc else True
        if self._task is not None and not self._task.done():
            done, _ = await asyncio.wait({self._task}, timeout=5)
            if done:
                self._ledger.release(self._reservation, drained=drained)
            else:
                self._ledger.release(self._reservation, drained=False)
                self._task.add_done_callback(
                    lambda task: self._ledger.release(
                        self._reservation, drained=drained and not task.cancelled()
                    )
                )
        self._active = None

    def check_health(self) -> None:
        if self._closed:
            from vllm.v1.engine.exceptions import EngineDeadError

            raise EngineDeadError()

    async def collective_rpc_async(self, method, timeout=None, args=(), kwargs=None):
        raise NotImplementedError(f"MiniCPM-o backend does not implement collective RPC {method}")

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            proc = self._proc
        drained = self._kill_process(proc) if proc else True
        self._output = None
        task = self._task
        if task is not None and not task.done():
            self._ledger.release(self._reservation, drained=False)
            task.add_done_callback(
                lambda done: self._ledger.release(
                    self._reservation, drained=drained and not done.cancelled()
                )
            )
        else:
            self._ledger.release(self._reservation, drained=drained)
