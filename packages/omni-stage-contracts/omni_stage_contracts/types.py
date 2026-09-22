"""Versioned host-copy contracts shared by Python and native stage adapters."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
FEATURES = frozenset({"host-copy", "request-epochs"})
_ITEM_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float16": 2,
    "float32": 4,
    "float64": 8,
}


def negotiate(version: int, required: list[str] | tuple[str, ...] = ()) -> None:
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported stage protocol version: {version!r}")
    missing = set(required) - FEATURES
    if missing:
        raise ValueError(f"unsupported required stage features: {sorted(missing)}")


@dataclass(frozen=True)
class BufferRef:
    object_id: str
    owner: str
    generation: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    memory_domain: str = "host-copy"
    layout: str = "C"
    ready: bool = True

    def __post_init__(self) -> None:
        if not self.object_id or not self.owner or not self.generation:
            raise ValueError("buffer identity/owner/generation are required")
        if self.memory_domain != "host-copy" or self.layout != "C" or not self.ready:
            raise ValueError("v1 only supports ready, contiguous copied host buffers")
        if self.dtype not in _ITEM_SIZES:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if len(self.shape) > 32 or any(type(x) is not int or x < 0 for x in self.shape):
            raise ValueError("invalid tensor dimensions")
        expected = math.prod(self.shape) * _ITEM_SIZES[self.dtype]
        if type(self.nbytes) is not int or self.nbytes != expected or self.nbytes > 2 << 30:
            raise ValueError("tensor shape/dtype/byte length mismatch or payload exceeds limit")


@dataclass(frozen=True)
class StateHandle:
    session_id: str
    backend: str
    artifact_id: str
    layout_version: int = 1
    epoch: int = 0
    replayable: bool = False
    migratable: bool = False
    backend_instance_id: str = ""
    worker_generation: str = ""

    def accepts(self, other: StateHandle) -> bool:
        return all(
            getattr(self, k) == getattr(other, k)
            for k in (
                "session_id",
                "backend",
                "artifact_id",
                "layout_version",
                "epoch",
                "backend_instance_id",
                "worker_generation",
            )
        )

    def next_epoch(self) -> StateHandle:
        from dataclasses import replace

        return replace(self, epoch=self.epoch + 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageRequest:
    request_id: str
    stage_id: int
    epoch: int = 0
    worker_generation: str = ""
    operation: str = "run"
    inputs: tuple[BufferRef, ...] = ()

    def __post_init__(self):
        if not self.request_id or not self.worker_generation or not self.operation:
            raise ValueError("request identity, worker generation and operation are required")
        if any(type(n) is not int or n < 0 for n in (self.stage_id, self.epoch)):
            raise ValueError("stage_id and epoch must be nonnegative integers")


@dataclass(frozen=True)
class StageEvent:
    request_id: str
    stage_id: int
    epoch: int
    seq: int
    kind: str
    worker_generation: str
    buffers: tuple[BufferRef, ...] = ()
    terminal: bool = False
    error: str | None = None

    def __post_init__(self):
        if not self.request_id or not self.worker_generation or not self.kind:
            raise ValueError("event identity, worker generation and kind are required")
        if any(type(n) is not int or n < 0 for n in (self.stage_id, self.epoch, self.seq)):
            raise ValueError("stage_id, epoch and seq must be nonnegative integers")


@dataclass(frozen=True)
class DeviceDescriptor:
    """Physical identity is independent from per-domain execution routes.

    Unknown package/power/bandwidth relationships are empty, never inferred
    from an integrated/discrete label. Pool IDs identify backing constraints.
    """

    device_id: str
    kind: str
    domain_id: str
    memory_pool_ids: tuple[str, ...]
    integration: str = "unknown"
    package_id: str | None = None
    bandwidth_group_ids: tuple[str, ...] = ()
    power_domain_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"cpu", "gpu", "npu"}:
            raise ValueError(f"unknown device kind: {self.kind}")
        if self.integration not in {"integrated", "discrete", "unknown"}:
            raise ValueError("unknown integration class")
        if not self.device_id or not self.domain_id or not self.memory_pool_ids:
            raise ValueError("device, domain and physical memory constraints are required")
        if len(set(self.memory_pool_ids)) != len(self.memory_pool_ids):
            raise ValueError("duplicate physical memory constraint")


@dataclass(frozen=True)
class ArtifactManifest:
    """A complete artifact payload set; paths are relative to the manifest."""

    schema_version: int
    component: str
    files: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def read(cls, path: str | Path) -> ArtifactManifest:
        path = Path(path)
        manifest = cls(**json.loads(path.read_text(encoding="utf-8")))
        if manifest.schema_version != 1 or not manifest.files:
            raise ValueError("unsupported or empty artifact manifest")
        root = path.parent.resolve()
        for name, digest in manifest.files.items():
            if not isinstance(name, str) or "\\" in name or ":" in name:
                raise ValueError("artifact payloads must use portable relative POSIX paths")
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"artifact path leaves bundle: {name}")
            if not target.is_file() or file_digest(target) != digest:
                raise ValueError(f"artifact payload missing or digest mismatch: {name}")
        return manifest


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
