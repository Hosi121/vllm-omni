"""Source-checkout bridge; wheels use the canonical package directly."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "packages/omni-stage-contracts/omni_stage_contracts")]

from .types import (
    PROTOCOL_VERSION,
    ArtifactManifest,
    BufferRef,
    DeviceDescriptor,
    StageEvent,
    StageRequest,
    StateHandle,
    negotiate,
)

__all__ = [
    "PROTOCOL_VERSION",
    "ArtifactManifest",
    "BufferRef",
    "DeviceDescriptor",
    "StageEvent",
    "StageRequest",
    "StateHandle",
    "negotiate",
]
