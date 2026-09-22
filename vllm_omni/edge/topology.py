# SPDX-License-Identifier: Apache-2.0
"""Map discovery facts into physical constraints, without guessing performance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

from omni_stage_contracts import DeviceDescriptor


def describe_topology(
    devices,
    *,
    machine_id: str,
    domain_id: str,
    physical_ids: Mapping[str, str] | None = None,
    guest_pool: str | None = None,
) -> list[DeviceDescriptor]:
    """Normalize legacy discovery; caller reconciles cross-domain identities.

    Physical IDs must be shared across routes when discovery proves they refer
    to one GPU. Unknown bandwidth/power relationships stay empty. WSL quota is
    an additional constraint on host allocations, never another RAM capacity.
    """
    if not machine_id or not domain_id:
        raise ValueError("machine and domain IDs must be explicit")
    physical_ids = physical_ids or {}
    result = []
    for device in devices:
        extra = device.extra or {}
        physical = physical_ids.get(device.device_id, device.device_id)
        kind = {"gpu_integrated": "gpu", "gpu_discrete": "gpu"}.get(device.kind, device.kind)
        integrated = device.kind == "gpu_integrated"
        integration = "integrated" if integrated else "discrete" if device.kind == "gpu_discrete" else "unknown"
        pool = f"{machine_id}:ram" if device.memory_pool == "host_ram" else f"{machine_id}:vram:{physical}"
        pools = (pool,) + ((guest_pool,) if guest_pool and device.memory_pool == "host_ram" else ())
        result.append(
            DeviceDescriptor(
                device_id=f"{machine_id}:{physical}",
                kind=kind,
                domain_id=domain_id,
                memory_pool_ids=pools,
                integration=extra.get("integration", integration),
                package_id=extra.get("package_id"),
                bandwidth_group_ids=tuple(extra.get("bandwidth_group_ids", ())),
                power_domain_ids=tuple(extra.get("power_domain_ids", ())),
            )
        )
    return result


def topology_report(devices, **kwargs) -> list[dict]:
    return [asdict(device) for device in describe_topology(devices, **kwargs)]
