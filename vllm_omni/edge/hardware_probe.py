"""Hardware capability probe for edge deployments.

Produces a :class:`HardwareProfile` from ``/proc``, ``/sys`` and (optionally)
torch, and maps it to a coarse *hardware class* used to select deploy overlays
(``vllm_omni/deploy/edge/hardware/<class>.yaml``) and to derive engine
overrides (see :mod:`vllm_omni.edge.adapt`).

The probe reads from an injectable filesystem root so tests can feed captured
snapshots of other devices, and ``VLLM_OMNI_HW_PROFILE=<json>`` replaces the
probe entirely (used for emulating a device class on a development host).
"""

from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ENV_OVERRIDE = "VLLM_OMNI_HW_PROFILE"

HW_CLASS_JETSON = "jetson"
HW_CLASS_CUDA_DISCRETE = "cuda_discrete"
HW_CLASS_ARM64_CPU = "arm64_cpu"
HW_CLASS_X86_CPU = "x86_cpu"
HW_CLASS_NPU_PHONE = "npu_phone"
HW_CLASSES = (HW_CLASS_JETSON, HW_CLASS_CUDA_DISCRETE, HW_CLASS_ARM64_CPU, HW_CLASS_X86_CPU, HW_CLASS_NPU_PHONE)

# Flags we care about, normalized across x86 (``flags``) and ARM (``Features``).
_X86_FLAGS = ("avx2", "avx512f", "avx512_bf16", "avx512_vnni", "amx_bf16", "amx_tile", "amx_int8", "fma", "f16c")
_ARM_FLAGS = ("asimd", "asimdhp", "asimddp", "i8mm", "bf16", "sve", "sve2", "sme")
_FLAG_ALIASES = {"avx512f": "avx512", "asimd": "neon", "asimdhp": "fp16_arith", "asimddp": "dotprod"}


@dataclass
class HardwareProfile:
    arch: str = "unknown"  # x86_64 | aarch64 | ...
    cpu_model: str = ""
    cpu_flags: list[str] = field(default_factory=list)
    n_logical: int = 0
    n_physical: int = 0
    clusters: list[dict[str, Any]] = field(default_factory=list)  # [{"max_khz": int, "cpus": [int, ...]}], big first
    numa_nodes: int = 1
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    accelerator: str = "none"  # none | cuda_discrete | cuda_unified | npu:<vendor>
    gpu_name: str = ""
    gpu_mem_bytes: int = 0
    gpu_sm_count: int = 0
    gpu_capability: list[int] = field(default_factory=list)
    supports_bf16: bool = False
    supports_fp16: bool = True
    thermal_zones: int = 0
    source: str = "probe"

    # ---------------------------------------------------------------- helpers
    def has(self, *flags: str) -> bool:
        return all(f in self.cpu_flags for f in flags)

    @property
    def big_cpus(self) -> list[int]:
        return list(self.clusters[0]["cpus"]) if self.clusters else list(range(self.n_logical))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HardwareProfile:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> HardwareProfile:
        return cls.from_dict(json.loads(text))


# ------------------------------------------------------------------ parsing
def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def parse_cpuinfo(text: str) -> tuple[str, list[str], int]:
    """Return (model name, normalized flags, logical cpu count) from /proc/cpuinfo."""
    model = ""
    flags: set[str] = set()
    n_logical = 0
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "processor":
            n_logical += 1
        elif key in ("model name", "hardware", "cpu model") and not model:
            model = value
        elif key in ("flags", "features"):
            for tok in value.split():
                if tok in _X86_FLAGS or tok in _ARM_FLAGS:
                    flags.add(_FLAG_ALIASES.get(tok, tok))
    return model, sorted(flags), n_logical


def parse_meminfo(text: str) -> tuple[int, int]:
    total = avail = 0
    for line in text.splitlines():
        m = re.match(r"^(MemTotal|MemAvailable):\s+(\d+)\s*kB", line)
        if m:
            val = int(m.group(2)) * 1024
            if m.group(1) == "MemTotal":
                total = val
            else:
                avail = val
    return total, avail


def parse_clusters(cpu_max_khz: dict[int, int]) -> list[dict[str, Any]]:
    """Group CPUs by max frequency (descending) -> big.LITTLE clusters."""
    groups: dict[int, list[int]] = {}
    for cpu, khz in cpu_max_khz.items():
        groups.setdefault(khz, []).append(cpu)
    return [{"max_khz": khz, "cpus": sorted(cpus)} for khz, cpus in sorted(groups.items(), key=lambda kv: -kv[0])]


def _sysfs_cpu_dirs(root: Path) -> list[Path]:
    base = root / "sys" / "devices" / "system" / "cpu"
    return sorted((p for p in base.glob("cpu[0-9]*") if p.is_dir()), key=lambda p: int(p.name[3:]))


def _physical_cores(root: Path, n_logical: int) -> int:
    ids: set[tuple[str, str]] = set()
    for cpu in _sysfs_cpu_dirs(root):
        core = _read(cpu / "topology" / "core_id")
        pkg = _read(cpu / "topology" / "physical_package_id")
        if core is not None:
            ids.add(((pkg or "0").strip(), core.strip()))
    return len(ids) if ids else n_logical


def _cpu_max_khz(root: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    for cpu in _sysfs_cpu_dirs(root):
        raw = _read(cpu / "cpufreq" / "cpuinfo_max_freq")
        if raw and raw.strip().isdigit():
            out[int(cpu.name[3:])] = int(raw.strip())
    return out


def _numa_nodes(root: Path) -> int:
    base = root / "sys" / "devices" / "system" / "node"
    nodes = [p for p in base.glob("node[0-9]*")] if base.exists() else []
    return max(1, len(nodes))


def _thermal_zones(root: Path) -> int:
    base = root / "sys" / "class" / "thermal"
    return len(list(base.glob("thermal_zone*"))) if base.exists() else 0


def _is_tegra(root: Path) -> bool:
    return (
        (root / "etc" / "nv_tegra_release").exists()
        or (root / "proc" / "device-tree" / "compatible").exists()
        and "tegra" in (_read(root / "proc" / "device-tree" / "compatible") or "")
    )


def _torch_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import torch

        if not torch.cuda.is_available():
            return info
        props = torch.cuda.get_device_properties(0)
        info.update(
            available=True,
            name=props.name,
            mem_bytes=int(props.total_memory),
            sm_count=int(getattr(props, "multi_processor_count", 0)),
            capability=[int(props.major), int(props.minor)],
            bf16=bool(torch.cuda.is_bf16_supported()),
        )
    except Exception:
        return info
    return info


# -------------------------------------------------------------------- probe
def probe(root: str | Path = "/", use_torch: bool = True) -> HardwareProfile:
    """Probe the current machine (or a snapshot rooted at ``root``)."""
    root = Path(root)
    model, flags, n_logical = parse_cpuinfo(_read(root / "proc" / "cpuinfo") or "")
    ram_total, ram_avail = parse_meminfo(_read(root / "proc" / "meminfo") or "")
    arch = platform.machine() if root == Path("/") else _arch_from_flags(flags, model)
    profile = HardwareProfile(
        arch=arch,
        cpu_model=model,
        cpu_flags=flags,
        n_logical=n_logical or os.cpu_count() or 1,
        n_physical=_physical_cores(root, n_logical or os.cpu_count() or 1),
        clusters=parse_clusters(_cpu_max_khz(root)),
        numa_nodes=_numa_nodes(root),
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_avail,
        thermal_zones=_thermal_zones(root),
    )
    gpu = _torch_gpu_info() if (use_torch and root == Path("/")) else {"available": False}
    tegra = _is_tegra(root)
    if gpu.get("available"):
        profile.accelerator = "cuda_unified" if tegra else "cuda_discrete"
        profile.gpu_name = gpu["name"]
        profile.gpu_mem_bytes = gpu["mem_bytes"]
        profile.gpu_sm_count = gpu["sm_count"]
        profile.gpu_capability = gpu["capability"]
        profile.supports_bf16 = gpu["bf16"]
    elif tegra:
        profile.accelerator = "cuda_unified"
        profile.gpu_mem_bytes = ram_total  # unified memory
        profile.supports_bf16 = True
    else:
        profile.supports_bf16 = _cpu_bf16(profile)
    profile.supports_fp16 = (
        True if profile.accelerator.startswith("cuda") else profile.has("fp16_arith") or arch == "x86_64"
    )
    return profile


def _arch_from_flags(flags: list[str], model: str) -> str:
    if any(f in flags for f in ("neon", "dotprod", "sve", "i8mm", "fp16_arith")):
        return "aarch64"
    if any(f in flags for f in ("avx2", "avx512", "amx_tile", "fma")):
        return "x86_64"
    return "unknown"


def _cpu_bf16(profile: HardwareProfile) -> bool:
    if profile.arch == "x86_64":
        return profile.has("avx512_bf16") or profile.has("amx_bf16")
    if profile.arch == "aarch64":
        return profile.has("bf16")
    return False


def hardware_class(profile: HardwareProfile) -> str:
    acc = profile.accelerator
    if acc.startswith("npu"):
        return HW_CLASS_NPU_PHONE
    if acc == "cuda_unified":
        return HW_CLASS_JETSON
    if acc == "cuda_discrete":
        return HW_CLASS_CUDA_DISCRETE
    if profile.arch == "aarch64":
        return HW_CLASS_ARM64_CPU
    return HW_CLASS_X86_CPU


def load_profile(path: str | Path | None = None, use_torch: bool = True) -> HardwareProfile:
    """Env override (``VLLM_OMNI_HW_PROFILE``) or explicit path wins over probing."""
    override = path or os.environ.get(ENV_OVERRIDE)
    if override:
        p = HardwareProfile.from_json(Path(override).read_text(encoding="utf-8"))
        p.source = f"override:{override}"
        return p
    return probe(use_torch=use_torch)


def describe(profile: HardwareProfile) -> str:
    cls = hardware_class(profile)
    big = profile.big_cpus
    return (
        f"class={cls} arch={profile.arch} cpu='{profile.cpu_model}' "
        f"logical={profile.n_logical} physical={profile.n_physical} "
        f"big_cpus={len(big)} flags={','.join(profile.cpu_flags)} ram={profile.ram_total_bytes / 2**30:.1f}GiB "
        f"accel={profile.accelerator} gpu='{profile.gpu_name}' gpu_mem={profile.gpu_mem_bytes / 2**30:.1f}GiB "
        f"bf16={profile.supports_bf16} source={profile.source}"
    )


if __name__ == "__main__":  # pragma: no cover - manual use
    prof = load_profile()
    print(describe(prof))
    print(prof.to_json())
