"""Tests for vllm_omni.edge.hardware_probe using synthetic /proc and /sys snapshots (CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_omni.edge import hardware_probe as hp

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

X86_CPUINFO = """processor\t: 0
model name\t: Intel(R) Xeon(R) Platinum 8480C
flags\t\t: fpu sse avx2 avx512f avx512_bf16 amx_bf16 amx_tile fma f16c
processor\t: 1
model name\t: Intel(R) Xeon(R) Platinum 8480C
flags\t\t: fpu sse avx2 avx512f avx512_bf16 amx_bf16 amx_tile fma f16c
"""

ARM_CPUINFO = """processor\t: 0
Features\t: fp asimd evtstrm aes asimdhp asimddp
processor\t: 1
Features\t: fp asimd evtstrm aes asimdhp asimddp
processor\t: 2
Features\t: fp asimd evtstrm aes asimdhp asimddp i8mm bf16
processor\t: 3
Features\t: fp asimd evtstrm aes asimdhp asimddp i8mm bf16
Hardware\t: Rockchip RK3588
"""

MEMINFO = "MemTotal:       16000000 kB\nMemFree:         1000000 kB\nMemAvailable:   12000000 kB\n"


def _write_sysfs(root: Path, cpus: dict[int, tuple[int, int]], numa: int = 1, thermal: int = 2, tegra: bool = False):
    for cpu, (max_khz, core_id) in cpus.items():
        d = root / "sys" / "devices" / "system" / "cpu" / f"cpu{cpu}"
        (d / "cpufreq").mkdir(parents=True, exist_ok=True)
        (d / "topology").mkdir(parents=True, exist_ok=True)
        (d / "cpufreq" / "cpuinfo_max_freq").write_text(str(max_khz))
        (d / "topology" / "core_id").write_text(str(core_id))
        (d / "topology" / "physical_package_id").write_text("0")
    for n in range(numa):
        (root / "sys" / "devices" / "system" / "node" / f"node{n}").mkdir(parents=True, exist_ok=True)
    for t in range(thermal):
        (root / "sys" / "class" / "thermal" / f"thermal_zone{t}").mkdir(parents=True, exist_ok=True)
    if tegra:
        (root / "etc").mkdir(parents=True, exist_ok=True)
        (root / "etc" / "nv_tegra_release").write_text("# R36 (release), REVISION: 4.0")


def _snapshot(tmp_path: Path, cpuinfo: str, meminfo: str = MEMINFO, **kw) -> Path:
    root = tmp_path / "root"
    (root / "proc").mkdir(parents=True)
    (root / "proc" / "cpuinfo").write_text(cpuinfo)
    (root / "proc" / "meminfo").write_text(meminfo)
    _write_sysfs(root, **kw)
    return root


def test_x86_amx_snapshot(tmp_path):
    root = _snapshot(tmp_path, X86_CPUINFO, cpus={0: (3800000, 0), 1: (3800000, 0)}, numa=2)
    p = hp.probe(root, use_torch=False)
    assert p.arch == "x86_64"
    assert p.has("avx2", "avx512", "avx512_bf16", "amx_bf16", "amx_tile")
    assert p.n_logical == 2 and p.n_physical == 1  # two hyper-threads of one core
    assert p.numa_nodes == 2 and p.thermal_zones == 2
    assert p.ram_total_bytes == 16000000 * 1024 and p.ram_available_bytes == 12000000 * 1024
    assert p.accelerator == "none" and p.supports_bf16 is True
    assert hp.hardware_class(p) == hp.HW_CLASS_X86_CPU
    assert p.clusters == [{"max_khz": 3800000, "cpus": [0, 1]}]


def test_arm_big_little_snapshot(tmp_path):
    root = _snapshot(tmp_path, ARM_CPUINFO, cpus={0: (1800000, 0), 1: (1800000, 1), 2: (2400000, 2), 3: (2400000, 3)})
    p = hp.probe(root, use_torch=False)
    assert p.arch == "aarch64"
    assert p.has("neon", "fp16_arith", "dotprod", "i8mm", "bf16")
    assert p.cpu_model == "Rockchip RK3588"
    assert p.clusters[0] == {"max_khz": 2400000, "cpus": [2, 3]}  # big cluster first
    assert p.big_cpus == [2, 3]
    assert p.n_physical == 4
    assert hp.hardware_class(p) == hp.HW_CLASS_ARM64_CPU
    assert p.supports_bf16 is True and p.supports_fp16 is True


def test_tegra_snapshot_is_jetson_with_unified_memory(tmp_path):
    root = _snapshot(tmp_path, ARM_CPUINFO, cpus={0: (2200000, 0)}, tegra=True)
    p = hp.probe(root, use_torch=False)
    assert p.accelerator == "cuda_unified"
    assert p.gpu_mem_bytes == p.ram_total_bytes
    assert hp.hardware_class(p) == hp.HW_CLASS_JETSON


def test_override_env_and_roundtrip(tmp_path, monkeypatch):
    prof = hp.HardwareProfile(
        arch="aarch64", cpu_flags=["neon"], n_logical=8, accelerator="npu:qualcomm", ram_total_bytes=8 * 2**30
    )
    path = tmp_path / "hw.json"
    path.write_text(prof.to_json())
    monkeypatch.setenv(hp.ENV_OVERRIDE, str(path))
    loaded = hp.load_profile(use_torch=False)
    assert loaded.source.startswith("override:")
    assert loaded.accelerator == "npu:qualcomm"
    assert hp.hardware_class(loaded) == hp.HW_CLASS_NPU_PHONE
    # unknown keys are ignored on load
    d = json.loads(prof.to_json())
    d["future_field"] = 1
    assert hp.HardwareProfile.from_dict(d).n_logical == 8
    assert "class=npu_phone" in hp.describe(loaded)


def test_live_probe_runs_on_this_host():
    p = hp.probe(use_torch=False)
    assert p.n_logical >= 1 and p.ram_total_bytes > 0
    assert hp.hardware_class(p) in hp.HW_CLASSES
