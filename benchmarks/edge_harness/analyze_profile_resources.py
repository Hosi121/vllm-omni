# SPDX-License-Identifier: Apache-2.0
"""Summarize raw host telemetry and traces without adding overlapping timings."""

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from summarize_e2e_profiles import read_lines, stats


def interval_union(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        total += stop - start if end is None or start >= end else max(0, stop - end)
        end = stop if end is None else max(end, stop)
    return total


def trace_events(path):
    """Read one event at a time, avoiding a multi-GB decoded trace document."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        buffer = ""
        while True:
            block = stream.read(65536)
            if not block:
                raise ValueError("No traceEvents array")
            buffer += block
            match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
            if match:
                buffer = buffer[match.end() :]
                break
            if len(buffer) > 8 * 1024 * 1024:
                raise ValueError("Trace header exceeds 8 MiB")
        decoder = json.JSONDecoder()
        while True:
            buffer = buffer.lstrip(" \t\r\n,")
            if buffer.startswith("]"):
                return
            try:
                event, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                block = stream.read(65536)
                if not block:
                    raise ValueError("Truncated traceEvents array") from None
                buffer += block
                if len(buffer) > 16 * 1024 * 1024:
                    raise ValueError("One trace event exceeds 16 MiB") from None
                continue
            buffer = buffer[end:]
            yield event


def trace_summary(path):
    categories, ops = defaultdict(list), defaultdict(list)
    for event in trace_events(path):
        if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)):
            continue
        categories[event.get("cat", "uncategorized")].append((event["ts"], event["ts"] + event["dur"]))
        ops[(event.get("cat", ""), event.get("name", ""))].append(float(event["dur"]))
    return {
        "source": str(path),
        "scope": "one process trace; event duration sums overlap and are not E2E time",
        "categories": {
            name: {
                "events": len(group),
                "duration_sum_us": sum(b - a for a, b in group),
                "busy_union_us": interval_union(group),
            }
            for name, group in categories.items()
        },
        "largest_ops_by_duration_sum": sorted(
            [
                {"category": cat, "name": name, "duration_sum_us": sum(values), "duration_us": stats(values)}
                for (cat, name), values in ops.items()
            ],
            key=lambda r: r["duration_sum_us"],
            reverse=True,
        )[:40],
    }


def telemetry_summary(path, start=None, stop=None):
    rows = [
        r for r in read_lines(path) if (start is None or r["unix"] >= start) and (stop is None or r["unix"] <= stop)
    ]
    series = defaultdict(list)
    power = []
    for row in rows:
        series["host_available_bytes"].append(row["host_memory"]["available"])
        series["host_used_bytes"].append(row["host_memory"]["used"])
        series["host_cpu_percent"].append(row.get("host_cpu_percent"))
        fields = row.get("gpu_csv", "").split(",")
        if len(fields) != 10:
            continue
        for index, name in [
            (2, "gpu_memory_mib"),
            (4, "gpu_temperature_c"),
            (5, "gpu_power_w"),
            (6, "gpu_sm_clock_mhz"),
            (7, "gpu_mem_clock_mhz"),
            (8, "gpu_utilization_percent"),
        ]:
            try:
                value = float(fields[index])
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            series[name].append(value)
            if index == 5:
                power.append((row["unix"], value))
    gaps = [b["unix"] - a["unix"] for a, b in zip(rows, rows[1:])]
    intervals = [(a, b) for a, b in zip(power, power[1:]) if 0 < b[0] - a[0] <= 5]
    energy = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in intervals)
    return {
        "source": str(path),
        "samples": len(rows),
        "span_s": rows[-1]["unix"] - rows[0]["unix"] if len(rows) >= 2 else None,
        "requested_window": {"start": start, "stop": stop},
        "sampling_gap_s": stats(gaps),
        "gaps_over_5s": sum(g > 5 for g in gaps),
        "metrics": {key: stats(values) for key, values in series.items()},
        "ac_states": sorted({r["power"]["ACLineStatus"] for r in rows if "power" in r}),
        "whole_gpu_energy_j_over_observed_intervals": energy if intervals else None,
        "energy_observed_duration_s": sum(b[0] - a[0] for a, b in intervals),
        "energy_scope": (
            "Whole NVIDIA GPU incl. other processes; trapezoids skip gaps >5s; not whole-system, CPU or NPU energy."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--skip-traces", action="store_true", help="Summarize telemetry only while inference timing is still running."
    )
    a = p.parse_args()
    result = {"telemetry": {}, "traces": [], "trace_analysis": "deferred" if a.skip_traces else "requested"}
    for host in sorted(a.runs.glob("*-host")):
        raw = host / "telemetry.jsonl"
        if not raw.is_file():
            continue
        name = host.name.removesuffix("-host")
        record = {"whole_run": telemetry_summary(raw)}
        report_path = a.runs / name / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            host_report_path = host / "status.json"
            host_report = json.loads(host_report_path.read_text(encoding="utf-8")) if host_report_path.is_file() else {}
            rows = read_lines(a.runs / name / "requests.jsonl")
            warmups = [r for r in rows if r.get("phase") == "warmup"]
            launch, report_start = host_report.get("start_unix"), report.get("start_unix")
            log_path = host / "command.log"
            timing_lines = []
            log_lines = []
            if log_path.is_file():
                log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                timing_lines = [
                    line
                    for line in log_lines
                    if any(
                        marker in line
                        for marker in (
                            "Loading model weights took",
                            "Model loading took",
                            "init engine (",
                            "torch.compile took",
                            "Captured talker_mtp graphs",
                            "engine startup completed",
                        )
                    )
                ]
            record["lifecycle_log_observations"] = {
                "scope": "Counts are log lines, not request counts or a correctness verdict. Inspect source logs.",
                "markers": {},
            }
            for marker in ("Dropping output for unknown req", "force killing remaining process "):
                matches = [line for line in log_lines if marker in line]
                record["lifecycle_log_observations"]["markers"][marker] = {
                    "count": len(matches),
                    "examples": matches[:3] + matches[-3:] if len(matches) > 6 else matches,
                }
            first_submit = warmups[0].get("submitted_unix") if warmups else None
            first_latency = warmups[0].get("ttft_s") if warmups else None
            if first_latency is None and warmups and warmups[0].get("ttfa_ms") is not None:
                first_latency = warmups[0]["ttfa_ms"] / 1000
            record["startup"] = {
                "cache_condition": report.get("cache_condition", "See source report"),
                "constructor_s": report.get("startup_s"),
                "host_launch_to_benchmark_report_start_s": report_start - launch if launch and report_start else None,
                "host_launch_to_first_request_s": first_submit - launch if first_submit and launch else None,
                "host_launch_to_first_output_s": (
                    first_submit - launch + first_latency
                    if first_submit and launch and first_latency is not None
                    else None
                ),
                "inclusive_startup_scope": (
                    "Host launch through first warmup submission/output includes imports, artifact hashing, "
                    "plan construction and engine startup; uses recorded host/engine wall timestamps. "
                    "Constructor timing alone excludes preparation before the constructor."
                ),
                "pre_constructor_scope": (
                    "Host launch to report creation includes Python/import/preparation work; "
                    "not isolated OS process creation."
                ),
                "first_warmup": {
                    key: warmups[0].get(key)
                    for key in (
                        "request_id",
                        "length_band",
                        "concurrency",
                        "wall_s",
                        "ttft_s",
                        "total_wall_s",
                        "ttfa_ms",
                        "rtf_total",
                    )
                }
                if warmups
                else None,
                "runtime_loading_compilation_evidence": timing_lines,
                "scope": (
                    "Runtime-reported component lines retain original boundaries; "
                    "do not sum parallel-stage times into startup latency."
                ),
            }
            start, duration = report.get("sustained_start_unix"), report.get("sustained_wall_s")
            if start and duration:
                record["sustained"] = telemetry_summary(raw, start, start + duration)
                record["sustained_first_5min"] = telemetry_summary(raw, start, start + 300)
                record["sustained_last_5min"] = telemetry_summary(raw, start + duration - 300, start + duration)
                record["sustained_last_60s"] = telemetry_summary(raw, start + duration - 60, start + duration)
                requests = [r for r in rows if r.get("phase") == "sustained"]
                record["sustained_request_summary"] = {
                    "n": len(requests),
                    "rtf_above_one_requests": sum((r.get("rtf_total") or 0) > 1 for r in requests),
                    "simulated_playback_deficit_requests": sum((r.get("stall_at_ttfa_ms") or 0) > 0 for r in requests),
                    "rtf_total": stats([r.get("rtf_total") for r in requests]),
                    "scope": "All sustained requests; timing/real-time gates are separate from successful completion.",
                }
                record["sustained_request_trends"] = {}
                for label, begin, end in [
                    ("first_5min", start, start + 300),
                    ("last_5min", start + duration - 300, start + duration),
                    ("last_60s", start + duration - 60, start + duration),
                ]:
                    window = [r for r in requests if begin <= r.get("submitted_unix", -1) < end]
                    record["sustained_request_trends"][label] = {
                        "n": len(window),
                        "window_basis": "request submission time",
                        "metrics": {
                            metric: stats([r.get(metric) for r in window])
                            for metric in (
                                "wall_s",
                                "ttft_s",
                                "decode_tok_per_s",
                                "total_wall_s",
                                "ttfa_ms",
                                "rtf_total",
                                "stall_at_ttfa_ms",
                            )
                        },
                    }
        result["telemetry"][name] = record
    for path in [] if a.skip_traces else sorted(a.runs.glob("*trace*/traces/**/*")):
        if path.is_file() and (path.name.endswith(".json") or path.name.endswith(".json.gz")):
            report_path = a.runs / path.relative_to(a.runs).parts[0] / "report.json"
            if (
                not report_path.is_file()
                or json.loads(report_path.read_text(encoding="utf-8")).get("status") != "completed"
            ):
                result["traces"].append(
                    {
                        "source": str(path),
                        "status": "not_analyzed",
                        "reason": "Diagnostic has no completed report; trace may be incomplete.",
                    }
                )
                continue
            try:
                result["traces"].append(trace_summary(path))
            except Exception as error:
                result["traces"].append({"source": str(path), "error": repr(error)})
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Summarized {len(result['telemetry'])} telemetry runs and {len(result['traces'])} traces")


if __name__ == "__main__":
    main()
