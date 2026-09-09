# Edge benchmark results

Committed JSON produced by the edge branch benchmarks (kept small; WAVs are never committed):

- `stream_latency/<model>/<deploy>/<timestamp>[_tag].json` — `stream_latency_bench.py` output (schema 1): `meta` (model, deploy config, git SHA, GPU), `init` (`init_s`, optional init timeline), `resources` (peak RSS, peak GPU MiB, sampling mode), `requests[]` (per-chunk arrivals, TTFA, playback-start, stall at TTFA, RTF streamed / total) and `summary` (p50/p90/max).
- `init_profile/<timestamp>/summary.json` — `init_profile.py` comparison of initialization phases per configuration.
- `hw_emulation/`, `scheduling/` — per-work-package comparison tables.

Metric definitions live in `benchmarks/tts/stream_latency_metrics.py`.
