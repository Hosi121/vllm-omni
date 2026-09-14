# Environment, pins and reproducing a number

Every figure in this repository was produced with the versions below. A result
whose environment was not recorded cannot be retracted later, only doubted, so
the harness stamps these into each artifact automatically — see the
`provenance` block in any file under `benchmarks/edge_harness/results/`.

## vLLM is **not** modified

Worth stating plainly, because it is the obvious assumption and it is wrong.

This fork changes vLLM's *behaviour* on CPU, but the vLLM tree and the installed
wheel are stock. The changes are runtime patches applied at import through
vLLM's own extension points, so they survive a wheel upgrade instead of being
lost on it:

| patch (`vllm_omni/patch.py`) | what it changes |
|---|---|
| `_patch_cpu_aot_compile_cache_load` | vLLM's CPU path fails to load its own AOT artifact (`finalize_loading` missing on a plain function), treats it as a miss, and recompiles on every start |
| `_patch_cpu_explicit_kv_cache_memory` | `CpuPlatform.check_and_update_config` overwrites `kv_cache_memory_bytes` from `VLLM_CPU_KVCACHE_SPACE` unconditionally, so a per-stage budget had no effect at all — this is what made "KV sizing is not a memory lever" look true |
| `_patch_cpu_release_memory_after_load` | returns freed allocator arenas to the OS after weight load |

Other extension points used, rather than edits: the quantization-config registry
(`register_quantization_config` for `cpu_int4`), the IR op registry, platform
hooks, and worker method wrapping.

So there is no vLLM patch series to apply. Pin the version and let `patch.py`
do the rest.

## Pins

| component | pin |
|---|---|
| vLLM (installed) | `0.28.0+cpu` wheel |
| vLLM (source read for this work) | `51da0ca66c8065619c79e35dff97aa99aeaf5644` (`v0.28.1rc0`) |
| llama.cpp (baseline) | `e71b80510c848c00175924ecf3c40333ccae8eb5` (2026-09-07) |
| Python | 3.12 |
| torch | CPU build, as pulled by the vLLM CPU wheel |

```bash
pip install --torch-backend cpu \
  https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl
VLLM_TARGET_DEVICE=cpu pip install -e .          # this fork
```

## Machine

Dual-socket Xeon Platinum 8480C (Sapphire Rapids, 56 cores/socket, AMX-BF16 and
AMX-INT8), 2 TB RAM, NUMA nodes 0 = CPUs 0-55/112-167 and 1 = 56-111/168-223.

It is **shared**, which shapes everything: end-to-end throughput swings ~30% run
to run, per-op self time reproduces to ~5%, `RssAnon` to 0.2%. Runs are pinned
with `numactl --physcpubind=<24 cores> --membind=<node>` and the harness refuses
to start when those cores are already busy.

## Reproducing a published number

```bash
# memory: any of the four levers, as its own A/B
python benchmarks/edge_harness/bench_harness.py --spec <arms>.json \
    --passes 3 --metric rss_anon_tree_mb --cores 56-79 --out summary.json

# where each number came from
python -c "import json,sys; print(json.load(open(sys.argv[1]))['provenance'])" \
    benchmarks/edge_harness/results/<file>.json
```

Two settings change what a memory or profile number *means*, and both have
produced a wrong published conclusion by being left at their default:
`--context-len` (without it the KV cache is allocated and never touched) and
`--prompt-len` (the default of 1 profiles an empty attention cache).

## Retracted artifacts

Files under `benchmarks/edge_harness/results/` carrying `"retracted": true` are
kept deliberately rather than deleted, with the reason and the superseding file
named. `profile_w4a8_g128_p{1,2}.json` are the clearest case: both show
`_C::onednn_mm` at ~2.97 ms/step, the signature of a checkpoint whose output
head was left in bf16, and every W4A8 number derived from them is void.
