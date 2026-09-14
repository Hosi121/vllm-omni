# vllm-embedding

A fork of [vLLM-Omni](https://github.com/vllm-project/vllm-omni) for running
multimodal models on **edge CPUs** — machines with tens of gigabytes of RAM and
no GPU, rather than servers.

The upstream project targets datacentre inference. Getting the same models onto
a mini-PC or a gateway box turns out to be less about kernels than about three
things nobody profiles: how much memory is resident while the model *loads*,
how much is resident that nothing ever reads, and whether the number you just
measured came from the code path you thought it did.

Upstream's own README is preserved at [docs/README.vllm-omni.md](docs/README.vllm-omni.md).

## What this fork adds

| area | change |
|---|---|
| 4-bit CPU quantization | `cpu_int4` — a W4A16 path built on PyTorch's tinygemm kernel, with the prefill path reading the kernel's own weight layout instead of a duplicate |
| Memory | RoPE tables sized to the servable context; weights repacked in blocks; KV budget that is actually honoured on CPU |
| Device routing | `vllm_omni/edge/weight_path.py` picks the weight format per device class, from measurements rather than assumption |
| Measurement | `benchmarks/edge_harness/` — the benchmark, profiling, memory-attribution and fidelity tooling every number here was produced with |

## Headline results

Spark-X2.5-1.7B, Xeon Platinum 8480C, 24 pinned node-local cores, W4A16 g128.
Every figure is a median of at least three interleaved passes against a control
arm, on a shared host where end-to-end throughput swings ~30% run to run.

**Memory** — each measured as its own A/B with the alternative reachable by a flag:

| lever | saving | cost |
|---|---|---|
| KV budget 1 GiB → 192 MiB | **843 MB** | fewer cached tokens |
| RoPE table capped to `max_model_len` | **618 MB** | none |
| One packed weight layout instead of two | **615 MB** | ~1.6× prefill |
| Blocked weight repack | **489 MB** of load-time peak | none |

These overlap and **must not be added** — see [docs/edge/results.md](docs/edge/results.md).

**Speed**, same machine:

| path | decode | prefill (pp512) |
|---|---|---|
| W4A8 (AMX `int4_scaled_mm_cpu`) | 74.4 tok/s | — |
| W4A16 (tinygemm) | 71.6 tok/s | 2190 tok/s |
| `cpu_gemm_wna16` fallback | 32.3 tok/s | — |
| llama.cpp Q4_K_M (reference) | 90.0 tok/s | 1121 tok/s |

## Why the numbers here are worth reading

This project has retracted five conclusions, every one of them a measurement
error rather than a coding error. The harness exists because of them:

- a "3.1 GB saving" that was `VmRSS` counting the mmap'd checkpoint;
- W4A8 timings taken on a checkpoint whose output head was silently left in bf16;
- a memory benchmark reading `/proc/self/status` while the model ran in a *worker process*;
- "KV sizing saves only 188 MB" — measured through a code path that discarded the setting, so both arms ran the same cache;
- a 1.9× prefill cost quoted from the one pass of a grid whose other passes had crashed.

[docs/edge/methodology.md](docs/edge/methodology.md) is the resulting discipline,
and it is the most transferable thing in this repository.

## Quick start

```bash
# CPU build of vLLM, then this fork
pip install vllm==0.28.0+cpu
VLLM_TARGET_DEVICE=cpu pip install -e .

# a right-sized edge deployment
VLLM_CPU_KVCACHE_SPACE=1 python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='<your-cpu_int4-checkpoint>', quantization='cpu_int4',
          dtype='bfloat16', max_model_len=2048, max_num_seqs=1,
          kv_cache_memory_bytes=192 << 20)   # the single largest memory lever
print(llm.generate(['Hello'], SamplingParams(max_tokens=32))[0].outputs[0].text)
"
```

See [docs/edge/cpu-4bit.md](docs/edge/cpu-4bit.md) for building a checkpoint and
choosing between the 4-bit paths, and [ROADMAP.md](ROADMAP.md) for what is
measured, what is next, and what was tried and abandoned.

## Licence

Apache-2.0, inherited from vLLM-Omni. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
