# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Size the KV cache from the model instead of from a server default.

``VLLM_CPU_KVCACHE_SPACE`` defaults to a server-shaped budget and is parsed
with ``int()``, so it cannot even express an edge-sized cache -- the smallest
non-zero value it accepts is 1 GiB.

**Retraction (2026-09-13).** This module used to claim that right-sizing the
budget took resident memory from 8329 MB to 5254 MB. That number was wrong
twice over: it was read from ``VmRSS``, which counts the mmap'd checkpoint's
pages and moved 1-2 GB between identical runs, and the two arms were not the
same model file. Re-measured on ``RssAnon``, which repeats to 0.2%, the saving
is **188 MB**.

That is small because of *when* it was measured, not because the budget does
not matter: the benchmark generated 8 tokens, so almost no cache was ever
touched. Allocation is what this module sizes, and the two coincide only at
long context -- see the arithmetic below, where the flat and hybrid budgets at
32 k differ by 1323 MiB. Sizing the cache is worth re-opening on a benchmark
that actually fills it (harness item H3), and worth nothing on one that does
not.

A hybrid-attention model does not *need* a full-length cache in every layer.
Spark-X2.5 is 3 sliding layers (window 512) to 1 full layer over 28 layers, so
at a 32 k context its real cache is 3.8x smaller than a flat budget:

    flat      28 x 32768               = 917 504 token-layers
    hybrid    21 x 512 + 7 x 32768     = 240 128 token-layers

But vLLM's CPU path did not take that discount when this was measured: for a
4 GiB budget it reported 71 527 tokens, i.e. 56 KiB/token, which is the flat
figure. That observation now needs re-checking rather than trusting --
``CpuPlatform.support_hybrid_kv_cache()`` returns ``True`` in vLLM 0.28
(``vllm/platforms/cpu.py:545``), so either the model's attention groups are not
being reported as hybrid, or the budget is applied before that is known. Until
someone establishes which, the conservative number is the one to pass. So
``bytes_total`` is the **flat** number -- the one that is safe to pass today --
and the hybrid number is reported alongside it as ``bytes_total_hybrid``, for
a backend that accounts per layer type. A budget that is too small is not a
graceful degradation: vLLM preempts by recompute, which on a single-stream
edge deployment is a stall.
"""

from dataclasses import dataclass
from typing import Any

_DTYPE_BYTES = {
    "float32": 4, "fp32": 4,
    "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "half": 2,
    "float8_e4m3fn": 1, "fp8": 1, "int8": 1,
}


def dtype_bytes(dtype: Any) -> int:
    name = getattr(dtype, "name", None) or str(dtype).replace("torch.", "")
    return _DTYPE_BYTES.get(name, 2)


@dataclass(frozen=True)
class KVBudget:
    bytes_total: int
    """Flat budget: safe to pass to vLLM today."""
    bytes_total_hybrid: int
    """What the model's own cache needs, if the backend accounts per layer type."""
    bytes_per_token_flat: int
    token_layers_hybrid: int
    token_layers_flat: int
    sliding_layers: int
    full_layers: int
    window: int | None

    @property
    def saving_vs_flat(self) -> float:
        """What a layer-type-aware backend would save; 1.0 when there is none."""
        return (
            self.token_layers_flat / self.token_layers_hybrid
            if self.token_layers_hybrid
            else 1.0
        )

    def summary(self) -> str:
        extra = ""
        if self.sliding_layers:
            extra = (
                f"; {self.bytes_total_hybrid / 2**20:.0f} MiB if the backend "
                f"accounts {self.sliding_layers} sliding@{self.window} layers "
                f"separately ({self.saving_vs_flat:.2f}x)"
            )
        return f"{self.bytes_total / 2**20:.0f} MiB flat{extra}"


def kv_budget_for(
    hf_config: Any,
    *,
    max_model_len: int,
    max_num_seqs: int = 1,
    kv_dtype: Any = "bfloat16",
    block_size: int = 16,
    headroom: float = 1.25,
) -> KVBudget:
    """Bytes of KV cache a deployment of this model actually needs.

    Reads ``layer_types``/``sliding_window`` when present, so hybrid-attention
    models are charged for their real cache and not for the longest layer.
    """
    n_layers = int(
        getattr(hf_config, "num_hidden_layers", None)
        or getattr(hf_config, "n_layer", 0)
    )
    if n_layers <= 0:
        raise ValueError("cannot read num_hidden_layers from the config")
    n_heads = int(getattr(hf_config, "num_attention_heads", 0) or 0)
    kv_heads = int(getattr(hf_config, "num_key_value_heads", 0) or n_heads or 1)
    head_dim = int(
        getattr(hf_config, "head_dim", 0)
        or (getattr(hf_config, "hidden_size", 0) // max(n_heads, 1))
    )
    if head_dim <= 0:
        raise ValueError("cannot read head_dim from the config")

    layer_types = list(getattr(hf_config, "layer_types", None) or [])
    window = getattr(hf_config, "sliding_window", None)
    sliding = sum(1 for t in layer_types if "sliding" in str(t)) if window else 0
    full = n_layers - sliding

    def rounded(tokens: int) -> int:
        return ((tokens + block_size - 1) // block_size) * block_size

    per_seq_full = rounded(max_model_len)
    # A sliding layer still has to hold the window, and never more than the
    # context: a 512-window layer at a 128-token context caches 128.
    per_seq_sliding = rounded(min(int(window), max_model_len)) if sliding else 0

    token_layers_hybrid = max_num_seqs * (
        full * per_seq_full + sliding * per_seq_sliding
    )
    token_layers_flat = max_num_seqs * n_layers * per_seq_full
    bytes_per_token_layer = kv_heads * head_dim * 2 * dtype_bytes(kv_dtype)

    return KVBudget(
        bytes_total=int(token_layers_flat * bytes_per_token_layer * headroom),
        bytes_total_hybrid=int(token_layers_hybrid * bytes_per_token_layer * headroom),
        bytes_per_token_flat=n_layers * bytes_per_token_layer,
        token_layers_hybrid=token_layers_hybrid,
        token_layers_flat=token_layers_flat,
        sliding_layers=sliding,
        full_layers=full,
        window=int(window) if sliding else None,
    )
