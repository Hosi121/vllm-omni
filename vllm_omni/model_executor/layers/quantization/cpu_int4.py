# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""W4A16 group-wise weights for vLLM's CPU backend.

Single-stream decode on a CPU is a bandwidth problem: the whole weight matrix
is read to produce one token, so the bits per weight, not the FLOPs, set the
token rate.  Four bits is how llama.cpp's Q4_K_M builds win single-stream
decode against an engine that beats them everywhere else.

vLLM 0.28 does reach four bits on CPU, through the GPTQ/AWQ kernel selector
(`int4_scaled_mm_cpu`, AMX W4A8 -- see `gptq_cpu_export`).  That path pins the
zero point at 8, so its grid is symmetric and its activations are int8.  This
one is the other trade: an asymmetric min/max grid and bf16 activations, which
costs some speed and buys back fidelity.

Weights are quantized to 4 bits on groups of ``group_size`` columns with a
bf16 scale and offset per group; at the default group of 64 that is 4.5 bits
per weight, the same budget Q4_K_M spends.  The GEMM is PyTorch's ``_weight_int4pack_mm_for_cpu`` tinygemm
kernel, which has AVX-512 paths and, unlike the fp32 entry point, keeps
bf16 activations in bf16 end to end.

Checkpoints carry, per quantized linear:

    <name>.weight_packed   uint8  [N, K // 2]  two 4-bit codes per byte,
                                               byte j = w[2j] | w[2j+1] << 4
    <name>.weight_scale    bf16   [N, K // group]
    <name>.weight_zero     bf16   [N, K // group]

and dequantize as ``w = (q - 8) * scale + zero``.  The row-major nibble layout
is the portable one; the tinygemm layout the kernel wants blends rows within a
block of 16 and is built in ``process_weights_after_loading``.

The tinygemm kernel dequantizes once per row of activations, so its cost is
linear in the batch and it loses to AMX bf16 above about eight rows -- fine
for decode, ruinous for prefill.  Large batches therefore take a second path:
the weight is unpacked to bf16 once per call and handed to the ordinary
oneDNN/AMX GEMM, which costs one extra pass over the weights and wins back
prefill.  That path reads a row-major copy, so with ``prefill_dequant`` on
(the default) a layer holds both layouts -- 8 bits per weight resident, but
still 4.5 bits per weight *read* in a decode step, which is what sets the
token rate.

Measured on Spark-X2.5-1.7B (Xeon 8480C, anonymous RSS, median of three
interleaved passes, spread under 1.5%):

    prefill_dequant   resident      prefill        decode
    true (default)     3542 MB     3252 tok/s    76.9 tok/s
    false              2802 MB      448 tok/s    78.5 tok/s

So the second layout costs **740 MB** and buys **7.3x prefill**, because the
tinygemm kernel dequantizes once per row of activations and its cost is linear
in the batch. Keep it on unless memory is tighter than prefill -- on a device
where it is, ``prefill_dequant: false`` is the smallest resident build here.

``dequant_threshold`` defaults to 2, i.e. only a single-row step takes the
tinygemm path.  That is not just a speed choice.  On a Sapphire Rapids host,
running the tinygemm kernel on a multi-row batch and then vLLM's
``cpu_attention_with_kv_cache`` in the same forward kills the process with
SIGILL inside the attention kernel -- the signature of an AMX instruction
issued with no valid tile configuration loaded.  The same prompt is fine in
bf16, and fine at any batch size once every multi-row GEMM goes through the
dequantize path, so the two kernels disagree about who owns the tile state.
Single-row steps never trip it, which is why the crossover sits at 2 rather
than at the ~8 rows where the dequantize path starts winning on speed.
Batched decode therefore pays the dequantize path; single-stream decode, the
case 4-bit weights exist for, does not.
"""

from typing import Any

import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

# The tinygemm packer refuses anything else.
PACK_N_ALIGN = 16
INNER_K_TILES = 1


def pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """[N, K] codes in [0, 15] -> [N, K // 2] uint8, low nibble first."""
    q = q.to(torch.uint8)
    lo, hi = q[:, 0::2], q[:, 1::2]
    return (lo | (hi << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """[N, K // 2] uint8 -> [N, K] int32 codes in [0, 15]."""
    n, half = packed.shape
    out = torch.empty(n, half * 2, dtype=torch.int32)
    out[:, 0::2] = (packed & 0x0F).to(torch.int32)
    out[:, 1::2] = (packed >> 4).to(torch.int32)
    return out


def quantize_groupwise(w: torch.Tensor, group: int):
    """Min/max 4-bit quantization of [N, K] -> (packed, scale, zero).

    ``scale = (max - min) / 15`` and ``zero = min + 8 * scale`` make the
    kernel's ``(q - 8) * scale + zero`` reproduce the group's extremes
    exactly, which is the same asymmetric grid llama.cpp's K-quants use.
    """
    n, k = w.shape
    if k % group:
        raise ValueError(f"input dim {k} is not a multiple of group size {group}")
    wf = w.float().reshape(n, k // group, group)
    wmin, wmax = wf.amin(dim=-1), wf.amax(dim=-1)
    scale = (wmax - wmin).clamp(min=1e-9) / 15.0
    zero = wmin + 8.0 * scale
    q = ((wf - wmin.unsqueeze(-1)) / scale.unsqueeze(-1)).round_().clamp_(0, 15)
    return pack_nibbles(q.reshape(n, k)), scale, zero


def _to_tinygemm(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
    """Row-major nibbles + [N, G] scales -> kernel layout + [G, N, 2] scales."""
    codes = unpack_nibbles(packed)
    inner = torch.ops.aten._convert_weight_to_int4pack_for_cpu(codes, INNER_K_TILES)
    sz = torch.stack([scale.float(), zero.float()], dim=-1)  # [N, G, 2]
    sz = sz.transpose(0, 1).contiguous().to(torch.bfloat16)  # [G, N, 2]
    return inner, sz


def to_split_halves(packed: torch.Tensor) -> torch.Tensor:
    """Interleaved nibbles -> byte j holding w[j] and w[j + K/2].

    The on-disk layout interleaves so that a tensor-parallel split along K
    stays a plain slice.  Once the shard is local, splitting the halves makes
    both nibble planes contiguous runs, and the dequantize loop drops its
    lane shuffle: measured 3.1 ms per layer instead of 4.5.
    """
    codes = unpack_nibbles(packed)
    n, k = codes.shape
    lo, hi = codes[:, : k // 2], codes[:, k // 2 :]
    return (lo.to(torch.uint8) | (hi.to(torch.uint8) << 4)).contiguous()


def _dequantize_halves(
    halves: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group: int,
) -> torch.Tensor:
    n, half = halves.shape
    g2 = half // group
    lo = (halves & 0x0F).to(torch.bfloat16).reshape(n, g2, group)
    hi = (halves >> 4).to(torch.bfloat16).reshape(n, g2, group)
    a = (lo - 8.0) * scale[:, :g2].unsqueeze(-1) + zero[:, :g2].unsqueeze(-1)
    b = (hi - 8.0) * scale[:, g2:].unsqueeze(-1) + zero[:, g2:].unsqueeze(-1)
    return torch.cat([a.reshape(n, half), b.reshape(n, half)], dim=1)


_compiled_dequant = None


def dequantize(halves, scale, zero, group):
    """bf16 weight from the row-major halves; compiled, so it is one pass."""
    global _compiled_dequant
    if _compiled_dequant is None:
        # Compiled outside any vLLM graph: this runs inside an opaque custom
        # op, so dynamo never traces it and never guards on the batch size.
        _compiled_dequant = torch.compile(_dequantize_halves, dynamic=False)
    return _compiled_dequant(halves, scale, zero, group)


def _cpu_int4_linear(
    x: torch.Tensor,
    int4pack: torch.Tensor,
    scales_and_zeros: torch.Tensor,
    halves: torch.Tensor | None,
    scale: torch.Tensor | None,
    zero: torch.Tensor | None,
    group: int,
    threshold: int,
) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    if halves is not None and x.shape[0] >= threshold:
        weight = dequantize(halves, scale, zero, group)
        return torch.nn.functional.linear(x, weight)
    return torch.ops.aten._weight_int4pack_mm_for_cpu(
        x, int4pack, group, scales_and_zeros
    )


def _cpu_int4_linear_fake(
    x: torch.Tensor,
    int4pack: torch.Tensor,
    scales_and_zeros: torch.Tensor,
    halves: torch.Tensor | None,
    scale: torch.Tensor | None,
    zero: torch.Tensor | None,
    group: int,
    threshold: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], int4pack.shape[0]), dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="cpu_int4_linear",
    op_func=_cpu_int4_linear,
    mutates_args=[],
    fake_impl=_cpu_int4_linear_fake,
    dispatch_key="CPU",
)


@register_quantization_config("cpu_int4")
class CPUInt4Config(QuantizationConfig):
    """Group-wise 4-bit weight-only quantization for the CPU backend."""

    def __init__(
        self,
        group_size: int = 64,
        ignore: list[str] | None = None,
        quantize_embedding: bool = True,
        prefill_dequant: bool = True,
        dequant_threshold: int = 2,
        use_custom_op: bool = True,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.ignore = list(ignore or [])
        self.quantize_embedding = quantize_embedding
        self.prefill_dequant = prefill_dequant
        self.dequant_threshold = dequant_threshold
        self.use_custom_op = use_custom_op

    def __repr__(self) -> str:
        return (
            f"CPUInt4Config(group_size={self.group_size}, "
            f"ignore={self.ignore}, quantize_embedding={self.quantize_embedding}, "
            f"prefill_dequant={self.prefill_dequant}, "
            f"dequant_threshold={self.dequant_threshold}, "
            f"use_custom_op={self.use_custom_op})"
        )

    def get_name(self) -> str:
        return "cpu_int4"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # The fp32 entry point of the tinygemm kernel has no vectorized path
        # and runs ~30x slower, so bf16 is the only activation type offered.
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CPUInt4Config":
        return cls(
            group_size=cls.get_from_keys_or(config, ["group_size"], 64),
            ignore=cls.get_from_keys_or(config, ["ignore"], []),
            quantize_embedding=cls.get_from_keys_or(
                config, ["quantize_embedding"], True
            ),
            prefill_dequant=cls.get_from_keys_or(config, ["prefill_dequant"], True),
            dequant_threshold=cls.get_from_keys_or(config, ["dequant_threshold"], 2),
            use_custom_op=cls.get_from_keys_or(config, ["use_custom_op"], True),
        )

    def _is_ignored(self, prefix: str) -> bool:
        return any(prefix == p or prefix.endswith("." + p) for p in self.ignore)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        if isinstance(layer, VocabParallelEmbedding):
            if not self.quantize_embedding or self._is_ignored(prefix):
                return None
            return CPUInt4EmbeddingMethod(self)
        if isinstance(layer, LinearBase):
            if self._is_ignored(prefix):
                return UnquantizedLinearMethod()
            return CPUInt4LinearMethod(self)
        return None


def _create_int4_weights(
    method: "CPUInt4LinearMethod",
    layer: torch.nn.Module,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    params_dtype: torch.dtype,
    extra_weight_attrs: dict,
) -> None:
    group = method.quant_config.group_size
    out_size = sum(output_partition_sizes)
    if input_size_per_partition % group:
        raise ValueError(
            f"input dim {input_size_per_partition} is not a multiple of "
            f"group size {group}"
        )
    if out_size % PACK_N_ALIGN:
        raise ValueError(
            f"output dim {out_size} is not a multiple of {PACK_N_ALIGN}, which "
            "the tinygemm packer requires"
        )
    weight_loader = extra_weight_attrs.pop("weight_loader")

    packed = PackedvLLMParameter(
        data=torch.empty(out_size, input_size_per_partition // 2, dtype=torch.uint8),
        input_dim=1,
        output_dim=0,
        packed_dim=1,
        packed_factor=2,
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight_packed", packed)
    for name in ("weight_scale", "weight_zero"):
        p = GroupQuantScaleParameter(
            data=torch.empty(
                out_size, input_size_per_partition // group, dtype=torch.bfloat16
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter(name, p)
    layer.input_size_per_partition = input_size_per_partition
    layer.output_size_per_partition = out_size
    set_weight_attrs(packed, extra_weight_attrs)


def _finalize_int4(
    layer: torch.nn.Module, keep_row_major: bool, prefill_dequant: bool
) -> None:
    """Repack into the layouts the two GEMM paths want; once per layer."""
    if getattr(layer, "_cpu_int4_ready", False):
        return
    packed = layer.weight_packed.data
    scale = layer.weight_scale.data
    zero = layer.weight_zero.data
    inner, sz = _to_tinygemm(packed, scale, zero)
    layer.weight_int4pack = torch.nn.Parameter(inner, requires_grad=False)
    layer.weight_scales_and_zeros = torch.nn.Parameter(sz, requires_grad=False)
    if prefill_dequant:
        layer.weight_halves = torch.nn.Parameter(
            to_split_halves(packed), requires_grad=False
        )
        layer.weight_group_scale = torch.nn.Parameter(scale.clone(), requires_grad=False)
        layer.weight_group_zero = torch.nn.Parameter(zero.clone(), requires_grad=False)
    else:
        layer.weight_halves = None
        layer.weight_group_scale = None
        layer.weight_group_zero = None
    if keep_row_major:
        # Only the vocab table needs this: the tinygemm layout interleaves
        # rows inside blocks of 16, so a single-token embedding lookup cannot
        # address it. One row is a few hundred bytes, so keeping the portable
        # copy costs no bandwidth at decode.
        layer.weight_packed = torch.nn.Parameter(packed, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        layer.weight_zero = torch.nn.Parameter(zero, requires_grad=False)
    else:
        del layer.weight_packed
        del layer.weight_scale
        del layer.weight_zero
    layer._cpu_int4_ready = True


def _int4_mm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    group: int,
    threshold: int,
    custom_op: bool = True,
) -> torch.Tensor:
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    if x2d.dtype != torch.bfloat16:
        # Static at trace time: the config advertises bf16 activations only,
        # and the fp32 entry point of the tinygemm kernel has no vectorized
        # path (~30x slower), so never fall into it by accident.
        x2d = x2d.to(torch.bfloat16)
    if custom_op:
        # Opaque to dynamo, so the batch-size branch never becomes a guard on
        # vLLM's single traced graph. Costs ~5 us a call in dispatch.
        out = torch.ops.vllm.cpu_int4_linear(
            x2d,
            layer.weight_int4pack,
            layer.weight_scales_and_zeros,
            layer.weight_halves,
            layer.weight_group_scale,
            layer.weight_group_zero,
            group,
            threshold,
        )
    elif layer.weight_halves is not None and x2d.shape[0] >= threshold:
        out = torch.nn.functional.linear(
            x2d,
            _dequantize_halves(
                layer.weight_halves,
                layer.weight_group_scale,
                layer.weight_group_zero,
                group,
            ),
        )
    else:
        out = torch.ops.aten._weight_int4pack_mm_for_cpu(
            x2d, layer.weight_int4pack, group, layer.weight_scales_and_zeros
        )
    return out.reshape(*orig_shape[:-1], out.shape[-1])


class CPUInt4LinearMethod(QuantizeMethodBase):
    """Group-wise 4-bit linear."""

    def __init__(self, quant_config: CPUInt4Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        _create_int4_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            params_dtype,
            extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _finalize_int4(
            layer,
            keep_row_major=False,
            prefill_dequant=self.quant_config.prefill_dequant,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = _int4_mm(
            layer,
            x,
            self.quant_config.group_size,
            self.quant_config.dequant_threshold,
            self.quant_config.use_custom_op,
        )
        if bias is not None:
            out = out + bias
        return out


class CPUInt4EmbeddingMethod(CPUInt4LinearMethod):
    """The same weights, used as a vocab table and as the output head.

    With tied embeddings these are one tensor, so quantizing it pays twice:
    the lm_head GEMM is the single largest read in a decode step, and the
    table itself is 16% of the model's bytes.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        _create_int4_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            params_dtype,
            extra_weight_attrs,
        )
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # No dequantize path for the vocab table: the head only ever sees the
        # tokens being sampled (one row per sequence), and unpacking a
        # 131072 x 2048 matrix to bf16 would cost more than the GEMM saves.
        _finalize_int4(layer, keep_row_major=True, prefill_dequant=False)

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: torch.nn.Module):
        """Share every quantized tensor, not just ``weight``."""
        for name in (
            "weight_packed",
            "weight_scale",
            "weight_zero",
            "weight_int4pack",
            "weight_scales_and_zeros",
            "weight_halves",
            "weight_group_scale",
            "weight_group_zero",
        ):
            src = getattr(embed_tokens, name, None)
            if src is not None:
                setattr(layer, name, src)
        layer._cpu_int4_tied_to = embed_tokens
        return layer

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        group = self.quant_config.group_size
        rows = layer.weight_packed.data.index_select(0, input_.reshape(-1))
        codes = unpack_nibbles(rows).float()
        n, k = codes.shape
        scale = layer.weight_scale.data.index_select(0, input_.reshape(-1)).float()
        zero = layer.weight_zero.data.index_select(0, input_.reshape(-1)).float()
        deq = (codes.reshape(n, k // group, group) - 8.0) * scale.unsqueeze(
            -1
        ) + zero.unsqueeze(-1)
        out = deq.reshape(n, k).to(torch.bfloat16)
        return out.reshape(*input_.shape, k)
