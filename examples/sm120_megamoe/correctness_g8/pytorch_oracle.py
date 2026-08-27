"""Pure-PyTorch mathematical oracle for SM120 FP8/FP4 routed experts.

The implementation intentionally does not call a DeepGEMM kernel.  It accepts
the quantized tensors consumed by the CUDA baseline, explicitly dequantizes
them, and composes W1, SwiGLU, intermediate FP8 quantization, W2, and combine.

It is a numerical oracle, not a bit-exact CUDA-fast-math emulator.  In
particular, ``torch.nn.functional.silu`` does not reproduce ``__expf`` plus
``rcp.approx.ftz`` bit for bit.  Callers must use a documented error metric and
tolerance.  The frozen exact sparse oracle remains a complementary stage gate.
"""

from __future__ import annotations

import torch


_REFERENCE_SEED = 20260819
_FP8_CODES = (0x28, 0x30, 0x38, 0x40, 0xA8, 0xB0, 0xB8, 0xC0)
_FP4_CODES = (1, 2, 3, 4, 5, 6, 9, 10)


def decode_ue8m0(exponents: torch.Tensor) -> torch.Tensor:
    """Decode one UE8M0 exponent byte per scale to float32."""

    if exponents.dtype != torch.uint8:
        raise TypeError("UE8M0 exponents must be uint8")
    return (exponents.to(torch.int32) << 23).view(torch.float32)


def decode_fp8_e4m3(
    values: torch.Tensor, scale_exponents: torch.Tensor, *, gran_k: int = 32
) -> torch.Tensor:
    """Decode a row-major FP8 tensor with one UE8M0 scale per K group."""

    if values.ndim != 2 or values.shape[1] % gran_k != 0:
        raise ValueError("FP8 values must be 2D with K divisible by gran_k")
    if values.dtype == torch.uint8:
        values = values.view(torch.float8_e4m3fn)
    if values.dtype != torch.float8_e4m3fn:
        raise TypeError("FP8 values must be uint8 codes or torch.float8_e4m3fn")
    expected = (values.shape[0], values.shape[1] // gran_k)
    if tuple(scale_exponents.shape) != expected:
        raise ValueError(f"FP8 scale shape must be {expected}")
    scales = decode_ue8m0(scale_exponents)
    return values.float() * scales.repeat_interleave(gran_k, dim=1)


def decode_packed_fp4_e2m1(
    packed: torch.Tensor, scale_exponents: torch.Tensor, *, gran_k: int = 32
) -> torch.Tensor:
    """Decode row-major, low-nibble-first E2M1 with UE8M0 K-group scales."""

    if packed.ndim != 2 or packed.shape[1] * 2 % gran_k != 0:
        raise ValueError("packed FP4 values must be 2D with K divisible by gran_k")
    if packed.dtype not in (torch.int8, torch.uint8):
        raise TypeError("packed FP4 values must be int8 or uint8")
    rows, packed_k = packed.shape
    k = packed_k * 2
    expected = (rows, k // gran_k)
    if tuple(scale_exponents.shape) != expected:
        raise ValueError(f"FP4 scale shape must be {expected}")
    codes = torch.empty((rows, k), dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = (packed >> 4) & 0x0F
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    indices = (codes & 7).to(torch.long)
    values = magnitudes[indices]
    values = torch.where(((codes & 8) != 0) & (indices != 0), -values, values)
    scales = decode_ue8m0(scale_exponents)
    return values * scales.repeat_interleave(gran_k, dim=1)


def quantize_intermediate_fp8(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw E4M3 codes and UE8M0 exponents for the frozen K32 rule."""

    if values.ndim != 2 or values.shape[1] % 32 != 0:
        raise ValueError("intermediate values must be 2D with K divisible by 32")
    grouped = values.float().view(values.shape[0], -1, 32)
    amax = grouped.abs().amax(dim=2)
    raw = (amax * (1.0 / 448.0)).contiguous()
    bits = raw.view(torch.int32)
    exponent = ((bits >> 23) & 255) + (((bits & 0x7FFFFF) + 0x7FFFFF) >> 23)
    exponent = exponent.clamp(max=254).to(torch.uint8)
    inverse = ((254 - exponent.to(torch.int32)) << 23).view(torch.float32)
    quantized = (grouped * inverse.unsqueeze(2)).to(torch.float8_e4m3fn)
    return quantized.view(torch.uint8).view_as(values), exponent


def _quantize_intermediate_fp8(values: torch.Tensor) -> torch.Tensor:
    """Quantize/dequantize rows at K32 using the frozen host's UE8M0 rule."""

    codes, exponents = quantize_intermediate_fp8(values)
    return decode_fp8_e4m3(codes, exponents, gran_k=32)


def reference_route_stages(
    x_fp8: torch.Tensor,
    x_scale_exponents: torch.Tensor,
    route_weight: torch.Tensor | float,
    w1_fp4: torch.Tensor,
    w1_scale_exponents: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_scale_exponents: torch.Tensor,
    *,
    activation_clamp: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Evaluate one exported route and retain every production stage."""

    if x_fp8.ndim != 1 or x_scale_exponents.ndim != 1:
        raise ValueError("one route requires one activation row and scale row")
    if w1_fp4.ndim != 2 or w2_fp4.ndim != 2:
        raise ValueError("one route requires one expert's W1 and W2")
    hidden = x_fp8.numel()
    physical_w1_n = w1_fp4.shape[0]
    intermediate = physical_w1_n // 2
    if physical_w1_n % 16 != 0 or w1_fp4.shape[1] * 2 != hidden:
        raise ValueError("exported W1 shape is inconsistent")
    if w2_fp4.shape[1] * 2 != intermediate:
        raise ValueError("exported W2 shape is inconsistent")

    x = decode_fp8_e4m3(
        x_fp8.unsqueeze(0), x_scale_exponents.unsqueeze(0), gran_k=32
    )[0]
    w1 = decode_packed_fp4_e2m1(
        w1_fp4, w1_scale_exponents, gran_k=32
    )
    w1_bf16 = torch.mv(w1, x).to(torch.bfloat16)
    logical_n = torch.arange(intermediate, device=x.device)
    gate_physical = (logical_n // 8) * 16 + (logical_n & 7)
    up_physical = gate_physical + 8
    gate = w1_bf16.float()[gate_physical].clamp(max=activation_clamp)
    up = w1_bf16.float()[up_physical].clamp(
        min=-activation_clamp, max=activation_clamp
    )
    weight = torch.as_tensor(route_weight, dtype=torch.float32, device=x.device)
    swiglu = torch.nn.functional.silu(gate) * up * weight
    intermediate_codes, intermediate_sf = quantize_intermediate_fp8(
        swiglu.unsqueeze(0)
    )
    intermediate_dequant = decode_fp8_e4m3(
        intermediate_codes, intermediate_sf, gran_k=32
    )[0]
    w2 = decode_packed_fp4_e2m1(
        w2_fp4, w2_scale_exponents, gran_k=32
    )
    w2_bf16 = torch.mv(w2, intermediate_dequant).to(torch.bfloat16)
    return {
        "w1_bf16": w1_bf16,
        "swiglu": swiglu,
        "intermediate_fp8": intermediate_codes[0],
        "intermediate_sf": intermediate_sf[0],
        "intermediate_dequant": intermediate_dequant,
        "w2_bf16": w2_bf16,
    }


def reference_megamoe(
    x_fp8: torch.Tensor,
    x_scale_exponents: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_scale_exponents: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_scale_exponents: torch.Tensor,
    *,
    activation_clamp: float = 10.0,
) -> torch.Tensor:
    """Compose the fixed-layout routed-expert operation without DeepGEMM.

    Shapes are ``x[M,H]``, ``topk[M,K]``, ``w1[E,2I,H/2]`` packed, and
    ``w2[E,O,I/2]`` packed.  W1's physical N layout is the SM120 donor layout:
    groups of eight logical columns interleave the gate and up branches.
    Scale tensors hold unpacked UE8M0 exponent bytes at K32 granularity.
    """

    if x_fp8.ndim != 2 or topk_idx.ndim != 2:
        raise ValueError("x and topk_idx must be 2D")
    if topk_weights.shape != topk_idx.shape:
        raise ValueError("topk weights and indices must have identical shapes")
    if x_fp8.shape[0] != topk_idx.shape[0]:
        raise ValueError("x and topk must have the same row count")
    if w1_fp4.ndim != 3 or w2_fp4.ndim != 3:
        raise ValueError("weights must be E-by-N-by-packed-K")
    experts, physical_w1_n, packed_h = w1_fp4.shape
    if physical_w1_n % 2 != 0 or physical_w1_n % 16 != 0:
        raise ValueError("W1 physical N must be divisible by 16")
    intermediate = physical_w1_n // 2
    hidden = packed_h * 2
    if x_fp8.shape[1] != hidden:
        raise ValueError("x hidden size does not match W1")
    if w2_fp4.shape[0] != experts or w2_fp4.shape[2] * 2 != intermediate:
        raise ValueError("W2 expert/K dimensions do not match W1")
    output = w2_fp4.shape[1]

    x = decode_fp8_e4m3(x_fp8, x_scale_exponents, gran_k=32)
    combined = torch.zeros(
        (x.shape[0], output), dtype=torch.float32, device=x.device
    )
    logical_n = torch.arange(intermediate, device=x.device)
    group = logical_n // 8
    lane = logical_n & 7
    gate_physical = group * 16 + lane
    up_physical = gate_physical + 8

    for token in range(x.shape[0]):
        for slot in range(topk_idx.shape[1]):
            expert = int(topk_idx[token, slot])
            if expert < 0:
                continue
            if expert >= experts:
                raise ValueError(f"expert {expert} is outside [0,{experts})")
            w1 = decode_packed_fp4_e2m1(
                w1_fp4[expert], w1_scale_exponents[expert], gran_k=32
            )
            w1_bf16 = torch.mv(w1, x[token]).to(torch.bfloat16).float()
            gate = w1_bf16[gate_physical].clamp(max=activation_clamp)
            up = w1_bf16[up_physical].clamp(
                min=-activation_clamp, max=activation_clamp
            )
            routed = (
                torch.nn.functional.silu(gate)
                * up
                * topk_weights[token, slot].float()
            )
            w2_input = _quantize_intermediate_fp8(routed.unsqueeze(0))[0]
            w2 = decode_packed_fp4_e2m1(
                w2_fp4[expert], w2_scale_exponents[expert], gran_k=32
            )
            partial = torch.mv(w2, w2_input).to(torch.bfloat16).float()
            combined[token] += partial
    return combined.to(torch.bfloat16)


def relative_l2(observed: torch.Tensor, expected: torch.Tensor) -> float:
    """Normalized L2 metric used by the independent numerical gate."""

    if observed.shape != expected.shape:
        raise ValueError("observed and expected shapes differ")
    observed_f = observed.float()
    expected_f = expected.float()
    denominator = torch.linalg.vector_norm(expected_f).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(observed_f - expected_f) / denominator)


def _pseudo_fp8_codes(count: int, rank: int, device: torch.device) -> torch.Tensor:
    """Reproduce the frozen host's uint64 fixture mixer with signed int64.

    Only bits 0..2 and 29..31 are observed after the xor.  Arithmetic and
    logical right shift therefore have identical low three bits, while int64
    multiplication supplies the required modulo-2**64 wraparound.
    """

    index = torch.arange(count, dtype=torch.int64, device=device)
    golden = -7046029254386353131  # 0x9e3779b97f4a7c15 as signed int64
    rank_mix = -4658895280553007687  # 0xbf58476d1ce4e5b9 as signed int64
    rank_term = ((rank + 1) * rank_mix) & 0xFFFFFFFFFFFFFFFF
    if rank_term >= 1 << 63:
        rank_term -= 1 << 64
    mixed = (index + _REFERENCE_SEED) * golden + rank_term
    selector = (mixed ^ (mixed >> 29)) & 7
    table = torch.tensor(_FP8_CODES, dtype=torch.uint8, device=device)
    return table[selector]


def _pseudo_fp4_codes(
    expert: int, n: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """Reproduce the frozen host's uint32 dense FP4 fixture mixer."""

    mixed = (
        expert * 1315423911
        + n * 2654435761
        + k * 2246822519
        + _REFERENCE_SEED
    ) & 0xFFFFFFFF
    selector = (mixed ^ (mixed >> 13)) & 7
    table = torch.tensor(_FP4_CODES, dtype=torch.uint8, device=n.device)
    return table[selector]


def _dense_fixture_matrix_chunk(
    expert: int,
    n_begin: int,
    n_end: int,
    k_extent: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate and dequantize one N chunk without materializing an expert."""

    n = torch.arange(n_begin, n_end, dtype=torch.int64, device=device)[:, None]
    k = torch.arange(k_extent, dtype=torch.int64, device=device)[None, :]
    codes = _pseudo_fp4_codes(expert, n, k)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=device,
    )
    values = magnitudes[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -values, values)
    exponents = 126 + (expert + n * 3 + (k // 32)) % 4
    scales = (exponents.to(torch.int32) << 23).view(torch.float32)
    return values * scales


def _dense_fixture_mv(
    vector: torch.Tensor,
    expert: int,
    n_extent: int,
    *,
    chunk_n: int,
) -> torch.Tensor:
    pieces = []
    for n_begin in range(0, n_extent, chunk_n):
        n_end = min(n_begin + chunk_n, n_extent)
        matrix = _dense_fixture_matrix_chunk(
            expert, n_begin, n_end, vector.numel(), vector.device
        )
        pieces.append(torch.mv(matrix, vector.float()))
    return torch.cat(pieces)


def deterministic_fixture_inputs(
    rank: int,
    world_size: int,
    active_rows: int,
    *,
    route_mode: str = "balanced",
    mask_period: int = 0,
    hidden: int = 7168,
    top_k: int = 6,
    local_experts: int = 48,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recreate the quantized input/routing tensors of the frozen host."""

    if world_size not in (1, 2, 4, 8):
        raise ValueError("world_size must be one of 1, 2, 4, 8")
    if active_rows < 1 or active_rows > 2048:
        raise ValueError("active_rows must be in [1, 2048]")
    if route_mode not in ("balanced", "skewed", "empty"):
        raise ValueError("unsupported route mode")
    device = torch.device(device)
    x_codes = _pseudo_fp8_codes(active_rows * hidden, rank, device).view(
        active_rows, hidden
    )
    word_index = torch.arange(
        active_rows * (hidden // 128), dtype=torch.int64, device=device
    )
    base = 126 + ((word_index + rank) & 1)
    x_scale_exponents = torch.stack(
        tuple((base + byte).to(torch.uint8) for byte in range(4)), dim=1
    ).view(active_rows, hidden // 32)
    topk_idx = torch.full(
        (active_rows, top_k), -1, dtype=torch.int32, device=device
    )
    topk_weights = torch.zeros(
        (active_rows, top_k), dtype=torch.float32, device=device
    )
    active_experts = world_size * local_experts
    weights = (0.5, 0.25, 0.125, 0.5, 0.25, 0.125)
    for token in range(active_rows):
        for slot in range(top_k):
            route = token * top_k + slot
            if mask_period > 0 and route % mask_period == 0:
                continue
            if route_mode == "balanced":
                expert = (token * 17 + slot * 53 + rank * 97) % active_experts
            elif route_mode == "skewed":
                expert = (token + slot * 3 + rank) % min(8, active_experts)
            else:
                expert = slot
            topk_idx[token, slot] = expert
            topk_weights[token, slot] = weights[slot]
    return x_codes, x_scale_exponents, topk_idx, topk_weights


def reference_deterministic_dense_fixture(
    rank: int,
    world_size: int,
    active_rows: int,
    *,
    route_mode: str = "balanced",
    mask_period: int = 0,
    hidden: int = 7168,
    intermediate: int = 3072,
    output: int = 7168,
    activation_clamp: float = 10.0,
    chunk_n: int = 128,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Pure-PyTorch full-dimension reference for the deterministic fixture.

    Weight chunks are generated from the pinned host formula just before each
    matrix-vector product.  This keeps peak memory bounded while exercising
    every K contribution of the production H7168/I3072/O7168 shape.
    """

    device = torch.device(device)
    x_codes, x_sf, topk_idx, topk_weights = deterministic_fixture_inputs(
        rank,
        world_size,
        active_rows,
        route_mode=route_mode,
        mask_period=mask_period,
        hidden=hidden,
        device=device,
    )
    x = decode_fp8_e4m3(x_codes, x_sf)
    combined = torch.zeros((active_rows, output), dtype=torch.float32, device=device)
    logical_n = torch.arange(intermediate, device=device)
    gate_physical = (logical_n // 8) * 16 + (logical_n & 7)
    up_physical = gate_physical + 8
    for token in range(active_rows):
        for slot in range(topk_idx.shape[1]):
            expert = int(topk_idx[token, slot])
            if expert < 0:
                continue
            w1_bf16 = _dense_fixture_mv(
                x[token], expert, 2 * intermediate, chunk_n=chunk_n
            ).to(torch.bfloat16).float()
            gate = w1_bf16[gate_physical].clamp(max=activation_clamp)
            up = w1_bf16[up_physical].clamp(
                min=-activation_clamp, max=activation_clamp
            )
            routed = (
                torch.nn.functional.silu(gate)
                * up
                * topk_weights[token, slot]
            )
            w2_input = _quantize_intermediate_fp8(routed.unsqueeze(0))[0]
            partial = _dense_fixture_mv(
                w2_input, expert, output, chunk_n=chunk_n
            ).to(torch.bfloat16).float()
            combined[token] += partial
    return combined.to(torch.bfloat16).cpu()
