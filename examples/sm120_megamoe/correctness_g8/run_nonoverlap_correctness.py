#!/usr/bin/env python3
"""Compare the frozen SM120 output with DeepEP + DeepGEMM non-overlap.

This is the secondary differential gate.  It deliberately uses only one
expanded DeepEP dispatch and one combine.  Cached/overlapped DeepEP paths are
outside this baseline contract.  The two routed expert layers use the public
SM120 grouped FP8xFP4 DeepGEMM API; SwiGLU and the K32 intermediate requant are
spelled out in PyTorch so the test does not depend on TileLang.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from pytorch_oracle import deterministic_fixture_inputs


CASE_ID = "world8-r1-dense-balanced"
DENSE_CASE_ID = "p8-r1-random-balanced"
WORLD_SIZE = 8
ACTIVE_ROWS = 1
HIDDEN = 7168
INTERMEDIATE = 3072
OUTPUT = 7168
TOP_K = 6
EXPERTS = 384
LOCAL_EXPERTS = 48
REFERENCE_SEED = 20260819
RELATIVE_L2_LIMIT = 0.01
MAX_ABS_REFERENCE_FRACTION_LIMIT = 0.02


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dense_outputs(receipt_path: Path) -> tuple[str, dict[int, Path]]:
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema")
        != "deepgemm-sm120-megamoe-stage1-dense-tensor-correctness-v1"
    ):
        raise ValueError("unsupported dense receipt schema")
    if receipt.get("independent_dense_matrix_passed") is not True:
        raise ValueError("dense tensor matrix did not pass")
    if receipt.get("same_exported_inputs_consumed_by_pytorch") is not True:
        raise ValueError("dense receipt did not consume exported production inputs")
    cases = [
        case for case in receipt.get("cases", [])
        if case.get("id") == DENSE_CASE_ID
    ]
    if len(cases) != 1 or cases[0].get("status") != "pass":
        raise ValueError(f"dense receipt must contain one passing {DENSE_CASE_ID}")
    case = cases[0]
    if case.get("world_size") != WORLD_SIZE or case.get("active_rows") != ACTIVE_ROWS:
        raise ValueError("dense receipt P8/R1 shape changed")
    paths: dict[int, Path] = {}
    for record in case.get("rank_records", []):
        rank = int(record["rank"])
        epochs = record.get("epochs", [])
        if len(epochs) != 3:
            raise ValueError(f"rank {rank} dense epochs are incomplete")
        outputs = [epoch.get("combine_output", {}) for epoch in epochs]
        if len({value.get("sha256") for value in outputs}) != 1:
            raise ValueError(f"rank {rank} dense output changed across epochs")
        path = Path(outputs[0]["path"])
        if not path.is_file() or sha256_file(path) != outputs[0]["sha256"]:
            raise ValueError(f"rank {rank} production output hash mismatch")
        paths[rank] = path.resolve()
    if set(paths) != set(range(WORLD_SIZE)):
        raise ValueError("dense receipt does not contain all eight ranks")
    return str(receipt["binary_sha256"]), paths


def _logical_to_physical_w1(n: torch.Tensor) -> torch.Tensor:
    logical = n % INTERMEDIATE
    branch = n // INTERMEDIATE
    return (logical // 8) * 16 + (logical & 7) + branch * 8


@torch.inference_mode()
def _fixture_weight(
    rank: int,
    *,
    is_w1: bool,
    n_chunk: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the exact packed FP4/UE8M0 local-expert fixture."""

    n_extent = 2 * INTERMEDIATE if is_w1 else OUTPUT
    k_extent = HIDDEN if is_w1 else INTERMEDIATE
    device = torch.device("cuda")
    packed = torch.empty(
        (LOCAL_EXPERTS, n_extent, k_extent // 2),
        dtype=torch.int8,
        device=device,
    )
    scales = torch.empty(
        (LOCAL_EXPERTS, n_extent, k_extent // 32),
        dtype=torch.float32,
        device=device,
    )
    experts = torch.arange(
        rank * LOCAL_EXPERTS,
        (rank + 1) * LOCAL_EXPERTS,
        dtype=torch.int64,
        device=device,
    )[:, None, None]
    k_pair = torch.arange(k_extent // 2, dtype=torch.int64, device=device)[
        None, None, :
    ]
    k32 = torch.arange(k_extent // 32, dtype=torch.int64, device=device)[
        None, None, :
    ]
    code_table = torch.tensor(
        [1, 2, 3, 4, 5, 6, 9, 10], dtype=torch.uint8, device=device
    )
    for begin in range(0, n_extent, n_chunk):
        end = min(begin + n_chunk, n_extent)
        logical_n = torch.arange(begin, end, dtype=torch.int64, device=device)
        physical_n = (
            _logical_to_physical_w1(logical_n) if is_w1 else logical_n
        )[None, :, None]

        def code(k: torch.Tensor) -> torch.Tensor:
            mixed = (
                experts * 1315423911
                + physical_n * 2654435761
                + k * 2246822519
                + REFERENCE_SEED
            ) & 0xFFFFFFFF
            return code_table[(mixed ^ (mixed >> 13)) & 7]

        low = code(2 * k_pair)
        high = code(2 * k_pair + 1)
        packed_chunk = low | (high << 4)
        packed[:, begin:end].copy_(packed_chunk.view(torch.int8))

        exponents = (
            126 + (experts + physical_n * 3 + k32) % 4
        ).to(torch.int32)
        scales[:, begin:end].copy_((exponents << 23).view(torch.float32))
    return packed, scales


@torch.inference_mode()
def _quantize_intermediate(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the frozen epilogue's FP8/K32 UE8M0 rule."""

    grouped = values.float().view(values.shape[0], -1, 32)
    amax = grouped.abs().amax(dim=2)
    raw = (amax * (1.0 / 448.0)).contiguous()
    bits = raw.view(torch.int32)
    exponent = ((bits >> 23) & 255) + (((bits & 0x7FFFFF) + 0x7FFFFF) >> 23)
    exponent = exponent.clamp(max=254).to(torch.uint8)
    scale = (exponent.to(torch.int32) << 23).view(torch.float32)
    inverse = ((254 - exponent.to(torch.int32)) << 23).view(torch.float32)
    quantized = (grouped * inverse.unsqueeze(2)).to(torch.float8_e4m3fn)
    return quantized.view_as(values).contiguous(), scale.contiguous()


def _nccl_version() -> int:
    root = Path(os.environ["EP_NCCL_ROOT_DIR"])
    library = ctypes.CDLL(str(root / "lib" / "libnccl.so.2"))
    value = ctypes.c_int()
    status = library.ncclGetVersion(ctypes.byref(value))
    if status != 0:
        raise RuntimeError(f"ncclGetVersion failed: {status}")
    return int(value.value)


def _read_bf16(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    if len(raw) != ACTIVE_ROWS * OUTPUT * 2:
        raise ValueError(f"unexpected production output size: {path}")
    return (
        torch.frombuffer(bytearray(raw), dtype=torch.uint16)
        .view(torch.bfloat16)
        .reshape(ACTIVE_ROWS, OUTPUT)
        .cuda()
    )


@torch.inference_mode()
def _worker(
    local_rank: int,
    artifact_dir_text: str,
    production_paths: list[str],
    num_sms: int,
    num_qps: int,
    allow_hybrid_mode: bool,
) -> None:
    import deep_ep
    import deep_gemm
    from deep_gemm.utils.dist import init_dist

    rank, world_size, group = init_dist(local_rank, WORLD_SIZE)
    if rank != local_rank or world_size != WORLD_SIZE:
        raise RuntimeError("unexpected distributed rank mapping")
    artifact_dir = Path(artifact_dir_text)
    buffer = None
    started = time.monotonic()
    try:
        x_codes, x_sf_bytes, topk_idx, topk_weights = deterministic_fixture_inputs(
            rank,
            WORLD_SIZE,
            ACTIVE_ROWS,
            route_mode="balanced",
            mask_period=0,
            device="cuda",
        )
        x = (
            x_codes.view(torch.float8_e4m3fn),
            x_sf_bytes.contiguous().view(torch.int32),
        )
        topk_idx = topk_idx.to(deep_ep.topk_idx_t)
        cumulative_stats = torch.zeros(
            LOCAL_EXPERTS, dtype=torch.int32, device="cuda"
        )
        buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=2048,
            hidden=HIDDEN,
            num_topk=TOP_K,
            use_fp8_dispatch=True,
            allow_hybrid_mode=allow_hybrid_mode,
            allow_multiple_reduction=False,
            num_allocated_qps=num_qps,
            explicitly_destroy=True,
            num_gpu_timeout_secs=180,
            num_cpu_timeout_secs=180,
        )
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        recv_x, _, recv_weights, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            cumulative_local_expert_recv_stats=cumulative_stats,
            num_experts=EXPERTS,
            expert_alignment=alignment,
            num_sms=num_sms,
            num_qps=num_qps,
            do_cpu_sync=False,
            do_handle_copy=False,
            do_expand=True,
            use_tma_aligned_col_major_sf=True,
        )
        num_recv_tokens = int(recv_x[0].shape[0])
        if recv_weights is None or num_recv_tokens <= 0:
            raise RuntimeError("DeepEP expanded dispatch returned no routed rows")

        w1 = _fixture_weight(rank, is_w1=True)
        w2 = _fixture_weight(rank, is_w1=False)
        l1_y = torch.empty(
            (num_recv_tokens, 2 * INTERMEDIATE),
            dtype=torch.bfloat16,
            device="cuda",
        )
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            recv_x,
            w1,
            l1_y,
            handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True,
            recipe=(1, 1, 32),
        )
        gate = l1_y[:, :INTERMEDIATE].float().clamp(max=10.0)
        up = l1_y[:, INTERMEDIATE:].float().clamp(min=-10.0, max=10.0)
        routed = torch.nn.functional.silu(gate) * up
        routed.mul_(recv_weights.reshape(-1, 1).float())
        l2_x = _quantize_intermediate(routed)
        l2_y = torch.empty(
            (num_recv_tokens, OUTPUT), dtype=torch.bfloat16, device="cuda"
        )
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            l2_x,
            w2,
            l2_y,
            handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True,
            recipe=(1, 1, 32),
        )
        observed = buffer.combine(
            l2_y,
            handle=handle,
            num_sms=num_sms,
            num_qps=num_qps,
        )[0]
        torch.cuda.synchronize()
        expected = _read_bf16(Path(production_paths[rank]))
        difference = (observed.float() - expected.float()).abs()
        reference_norm = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        relative_l2 = float(torch.linalg.vector_norm(difference) / reference_norm)
        max_abs = float(difference.max())
        reference_max = float(expected.float().abs().max())
        max_fraction = max_abs / max(reference_max, 1e-12)
        observed_bits = observed.view(torch.uint16).to(torch.int32)
        expected_bits = expected.view(torch.uint16).to(torch.int32)
        exact_mismatches = int(torch.count_nonzero(observed_bits != expected_bits))
        nonzero_count = int(torch.count_nonzero(observed_bits & 0x7FFF))
        passed = (
            relative_l2 <= RELATIVE_L2_LIMIT
            and max_fraction <= MAX_ABS_REFERENCE_FRACTION_LIMIT
            and nonzero_count > 0
        )
        output_bytes = observed.cpu().view(torch.uint16).numpy().tobytes()
        output_path = artifact_dir / f"rank{rank}.nonoverlap.bf16"
        output_path.write_bytes(output_bytes)
        record = {
            "rank": rank,
            "world_size": world_size,
            "status": "pass" if passed else "fail",
            "mode": "DeepEP-direct-expanded+DeepGEMM-SM120-grouped",
            "num_sms": num_sms,
            "num_qps": num_qps,
            "allow_hybrid_mode": allow_hybrid_mode,
            "alignment": alignment,
            "num_recv_tokens": num_recv_tokens,
            "psum_num_recv_tokens_per_expert": (
                handle.psum_num_recv_tokens_per_expert.cpu().tolist()
            ),
            "relative_l2": relative_l2,
            "max_abs": max_abs,
            "max_abs_reference_fraction": max_fraction,
            "exact_bf16_mismatches": exact_mismatches,
            "observed_nonzero_count": nonzero_count,
            "observed_sha256": sha256_bytes(output_bytes),
            "production_sha256": sha256_file(Path(production_paths[rank])),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "deep_ep_version": deep_ep.__version__,
            "deep_gemm_version": deep_gemm.__version__,
            "nccl_version_integer": _nccl_version(),
            "elapsed_seconds": time.monotonic() - started,
        }
        (artifact_dir / f"rank{rank}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        if not passed:
            raise AssertionError(
                f"rank {rank} non-overlap mismatch: relative_l2={relative_l2}, "
                f"max_fraction={max_fraction}"
            )
        dist.barrier()
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-sms", type=int, default=64)
    parser.add_argument("--num-qps", type=int, default=9)
    parser.add_argument("--allow-hybrid-mode", action="store_true")
    args = parser.parse_args()
    if args.num_sms <= 0 or args.num_qps <= 0:
        parser.error("--num-sms and --num-qps must be positive")
    if "EP_NCCL_ROOT_DIR" not in os.environ:
        parser.error("EP_NCCL_ROOT_DIR must pin the DeepEP NCCL runtime")
    dense_receipt = args.dense_receipt.resolve()
    production_binary_sha256, production_paths = _dense_outputs(dense_receipt)
    output = args.output.resolve()
    artifact_dir = output.parent / f"{output.stem}.artifacts" / f"run-{time.time_ns()}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_free_port()),
            "WORLD_SIZE": "1",
            "RANK": "0",
        }
    )
    started = time.monotonic()
    torch.multiprocessing.spawn(
        _worker,
        args=(
            str(artifact_dir),
            [str(production_paths[rank]) for rank in range(WORLD_SIZE)],
            args.num_sms,
            args.num_qps,
            args.allow_hybrid_mode,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )
    rank_records = [
        json.loads((artifact_dir / f"rank{rank}.json").read_text())
        for rank in range(WORLD_SIZE)
    ]
    passed = all(record.get("status") == "pass" for record in rank_records)
    receipt = {
        "schema": "deepgemm-sm120-megamoe-nonoverlap-correctness-v1",
        "status": "pass" if passed else "fail",
        "case_id": CASE_ID,
        "world_size": WORLD_SIZE,
        "active_rows": ACTIVE_ROWS,
        "shape": {
            "experts": EXPERTS,
            "local_experts": LOCAL_EXPERTS,
            "top_k": TOP_K,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "output": OUTPUT,
        },
        "dense_receipt": str(dense_receipt),
        "dense_receipt_sha256": sha256_file(dense_receipt),
        "production_binary_sha256": production_binary_sha256,
        "artifacts_dir": str(artifact_dir),
        "rank_records": rank_records,
        "rank_count": len(rank_records),
        "tolerances": {
            "relative_l2_max": RELATIVE_L2_LIMIT,
            "max_abs_reference_fraction_max": MAX_ABS_REFERENCE_FRACTION_LIMIT,
        },
        "deep_ep_dispatch_combine": passed,
        "deep_gemm_sm120_grouped_w1_w2": passed,
        "secondary_non_overlapped_differential_passed": passed,
        "authority": "secondary-shared-DeepGEMM-compute",
        "functional_qualified": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    json.dump(receipt, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
