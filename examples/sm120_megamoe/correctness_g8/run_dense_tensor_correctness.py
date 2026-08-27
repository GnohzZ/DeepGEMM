#!/usr/bin/env python3
"""Qualify exported SM120 MegaMoE production tensors with pure PyTorch."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


TENSOR_PREFIX = "DENSE_TENSOR_JSON="
RANK_PREFIX = "RANK_RESULT_JSON="
SUMMARY_PREFIX = "RESULT_JSON="
HIDDEN = 7168
INTERMEDIATE = 3072
OUTPUT = 7168
TOP_K = 6
LOCAL_EXPERTS = 48
W1_PHYSICAL_N = 2 * INTERMEDIATE
W1_K_BLOCKS = HIDDEN // 512
W2_K_BLOCKS = INTERMEDIATE // 512
RELATIVE_L2_LIMIT = 0.01
MAX_ABS_REFERENCE_FRACTION_LIMIT = 0.02

CASES = (
    {
        "id": "p1-r1-random-balanced",
        "world_size": 1,
        "active_rows": 1,
        "route_mode": "balanced",
        "mask_period": 0,
        "variant": "random",
        "topk_permute": False,
        "coverage": ("random_dense", "balanced_routing", "p1"),
        "exact_zero": (),
    },
    {
        "id": "p2-r1-random-balanced",
        "world_size": 2,
        "active_rows": 1,
        "route_mode": "balanced",
        "mask_period": 0,
        "variant": "random",
        "topk_permute": False,
        "coverage": ("random_dense", "balanced_routing", "p2"),
        "exact_zero": (),
    },
    {
        "id": "p2-r2-random-skewed",
        "world_size": 2,
        "active_rows": 2,
        "route_mode": "skewed",
        "mask_period": 0,
        "variant": "random",
        "topk_permute": False,
        "coverage": ("random_dense", "skewed_routing"),
        "exact_zero": (),
    },
    {
        "id": "p2-r2-random-balanced-partial-mask2",
        "world_size": 2,
        "active_rows": 2,
        "route_mode": "balanced",
        "mask_period": 2,
        "variant": "random",
        "topk_permute": False,
        "coverage": ("partial_mask",),
        "exact_zero": (),
    },
    {
        "id": "p2-r2-random-balanced-all-mask1",
        "world_size": 2,
        "active_rows": 2,
        "route_mode": "balanced",
        "mask_period": 1,
        "variant": "random",
        "topk_permute": False,
        "coverage": ("all_mask",),
        "exact_zero": ("combine_output",),
    },
    {
        "id": "p1-r1-zero-activation",
        "world_size": 1,
        "active_rows": 1,
        "route_mode": "balanced",
        "mask_period": 0,
        "variant": "zero_activation",
        "topk_permute": False,
        "coverage": ("zero_activation",),
        "exact_zero": ("w1_bf16", "intermediate_fp8", "w2_bf16", "combine_output"),
    },
    {
        "id": "p1-r1-zero-w1",
        "world_size": 1,
        "active_rows": 1,
        "route_mode": "balanced",
        "mask_period": 0,
        "variant": "zero_w1",
        "topk_permute": False,
        "coverage": ("zero_w1",),
        "exact_zero": ("w1_bf16", "intermediate_fp8", "w2_bf16", "combine_output"),
    },
    {
        "id": "p1-r1-zero-w2",
        "world_size": 1,
        "active_rows": 1,
        "route_mode": "balanced",
        "mask_period": 0,
        "variant": "zero_w2",
        "topk_permute": False,
        "coverage": ("zero_w2",),
        "exact_zero": ("w2_bf16", "combine_output"),
    },
    {
        "id": "p2-r1-random-balanced-topk-permuted",
        "world_size": 2,
        "active_rows": 1,
        "route_mode": "balanced",
        "mask_period": 0,
        "variant": "random",
        "topk_permute": True,
        "coverage": ("topk_permutation",),
        "exact_zero": (),
    },
    {
        "id": "p8-r1-random-balanced",
        "world_size": 8,
        "active_rows": 1,
        "route_mode": "balanced",
        "mask_period": 0,
        "variant": "random",
        "topk_permute": False,
        "coverage": ("random_dense", "balanced_routing", "p8"),
        "exact_zero": (),
    },
)
REQUIRED_CASE_IDS = frozenset(case["id"] for case in CASES)
REQUIRED_COVERAGE = frozenset(
    {
        "random_dense",
        "balanced_routing",
        "skewed_routing",
        "partial_mask",
        "all_mask",
        "zero_activation",
        "zero_w1",
        "zero_w2",
        "topk_permutation",
        "repeated_slot_0_1_0",
        "p1",
        "p2",
        "p8",
    }
)
INPUT_NAMES = frozenset(
    {
        "x_fp8",
        "x_sf",
        "topk_idx",
        "topk_weights",
        "active_experts",
        "w1_fp4",
        "w1_sf",
        "w2_fp4",
        "w2_sf",
    }
)
STAGE_NAMES = frozenset(
    {
        "meta_source_rank",
        "meta_token",
        "meta_slot",
        "grouped_layout",
        "w1_bf16",
        "intermediate_fp8",
        "intermediate_sf",
        "w2_bf16",
        "combine_output",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def prefixed_json(output: str, prefix: str) -> list[dict[str, Any]]:
    values = []
    for line in output.splitlines():
        if line.startswith(prefix):
            value = json.loads(line[len(prefix) :])
            require(isinstance(value, dict), f"{prefix} payload must be an object")
            values.append(value)
    return values


def validate_transport(payload: dict[str, Any], case: dict[str, Any], rank: int) -> None:
    expected = {
        "rank": rank,
        "world_size": case["world_size"],
        "active_rows": case["active_rows"],
        "oracle": "dense_external",
        "route_mode": case["route_mode"],
        "mask_period": case["mask_period"],
        "epoch_slots": [0, 1, 0],
        "actual_production_launches": 3,
        "diagnostic_oracle_launches": 0,
        "output_guard_mismatches": 0,
        "protocol_error": 0,
        "owner_mismatches": 0,
        "counter_mismatches": 0,
        "signal_mismatches": 0,
        "ack_signal_mismatches": 0,
        "ready_audit_mismatches": 0,
        "launch_mismatches": 0,
        "functional_qualified": False,
        "status": "pass",
    }
    for field, value in expected.items():
        require(payload.get(field) == value,
                f"rank {rank} transport {field} changed: {payload.get(field)!r}")
    routes = payload.get("epoch_route_totals")
    expected_routes = payload.get("expected_received_routes")
    require(routes == [expected_routes, expected_routes, expected_routes],
            f"rank {rank} route totals changed across epochs")


def _communicate(process: subprocess.Popen[str], timeout: float) -> tuple[str, str]:
    return process.communicate(timeout=timeout)


def validate_tensor_record(record: dict[str, Any], rank: int) -> None:
    require(record.get("rank") == rank, "tensor rank changed")
    require(record.get("authority") == "production-buffer", "tensor authority changed")
    require(isinstance(record.get("shape"), list), "tensor shape is missing")
    require(all(isinstance(value, int) and value >= 0 for value in record["shape"]),
            "tensor shape is invalid")
    path = Path(record.get("path", ""))
    require(path.is_file(), f"tensor is missing: {path}")
    require(path.stat().st_size == record.get("bytes"), f"tensor size changed: {path}")
    require(sha256(path) == record.get("sha256"), f"tensor hash changed: {path}")


def launch_case(
    binary: Path,
    case: dict[str, Any],
    devices: list[str],
    output_prefix: Path,
    timeout: float,
) -> list[dict[str, Any]]:
    world_size = int(case["world_size"])
    require(len(devices) >= world_size, f"{case['id']} requires {world_size} GPUs")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    selected = devices[:world_size]
    unique_id = output_prefix.with_suffix(".nccl-id")
    processes: list[subprocess.Popen[str]] = []
    for rank in range(world_size):
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": ",".join(selected),
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
                "LOCAL_DEVICE": str(rank),
                "NCCL_UNIQUE_ID_FILE": str(unique_id),
                "NCCL_GIN_TYPE": env.get("NCCL_GIN_TYPE", "3"),
                "NCCL_NET_PLUGIN": env.get("NCCL_NET_PLUGIN", "spcx"),
                "CAKE_ACTIVE_ROWS": str(case["active_rows"]),
                "CAKE_MASK_PERIOD": str(case["mask_period"]),
                "CAKE_ROUTE_MODE": str(case["route_mode"]),
                "CAKE_ORACLE": "dense_external",
                "CAKE_DENSE_VARIANT": str(case["variant"]),
                "CAKE_DENSE_TOPK_PERMUTE": "1" if case["topk_permute"] else "0",
                "CAKE_DENSE_OUTPUT_PREFIX": str(output_prefix),
            }
        )
        processes.append(
            subprocess.Popen(
                [str(binary)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=world_size) as executor:
            futures = [executor.submit(_communicate, process, timeout) for process in processes]
            outputs = [future.result() for future in futures]
    except BaseException:
        for process in processes:
            process.kill()
        raise

    rank_records = []
    for rank, (process, (stdout, stderr)) in enumerate(zip(processes, outputs)):
        if process.returncode != 0:
            raise RuntimeError(
                f"{case['id']} rank {rank} exited {process.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        ranks = prefixed_json(stdout, RANK_PREFIX)
        summaries = prefixed_json(stdout, SUMMARY_PREFIX)
        tensors = prefixed_json(stdout, TENSOR_PREFIX)
        require(len(ranks) == len(summaries) == 1, f"rank {rank} summary is incomplete")
        validate_transport(ranks[0], case, rank)
        summary = summaries[0]
        for field, value in {
            "rank": rank,
            "world_size": world_size,
            "failures": 0,
            "functional_qualified": False,
            "status": "pass",
            "launch": "multi_process",
        }.items():
            require(summary.get(field) == value, f"rank {rank} summary {field} changed")

        records: dict[tuple[str, int], dict[str, Any]] = {}
        for tensor in tensors:
            validate_tensor_record(tensor, rank)
            key = (str(tensor.get("name")), int(tensor.get("epoch", -2)))
            require(key not in records, f"rank {rank} duplicated tensor {key}")
            records[key] = tensor
        expected_keys = {(name, -1) for name in INPUT_NAMES}
        expected_keys.update((name, epoch) for name in STAGE_NAMES for epoch in range(3))
        require(set(records) == expected_keys, f"rank {rank} tensor set is incomplete")

        output_hashes = [
            records[("combine_output", epoch)]["sha256"] for epoch in range(3)
        ]
        require(len(set(output_hashes)) == 1,
                f"rank {rank} combine output changed across [0,1,0]")
        rank_records.append(
            {
                "rank": rank,
                "transport": ranks[0],
                "inputs": {name: records[(name, -1)] for name in sorted(INPUT_NAMES)},
                "epochs": [
                    {name: records[(name, epoch)] for name in sorted(STAGE_NAMES)}
                    for epoch in range(3)
                ],
                "stdout_stderr_sha256": hashlib.sha256(
                    (stdout + "\n<STDERR>\n" + stderr).encode()
                ).hexdigest(),
            }
        )
    return rank_records


def load_oracle(script_dir: Path):
    path = script_dir / "pytorch_oracle.py"
    spec = importlib.util.spec_from_file_location("sm120_tensor_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tensor(torch: Any, record: dict[str, Any]) -> Any:
    shape = tuple(record["shape"])
    count = math.prod(shape)
    dtype_name = record["dtype"]
    if count == 0:
        dtype = torch.int32 if dtype_name == "int32" else torch.uint8
        return torch.empty(shape, dtype=dtype)
    if dtype_name == "int32":
        tensor = torch.from_file(record["path"], shared=False, size=count, dtype=torch.int32)
    elif dtype_name == "float32":
        tensor = torch.from_file(record["path"], shared=False, size=count, dtype=torch.float32)
    elif dtype_name == "bf16-le":
        tensor = torch.from_file(record["path"], shared=False, size=count, dtype=torch.uint16)
        tensor = tensor.view(torch.bfloat16)
    else:
        tensor = torch.from_file(record["path"], shared=False, size=count, dtype=torch.uint8)
    return tensor.reshape(shape)


def numeric_metrics(torch: Any, observed: Any, expected: Any) -> dict[str, Any]:
    require(tuple(observed.shape) == tuple(expected.shape), "numeric shapes differ")
    observed_f = observed.float()
    expected_f = expected.float()
    difference = (observed_f - expected_f).abs()
    denominator = torch.linalg.vector_norm(expected_f).clamp_min(1e-12)
    relative_l2 = float(torch.linalg.vector_norm(observed_f - expected_f) / denominator)
    max_abs = float(difference.max()) if difference.numel() else 0.0
    reference_max = float(expected_f.abs().max()) if expected_f.numel() else 0.0
    max_fraction = max_abs / max(reference_max, 1e-12)
    metrics = {
        "relative_l2": relative_l2,
        "max_abs": max_abs,
        "max_abs_reference_fraction": max_fraction,
        "status": "pass",
    }
    require(relative_l2 <= RELATIVE_L2_LIMIT, f"relative-L2 failed: {relative_l2}")
    require(max_fraction <= MAX_ABS_REFERENCE_FRACTION_LIMIT,
            f"max-abs fraction failed: {max_fraction}")
    if observed.dtype == torch.bfloat16 and expected.dtype == torch.bfloat16:
        observed_bits = observed.view(torch.uint16).to(torch.int32)
        expected_bits = expected.view(torch.uint16).to(torch.int32)
        metrics["exact_bf16_mismatches"] = int(
            torch.count_nonzero(observed_bits != expected_bits)
        )
    return metrics


def exact_zero_count(torch: Any, tensor: Any) -> int:
    if tensor.dtype == torch.bfloat16:
        bits = tensor.view(torch.uint16).to(torch.int32)
        return int(torch.count_nonzero(bits & 0x7FFF))
    if tensor.dtype == torch.uint8:
        return int(torch.count_nonzero(tensor & 0x7F))
    return int(torch.count_nonzero(tensor))


def unpack_weight_scales(raw: Any, *, n_extent: int) -> Any:
    require(raw.ndim == 4 and raw.shape[2] == n_extent and raw.shape[3] == 4,
            "packed weight-scale shape changed")
    return raw.permute(0, 2, 1, 3).reshape(
        raw.shape[0], n_extent, raw.shape[1] * raw.shape[3]
    )


def tensor_sha256(torch: Any, *tensors: Any) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().contiguous().cpu().view(torch.uint8)
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def compare_case(
    case: dict[str, Any], rank_records: list[dict[str, Any]], script_dir: Path
) -> dict[str, Any]:
    oracle = load_oracle(script_dir)
    torch = oracle.torch
    inputs = []
    expert_locations: dict[int, tuple[int, int]] = {}
    for record in rank_records:
        loaded = {name: load_tensor(torch, tensor) for name, tensor in record["inputs"].items()}
        loaded["w1_sf_unpacked"] = unpack_weight_scales(
            loaded["w1_sf"], n_extent=W1_PHYSICAL_N
        )
        loaded["w2_sf_unpacked"] = unpack_weight_scales(
            loaded["w2_sf"], n_extent=OUTPUT
        )
        for index, expert in enumerate(loaded["active_experts"].tolist()):
            require(expert not in expert_locations, f"expert {expert} was dumped twice")
            expert_locations[int(expert)] = (record["rank"], index)
        inputs.append(loaded)

    def compare_owner(record: dict[str, Any]) -> dict[str, Any]:
        rank = int(record["rank"])
        device = torch.device(f"cuda:{rank}")
        route_metrics = []
        partials = []
        canonical_stages: dict[tuple[int, int, int, int], str] | None = None
        epoch_semantic_stage_sha256 = []
        epoch_route_counts = []
        total_padded_rows = []
        for epoch_index, epoch_records in enumerate(record["epochs"]):
            epoch = {
                name: load_tensor(torch, tensor)
                for name, tensor in epoch_records.items()
            }
            source_ranks = epoch["meta_source_rank"]
            tokens = epoch["meta_token"]
            slots = epoch["meta_slot"]
            local_experts = epoch["grouped_layout"]
            semantic_rows = (source_ranks >= 0).nonzero().flatten().tolist()
            semantic_stages: dict[tuple[int, int, int, int], str] = {}
            for row in semantic_rows:
                source = int(source_ranks[row])
                token = int(tokens[row])
                slot = int(slots[row])
                expert = rank * LOCAL_EXPERTS + int(local_experts[row])
                route_key = (source, token, slot, expert)
                require(route_key not in semantic_stages,
                        f"duplicated semantic route {route_key}")
                source_input = inputs[source]
                require(int(source_input["topk_idx"][token, slot, 0]) == expert,
                        "production route metadata does not match exported top-k")
                require(int(source_input["topk_idx"][token, slot, 1]) == 0,
                        "top-k high word changed")
                observed_w1 = epoch["w1_bf16"][row]
                observed_codes = epoch["intermediate_fp8"][row]
                observed_sf = epoch["intermediate_sf"][row]
                observed_w2 = epoch["w2_bf16"][row]
                semantic_stages[route_key] = tensor_sha256(
                    torch, observed_w1, observed_codes, observed_sf, observed_w2
                )
                if epoch_index != 0:
                    continue
                require(expert in expert_locations, f"expert {expert} weights are missing")
                weight_rank, weight_index = expert_locations[expert]
                require(weight_rank == rank, f"expert {expert} was dumped by the wrong owner")
                owner_input = inputs[weight_rank]
                stages = oracle.reference_route_stages(
                    source_input["x_fp8"][token].to(device),
                    source_input["x_sf"][token].to(device),
                    source_input["topk_weights"][token, slot].to(device),
                    owner_input["w1_fp4"][weight_index].to(device),
                    owner_input["w1_sf_unpacked"][weight_index].to(device),
                    owner_input["w2_fp4"][weight_index].to(device),
                    owner_input["w2_sf_unpacked"][weight_index].to(device),
                )
                observed_w1_gpu = observed_w1.to(device)
                observed_codes_gpu = observed_codes.to(device)
                observed_sf_gpu = observed_sf.to(device)
                observed_intermediate = oracle.decode_fp8_e4m3(
                    observed_codes_gpu.unsqueeze(0), observed_sf_gpu.unsqueeze(0),
                    gran_k=32,
                )[0]
                observed_w2_gpu = observed_w2.to(device)
                metrics = {
                    "source_rank": source,
                    "token": token,
                    "slot": slot,
                    "expert": expert,
                    "w1_bf16": numeric_metrics(
                        torch, observed_w1_gpu, stages["w1_bf16"]
                    ),
                    "intermediate": numeric_metrics(
                        torch, observed_intermediate, stages["intermediate_dequant"]
                    ),
                    "intermediate_fp8_code_mismatches": int(
                        torch.count_nonzero(
                            observed_codes_gpu != stages["intermediate_fp8"]
                        )
                    ),
                    "intermediate_sf_mismatches": int(
                        torch.count_nonzero(
                            observed_sf_gpu != stages["intermediate_sf"]
                        )
                    ),
                    "w2_bf16": numeric_metrics(
                        torch, observed_w2_gpu, stages["w2_bf16"]
                    ),
                    "status": "pass",
                }
                for name in case["exact_zero"]:
                    if name == "w1_bf16":
                        require(exact_zero_count(torch, observed_w1_gpu) == 0,
                                "W1 is not exact zero")
                    elif name == "intermediate_fp8":
                        require(exact_zero_count(torch, observed_codes_gpu) == 0,
                                "intermediate is not exact zero")
                    elif name == "w2_bf16":
                        require(exact_zero_count(torch, observed_w2_gpu) == 0,
                                "W2 is not exact zero")
                route_metrics.append(metrics)
                partials.append(
                    {
                        "source_rank": source,
                        "token": token,
                        "slot": slot,
                        "value": stages["w2_bf16"].cpu(),
                    }
                )
            if canonical_stages is None:
                canonical_stages = semantic_stages
            else:
                require(semantic_stages == canonical_stages,
                        f"rank {rank} semantic stages changed in epoch {epoch_index}")
            serialized_stages = sorted(
                ([*key], value) for key, value in semantic_stages.items()
            )
            epoch_semantic_stage_sha256.append(
                hashlib.sha256(
                    json.dumps(serialized_stages, separators=(",", ":")).encode()
                ).hexdigest()
            )
            epoch_route_counts.append(len(semantic_rows))
            total_padded_rows.append(int(source_ranks.numel()))
        return {
            "rank": rank,
            "total_padded_rows_per_epoch": total_padded_rows,
            "semantic_route_count_per_epoch": epoch_route_counts,
            "semantic_stage_sha256_per_epoch": epoch_semantic_stage_sha256,
            "semantic_stages_stable": len(set(epoch_semantic_stage_sha256)) == 1,
            "route_metrics": route_metrics,
            "partials": partials,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rank_records)) as executor:
        owner_results = list(executor.map(compare_owner, rank_records))

    combined = [
        torch.zeros((case["active_rows"], OUTPUT), dtype=torch.float32)
        for _ in range(case["world_size"])
    ]
    for owner in owner_results:
        for partial in owner.pop("partials"):
            combined[partial["source_rank"]][partial["token"]] += partial["value"].float()

    final_metrics = []
    output_hashes = []
    for source, record in enumerate(rank_records):
        expected = combined[source].to(torch.bfloat16)
        epoch_metrics = []
        for epoch_index, epoch in enumerate(record["epochs"]):
            observed = load_tensor(torch, epoch["combine_output"])
            metrics = numeric_metrics(torch, observed, expected)
            if "combine_output" in case["exact_zero"]:
                require(exact_zero_count(torch, observed) == 0,
                        "combine output is not exact zero")
            epoch_metrics.append({"epoch": epoch_index, **metrics})
        expected_sha = hashlib.sha256(expected.view(torch.uint16).numpy().tobytes()).hexdigest()
        final_metrics.append(
            {
                "rank": source,
                "reference_impl": "pure-pytorch-from-exported-production-inputs",
                "reference_sha256": expected_sha,
                "epochs": epoch_metrics,
                "status": "pass",
            }
        )
        output_hashes.append(record["epochs"][0]["combine_output"]["sha256"])

    input_hashes = sorted(
        (record["rank"], name, tensor["sha256"])
        for record in rank_records
        for name, tensor in record["inputs"].items()
    )
    stage_hashes = sorted(
        (record["rank"], epoch_index, name, tensor["sha256"])
        for record in rank_records
        for epoch_index, epoch in enumerate(record["epochs"])
        for name, tensor in epoch.items()
    )
    return {
        "owner_stage_comparisons": sorted(owner_results, key=lambda item: item["rank"]),
        "final_comparisons": final_metrics,
        "input_hash_aggregate": hashlib.sha256(
            json.dumps(input_hashes, separators=(",", ":")).encode()
        ).hexdigest(),
        "stage_output_hash_aggregate": hashlib.sha256(
            json.dumps(stage_hashes, separators=(",", ":")).encode()
        ).hexdigest(),
        "production_output_sha256": output_hashes,
        "status": "pass",
    }


def compare_topk_permutation(torch: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {case["id"]: case for case in cases}
    base = by_id["p2-r1-random-balanced"]
    permuted = by_id["p2-r1-random-balanced-topk-permuted"]
    rank_metrics = []
    for rank in range(2):
        base_tensor = load_tensor(torch, base["rank_records"][rank]["epochs"][0]["combine_output"])
        permuted_tensor = load_tensor(
            torch, permuted["rank_records"][rank]["epochs"][0]["combine_output"]
        )
        rank_metrics.append({"rank": rank, **numeric_metrics(torch, permuted_tensor, base_tensor)})
        require(
            base["rank_records"][rank]["inputs"]["x_fp8"]["sha256"]
            == permuted["rank_records"][rank]["inputs"]["x_fp8"]["sha256"],
            "top-k metamorphic pair changed activation input",
        )
        require(
            base["rank_records"][rank]["inputs"]["topk_idx"]["sha256"]
            != permuted["rank_records"][rank]["inputs"]["topk_idx"]["sha256"],
            "top-k permutation did not change top-k input",
        )
    return {"pair": [base["id"], permuted["id"]], "rank_metrics": rank_metrics, "status": "pass"}


def parse_devices(value: str) -> list[str]:
    devices = [part.strip() for part in value.split(",") if part.strip()]
    require(devices and len(set(devices)) == len(devices),
            "--devices must contain unique comma-separated ids")
    return devices


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--contract", type=Path, default=script_dir / "correctness-contract.json"
    )
    parser.add_argument("--adapter-source-manifest", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    selected = list(CASES)
    if args.case:
        requested = set(args.case)
        selected = [case for case in CASES if case["id"] in requested]
        missing = requested - {case["id"] for case in selected}
        if missing:
            parser.error(f"unknown cases: {', '.join(sorted(missing))}")
    if args.list:
        for case in selected:
            print(case["id"])
        return 0
    binary = args.binary.resolve()
    require(binary.is_file() and os.access(binary, os.X_OK), f"binary is not executable: {binary}")
    contract = load_json(args.contract)
    require(
        contract.get("schema") == "deepgemm-sm120-megamoe-g8-correctness-contract-v1",
        "unsupported correctness contract",
    )
    adapter_source = None
    if args.adapter_source_manifest is not None:
        adapter_source = load_json(args.adapter_source_manifest)
        require(
            adapter_source.get("schema")
            == "deepgemm-sm120-megamoe-controlled-dense-adapter-source-v1",
            "unsupported adapter source manifest",
        )
        require(adapter_source.get("qualified_sources_modified") is False,
                "qualified sources were modified")
        require(adapter_source.get("functional_qualified") is False,
                "adapter source manifest widened authority")
    devices = parse_devices(args.devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    if args.artifacts_dir is not None:
        root = args.artifacts_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(tempfile.mkdtemp(prefix="sm120-megamoe-stage1-dense-"))

    started = time.monotonic()
    case_receipts = []
    for case in selected:
        case_started = time.monotonic()
        rank_records = launch_case(
            binary, case, devices, root / case["id"] / "production", args.timeout_seconds
        )
        comparisons = compare_case(case, rank_records, script_dir)
        case_receipts.append(
            {
                **case,
                "coverage": list(case["coverage"]),
                "exact_zero": list(case["exact_zero"]),
                "rank_records": rank_records,
                "comparisons": comparisons,
                "elapsed_seconds": time.monotonic() - case_started,
                "status": "pass",
            }
        )

    completed_ids = {case["id"] for case in case_receipts}
    coverage = {item for case in case_receipts for item in case["coverage"]}
    if completed_ids == REQUIRED_CASE_IDS:
        coverage.add("repeated_slot_0_1_0")
    oracle = load_oracle(script_dir)
    topk_permutation = (
        compare_topk_permutation(oracle.torch, case_receipts)
        if {
            "p2-r1-random-balanced",
            "p2-r1-random-balanced-topk-permuted",
        }.issubset(completed_ids)
        else None
    )
    matrix_complete = completed_ids == REQUIRED_CASE_IDS and coverage == REQUIRED_COVERAGE
    if matrix_complete:
        require(adapter_source is not None,
                "the full matrix requires --adapter-source-manifest")
    matrix_passed = matrix_complete and adapter_source is not None
    receipt = {
        "schema": "deepgemm-sm120-megamoe-stage1-dense-tensor-correctness-v1",
        "status": "pass",
        "evidence_source_sha256": {
            name: sha256(script_dir / name)
            for name in (
                "build_dense_adapter.py",
                "pytorch_oracle.py",
                "run_dense_tensor_correctness.py",
            )
        },
        "binary_sha256": sha256(binary),
        "contract_sha256": sha256(args.contract),
        "adapter_source_manifest_sha256": (
            sha256(args.adapter_source_manifest)
            if args.adapter_source_manifest is not None
            else None
        ),
        "adapter_source": adapter_source,
        "artifacts_dir": str(root),
        "artifacts_retained": True,
        "case_count": len(case_receipts),
        "rank_record_count": sum(len(case["rank_records"]) for case in case_receipts),
        "required_case_ids": sorted(REQUIRED_CASE_IDS),
        "completed_case_ids": sorted(completed_ids),
        "required_coverage": sorted(REQUIRED_COVERAGE),
        "completed_coverage": sorted(coverage),
        "production_tensor_adapter_complete": matrix_passed,
        "same_exported_inputs_consumed_by_pytorch": matrix_passed,
        "independent_dense_matrix_passed": matrix_passed,
        "qualified_sources_modified": (
            adapter_source.get("qualified_sources_modified")
            if adapter_source is not None
            else None
        ),
        "topk_permutation_metamorphic": topk_permutation,
        "tolerances": {
            "relative_l2_max": RELATIVE_L2_LIMIT,
            "max_abs_reference_fraction_max": MAX_ABS_REFERENCE_FRACTION_LIMIT,
        },
        "cases": case_receipts,
        "functional_qualified": False,
        "qualification_blockers": [
            "current-binary P8 sparse replay is a separate Stage-2 gate",
            "the P8 DeepEP+DeepGEMM differential is a separate Stage-2 gate",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "case_count": receipt["case_count"],
                    "rank_record_count": receipt["rank_record_count"],
                    "independent_dense_matrix_passed": receipt[
                        "independent_dense_matrix_passed"
                    ],
                    "receipt": str(args.output.resolve()),
                },
                sort_keys=True,
            )
        )
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
