#!/usr/bin/env python3
"""Runtime correctness campaign for the frozen SM120 MegaMoE G8 baseline.

This runner does not turn the built-in sparse CUDA oracle into a functional
qualification.  Its receipt keeps the missing independent dense gate explicit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


_RANK_PREFIX = "RANK_RESULT_JSON="
_SUMMARY_PREFIX = "RESULT_JSON="
_ZERO_FIELDS = (
    "protocol_error",
    "owner_mismatches",
    "counter_mismatches",
    "signal_mismatches",
    "ack_signal_mismatches",
    "ready_audit_mismatches",
    "w1_bf16_mismatches",
    "requant_fp8_sf_mismatches",
    "w2_bf16_partial_mismatches",
    "output_mismatches",
    "output_guard_mismatches",
    "launch_mismatches",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_prefixed_json(output: str, prefix: str) -> dict[str, Any]:
    records = []
    for line in output.splitlines():
        if line.startswith(prefix):
            value = json.loads(line[len(prefix) :])
            if not isinstance(value, dict):
                raise ValueError(f"{prefix} payload must be an object")
            records.append(value)
    if len(records) != 1:
        raise ValueError(f"expected one {prefix} record, observed {len(records)}")
    return records[0]


def load_contract(path: Path, repository: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text())
    if contract.get("schema") != "deepgemm-sm120-megamoe-g8-correctness-contract-v1":
        raise ValueError("unsupported correctness contract schema")
    baseline = contract["baseline"]
    manifest_path = repository / baseline["qualified_manifest"]
    if _sha256(manifest_path) != baseline["qualified_manifest_sha256"]:
        raise ValueError("qualified manifest hash does not match the contract")
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest["artifacts"]
    if artifacts["cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu"] != baseline["kernel_sha256"]:
        raise ValueError("kernel hash does not match the qualified manifest")
    host_name = "deepgemm_fp8_fp4_mega_moe_sm120_production_canonical_fused_ready_chunk8_host.cu"
    if artifacts[host_name] != baseline["correctness_host_sha256"]:
        raise ValueError("correctness host hash does not match the qualified manifest")
    cases = contract.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("contract must contain a non-empty case matrix")
    ids = [case.get("id") for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("case ids must be unique")
    return contract


def validate_rank_payload(
    payload: dict[str, Any], case: dict[str, Any], rank: int
) -> None:
    exact = {
        "rank": rank,
        "world_size": case["world_size"],
        "active_rows": case["active_rows"],
        "oracle": case["oracle"],
        "route_mode": case["route_mode"],
        "mask_period": case["mask_period"],
        "epoch_slots": [0, 1, 0],
        "launch_count_per_epoch": 1,
        "kernel_count": 1,
        "requested_ctas": 110,
        "actual_ctas": 110,
        "threads_per_cta": 384,
        "dynamic_smem_bytes": 94208,
        "actual_production_launches": 3,
        "gin_signal_count": 24,
        "service_cta": 0,
        "worker_ctas": 109,
        "status": "pass",
    }
    for field, expected in exact.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"rank {rank} {field} mismatch: {payload.get(field)!r} != {expected!r}"
            )
    required_true = (
        "single_entry",
        "full_shape",
        "direct_canonical_donor",
        "exact_bf16_equal",
        "stage_oracle_installed",
        "post_combine_ack",
        "one_pointer_params",
        "ready_driven",
        "chunked_task_claim",
        "task_major_chunk_issuance",
        "forced_w1_opportunity_after_each_early_w2_chunk",
    )
    for field in required_true:
        if payload.get(field) is not True:
            raise ValueError(f"rank {rank} did not prove {field}")
    required_false = (
        "barrier_ordered",
        "runtime_register_repartition_qualified",
        "resource_qualified",
        "production_compute_comparable",
        "functional_qualified",
    )
    for field in required_false:
        if payload.get(field) is not False:
            raise ValueError(f"rank {rank} weakened the false {field} boundary")
    for field in _ZERO_FIELDS:
        if payload.get(field) != 0:
            raise ValueError(f"rank {rank} failed {field}: {payload.get(field)!r}")
    routes = payload.get("epoch_route_totals")
    expected_routes = payload.get("expected_received_routes")
    if routes != [expected_routes, expected_routes, expected_routes]:
        raise ValueError(f"rank {rank} route totals changed across epochs")
    stages = payload.get("stage_mismatches_per_epoch")
    if stages != [[0, 0, 0], [0, 0, 0], [0, 0, 0]]:
        raise ValueError(f"rank {rank} stage mismatch matrix is not zero")


def validate_summary_payload(
    payload: dict[str, Any], case: dict[str, Any], rank: int
) -> None:
    expected = {
        "rank": rank,
        "world_size": case["world_size"],
        "launch_count": 1,
        "stage_oracle_installed": True,
        "functional_qualified": False,
        "failures": 0,
        "status": "pass",
        "launch": "multi_process",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"rank {rank} summary {field} mismatch: {payload.get(field)!r} != {value!r}"
            )


def _communicate(
    process: subprocess.Popen[str], timeout: float
) -> tuple[str, str]:
    return process.communicate(timeout=timeout)


def run_case(
    binary: Path,
    case: dict[str, Any],
    devices: list[str],
    timeout: float,
) -> dict[str, Any]:
    world_size = int(case["world_size"])
    if len(devices) < world_size:
        raise ValueError(
            f"case {case['id']} requires {world_size} devices, only {len(devices)} supplied"
        )
    selected_devices = devices[:world_size]
    processes: list[subprocess.Popen[str]] = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="sm120-megamoe-g8-") as directory:
        unique_id_path = str(Path(directory) / "nccl.unique-id")
        for rank in range(world_size):
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": ",".join(selected_devices),
                    "WORLD_SIZE": str(world_size),
                    "RANK": str(rank),
                    "LOCAL_DEVICE": str(rank),
                    "NCCL_UNIQUE_ID_FILE": unique_id_path,
                    "CAKE_ACTIVE_ROWS": str(case["active_rows"]),
                    "CAKE_MASK_PERIOD": str(case["mask_period"]),
                    "CAKE_ROUTE_MODE": str(case["route_mode"]),
                    "CAKE_ORACLE": str(case["oracle"]),
                }
            )
            env.setdefault("NCCL_GIN_TYPE", "3")
            env.setdefault("NCCL_NET_PLUGIN", "spcx")
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
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=world_size
            ) as executor:
                futures = [executor.submit(_communicate, process, timeout) for process in processes]
                outputs = [future.result() for future in futures]
        except BaseException:
            for process in processes:
                process.kill()
            raise

    rank_records = []
    output_sha256 = []
    for rank, (process, (stdout, stderr)) in enumerate(zip(processes, outputs)):
        if process.returncode != 0:
            raise RuntimeError(
                f"case {case['id']} rank {rank} exited {process.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        rank_payload = _one_prefixed_json(stdout, _RANK_PREFIX)
        summary_payload = _one_prefixed_json(stdout, _SUMMARY_PREFIX)
        validate_rank_payload(rank_payload, case, rank)
        validate_summary_payload(summary_payload, case, rank)
        rank_records.append(rank_payload)
        output_sha256.append(
            hashlib.sha256((stdout + "\n<STDERR>\n" + stderr).encode()).hexdigest()
        )
    return {
        "id": case["id"],
        "status": "pass",
        "world_size": world_size,
        "devices": selected_devices,
        "elapsed_seconds": time.monotonic() - started,
        "rank_count": len(rank_records),
        "rank_output_sha256": output_sha256,
        "rank_records": rank_records,
    }


def parse_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain unique comma-separated GPU ids")
    return devices


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repository = script_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument(
        "--contract", type=Path, default=script_dir / "correctness-contract.json"
    )
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tier", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    contract = load_contract(args.contract.resolve(), repository)
    cases = contract["cases"]
    if args.case:
        selected = [case for case in cases if case["id"] in set(args.case)]
        missing = set(args.case) - {case["id"] for case in selected}
        if missing:
            parser.error(f"unknown cases: {', '.join(sorted(missing))}")
    elif args.tier == "smoke":
        selected = [case for case in cases if case["tier"] == "smoke"]
    else:
        selected = cases
    if args.list:
        for case in selected:
            print(case["id"])
        return 0
    if args.binary is None:
        parser.error("--binary is required unless --list is used")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"binary is not executable: {binary}")
    devices = parse_devices(args.devices)
    records = [
        run_case(binary, case, devices, args.timeout_seconds) for case in selected
    ]
    receipt = {
        "schema": "deepgemm-sm120-megamoe-g8-runtime-correctness-v1",
        "status": "pass",
        "contract_sha256": _sha256(args.contract.resolve()),
        "binary_sha256": _sha256(binary),
        "case_count": len(records),
        "rank_record_count": sum(record["rank_count"] for record in records),
        "cases": records,
        "dynamic_sparse_matrix_passed": True,
        "independent_dense_matrix_passed": False,
        "functional_qualified": False,
        "qualification_blocker": (
            "production tensor extraction plus dense PyTorch and non-overlapped "
            "differential adapters have not run"
        ),
    }
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
