#!/usr/bin/env python3
"""Validate phase-two evidence and freeze the current SM120 MegaMoE baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


P8_CASES = {
    "world8-r113-distinct-balanced-mask0-c110",
    "world8-r2048-distinct-balanced-mask0-c110",
}
DENSE_CASES = {
    "p1-r1-random-balanced",
    "p2-r1-random-balanced",
    "p2-r2-random-skewed",
    "p2-r2-random-balanced-partial-mask2",
    "p2-r2-random-balanced-all-mask1",
    "p1-r1-zero-activation",
    "p1-r1-zero-w1",
    "p1-r1-zero-w2",
    "p2-r1-random-balanced-topk-permuted",
    "p8-r1-random-balanced",
}
DENSE_COVERAGE = {
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
ZERO_FIELDS = (
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
KERNEL = "kernel_cake_sm120_production_canonical_fused_ready_chunk8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_sparse(
    receipt: dict[str, Any], contract_hash: str, binary_hash: str
) -> dict[str, Any]:
    require(
        receipt.get("schema") == "deepgemm-sm120-megamoe-g8-runtime-correctness-v1",
        "unsupported sparse receipt schema",
    )
    require(receipt.get("status") == "pass", "sparse receipt did not pass")
    require(receipt.get("dynamic_sparse_matrix_passed") is True, "sparse matrix failed")
    require(receipt.get("functional_qualified") is False, "sparse authority was widened")
    require(receipt.get("contract_sha256") == contract_hash, "sparse contract hash changed")
    require(receipt.get("binary_sha256") == binary_hash, "sparse binary hash changed")
    cases = receipt.get("cases")
    require(isinstance(cases, list), "sparse cases are missing")
    require({case.get("id") for case in cases} == P8_CASES, "P8 sparse cases are incomplete")
    require(receipt.get("case_count") == 2, "P8 sparse case count must be two")
    require(receipt.get("rank_record_count") == 16, "P8 sparse rank count must be 16")
    mismatch_sum = 0
    for case in cases:
        require(case.get("status") == "pass", f"sparse case {case.get('id')} failed")
        require(case.get("world_size") == 8, "sparse P8 case has the wrong world size")
        ranks = case.get("rank_records")
        require(isinstance(ranks, list) and len(ranks) == 8, "sparse P8 rank set is incomplete")
        require({rank.get("rank") for rank in ranks} == set(range(8)), "sparse rank ids changed")
        for rank in ranks:
            require(rank.get("status") == "pass", "a sparse rank failed")
            require(rank.get("epoch_slots") == [0, 1, 0], "sparse slot replay changed")
            require(rank.get("exact_bf16_equal") is True, "sparse BF16 result is not exact")
            require(rank.get("output_guard_mismatches") == 0, "sparse output guard failed")
            for field in ZERO_FIELDS:
                value = rank.get(field)
                require(value == 0, f"sparse {field} is {value!r}")
                mismatch_sum += value
    return {
        "case_ids": sorted(P8_CASES),
        "case_count": 2,
        "rank_count": 16,
        "epochs_per_rank": 3,
        "slot_sequence": [0, 1, 0],
        "mismatch_sum": mismatch_sum,
        "exact_bf16": True,
    }


def validate_dense(
    receipt: dict[str, Any], contract_hash: str, binary_hash: str
) -> dict[str, Any]:
    require(
        receipt.get("schema")
        == "deepgemm-sm120-megamoe-stage1-dense-tensor-correctness-v1",
        "unsupported dense receipt schema",
    )
    require(receipt.get("status") == "pass", "dense receipt did not pass")
    require(receipt.get("contract_sha256") == contract_hash, "dense contract hash changed")
    require(
        receipt.get("production_tensor_adapter_complete") is True,
        "production tensor adapter is incomplete",
    )
    require(
        receipt.get("same_exported_inputs_consumed_by_pytorch") is True,
        "PyTorch did not consume the exported production inputs",
    )
    require(receipt.get("independent_dense_matrix_passed") is True, "dense matrix failed")
    require(receipt.get("functional_qualified") is False, "dense authority was widened")
    require(receipt.get("qualified_sources_modified") is False,
            "qualified sources were modified")
    require(receipt.get("binary_sha256") == binary_hash, "dense binary hash changed")
    evidence_sources = receipt.get("evidence_source_sha256", {})
    script_dir = Path(__file__).resolve().parent
    for name in (
        "build_dense_adapter.py",
        "pytorch_oracle.py",
        "run_dense_tensor_correctness.py",
    ):
        require(evidence_sources.get(name) == sha256(script_dir / name),
                f"dense evidence source changed: {name}")
    require(set(receipt.get("required_case_ids", [])) == DENSE_CASES, "dense case plan changed")
    require(set(receipt.get("completed_case_ids", [])) == DENSE_CASES,
            "dense completed case set changed")
    require(set(receipt.get("required_coverage", [])) == DENSE_COVERAGE,
            "dense coverage plan changed")
    require(set(receipt.get("completed_coverage", [])) == DENSE_COVERAGE,
            "dense coverage is incomplete")
    source = receipt.get("adapter_source", {})
    require(
        source.get("schema")
        == "deepgemm-sm120-megamoe-controlled-dense-adapter-source-v1",
        "adapter source manifest is missing",
    )
    require(source.get("qualified_sources_modified") is False,
            "adapter source manifest reports frozen source changes")
    require(source.get("functional_qualified") is False,
            "adapter source manifest widened authority")
    cases = receipt.get("cases")
    require(isinstance(cases, list), "dense cases are missing")
    require({case.get("id") for case in cases} == DENSE_CASES, "dense cases are incomplete")
    require(receipt.get("case_count") == 10, "dense case count must be ten")
    require(receipt.get("rank_record_count") == 22, "dense rank count must be 22")
    tolerances = receipt.get("tolerances", {})
    l2_limit = tolerances.get("relative_l2_max")
    fraction_limit = tolerances.get("max_abs_reference_fraction_max")
    require(isinstance(l2_limit, (int, float)), "dense relative-L2 limit is missing")
    require(isinstance(fraction_limit, (int, float)), "dense max-abs limit is missing")
    route_comparison_count = 0
    output_epoch_count = 0
    bit_mismatches = 0
    max_relative_l2 = 0.0
    max_abs_fraction = 0.0
    for case in cases:
        require(case.get("status") == "pass", f"dense case {case.get('id')} failed")
        ranks = case.get("rank_records")
        require(isinstance(ranks, list) and len(ranks) == case.get("world_size"),
                "dense production rank set is incomplete")
        require({item.get("rank") for item in ranks} == set(range(case["world_size"])),
                "dense production rank ids changed")
        comparisons = case.get("comparisons", {})
        require(comparisons.get("status") == "pass", "dense comparisons failed")
        owners = comparisons.get("owner_stage_comparisons")
        require(isinstance(owners, list) and len(owners) == case.get("world_size"),
                "dense owner comparison set is incomplete")
        require({item.get("rank") for item in owners} == set(range(case["world_size"])),
                "dense owner comparison rank ids changed")
        for owner in owners:
            require(owner.get("semantic_stages_stable") is True,
                    "semantic stages changed across epochs")
            semantic_hashes = owner.get("semantic_stage_sha256_per_epoch")
            require(isinstance(semantic_hashes, list) and len(semantic_hashes) == 3,
                    "semantic epoch hashes are incomplete")
            require(len(set(semantic_hashes)) == 1,
                    "semantic epoch hashes changed")
            for route in owner.get("route_metrics", []):
                route_comparison_count += 1
                require(route.get("status") == "pass", "a dense route comparison failed")
                for stage in ("w1_bf16", "intermediate", "w2_bf16"):
                    stage_metrics = route.get(stage, {})
                    relative_l2 = float(stage_metrics.get("relative_l2", float("inf")))
                    abs_fraction = float(
                        stage_metrics.get("max_abs_reference_fraction", float("inf"))
                    )
                    require(relative_l2 <= l2_limit,
                            f"dense {stage} relative-L2 tolerance failed")
                    require(abs_fraction <= fraction_limit,
                            f"dense {stage} max-abs tolerance failed")
                    max_relative_l2 = max(max_relative_l2, relative_l2)
                    max_abs_fraction = max(max_abs_fraction, abs_fraction)
                    bit_mismatches += int(stage_metrics.get("exact_bf16_mismatches", 0))
        final = comparisons.get("final_comparisons")
        require(isinstance(final, list) and len(final) == case.get("world_size"),
                "dense final comparison set is incomplete")
        require({item.get("rank") for item in final} == set(range(case["world_size"])),
                "dense final comparison rank ids changed")
        for comparison in final:
            require(comparison.get("status") == "pass", "a dense final comparison failed")
            require(
                comparison.get("reference_impl")
                == "pure-pytorch-from-exported-production-inputs",
                "dense oracle is not the independent PyTorch implementation",
            )
            epochs = comparison.get("epochs")
            require(isinstance(epochs, list) and len(epochs) == 3, "dense epochs are incomplete")
            require([epoch.get("epoch") for epoch in epochs] == [0, 1, 2], "dense epochs changed")
            for epoch in epochs:
                output_epoch_count += 1
                require(epoch.get("status") == "pass", "a dense epoch failed")
                relative_l2 = float(epoch.get("relative_l2", float("inf")))
                abs_fraction = float(epoch.get("max_abs_reference_fraction", float("inf")))
                require(relative_l2 <= l2_limit, "dense relative-L2 tolerance failed")
                require(abs_fraction <= fraction_limit, "dense max-abs tolerance failed")
                bit_mismatches += int(epoch.get("exact_bf16_mismatches", 0))
                max_relative_l2 = max(max_relative_l2, relative_l2)
                max_abs_fraction = max(max_abs_fraction, abs_fraction)
    return {
        "case_ids": sorted(DENSE_CASES),
        "case_count": 10,
        "rank_count": 22,
        "route_comparison_count": route_comparison_count,
        "output_epoch_count": output_epoch_count,
        "oracle": "pure-pytorch-from-exported-production-inputs",
        "p8_dense_case": "p8-r1-random-balanced",
        "production_tensor_adapter": True,
        "semantic_stages_stable": True,
        "pass": True,
        "exact_bf16_mismatch_count": bit_mismatches,
        "max_relative_l2": max_relative_l2,
        "max_abs_reference_fraction": max_abs_fraction,
        "tolerances": tolerances,
    }


def validate_nonoverlap(
    receipt: dict[str, Any], dense_receipt_hash: str, dense_binary_hash: str
) -> dict[str, Any]:
    require(
        receipt.get("schema") == "deepgemm-sm120-megamoe-nonoverlap-correctness-v1",
        "unsupported non-overlap receipt schema",
    )
    require(receipt.get("status") == "pass", "non-overlap receipt did not pass")
    require(receipt.get("world_size") == 8, "non-overlap differential is not P8")
    require(receipt.get("rank_count") == 8, "non-overlap rank count must be eight")
    require(receipt.get("case_id") == "world8-r1-dense-balanced", "non-overlap case changed")
    require(
        receipt.get("secondary_non_overlapped_differential_passed") is True,
        "non-overlap differential failed",
    )
    require(receipt.get("deep_ep_dispatch_combine") is True, "DeepEP path was not exercised")
    require(receipt.get("deep_gemm_sm120_grouped_w1_w2") is True, "DeepGEMM path was not exercised")
    require(receipt.get("functional_qualified") is False, "non-overlap authority was widened")
    require(receipt.get("dense_receipt_sha256") == dense_receipt_hash, "dense receipt link changed")
    require(receipt.get("production_binary_sha256") == dense_binary_hash, "production link changed")
    ranks = receipt.get("rank_records")
    require(isinstance(ranks, list) and len(ranks) == 8, "non-overlap ranks are incomplete")
    require({rank.get("rank") for rank in ranks} == set(range(8)), "non-overlap rank ids changed")
    nccl_versions = set()
    for rank in ranks:
        require(rank.get("status") == "pass", "a non-overlap rank failed")
        require(rank.get("world_size") == 8, "a non-overlap rank has the wrong world size")
        require(rank.get("relative_l2") == 0, "non-overlap relative-L2 is not exact")
        require(rank.get("max_abs") == 0, "non-overlap max-abs is not exact")
        require(rank.get("exact_bf16_mismatches") == 0, "non-overlap BF16 mismatch")
        require(rank.get("production_sha256") == rank.get("observed_sha256"),
                "non-overlap output hash mismatch")
        nccl_versions.add(rank.get("nccl_version_integer"))
    require(nccl_versions == {23007}, "non-overlap did not use NCCL 2.30.7")
    return {
        "case_id": "world8-r1-dense-balanced",
        "world_size": 8,
        "rank_count": 8,
        "path": "DeepEP-direct-expanded+DeepGEMM-SM120-grouped",
        "authority": receipt.get("authority"),
        "nccl_version_integer": 23007,
        "exact_bf16_mismatch_count": 0,
        "pass": True,
    }


def validate_performance(receipt: dict[str, Any], kernel_source_hash: str) -> dict[str, Any]:
    require(receipt.get("schema_version") == 1, "unsupported performance receipt schema")
    require(receipt.get("world_size") == 8, "performance baseline is not P8")
    require(receipt.get("active_rows") == 2048, "performance baseline is not R2048")
    require(receipt.get("warmup_launches") == 5, "performance warmup count changed")
    require(receipt.get("repeat_launches") == 20, "performance repeat count changed")
    require(receipt.get("single_launch_full_chain") is True, "performance chain was split")
    require(receipt.get("candidate_source_sha256") == kernel_source_hash,
            "performance kernel source hash changed")
    ranks = receipt.get("per_rank")
    require(isinstance(ranks, list) and len(ranks) == 8, "performance ranks are incomplete")
    require({rank.get("rank") for rank in ranks} == set(range(8)), "performance rank ids changed")
    mismatch_fields = (
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
        "precheck_failures",
        "postcheck_failures",
    )
    for rank in ranks:
        require(rank.get("correctness_status") == "pass", "performance pre/postcheck failed")
        for field in mismatch_fields:
            require(rank.get(field) == 0, f"performance {field} is not zero")
    return {
        "world_size": 8,
        "active_rows": 2048,
        "warmup_launches": 5,
        "repeat_launches": 20,
        "mean_ms": receipt["distributed_mean_ms"],
        "p50_ms": receipt["distributed_p50_ms"],
        "p95_ms": receipt["distributed_p95_ms"],
        "min_ms": receipt["distributed_min_ms"],
        "max_ms": receipt["distributed_max_ms"],
        "aggregate_tensor_tflops": receipt["aggregate_tensor_tflops"],
        "single_launch_full_chain": True,
        "pre_post_correctness": "pass",
    }


def run_output(arguments: list[str]) -> bytes:
    result = subprocess.run(arguments, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout


def audit_resources(
    binary: Path, compiler_log: Path, cuobjdump: Path, resource_snapshot: Path
) -> dict[str, Any]:
    resource_output = run_output([str(cuobjdump), "--dump-resource-usage", str(binary)])
    resource_snapshot.parent.mkdir(parents=True, exist_ok=True)
    resource_snapshot.write_bytes(resource_output)
    resource_text = resource_output.decode(errors="replace")
    pattern = re.compile(
        rf"Function {re.escape(KERNEL)}:\s*\n"
        r"\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)"
    )
    match = pattern.search(resource_text)
    require(match is not None, "production kernel resource entry was not found")
    registers, stack, shared, local = (int(value) for value in match.groups())

    compiler_text = compiler_log.read_text()
    compiler_pattern = re.compile(
        rf"Function properties for {re.escape(KERNEL)}\s*\n"
        r"\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\s*\n"
        r"ptxas info\s*: Used (\d+) registers, used (\d+) barriers, "
        r"(\d+) bytes cumulative stack size, (\d+) bytes smem"
    )
    compiler_match = compiler_pattern.search(compiler_text)
    require(compiler_match is not None, "production kernel ptxas entry was not found")
    (
        ptxas_stack,
        spill_stores,
        spill_loads,
        ptxas_registers,
        barriers,
        cumulative_stack,
        ptxas_smem,
    ) = (int(value) for value in compiler_match.groups())
    require(registers == ptxas_registers, "cuobjdump and ptxas register counts disagree")
    require(stack == ptxas_stack == cumulative_stack, "stack reports disagree")
    sass = run_output([str(cuobjdump), "--dump-sass", str(binary)])
    return {
        "architecture": "sm_120a",
        "kernel": KERNEL,
        "registers": registers,
        "stack_bytes": stack,
        "spill_store_bytes": spill_stores,
        "spill_load_bytes": spill_loads,
        "local_bytes": local,
        "barriers": barriers,
        "cuobjdump_static_shared_bytes": shared,
        "ptxas_smem_bytes": ptxas_smem,
        "dynamic_shared_memory_bytes": 94208,
        "resource_report_sha256": hashlib.sha256(resource_output).hexdigest(),
        "compiler_log_sha256": sha256(compiler_log),
        "full_sass_sha256": hashlib.sha256(sass).hexdigest(),
        "audited": True,
        "historical_register_equivalence_required": False,
    }


def git(repository: Path, *arguments: str) -> str:
    return run_output(["git", "-C", str(repository), *arguments]).decode().strip()


def git_is_ancestor(repository: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    result = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", ancestor, descendant],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def parse_nvcc_release(version_output: str) -> str:
    match = re.search(r"release\s+([0-9]+\.[0-9]+),\s+V([0-9]+\.[0-9]+\.[0-9]+)", version_output)
    require(match is not None, "nvcc version output is not recognized")
    return match.group(2)


def relative_or_absolute(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.resolve()))
    except ValueError:
        return str(path.resolve())


def receipt_record(path: Path, repository: Path) -> dict[str, str]:
    return {"path": relative_or_absolute(path, repository), "sha256": sha256(path)}


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repository = script_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=script_dir / "correctness-contract.json")
    parser.add_argument("--sparse-receipt", type=Path,
                        default=script_dir / "stage2-p8-sparse-qualified-current.json")
    parser.add_argument("--dense-receipt", type=Path,
                        default=script_dir / "stage1-dense-tensor-qualified-current.json")
    parser.add_argument("--nonoverlap-receipt", type=Path,
                        default=script_dir / "stage2-nonoverlap-full-current.json")
    parser.add_argument("--performance-receipt", type=Path, required=True)
    parser.add_argument("--correctness-binary", type=Path, required=True)
    parser.add_argument("--dense-binary", type=Path, required=True)
    parser.add_argument("--performance-binary", type=Path, required=True)
    parser.add_argument("--deepgemm-extension", type=Path, required=True)
    parser.add_argument("--deepep-extension", type=Path, required=True)
    parser.add_argument("--nccl-library", type=Path, required=True)
    parser.add_argument("--performance-compiler-log", type=Path, required=True)
    parser.add_argument(
        "--performance-receipt-snapshot",
        type=Path,
        default=script_dir / "stage2-performance-current.json",
    )
    parser.add_argument(
        "--performance-compiler-log-snapshot",
        type=Path,
        default=script_dir / "stage2-performance-compiler-current.log",
    )
    parser.add_argument(
        "--resource-report-snapshot",
        type=Path,
        default=script_dir / "stage2-performance-resource-current.txt",
    )
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--cuobjdump", type=Path,
                        default=Path("/usr/local/cuda/bin/cuobjdump"))
    parser.add_argument("--deepep-commit",
                        default="01dc3aaac82068020353dce2c302e38153c0bfaa")
    parser.add_argument("--output", type=Path,
                        default=script_dir / "baseline-manifest.json")
    args = parser.parse_args()

    paths = (
        args.contract,
        args.sparse_receipt,
        args.dense_receipt,
        args.nonoverlap_receipt,
        args.performance_receipt,
        args.correctness_binary,
        args.dense_binary,
        args.performance_binary,
        args.deepgemm_extension,
        args.deepep_extension,
        args.nccl_library,
        args.performance_compiler_log,
        args.nvcc,
        args.cuobjdump,
    )
    for path in paths:
        require(path.is_file(), f"required input is missing: {path}")

    contract = load_json(args.contract)
    require(
        contract.get("schema") == "deepgemm-sm120-megamoe-g8-correctness-contract-v1",
        "unsupported correctness contract",
    )
    baseline_commit = contract["baseline"]["repository_commit"]
    require(
        git_is_ancestor(repository, baseline_commit),
        "contract baseline is not an ancestor of the current DeepGEMM commit",
    )
    qualified_manifest_path = repository / contract["baseline"]["qualified_manifest"]
    require(sha256(qualified_manifest_path) == contract["baseline"]["qualified_manifest_sha256"],
            "frozen qualification manifest hash changed")
    qualified_manifest = load_json(qualified_manifest_path)
    qualified_dir = qualified_manifest_path.parent
    source_files = {
        "kernel": qualified_dir / "cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu",
        "correctness_host": qualified_dir
        / "deepgemm_fp8_fp4_mega_moe_sm120_production_canonical_fused_ready_chunk8_host.cu",
        "performance_host": qualified_dir
        / "deepgemm_fp8_fp4_mega_moe_sm120_production_canonical_fused_ready_chunk8_perf_host.cu",
        "direct_math_donor": repository / qualified_manifest["direct_math_donor"]["path"],
    }
    expected_hashes = {
        "kernel": contract["baseline"]["kernel_sha256"],
        "correctness_host": contract["baseline"]["correctness_host_sha256"],
        "performance_host": qualified_manifest["artifacts"][source_files["performance_host"].name],
        "direct_math_donor": qualified_manifest["direct_math_donor"]["sha256"],
    }
    for name, path in source_files.items():
        require(sha256(path) == expected_hashes[name], f"{name} source hash changed")

    correctness_hash = sha256(args.correctness_binary)
    dense_hash = sha256(args.dense_binary)
    sparse_path = args.sparse_receipt.resolve()
    dense_path = args.dense_receipt.resolve()
    nonoverlap_path = args.nonoverlap_receipt.resolve()
    performance_path = args.performance_receipt.resolve()
    sparse = validate_sparse(load_json(sparse_path), sha256(args.contract), correctness_hash)
    dense = validate_dense(load_json(dense_path), sha256(args.contract), dense_hash)
    nonoverlap = validate_nonoverlap(load_json(nonoverlap_path), sha256(dense_path), dense_hash)
    performance = validate_performance(load_json(performance_path), expected_hashes["kernel"])
    args.performance_receipt_snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.performance_receipt_snapshot.write_bytes(performance_path.read_bytes())
    args.performance_compiler_log_snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.performance_compiler_log_snapshot.write_bytes(
        args.performance_compiler_log.read_bytes()
    )
    resources = audit_resources(
        args.performance_binary,
        args.performance_compiler_log_snapshot,
        args.cuobjdump,
        args.resource_report_snapshot,
    )
    nvcc_version = run_output([str(args.nvcc), "--version"]).decode().strip()
    cuda_compiler = parse_nvcc_release(nvcc_version)
    import torch

    manifest = {
        "schema": "deepgemm-sm120-megamoe-stage2-baseline-v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "SM120 DeepSeek-V4 Flash routed-expert MegaMoE, P8 baseline",
        "repository": {
            "deepgemm_commit": git(repository, "rev-parse", "HEAD"),
            "cutlass_commit": git(repository, "rev-parse", "HEAD:third-party/cutlass"),
            "fmt_commit": git(repository, "rev-parse", "HEAD:third-party/fmt"),
            "deepep_commit": args.deepep_commit,
        },
        "toolchain": {
            "cuda_compiler": cuda_compiler,
            "cuda_compiler_full_version": nvcc_version.splitlines()[-2:],
            "cuda_version_is_functional_blocker": False,
            "nccl_runtime": "2.30.7",
            "nccl_version_integer": 23007,
            "nccl_library_sha256": sha256(args.nccl_library),
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "gpu_architecture": "sm_120a",
        },
        "sources": {
            name: {
                "path": relative_or_absolute(path, repository),
                "sha256": expected_hashes[name],
            }
            for name, path in source_files.items()
        },
        "binaries": {
            "correctness": {
                "sha256": correctness_hash,
                "full_sass_sha256": hashlib.sha256(
                    run_output([str(args.cuobjdump), "--dump-sass", str(args.correctness_binary)])
                ).hexdigest(),
                "current_binary_runtime_replayed": True,
                "historical_sass_equivalence_inherited": False,
            },
            "dense_adapter": {"sha256": dense_hash},
            "performance": {
                "sha256": sha256(args.performance_binary),
                "full_sass_sha256": resources["full_sass_sha256"],
            },
            "deepgemm_python_extension": {"sha256": sha256(args.deepgemm_extension)},
            "deepep_python_extension": {"sha256": sha256(args.deepep_extension)},
        },
        "receipts": {
            "correctness_contract": receipt_record(args.contract, repository),
            "p8_sparse_current_binary": receipt_record(sparse_path, repository),
            "dense_independent_pytorch": receipt_record(dense_path, repository),
            "nonoverlap_deepep_deepgemm": receipt_record(nonoverlap_path, repository),
            "performance": receipt_record(args.performance_receipt_snapshot, repository),
            "performance_compiler_log": receipt_record(
                args.performance_compiler_log_snapshot, repository
            ),
            "performance_resource_report": receipt_record(
                args.resource_report_snapshot, repository
            ),
        },
        "correctness": {
            "p8_current_binary_replay": sparse,
            "dense_independent_oracle": dense,
            "secondary_non_overlapped_differential": nonoverlap,
        },
        "resources": resources,
        "performance_baseline": performance,
        "formal_qualification": {
            "functional_qualified": True,
            "p8_current_binary_replay": True,
            "protocol_signal_ack_guard_mismatch": 0,
            "dense_independent_oracle": "pass",
            "secondary_non_overlapped_differential": "pass",
            "resource_audited": True,
            "resource_qualified": False,
            "performance_baseline_frozen": True,
            "performance_qualified": False,
        },
        "authority_notes": {
            "functional_promotion": (
                "This manifest alone combines the three fail-closed receipts and promotes "
                "functional_qualified=true; no individual component receipt widens its authority."
            ),
            "resource_boundary": (
                "Resources are audited for the compiler recorded in this manifest. Cross-version "
                "register/SASS equivalence is not a functional requirement and runtime register "
                "repartition remains outside phase-two functional qualification."
            ),
            "performance_boundary": (
                "The measurement is the frozen optimization baseline, not a claim that an optimized "
                "candidate has passed a performance promotion threshold."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["formal_qualification"], indent=2, sort_keys=True))
    print(f"manifest={args.output.resolve()}")
    print(f"manifest_sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
