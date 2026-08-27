import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "examples" / "sm120_megamoe" / "correctness_g8"
RUNNER_PATH = CAMPAIGN / "run_correctness_matrix.py"
SPEC = importlib.util.spec_from_file_location("sm120_g8_correctness", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _load_freezer():
    path = CAMPAIGN / "freeze_baseline.py"
    spec = importlib.util.spec_from_file_location("sm120_g8_freezer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_correctness_contract_pins_frozen_baseline() -> None:
    contract = RUNNER.load_contract(
        CAMPAIGN / "correctness-contract.json", ROOT
    )
    assert contract["promotion_policy"] == {
        "static_pytest_is_functional_qualification": False,
        "dynamic_sparse_matrix_required": True,
        "independent_dense_matrix_required": True,
        "secondary_non_overlapped_differential_required": True,
        "all_world8_cases_must_be_runtime_replayed": True,
        "historical_sass_equivalence_is_runtime_replay": False,
        "functional_qualified_until_all_requirements_pass": False,
    }
    assert contract["oracle_boundary"]["dense_external_execution_present"] is True
    assert contract["dense_matrix_plan"]["required_primary_pytorch_cases"] == [
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
    ]
    assert [case["id"] for case in contract["cases"]] == [
        "world1-r1-distinct-balanced-c110",
        "world2-r1-distinct-balanced-c110",
        "world2-r1-zero-balanced-mask1-c110",
        "world2-r17-analytic-balanced-mask7-c110",
        "world2-r17-analytic-empty-mask0-c110",
        "world2-r17-analytic-skewed-mask0-c110",
        "world4-r128-distinct-balanced-mask0-c110",
        "world8-r113-distinct-balanced-mask0-c110",
        "world8-r2048-distinct-balanced-mask0-c110",
    ]


def _rank_payload() -> dict:
    return {
        "rank": 0,
        "world_size": 1,
        "active_rows": 1,
        "oracle": "distinct_k32",
        "route_mode": "balanced",
        "mask_period": 0,
        "epoch_slots": [0, 1, 0],
        "epoch_route_totals": [6, 6, 6],
        "expected_received_routes": 6,
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
        "stage_mismatches_per_epoch": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        **{field: 0 for field in RUNNER._ZERO_FIELDS},
        **{
            field: True
            for field in (
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
        },
        **{
            field: False
            for field in (
                "barrier_ordered",
                "runtime_register_repartition_qualified",
                "resource_qualified",
                "production_compute_comparable",
                "functional_qualified",
            )
        },
    }


def test_rank_validator_is_fail_closed() -> None:
    case = json.loads((CAMPAIGN / "correctness-contract.json").read_text())["cases"][0]
    payload = _rank_payload()
    RUNNER.validate_rank_payload(payload, case, 0)

    payload["signal_mismatches"] = 1
    with pytest.raises(ValueError, match="signal_mismatches"):
        RUNNER.validate_rank_payload(payload, case, 0)


def test_rank_validator_does_not_promote_sparse_oracle() -> None:
    case = json.loads((CAMPAIGN / "correctness-contract.json").read_text())["cases"][0]
    payload = _rank_payload()
    payload["functional_qualified"] = True
    with pytest.raises(ValueError, match="functional_qualified"):
        RUNNER.validate_rank_payload(payload, case, 0)


def _load_pytorch_oracle():
    path = CAMPAIGN / "pytorch_oracle.py"
    spec = importlib.util.spec_from_file_location("sm120_g8_pytorch_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pytorch_oracle_decodes_fp4_and_ue8m0_independently() -> None:
    oracle = _load_pytorch_oracle()
    packed = torch.tensor([[0x21, 0xA6]], dtype=torch.uint8)
    scales = torch.tensor([[127]], dtype=torch.uint8)
    decoded = oracle.decode_packed_fp4_e2m1(packed, scales, gran_k=4)
    assert decoded.tolist() == [[0.5, 1.0, 4.0, -1.0]]


def test_pytorch_oracle_composes_tiny_zero_route() -> None:
    oracle = _load_pytorch_oracle()
    hidden = intermediate = output = 32
    x = torch.full((1, hidden), 0x38, dtype=torch.uint8)
    x_sf = torch.full((1, 1), 127, dtype=torch.uint8)
    topk_idx = torch.tensor([[-1]], dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)
    w1 = torch.zeros((1, 2 * intermediate, hidden // 2), dtype=torch.uint8)
    w1_sf = torch.full((1, 2 * intermediate, 1), 127, dtype=torch.uint8)
    w2 = torch.zeros((1, output, intermediate // 2), dtype=torch.uint8)
    w2_sf = torch.full((1, output, 1), 127, dtype=torch.uint8)
    observed = oracle.reference_megamoe(
        x, x_sf, topk_idx, topk_weights, w1, w1_sf, w2, w2_sf
    )
    assert observed.dtype == torch.bfloat16
    assert observed.shape == (1, output)
    assert torch.count_nonzero(observed) == 0


def test_pytorch_oracle_composes_tiny_dense_nonzero_route() -> None:
    oracle = _load_pytorch_oracle()
    hidden = intermediate = output = 32
    # FP8 0x38 is 1.0 and FP4 nibble 0x1 is 0.5.  With unit UE8M0
    # scales, every W1 dot is 16.0, the clamp makes both SwiGLU inputs
    # 10.0, the intermediate quantize/dequantize value is 96.0, and
    # every W2 dot rounds to BF16 1536.0.
    x = torch.full((1, hidden), 0x38, dtype=torch.uint8)
    x_sf = torch.full((1, 1), 127, dtype=torch.uint8)
    topk_idx = torch.tensor([[0]], dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)
    w1 = torch.full(
        (1, 2 * intermediate, hidden // 2), 0x11, dtype=torch.uint8
    )
    w1_sf = torch.full((1, 2 * intermediate, 1), 127, dtype=torch.uint8)
    w2 = torch.full((1, output, intermediate // 2), 0x11, dtype=torch.uint8)
    w2_sf = torch.full((1, output, 1), 127, dtype=torch.uint8)
    observed = oracle.reference_megamoe(
        x, x_sf, topk_idx, topk_weights, w1, w1_sf, w2, w2_sf
    )
    torch.testing.assert_close(
        observed.float(), torch.full((1, output), 1536.0), rtol=0, atol=0
    )


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dense_adapter_derivation_is_hash_guarded_and_scoped() -> None:
    builder = _load_module("sm120_dense_builder", CAMPAIGN / "build_dense_adapter.py")
    qualified = ROOT / "examples" / "sm120_megamoe" / "qualified_g8"
    source = (qualified / builder.HOST).read_text()
    derived = builder.derive_host(source)
    assert "GENERATED CONTROLLED DENSE ADAPTER" in derived
    assert "CAKE_DENSE_OUTPUT_PREFIX" in derived
    assert '"x_fp8"' in derived
    assert '"w1_fp4"' in derived
    assert '"w1_bf16"' in derived
    assert '"intermediate_fp8"' in derived
    assert '"intermediate_sf"' in derived
    assert '"w2_bf16"' in derived
    assert '"combine_output"' in derived
    assert '"dense_external"' in derived
    assert "dense_external oracle is not installed" not in derived
    assert (qualified / builder.HOST).read_text() == source


def test_dense_transport_validator_rejects_promotion() -> None:
    dense = _load_module(
        "sm120_dense_runner_fail_closed",
        CAMPAIGN / "run_dense_tensor_correctness.py",
    )
    payload = _rank_payload()
    payload.update(
        {
            "oracle": "dense_external",
            "diagnostic_oracle_launches": 0,
            "functional_qualified": True,
        }
    )
    with pytest.raises(ValueError, match="functional_qualified"):
        dense.validate_transport(payload, dense.CASES[0], 0)


def test_dense_matrix_requires_p1_p2_and_p8() -> None:
    dense = _load_module(
        "sm120_dense_runner_required_cases",
        CAMPAIGN / "run_dense_tensor_correctness.py",
    )
    assert len(dense.REQUIRED_CASE_IDS) == 10
    assert "p1-r1-random-balanced" in dense.REQUIRED_CASE_IDS
    assert "p2-r2-random-skewed" in dense.REQUIRED_CASE_IDS
    assert "p8-r1-random-balanced" in dense.REQUIRED_CASE_IDS
    assert dense.REQUIRED_COVERAGE == {
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


def test_sparse_receipt_validation_is_fail_closed() -> None:
    freezer = _load_freezer()
    rank_records = [
        {
            "rank": rank,
            "status": "pass",
            "epoch_slots": [0, 1, 0],
            "exact_bf16_equal": True,
            **{field: 0 for field in freezer.ZERO_FIELDS},
        }
        for rank in range(8)
    ]
    sparse_receipt = {
        "schema": "deepgemm-sm120-megamoe-g8-runtime-correctness-v1",
        "status": "pass",
        "dynamic_sparse_matrix_passed": True,
        "functional_qualified": False,
        "contract_sha256": "contract",
        "binary_sha256": "binary",
        "case_count": 2,
        "rank_record_count": 16,
        "cases": [
            {
                "id": case_id,
                "status": "pass",
                "world_size": 8,
                "rank_records": json.loads(json.dumps(rank_records)),
            }
            for case_id in sorted(freezer.P8_CASES)
        ],
    }
    sparse = freezer.validate_sparse(sparse_receipt, "contract", "binary")
    assert sparse["mismatch_sum"] == 0

    broken = json.loads(json.dumps(sparse_receipt))
    broken["cases"][0]["rank_records"][0]["ack_signal_mismatches"] = 1
    with pytest.raises(ValueError, match="ack_signal_mismatches"):
        freezer.validate_sparse(broken, "contract", "binary")


def test_freezer_accepts_recorded_cuda_minor_version() -> None:
    freezer = _load_freezer()
    cuda_130 = "Cuda compilation tools, release 13.0, V13.0.88"
    cuda_133 = "Cuda compilation tools, release 13.3, V13.3.41"
    assert freezer.parse_nvcc_release(cuda_130) == "13.0.88"
    assert freezer.parse_nvcc_release(cuda_133) == "13.3.41"
