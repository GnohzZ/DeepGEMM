#!/usr/bin/env python3
"""Derive a dense-output host without modifying the qualified G8 sources.

The generated translation unit still includes the frozen production kernel and
shared fixture implementation.  The only derived-host changes enable the
already-defined ``dense_external`` fixture, skip the sparse-only diagnostics,
and dump each production epoch for an out-of-process PyTorch comparison.
Every input source is checked against the qualified manifest before derivation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HOST = "deepgemm_fp8_fp4_mega_moe_sm120_production_canonical_fused_ready_chunk8_host.cu"
SHARED = "deepgemm_fp8_fp4_mega_moe_sm120_production_host.cu"
KERNEL = "cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu"


DENSE_DUMP_HELPERS = r'''
void dense_write_host_tensor(const std::string& prefix, int rank,
                             const std::string& name, int epoch,
                             const void* data, std::size_t bytes,
                             const char* dtype, const std::string& shape) {
  const std::string epoch_suffix =
      epoch < 0 ? "" : ".epoch" + std::to_string(epoch);
  const std::string path = prefix + ".rank" + std::to_string(rank) + "." +
      name + epoch_suffix + ".bin";
  FILE* file = std::fopen(path.c_str(), "wb");
  const bool complete = file != nullptr &&
      (bytes == 0 || std::fwrite(data, 1, bytes, file) == bytes) &&
      std::fclose(file) == 0;
  if (!complete) {
    std::fprintf(stderr, "cannot write dense tensor %s\n", path.c_str());
    std::abort();
  }
  const std::uint8_t empty = 0;
  const auto* raw = bytes == 0 ? &empty :
      reinterpret_cast<const std::uint8_t*>(data);
  const std::string digest = sha256_hex(raw, bytes);
  std::printf(
      "DENSE_TENSOR_JSON={\"rank\":%d,\"name\":\"%s\","
      "\"epoch\":%d,\"path\":\"%s\",\"bytes\":%zu,"
      "\"dtype\":\"%s\",\"shape\":%s,\"sha256\":\"%s\","
      "\"authority\":\"production-buffer\"}\n",
      rank, name.c_str(), epoch, path.c_str(), bytes, dtype, shape.c_str(),
      digest.c_str());
  std::fflush(stdout);
}

void dense_dump_device_tensor(const std::string& prefix, int rank,
                              const std::string& name, int epoch,
                              const void* device, std::size_t bytes,
                              const char* dtype,
                              const std::string& shape) {
  std::vector<std::uint8_t> host(bytes);
  if (bytes != 0) {
    CUDA_CHECK(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
  }
  dense_write_host_tensor(prefix, rank, name, epoch, host.data(), bytes,
                          dtype, shape);
}

void dense_dump_intermediate_scales(const std::string& prefix, int rank,
                                    int epoch, const void* device,
                                    int rows, int groups, int max_rows) {
  // Production stores UE8M0 scales in the TMA-friendly physical layout
  // [groups / 4, max_rows, 4].  Export only semantic rows, in logical
  // [rows, groups] order, while preserving the bytes actually produced.
  const std::size_t storage_bytes =
      static_cast<std::size_t>(groups / 4) * max_rows * 4;
  std::vector<std::uint8_t> storage(storage_bytes);
  CUDA_CHECK(cudaMemcpy(storage.data(), device, storage_bytes,
                        cudaMemcpyDeviceToHost));
  std::vector<std::uint8_t> logical(
      static_cast<std::size_t>(rows) * groups);
  for (int row = 0; row < rows; ++row) {
    for (int group = 0; group < groups; ++group) {
      const std::size_t physical =
          (static_cast<std::size_t>(group >> 2) * max_rows + row) * 4 +
          (group & 3);
      logical[static_cast<std::size_t>(row) * groups + group] =
          storage[physical];
    }
  }
  dense_write_host_tensor(
      prefix, rank, "intermediate_sf", epoch, logical.data(), logical.size(),
      "ue8m0-u8", "[" + std::to_string(rows) + "," +
          std::to_string(groups) + "]");
}

std::vector<std::uint8_t> dense_collect_expert_slices(
    const void* device, const std::vector<int>& experts, int rank,
    std::size_t bytes_per_expert) {
  std::vector<std::uint8_t> host(experts.size() * bytes_per_expert);
  const auto* base = reinterpret_cast<const std::uint8_t*>(device);
  for (std::size_t index = 0; index < experts.size(); ++index) {
    const int local_expert = experts[index] - rank * kLocalExperts;
    if (local_expert < 0 || local_expert >= kLocalExperts) {
      std::fprintf(stderr, "dense expert %d is not local to rank %d\n",
                   experts[index], rank);
      std::abort();
    }
    CUDA_CHECK(cudaMemcpy(host.data() + index * bytes_per_expert,
                          base + local_expert * bytes_per_expert,
                          bytes_per_expert, cudaMemcpyDeviceToHost));
  }
  return host;
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    observed = source.count(old)
    if observed != 1:
        raise ValueError(f"{label}: expected one source anchor, observed {observed}")
    return source.replace(old, new, 1)


def derive_host(source: str) -> str:
    source = replace_once(
        source,
        "namespace {\n\nconstexpr int kCanonicalTaskM = 64;",
        "namespace {\n" + DENSE_DUMP_HELPERS +
        "\nconstexpr int kCanonicalTaskM = 64;",
        "dense tensor dump helpers",
    )
    source = replace_once(
        source,
        '''  if (pattern == OraclePattern::kDenseExternal) {
    std::fprintf(stderr,
                 "canonical fused ready chunk8 dense_external oracle is not installed; "
                 "fail closed\\n");
    std::abort();
  }
''',
        '''  // The controlled adapter enables the dense fixture already defined by the
  // frozen shared host.  Numerical authority remains out of process.
''',
        "dense fail-closed gate",
    )
    source = replace_once(
        source,
        '''  const OraclePattern pattern = oracle_pattern_from_string(oracle_name);
  // The controlled adapter enables the dense fixture already defined by the
  // frozen shared host.  Numerical authority remains out of process.
''',
        '''  const OraclePattern pattern = oracle_pattern_from_string(oracle_name);
  const std::string dense_variant = parse_choice_env(
      "CAKE_DENSE_VARIANT", "random",
      {"random", "zero_activation", "zero_w1", "zero_w2"});
  const bool dense_topk_permute =
      parse_long_env("CAKE_DENSE_TOPK_PERMUTE", 0, 0, 1) != 0;
  // The controlled adapter enables the dense fixture already defined by the
  // frozen shared host.  Numerical authority remains out of process.
''',
        "dense case controls",
    )
    source = replace_once(
        source,
        '''
  int expected_received_routes = 0;
''',
        '''
  if (pattern == OraclePattern::kDenseExternal && dense_topk_permute) {
    for (int token = 0; token < active_rows; ++token) {
      for (int slot = 0; slot < kTopK / 2; ++slot) {
        const int other = kTopK - 1 - slot;
        const int left = token * kTopK + slot;
        const int right = token * kTopK + other;
        std::swap(host_topk[left * 2], host_topk[right * 2]);
        std::swap(host_topk[left * 2 + 1], host_topk[right * 2 + 1]);
        std::swap(host_weights[left], host_weights[right]);
      }
    }
  }

  int expected_received_routes = 0;
''',
        "top-k permutation",
    )
    source = replace_once(
        source,
        '''  DeviceBuffers b;
  CanonicalBuffers canonical;
''',
        '''  std::vector<int> dense_active_experts;
  if (pattern == OraclePattern::kDenseExternal) {
    for (int source = 0; source < world_size; ++source) {
      for (int token = 0; token < active_rows; ++token) {
        for (int slot = 0; slot < kTopK; ++slot) {
          const int route = token * kTopK + slot;
          if (mask_period > 0 && route % mask_period == 0) continue;
          const int expert =
              route_expert(token, slot, source, world_size, route_mode);
          if (expert / kLocalExperts == rank)
            dense_active_experts.push_back(expert);
        }
      }
    }
    std::sort(dense_active_experts.begin(), dense_active_experts.end());
    dense_active_experts.erase(
        std::unique(dense_active_experts.begin(), dense_active_experts.end()),
        dense_active_experts.end());
  }

  DeviceBuffers b;
  CanonicalBuffers canonical;
''',
        "active dense experts",
    )
    source = replace_once(
        source,
        '''  initialize_weight_scales<<<kInitBlocks, kInitThreads>>>(
      b.w2_weight_sf, kW2WeightSfWords, rank, false);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
''',
        '''  initialize_weight_scales<<<kInitBlocks, kInitThreads>>>(
      b.w2_weight_sf, kW2WeightSfWords, rank, false);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  if (pattern == OraclePattern::kDenseExternal) {
    if (dense_variant == "zero_activation") {
      CUDA_CHECK(cudaMemset(b.x, 0,
          static_cast<std::size_t>(active_rows) * kHidden));
    } else if (dense_variant == "zero_w1") {
      CUDA_CHECK(cudaMemset(b.w1_weight, 0, kW1WeightBytes));
    } else if (dense_variant == "zero_w2") {
      CUDA_CHECK(cudaMemset(b.w2_weight, 0, kW2WeightBytes));
    }
    const char* output_prefix = std::getenv("CAKE_DENSE_OUTPUT_PREFIX");
    if (output_prefix == nullptr || output_prefix[0] == '\\0') {
      std::fprintf(stderr, "CAKE_DENSE_OUTPUT_PREFIX is required\\n");
      std::abort();
    }
    const std::string prefix(output_prefix);
    dense_dump_device_tensor(
        prefix, rank, "x_fp8", -1, b.x,
        static_cast<std::size_t>(active_rows) * kHidden, "fp8-e4m3-u8",
        "[" + std::to_string(active_rows) + "," +
            std::to_string(kHidden) + "]");
    dense_dump_device_tensor(
        prefix, rank, "x_sf", -1, b.x_sf,
        static_cast<std::size_t>(active_rows) * kW1KBlocks * 4,
        "ue8m0-u8", "[" + std::to_string(active_rows) + "," +
            std::to_string(kHidden / 32) + "]");
    dense_dump_device_tensor(
        prefix, rank, "topk_idx", -1, b.topk_idx,
        static_cast<std::size_t>(active_rows) * kTopK * 2 * sizeof(int),
        "int32", "[" + std::to_string(active_rows) + "," +
            std::to_string(kTopK) + ",2]");
    dense_dump_device_tensor(
        prefix, rank, "topk_weights", -1, b.topk_weights,
        static_cast<std::size_t>(active_rows) * kTopK * sizeof(float),
        "float32", "[" + std::to_string(active_rows) + "," +
            std::to_string(kTopK) + "]");
    dense_write_host_tensor(
        prefix, rank, "active_experts", -1, dense_active_experts.data(),
        dense_active_experts.size() * sizeof(int), "int32",
        "[" + std::to_string(dense_active_experts.size()) + "]");

    const std::size_t w1_bytes_per_expert =
        static_cast<std::size_t>(kW1PhysicalN) * kHidden / 2;
    const std::size_t w1_sf_bytes_per_expert =
        static_cast<std::size_t>(kW1KBlocks) * kW1PhysicalN * 4;
    const std::size_t w2_bytes_per_expert =
        static_cast<std::size_t>(kOutput) * kIntermediate / 2;
    const std::size_t w2_sf_bytes_per_expert =
        static_cast<std::size_t>(kW2KBlocks) * kOutput * 4;
    const auto dense_w1 = dense_collect_expert_slices(
        b.w1_weight, dense_active_experts, rank, w1_bytes_per_expert);
    const auto dense_w1_sf = dense_collect_expert_slices(
        b.w1_weight_sf, dense_active_experts, rank,
        w1_sf_bytes_per_expert);
    const auto dense_w2 = dense_collect_expert_slices(
        b.w2_weight, dense_active_experts, rank, w2_bytes_per_expert);
    const auto dense_w2_sf = dense_collect_expert_slices(
        b.w2_weight_sf, dense_active_experts, rank,
        w2_sf_bytes_per_expert);
    const std::string expert_count =
        std::to_string(dense_active_experts.size());
    dense_write_host_tensor(
        prefix, rank, "w1_fp4", -1, dense_w1.data(), dense_w1.size(),
        "fp4-e2m1-packed-u8", "[" + expert_count + "," +
            std::to_string(kW1PhysicalN) + "," +
            std::to_string(kHidden / 2) + "]");
    dense_write_host_tensor(
        prefix, rank, "w1_sf", -1, dense_w1_sf.data(), dense_w1_sf.size(),
        "ue8m0-packed-u32-bytes", "[" + expert_count + "," +
            std::to_string(kW1KBlocks) + "," +
            std::to_string(kW1PhysicalN) + ",4]");
    dense_write_host_tensor(
        prefix, rank, "w2_fp4", -1, dense_w2.data(), dense_w2.size(),
        "fp4-e2m1-packed-u8", "[" + expert_count + "," +
            std::to_string(kOutput) + "," +
            std::to_string(kIntermediate / 2) + "]");
    dense_write_host_tensor(
        prefix, rank, "w2_sf", -1, dense_w2_sf.data(), dense_w2_sf.size(),
        "ue8m0-packed-u32-bytes", "[" + expert_count + "," +
            std::to_string(kW2KBlocks) + "," +
            std::to_string(kOutput) + ",4]");
  }
''',
        "dense variants and input extraction",
    )
    source = replace_once(
        source,
        '''  const int reference_routes = active_rows * kTopK;
  const int ref_groups = reference_routes * (kIntermediate / 32);
  const int ref_blocks =
      std::max(1, std::min(kInitBlocks, (ref_groups + 3) / 4));
  reference_w1_requant<<<ref_blocks, kThreads, 0, stream>>>(
      b.x, b.x_sf, b.topk_idx, b.topk_weights, b.reference_intermediate,
      b.reference_intermediate_sf, active_rows, pattern);
  reference_w2<<<kInitBlocks, kInitThreads, 0, stream>>>(
      b.topk_idx, b.reference_intermediate, b.reference_intermediate_sf,
      b.reference_partials, active_rows, pattern);
  reference_combine<<<kInitBlocks, kInitThreads, 0, stream>>>(
      b.topk_idx, b.reference_partials, b.reference_output, active_rows);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<std::uint16_t> host_reference(
      static_cast<std::size_t>(active_rows) * kOutput);
  CUDA_CHECK(cudaMemcpy(host_reference.data(), b.reference_output,
                        host_reference.size() * sizeof(host_reference[0]),
                        cudaMemcpyDeviceToHost));
''',
        '''  std::vector<std::uint16_t> host_reference;
  if (pattern != OraclePattern::kDenseExternal) {
    const int reference_routes = active_rows * kTopK;
    const int ref_groups = reference_routes * (kIntermediate / 32);
    const int ref_blocks =
        std::max(1, std::min(kInitBlocks, (ref_groups + 3) / 4));
    reference_w1_requant<<<ref_blocks, kThreads, 0, stream>>>(
        b.x, b.x_sf, b.topk_idx, b.topk_weights, b.reference_intermediate,
        b.reference_intermediate_sf, active_rows, pattern);
    reference_w2<<<kInitBlocks, kInitThreads, 0, stream>>>(
        b.topk_idx, b.reference_intermediate, b.reference_intermediate_sf,
        b.reference_partials, active_rows, pattern);
    reference_combine<<<kInitBlocks, kInitThreads, 0, stream>>>(
        b.topk_idx, b.reference_partials, b.reference_output, active_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    host_reference.resize(static_cast<std::size_t>(active_rows) * kOutput);
    CUDA_CHECK(cudaMemcpy(host_reference.data(), b.reference_output,
                          host_reference.size() * sizeof(host_reference[0]),
                          cudaMemcpyDeviceToHost));
  }
''',
        "sparse reference setup",
    )
    source = replace_once(
        source,
        '''    // Diagnostics are deliberately outside the single production launch.
    // They cannot mutate any production buffer or weaken the
    // dispatch/result slot-credit boundary.
''',
        '''    if (pattern == OraclePattern::kDenseExternal) {
      const std::string prefix(std::getenv("CAKE_DENSE_OUTPUT_PREFIX"));
      const std::string rows = std::to_string(total_padded_rows);
      dense_dump_device_tensor(prefix, rank, "meta_source_rank", epoch_index,
          b.meta_source_rank, static_cast<std::size_t>(total_padded_rows) * 4,
          "int32", "[" + rows + "]");
      dense_dump_device_tensor(prefix, rank, "meta_token", epoch_index,
          b.meta_token, static_cast<std::size_t>(total_padded_rows) * 4,
          "int32", "[" + rows + "]");
      dense_dump_device_tensor(prefix, rank, "meta_slot", epoch_index,
          b.meta_slot, static_cast<std::size_t>(total_padded_rows) * 4,
          "int32", "[" + rows + "]");
      dense_dump_device_tensor(prefix, rank, "grouped_layout", epoch_index,
          canonical.grouped_layout,
          static_cast<std::size_t>(total_padded_rows) * 4, "int32",
          "[" + rows + "]");
      dense_dump_device_tensor(prefix, rank, "w1_bf16", epoch_index,
          canonical.w1_bf16,
          static_cast<std::size_t>(total_padded_rows) * kW1PhysicalN * 2,
          "bf16-le", "[" + rows + "," +
              std::to_string(kW1PhysicalN) + "]");
      dense_dump_device_tensor(prefix, rank, "intermediate_fp8", epoch_index,
          b.intermediate,
          static_cast<std::size_t>(total_padded_rows) * kIntermediate,
          "fp8-e4m3-u8", "[" + rows + "," +
              std::to_string(kIntermediate) + "]");
      dense_dump_intermediate_scales(
          prefix, rank, epoch_index, b.intermediate_sf, total_padded_rows,
          kIntermediate / 32, kCanonicalMaxPaddedRows);
      dense_dump_device_tensor(prefix, rank, "w2_bf16", epoch_index,
          canonical.w2_bf16,
          static_cast<std::size_t>(total_padded_rows) * kOutput * 2,
          "bf16-le", "[" + rows + "," + std::to_string(kOutput) + "]");
    }

    // Diagnostics are deliberately outside the single production launch.
    // They cannot mutate any production buffer or weaken the
    // dispatch/result slot-credit boundary.
''',
        "production stage extraction",
    )
    source = replace_once(
        source,
        "    if (total_padded_rows > 0) {\n",
        "    if (total_padded_rows > 0 &&\n        pattern != OraclePattern::kDenseExternal) {\n",
        "sparse stage diagnostics",
    )
    source = replace_once(
        source,
        '''    for (std::size_t i = 0; i < observed.size(); ++i) {
      output_mismatches += observed[i] != host_reference[i];
      __nv_bfloat16 actual{};
      __nv_bfloat16 expected{};
      std::memcpy(&actual, &observed[i], sizeof(actual));
      std::memcpy(&expected, &host_reference[i], sizeof(expected));
      max_abs_error = std::max(
          max_abs_error,
          std::abs(static_cast<double>(__bfloat162float(actual)) -
                   static_cast<double>(__bfloat162float(expected))));
    }
''',
        '''    if (pattern == OraclePattern::kDenseExternal) {
      const char* prefix = std::getenv("CAKE_DENSE_OUTPUT_PREFIX");
      if (prefix == nullptr || prefix[0] == '\\0') {
        std::fprintf(stderr, "CAKE_DENSE_OUTPUT_PREFIX is required\\n");
        std::abort();
      }
      const std::size_t output_bytes = observed.size() * sizeof(observed[0]);
      dense_write_host_tensor(
          prefix, rank, "combine_output", epoch_index, observed.data(),
          output_bytes, "bf16-le", "[" + std::to_string(active_rows) +
              "," + std::to_string(kOutput) + "]");
      /* Legacy output-only record replaced by the typed tensor record above.
      const std::string path = std::string(prefix) + ".rank" +
          std::to_string(rank) + ".epoch" + std::to_string(epoch_index) +
          ".bf16";
      FILE* file = std::fopen(path.c_str(), "wb");
      const bool complete = file != nullptr &&
          std::fwrite(observed.data(), sizeof(observed[0]), observed.size(), file) ==
              observed.size() && std::fclose(file) == 0;
      if (!complete) {
        std::fprintf(stderr, "cannot write dense output %s\\n", path.c_str());
        std::abort();
      }
      const std::string output_sha = sha256_hex(
          reinterpret_cast<const std::uint8_t*>(observed.data()), output_bytes);
      std::printf(
          "DENSE_OUTPUT_JSON={\\\"rank\\\":%d,\\\"world_size\\\":%d,"
          "\\\"epoch\\\":%d,\\\"path\\\":\\\"%s\\\",\\\"bytes\\\":%zu,"
          "\\\"sha256\\\":\\\"%s\\\",\\\"authority\\\":"
          "\\\"production-output-only\\\"}\\n",
          rank, world_size, epoch_index, path.c_str(), output_bytes,
          output_sha.c_str());
      std::fflush(stdout); */
    } else {
      for (std::size_t i = 0; i < observed.size(); ++i) {
        output_mismatches += observed[i] != host_reference[i];
        __nv_bfloat16 actual{};
        __nv_bfloat16 expected{};
        std::memcpy(&actual, &observed[i], sizeof(actual));
        std::memcpy(&expected, &host_reference[i], sizeof(expected));
        max_abs_error = std::max(
            max_abs_error,
            std::abs(static_cast<double>(__bfloat162float(actual)) -
                     static_cast<double>(__bfloat162float(expected))));
      }
    }
''',
        "production output extraction",
    )
    source = replace_once(
        source,
        '''      "CAKE_ORACLE", "distinct_k32", {"zero", "analytic", "distinct_k32"});
''',
        '''      "CAKE_ORACLE", "distinct_k32",
      {"zero", "analytic", "distinct_k32", "dense_external"});
''',
        "oracle parser",
    )
    banner = "// GENERATED CONTROLLED DENSE ADAPTER; DO NOT QUALIFY THIS FILE BY ITSELF.\n"
    return banner + source


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    qualified = script_dir.parent / "qualified_g8"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = qualified / "qualification-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest["artifacts"]
    for name in (HOST, SHARED, KERNEL):
        if sha256(qualified / name) != artifacts[name]:
            raise ValueError(f"frozen artifact hash mismatch: {name}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    derived = derive_host((qualified / HOST).read_text())
    (output_dir / HOST).write_text(derived)
    (output_dir / SHARED).write_bytes((qualified / SHARED).read_bytes())
    receipt = {
        "schema": "deepgemm-sm120-megamoe-controlled-dense-adapter-source-v1",
        "qualified_manifest_sha256": sha256(manifest_path),
        "frozen_host_sha256": artifacts[HOST],
        "frozen_shared_sha256": artifacts[SHARED],
        "frozen_kernel_sha256": artifacts[KERNEL],
        "derived_host_sha256": sha256(output_dir / HOST),
        "qualified_sources_modified": False,
        "functional_qualified": False,
    }
    (output_dir / "dense-adapter-source.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
