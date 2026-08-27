# SM120 MegaMoE G8 runtime correctness campaign

This directory extends the frozen `qualified_g8` evidence without changing any
qualified source or weakening its manifest hashes.  The contract deliberately
separates three kinds of evidence:

1. repository pytest checks are static artifact/receipt gates;
2. the frozen correctness executable dynamically exercises transport, staging,
   sparse fixture math, ring reuse, signals, acknowledgements, and guards;
3. an independent dense PyTorch oracle and a secondary non-overlapped
   DeepEP+DeepGEMM differential are required before setting
   `functional_qualified=true`.

The oracle authority is deliberately not flattened into one label:

- `pytorch_oracle.py` is the primary mathematically independent composition.
  It follows the pure-PyTorch BF16/FP32 reference strategy in
  `tests/test_mega_moe_sm90.py` and explicitly decodes FP8, FP4, and UE8M0.
  CUDA fast-math is not bit-identical to PyTorch, so this gate uses a documented
  numerical tolerance rather than exact BF16 equality.
- the non-overlapped baseline in `tests/test_mega_moe.py` is a valuable second
  end-to-end oracle, but it shares DeepGEMM grouped-GEMM compute with the code
  under test and is therefore not sufficient as the sole mathematical oracle.
- the frozen host's CUDA reference is exact and does not consume donor output,
  but its current fixtures are sparse and share fixture/decode helpers.

`build_dense_adapter.py` now provides a controlled adapter without changing any
file under `qualified_g8`.  It validates the frozen manifest hashes, derives a
temporary host that enables the already-defined deterministic dense fixture,
and exports the actual FP8 activation/UE8M0 scales, routing, FP4 W1/W2/scales,
W1 BF16, SwiGLU/requant FP8+UE8M0, W2 BF16 partials, and combine output.
`run_dense_tensor_correctness.py` feeds those exported production inputs to the
pure-PyTorch H7168/I3072/O7168 oracle and compares every production stage
fail-closed.

The formal dense matrix covers random balanced/skewed routing, partial/all
masking, zero activation/W1/W2, top-k permutation, P1/P2, and P8/R1.  Production
epochs use slots `[0,1,0]`; stage rows are normalized by route identity because
cross-rank arrival order is not part of the ABI.  P8/R113 and P8/R2048 are
separately replayed by the sparse transport/staging campaign, and the secondary
P8/R1 differential uses one direct expanded DeepEP dispatch, two public SM120
grouped DeepGEMM calls, and one DeepEP combine.

The `exact_bf16_equal` and `stage_oracle_installed` fields inherited from the
frozen rank JSON are not dense comparison evidence: sparse in-process
diagnostics are intentionally skipped by the adapter.  Only the out-of-process
PyTorch metrics and output hashes in the dense receipt have numerical
authority.

Generate the controlled source in a build directory:

```bash
python examples/sm120_megamoe/correctness_g8/build_dense_adapter.py \
  --output-dir build/sm120_megamoe_dense_adapter_source
```

Compile that generated host with the same SM120a, CUTLASS, CUDA, and pinned
NCCL 2.30.7 flags as the frozen correctness binary, with the generated source
directory placed before `qualified_g8` on the include path.  Then run the
formal tensor matrix on explicitly idle GPUs:

```bash
python examples/sm120_megamoe/correctness_g8/run_dense_tensor_correctness.py \
  --binary build/sm120_megamoe_g8_dense_adapter \
  --devices 0,1,2,3,4,5,6,7 \
  --adapter-source-manifest build/sm120_megamoe_dense_adapter_source/dense-adapter-source.json \
  --artifacts-dir /path/to/retained/dense-tensors \
  --output examples/sm120_megamoe/correctness_g8/stage1-dense-tensor-qualified-current.json
```

The receipt records binary/input/stage/output SHA-256 hashes, PyTorch reference
hashes, numerical metrics, exact-zero gates, and route-normalized epoch
stability.  Each component receipt deliberately leaves
`functional_qualified=false`; only `freeze_baseline.py` may combine all three
independent gates and promote the frozen baseline.

List the smoke matrix without requiring a GPU:

```bash
python examples/sm120_megamoe/correctness_g8/run_correctness_matrix.py \
  --tier smoke --list
```

Run the smoke matrix on explicitly selected idle GPUs:

```bash
python examples/sm120_megamoe/correctness_g8/run_correctness_matrix.py \
  --binary build/sm120_megamoe_g8_correctness \
  --devices 4,5 \
  --tier smoke \
  --output build/sm120_megamoe_g8_smoke.json
```

The full matrix requires eight idle SM120 GPUs and replays both historical P8
cases from the current binary:

```bash
python examples/sm120_megamoe/correctness_g8/run_correctness_matrix.py \
  --binary build/sm120_megamoe_g8_correctness \
  --devices 0,1,2,3,4,5,6,7 \
  --tier full \
  --output build/sm120_megamoe_g8_full.json
```

The runner launches one process per rank, validates every rank-level JSON field
fail-closed, and records the binary and output hashes.  A passing sparse matrix
receipt intentionally remains `functional_qualified=false` until the dense
external gate described by the contract exists and passes.

Run the secondary non-overlapped P8/R1 differential after installing DeepEP
and building its extension against the same NCCL runtime:

```bash
EP_NCCL_ROOT_DIR=/path/to/nccl-2.30.7 \
python examples/sm120_megamoe/correctness_g8/run_nonoverlap_correctness.py \
  --dense-receipt build/sm120_megamoe_dense.json \
  --output build/sm120_megamoe_nonoverlap.json
```

Freeze the phase-two baseline after all receipts and current binaries exist:

```bash
python examples/sm120_megamoe/correctness_g8/freeze_baseline.py \
  --performance-receipt /path/to/performance.json \
  --correctness-binary /path/to/sm120_megamoe_g8_correctness \
  --dense-binary /path/to/sm120_megamoe_g8_dense_adapter \
  --performance-binary /path/to/sm120_megamoe_g8_candidate_perf \
  --deepgemm-extension /path/to/deep_gemm/_C.so \
  --deepep-extension /path/to/deep_ep/_C.so \
  --nccl-library /path/to/nccl-2.30.7/libnccl.so.2 \
  --performance-compiler-log /path/to/build-performance.stderr.log
```

The freezer verifies source, binary, SASS, compiler-log, receipt, and runtime
hash links.  It records the active CUDA compiler rather than requiring a
specific CUDA minor version; cross-version SASS/register equivalence is not a
functional promotion requirement.  It also snapshots the performance receipt,
ptxas log, and resource report beside the manifest so the frozen evidence does
not depend on temporary build paths.
