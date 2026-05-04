#!/bin/bash
# Phase 8a build script — apply HEonGPU patch & rebuild.
# Run on pod: bash heongpu_patches/build_phase8a.sh
set -e

HEDIR=/workspace/HEonGPU
SRC=$HEDIR/src/lib/host/ckks
KSRC=$HEDIR/src/lib/kernel
INC=$HEDIR/src/include/heongpu/host/ckks
PATCHDIR=/workspace/repo/heongpu_patches

echo "==> Copy multiply_power_of_x.cu into kernel/"
cp $PATCHDIR/multiply_power_of_x.cu $KSRC/

echo "==> Patch operator.cuh: add multiply_power_of_x_inplace declaration"
# Insert just before the closing brace of HEOperator<Scheme::CKKS> class.
# Heuristic: find first occurrence of "using HEOperator<Scheme::CKKS>::apply_galois;"
# and insert our public method declaration above the class-using block,
# OR better: insert right before "    private:" of the CKKS operator class.
# The CKKS HEOperator class ends with a public 'using' block then either private members or '};'.
# Safe insertion: append right after the apply_galois_inplace inline definition.

# Use a unique marker to insert idempotently.
if ! grep -q "multiply_power_of_x_inplace" $INC/operator.cuh; then
  # Insert after line containing 'apply_galois(input1, input1, galois_key, galois_elt, options);' followed by '}'
  # We'll add a marker block. Find line number of the second-to-last `};` in the file (end of CKKS class).
  python3 - <<'PYEOF'
import re
p = "/workspace/HEonGPU/src/include/heongpu/host/ckks/operator.cuh"
src = open(p).read()
needle = "apply_galois(input1, input1, galois_key, galois_elt, options);\n        }"
ins = """apply_galois(input1, input1, galois_key, galois_elt, options);
        }

        /**
         * @brief NEXUS-port helper: in-place ct *= x^k mod (x^N+1).
         * Performs INTT, negacyclic shift, NTT-back via GPUNTT-1.0.
         */
        __host__ void multiply_power_of_x_inplace(
            Ciphertext<Scheme::CKKS>& input,
            int k,
            const ExecutionOptions& options = ExecutionOptions());"""
if "multiply_power_of_x_inplace" not in src:
    src = src.replace(needle, ins, 1)
    open(p, "w").write(src)
    print("operator.cuh patched")
else:
    print("operator.cuh already patched, skipping")
PYEOF
fi

echo "==> Patch operator.cu: add wrapper method calling impl"
if ! grep -q "multiply_power_of_x_inplace_impl" $SRC/operator.cu; then
  python3 - <<'PYEOF'
p = "/workspace/HEonGPU/src/lib/host/ckks/operator.cu"
src = open(p).read()
# Append new method at end of the file's namespace block, before the final closing brace of namespace heongpu
add = """
    // Forward declared here; defined in src/lib/kernel/multiply_power_of_x.cu
    __host__ void multiply_power_of_x_inplace_impl(
        Data64* ct_data,
        const Modulus64* moduli,
        const Root64* ntt_table,
        const Root64* intt_table,
        const Ninverse64* n_inverse,
        int n_power,
        int current_decomp_count,
        int Q_size,
        int k_in_2N,
        cudaStream_t stream);

    __host__ void HEOperator<Scheme::CKKS>::multiply_power_of_x_inplace(
        Ciphertext<Scheme::CKKS>& input,
        int k,
        const ExecutionOptions& options)
    {
        if (input.rescale_required_ || input.relinearization_required_)
        {
            throw std::invalid_argument(
                "multiply_power_of_x_inplace: ct must not require rescale/relin");
        }
        const int current_decomp_count = context_->Q_size - input.depth_;
        multiply_power_of_x_inplace_impl(
            input.data(),
            context_->modulus_->data(),
            context_->ntt_table_->data(),
            context_->intt_table_->data(),
            context_->n_inverse_->data(),
            context_->n_power,
            current_decomp_count,
            context_->Q_size,
            k,
            options.stream_);
    }
"""
# Insert just before the final closing brace of the namespace heongpu block.
# Find LAST occurrence of '} // namespace heongpu' (or '}' alone closing it).
# Conservative: find last "}\n" and insert above it.
# Robust: locate the "namespace heongpu" closing.
idx = src.rfind("} // namespace heongpu")
if idx == -1:
    # fall back: last standalone '}' on its own line
    lines = src.rstrip().split("\n")
    # last line should be '}'
    assert lines[-1].strip() == "}", "could not locate namespace closing"
    src = "\n".join(lines[:-1]) + "\n" + add + "\n}\n"
else:
    src = src[:idx] + add + "\n" + src[idx:]
open(p, "w").write(src)
print("operator.cu patched")
PYEOF
fi

echo "==> Add multiply_power_of_x.cu to CMakeLists if not already there"
CML=$HEDIR/src/CMakeLists.txt
# Check several potential cmake files for the kernel listing
for f in $(find $HEDIR/src -name CMakeLists.txt); do
  if grep -q "lib/kernel/multiplication.cu" "$f" && ! grep -q "multiply_power_of_x.cu" "$f"; then
    echo "  patching $f"
    sed -i 's|lib/kernel/multiplication.cu|lib/kernel/multiplication.cu\n    lib/kernel/multiply_power_of_x.cu|' "$f"
  fi
done

echo "==> Rebuild HEonGPU"
cd $HEDIR/build
cmake --build . -j$(nproc) 2>&1 | tail -40

echo "==> Install"
cmake --install . 2>&1 | tail -10

echo "==> Done. Now rebuild python extension."
cd /workspace/repo/fhe_thesis/encryption/heongpu_bindings/build_heongpu_py
cmake --build . -j$(nproc) 2>&1 | tail -20
