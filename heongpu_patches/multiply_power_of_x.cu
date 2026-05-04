// Phase 8a: multiply_power_of_x for HEonGPU CKKS.
//
// Implements ct ← ct · x^k mod (x^N + 1) by:
//   1. INTT each (poly, prime) pair → coefficient domain
//   2. Negacyclic shift kernel: new_coeff[(j+k) mod N] = old[j] * sign,
//      where sign = -1 if (j + k) >= N (wrapped under x^N = -1).
//   3. NTT back to NTT domain.
//
// Drop this file into HEonGPU/src/lib/kernel/ and add a friend-style
// patch to operator.cu that calls into it. We keep the kernel declared
// extern "C" so the call site doesn't need template instantiation.

#include <cstdint>
#include <gpuntt/common/modular_arith.cuh>
#include <gpuntt/common/common.cuh>
#include <gpuntt/ntt_merge/ntt.cuh>
#include <heongpu/host/ckks/operator.cuh>

namespace heongpu
{
    // Negacyclic shift in coefficient domain.
    // poly_data layout: (2 polys) × (Q_size primes) × N coefficients,
    // contiguous (poly-major). idx ranges over [0, 2*Q_size*N).
    __global__ void negacyclic_shift_kernel(
        const Data64* __restrict__ in_data,
        Data64* __restrict__       out_data,
        const Modulus64* __restrict__ moduli,
        int n_power,
        int Q_size,
        int k_in_2N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int N = 1 << n_power;
        int total = 2 * Q_size * N;
        if (idx >= total) return;

        int N_mask = N - 1;
        int j = idx & N_mask;
        int prime_poly = idx >> n_power;
        int prime = prime_poly % Q_size;

        int N2 = N << 1;
        int k = ((k_in_2N % N2) + N2) % N2;

        // Source coefficient s such that (s + k) mod 2N == j (indexing
        // outputs in the dest coordinate). Sign flips once for each
        // multiple of N crossed.
        // Equivalently: shift up by k, dest = (src + k) mod 2N, with
        // sign flip if dest crossed N.
        // We want the input value that flowed INTO output position j:
        //   src = (j - k) mod 2N
        //   if src < 0: src += 2N (handled by mod)
        //   wraps = ((j - k) - src) / N  → 0, 1, or 2
        int s_signed = j - k;
        int wraps;
        int s;
        if (s_signed >= 0)
        {
            s = s_signed;
            wraps = 0;
        }
        else
        {
            // s_signed in [-2N+1, -1]. Add 2N once to make positive.
            s = s_signed + N2;          // now s in [0, 2N)
            wraps = 1;                   // crossed boundary once via wrap
        }
        // s now in [0, 2N). If s >= N, subtract N and add another wrap.
        if (s >= N)
        {
            s -= N;
            wraps ^= 1;
        }

        const Data64 q_val = moduli[prime].value;
        Data64 v = in_data[(prime_poly << n_power) + s];
        if (wraps & 1)
        {
            v = (v == 0) ? 0 : (q_val - v);
        }
        out_data[idx] = v;
    }

    // Public-facing entry point. Defined here to keep the kernel and
    // its call-site in one translation unit (avoids extra header churn).
    // The caller (operator.cu) just forwards a few context fields.
    __host__ void multiply_power_of_x_inplace_impl(
        Data64* ct_data,
        const Modulus64* moduli,
        const Root64* ntt_table,
        const Root64* intt_table,
        const Ninverse64* n_inverse,
        int n_power,
        int current_decomp_count,   // number of primes still active for this ct
        int Q_size,                  // total #primes (for indexing)
        int k_in_2N,
        cudaStream_t stream)
    {
        const int N = 1 << n_power;
        const int active_total = 2 * current_decomp_count * N;

        // 1. INTT (NTT-domain → coefficient domain), only on active primes.
        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power      = n_power,
            .ntt_type     = gpuntt::INVERSE,
            .ntt_layout   = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse  = const_cast<Ninverse64*>(n_inverse),
            .stream       = stream};

        // Two polynomials × current_decomp_count primes worth of data.
        gpuntt::GPU_INTT_Inplace(
            ct_data,
            const_cast<Root64*>(intt_table),
            const_cast<Modulus64*>(moduli),
            cfg_intt,
            2 * current_decomp_count,   // batch size
            current_decomp_count);       // primes used per poly

        // 2. Negacyclic shift on coefficients into a temp buffer, then copy back.
        DeviceVector<Data64> temp(active_total, stream);

        const int block = 256;
        const int grid  = (active_total + block - 1) / block;
        negacyclic_shift_kernel<<<grid, block, 0, stream>>>(
            ct_data,
            temp.data(),
            moduli,
            n_power,
            current_decomp_count,
            k_in_2N);

        cudaMemcpyAsync(ct_data,
                        temp.data(),
                        active_total * sizeof(Data64),
                        cudaMemcpyDeviceToDevice,
                        stream);

        // 3. NTT back.
        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power      = n_power,
            .ntt_type     = gpuntt::FORWARD,
            .ntt_layout   = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse  = nullptr,
            .stream       = stream};

        gpuntt::GPU_NTT_Inplace(
            ct_data,
            const_cast<Root64*>(ntt_table),
            const_cast<Modulus64*>(moduli),
            cfg_ntt,
            2 * current_decomp_count,
            current_decomp_count);
    }

} // namespace heongpu
