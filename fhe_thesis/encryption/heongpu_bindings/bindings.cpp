// Copyright 2026 Gokay Gulsoy
// pybind11 wrapper for HEonGPU CKKS — Phase 1 smoke surface.
//
// Phase 1 goal: prove that we can drive HEonGPU from Python and get
// numerically correct CKKS arithmetic. The surface intentionally exposes
// only a small fraction of the C++ API; richer ops (rotate, bootstrap,
// matmul) will be added in Phase 2 once the build path is verified.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <cuda_runtime.h>

#include <heongpu/heongpu.hpp>

#include <memory>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

using Scheme = std::integral_constant<heongpu::Scheme, heongpu::Scheme::CKKS>;
constexpr auto SCHEME = heongpu::Scheme::CKKS;

// -----------------------------------------------------------------------------
// Lightweight RAII wrappers. HEonGPU's classes are templated on the scheme;
// we hide that behind plain Python classes for ergonomics.
// -----------------------------------------------------------------------------

struct CKKSContext {
    heongpu::HEContext<SCHEME> ctx;
    std::size_t poly_modulus_degree;

    CKKSContext(std::size_t N,
                const std::vector<int>& q_bits,
                const std::vector<int>& p_bits,
                bool sec_none)
        : ctx(sec_none
                  ? heongpu::GenHEContext<SCHEME>(heongpu::sec_level_type::none)
                  : heongpu::GenHEContext<SCHEME>()),
          poly_modulus_degree(N) {
        ctx->set_poly_modulus_degree(N);
        ctx->set_coeff_modulus_bit_sizes(q_bits, p_bits);
        ctx->generate();
    }

    void print_parameters() { ctx->print_parameters(); }
    std::size_t slot_count() const { return poly_modulus_degree / 2; }
};

struct SecretKey   { std::shared_ptr<heongpu::Secretkey<SCHEME>> sk; };
struct PublicKey   { std::shared_ptr<heongpu::Publickey<SCHEME>> pk; };
struct RelinKey    { std::shared_ptr<heongpu::Relinkey<SCHEME>>  rk; };
struct GaloisKey   { std::shared_ptr<heongpu::Galoiskey<SCHEME>> gk; };

struct Plaintext   { std::shared_ptr<heongpu::Plaintext<SCHEME>>  pt; };
struct Ciphertext  { std::shared_ptr<heongpu::Ciphertext<SCHEME>> ct; };

// NEXUS Phase 4: standalone CtoS/StoC at arbitrary chain levels.
// HEonGPU exposes precomputed encoding-transform contexts that hold
// V/V_inv plaintext matrices encoded for a chosen pair of levels.
// We can build several of these (one per level we want to chain at)
// and use the matching context inside coeff_to_slot / slot_to_coeff.
struct EncodingTransformContext {
    std::shared_ptr<heongpu::CKKSEncodingTransformContext> ctx{
        std::make_shared<heongpu::CKKSEncodingTransformContext>()};
    int  ctos_level()   const { return ctx->CtoS_level_; }
    int  stoc_level()   const { return ctx->StoC_level_; }
    int  ctos_piece()   const { return ctx->CtoS_piece_; }
    int  stoc_piece()   const { return ctx->StoC_piece_; }
    bool generated()    const { return ctx->generated_; }
    std::vector<int> key_indexs() const { return ctx->key_indexs_; }
};

// NEXUS Phase 5: CUDA stream wrapper for parallel op submission.
// Owning RAII handle. Create N of these from Python, submit
// independent ops on different streams, then sync when done.
struct CudaStream {
    cudaStream_t s = nullptr;
    bool owns = false;

    CudaStream() {
        cudaError_t err = cudaStreamCreate(&s);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamCreate failed: ") +
                                     cudaGetErrorString(err));
        }
        owns = true;
    }
    ~CudaStream() {
        if (owns && s) cudaStreamDestroy(s);
    }
    void synchronize() {
        cudaError_t err = cudaStreamSynchronize(s);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") +
                                     cudaGetErrorString(err));
        }
    }
    // No copy.
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

struct KeyGenerator {
    heongpu::HEKeyGenerator<SCHEME> kg;
    explicit KeyGenerator(CKKSContext& c) : kg(c.ctx) {}

    SecretKey generate_secret_key(CKKSContext& c) {
        auto sk = std::make_shared<heongpu::Secretkey<SCHEME>>(c.ctx);
        kg.generate_secret_key(*sk);
        return SecretKey{sk};
    }
    SecretKey generate_secret_key_h(CKKSContext& c, int hamming_weight) {
        auto sk = std::make_shared<heongpu::Secretkey<SCHEME>>(c.ctx, hamming_weight);
        kg.generate_secret_key(*sk);
        return SecretKey{sk};
    }
    PublicKey generate_public_key(CKKSContext& c, SecretKey& s) {
        auto pk = std::make_shared<heongpu::Publickey<SCHEME>>(c.ctx);
        kg.generate_public_key(*pk, *s.sk);
        return PublicKey{pk};
    }
    RelinKey generate_relin_key(CKKSContext& c, SecretKey& s) {
        auto rk = std::make_shared<heongpu::Relinkey<SCHEME>>(c.ctx);
        kg.generate_relin_key(*rk, *s.sk);
        return RelinKey{rk};
    }

    // Default Galois key — supports arbitrary rotations (large key set).
    GaloisKey generate_galois_key_default(CKKSContext& c, SecretKey& s) {
        auto gk = std::make_shared<heongpu::Galoiskey<SCHEME>>(c.ctx);
        kg.generate_galois_key(*gk, *s.sk);
        return GaloisKey{gk};
    }

    // Galois key restricted to a specific list of rotation offsets.
    GaloisKey generate_galois_key_shifts(CKKSContext& c, SecretKey& s,
                                         std::vector<int> shifts) {
        auto gk = std::make_shared<heongpu::Galoiskey<SCHEME>>(c.ctx, shifts);
        kg.generate_galois_key(*gk, *s.sk);
        return GaloisKey{gk};
    }
};

struct Encoder {
    heongpu::HEEncoder<SCHEME> enc;
    explicit Encoder(CKKSContext& c) : enc(c.ctx) {}

    Plaintext encode(CKKSContext& c, const std::vector<double>& v, double scale) {
        auto pt = std::make_shared<heongpu::Plaintext<SCHEME>>(c.ctx);
        std::vector<double> v_copy = v;  // HEonGPU encode wants a non-const ref
        enc.encode(*pt, v_copy, scale);
        return Plaintext{pt};
    }
    std::vector<double> decode(Plaintext& pt) {
        std::vector<double> out;
        enc.decode(out, *pt.pt);
        return out;
    }

    // ── NEXUS-style coefficient encoding (Phase 3) ────────────────
    // Encodes the input vector as the *coefficients* of the plaintext
    // polynomial (rather than as its slot vector). Used for coefficient-
    // packed matmul: pt_x · pt_W (mod X^N + 1) yields the convolution
    // of x with W in the coefficient domain.
    Plaintext encode_coeff(CKKSContext& c,
                           const std::vector<double>& v, double scale) {
        auto pt = std::make_shared<heongpu::Plaintext<SCHEME>>(c.ctx);
        std::vector<double> v_copy = v;
        enc.encode(*pt, v_copy, scale,
                   heongpu::ExecutionOptions(),
                   heongpu::encoding::COEFFICIENT);
        return Plaintext{pt};
    }
    std::vector<double> decode_coeff(Plaintext& pt) {
        // The decoder routes on plain.encoding_, which encode_coeff sets.
        std::vector<double> out;
        enc.decode(out, *pt.pt);
        return out;
    }
};

struct Encryptor {
    heongpu::HEEncryptor<SCHEME> enc;
    Encryptor(CKKSContext& c, PublicKey& pk) : enc(c.ctx, *pk.pk) {}

    Ciphertext encrypt(CKKSContext& c, Plaintext& pt) {
        auto ct = std::make_shared<heongpu::Ciphertext<SCHEME>>(c.ctx);
        enc.encrypt(*ct, *pt.pt);
        return Ciphertext{ct};
    }
};

struct Decryptor {
    heongpu::HEDecryptor<SCHEME> dec;
    Decryptor(CKKSContext& c, SecretKey& sk) : dec(c.ctx, *sk.sk) {}

    Plaintext decrypt(CKKSContext& c, Ciphertext& ct) {
        auto pt = std::make_shared<heongpu::Plaintext<SCHEME>>(c.ctx);
        dec.decrypt(*pt, *ct.ct);
        return Plaintext{pt};
    }
};

struct Operator {
    heongpu::HEArithmeticOperator<SCHEME> ops;
    heongpu::HEEncoder<SCHEME>* enc_ptr = nullptr;   // shared with Encoder
    CKKSContext* ctx_ptr = nullptr;
    double scale_cache = 0.0;
    Operator(CKKSContext& c, Encoder& enc)
        : ops(c.ctx, enc.enc), enc_ptr(&enc.enc), ctx_ptr(&c) {}

    // Phase 7b-BSGS: batched encode + mod-drop entirely in C++.
    // Releases the GIL once and runs all encodes back-to-back.
    // ``target_depth`` is the chain depth at which each plaintext should
    // sit when returned (the equivalent of N python mod_drop_inplace_pt
    // calls per plaintext).
    std::vector<Plaintext> encode_many_drop(
        const std::vector<std::vector<double>>& vs,
        double scale,
        int target_depth)
    {
        std::vector<Plaintext> out;
        out.reserve(vs.size());
        for (size_t i = 0; i < vs.size(); ++i) {
            auto pt = std::make_shared<heongpu::Plaintext<SCHEME>>(ctx_ptr->ctx);
            std::vector<double> v_copy = vs[i];
            enc_ptr->encode(*pt, v_copy, scale);
            while (pt->depth() < target_depth) {
                ops.mod_drop_inplace(*pt);
            }
            out.emplace_back(Plaintext{pt});
        }
        return out;
    }

    void add_inplace(Ciphertext& a, Ciphertext& b)               { ops.add_inplace(*a.ct, *b.ct); }
    void sub_inplace(Ciphertext& a, Ciphertext& b)               { ops.sub_inplace(*a.ct, *b.ct); }
    void multiply_inplace(Ciphertext& a, Ciphertext& b)          { ops.multiply_inplace(*a.ct, *b.ct); }
    void multiply_plain_inplace(Ciphertext& a, Plaintext& p)     { ops.multiply_plain_inplace(*a.ct, *p.pt); }
    void add_plain_inplace(Ciphertext& a, Plaintext& p)          { ops.add_plain_inplace(*a.ct, *p.pt); }
    void relinearize_inplace(Ciphertext& a, RelinKey& rk)        { ops.relinearize_inplace(*a.ct, *rk.rk); }
    void rescale_inplace(Ciphertext& a)                          { ops.rescale_inplace(*a.ct); }
    void mod_drop_inplace_ct(Ciphertext& a)                      { ops.mod_drop_inplace(*a.ct); }
    void mod_drop_inplace_pt(Plaintext& a)                       { ops.mod_drop_inplace(*a.pt); }
    int  depth(Ciphertext& a)                                    { return a.ct->depth(); }
    int  depth_pt(Plaintext& a)                                  { return a.pt->depth(); }

    // Sum of two ciphertexts that may be at different chain levels.
    // Drops the deeper-chain (higher coeff_modulus_count) operand down
    // to the shallower one, then performs add_inplace on the lhs.
    void add_inplace_match(Ciphertext& a, Ciphertext& b) {
        int da = a.ct->depth();
        int db = b.ct->depth();
        if (da == db) {
            ops.add_inplace(*a.ct, *b.ct);
            return;
        }
        // depth() grows as the chain shrinks; equalise to the larger value.
        if (da < db) {
            while (a.ct->depth() < db) ops.mod_drop_inplace(*a.ct);
            ops.add_inplace(*a.ct, *b.ct);
        } else {
            // Need to drop b without aliasing — shallow-copy via shared_ptr is fine
            // since add_inplace consumes b read-only on the GPU side.
            heongpu::Ciphertext<SCHEME> b_copy = *b.ct;
            while (b_copy.depth() < da) ops.mod_drop_inplace(b_copy);
            ops.add_inplace(*a.ct, b_copy);
        }
    }

    // Cyclic rotation by `shift` (positive = left, negative = right within slot vector).
    void rotate_rows_inplace(Ciphertext& a, GaloisKey& g, int shift) {
        ops.rotate_rows_inplace(*a.ct, *g.gk, shift);
    }

    // ── NEXUS Phase 5: stream-aware variants for parallel op submission ──
    // Identical semantics to the default-stream versions; the cudaStream_t
    // is taken from the supplied CudaStream wrapper. Caller is responsible
    // for synchronizing streams before reading results.
    void multiply_plain_inplace_s(Ciphertext& a, Plaintext& p, CudaStream& st) {
        heongpu::ExecutionOptions opts;
        opts.set_stream(st.s);
        ops.multiply_plain_inplace(*a.ct, *p.pt, opts);
    }
    void add_inplace_s(Ciphertext& a, Ciphertext& b, CudaStream& st) {
        heongpu::ExecutionOptions opts;
        opts.set_stream(st.s);
        ops.add_inplace(*a.ct, *b.ct, opts);
    }
    void rotate_rows_inplace_s(Ciphertext& a, GaloisKey& g, int shift,
                               CudaStream& st) {
        heongpu::ExecutionOptions opts;
        opts.set_stream(st.s);
        ops.rotate_rows_inplace(*a.ct, *g.gk, shift, opts);
    }
    void rescale_inplace_s(Ciphertext& a, CudaStream& st) {
        heongpu::ExecutionOptions opts;
        opts.set_stream(st.s);
        ops.rescale_inplace(*a.ct, opts);
    }
    void mod_drop_inplace_ct_s(Ciphertext& a, CudaStream& st) {
        heongpu::ExecutionOptions opts;
        opts.set_stream(st.s);
        ops.mod_drop_inplace(*a.ct, opts);
    }
    void mod_drop_inplace_pt_s(Plaintext& p, CudaStream& st) {
        heongpu::ExecutionOptions opts;
        opts.set_stream(st.s);
        ops.mod_drop_inplace(*p.pt, opts);
    }
    void relinearize_inplace_s(Ciphertext& a, RelinKey& rk, CudaStream& st) {
        heongpu::ExecutionOptions opts;
        opts.set_stream(st.s);
        ops.relinearize_inplace(*a.ct, *rk.rk, opts);
    }
    Ciphertext clone_ct_s(Ciphertext& a, CudaStream& /*st*/) {
        // copy-assign goes through the default stream internally; here we
        // expose the same as clone_ct but offer the API for symmetry.
        auto out = std::make_shared<heongpu::Ciphertext<SCHEME>>();
        *out = *a.ct;
        return Ciphertext{out};
    }

    // Bootstrapping: depth-refresh a CKKS ciphertext.
    // Caller must have already invoked generate_bootstrapping_params + generated
    // a Galois key from `bootstrapping_key_indexs()`.
    void generate_bootstrapping_params(double scale,
                                       int CtoS_piece, int StoC_piece,
                                       int taylor_number, bool less_key_mode) {
        heongpu::BootstrappingConfig cfg(CtoS_piece, StoC_piece,
                                         taylor_number, less_key_mode);
        ops.generate_bootstrapping_params(
            scale, cfg,
            heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);
    }

    std::vector<int> bootstrapping_key_indexs() {
        return ops.bootstrapping_key_indexs();
    }

    Ciphertext regular_bootstrapping(Ciphertext& a, GaloisKey& g, RelinKey& rk) {
        auto out = std::make_shared<heongpu::Ciphertext<SCHEME>>();
        *out = ops.regular_bootstrapping(*a.ct, *g.gk, *rk.rk);
        return Ciphertext{out};
    }

    // Shallow copy of a ciphertext that preserves all metadata (depth,
    // scale, encoding type). Used by Python wrappers that need an
    // out-of-place result.
    Ciphertext clone_ct(Ciphertext& a) {
        auto out = std::make_shared<heongpu::Ciphertext<SCHEME>>();
        *out = *a.ct;  // class-defined copy
        return Ciphertext{out};
    }

    // ── NEXUS Phase 3: encoding domain conversion ─────────────────
    // Both functions require generate_bootstrapping_params + a galois
    // key built from bootstrapping_key_indexs() to have been issued
    // ahead of time (HEonGPU stores the BSGS matrices on the operator).

    // coeff_to_slot returns 2 ciphertexts:
    //   result[0]: slots[0..N/2) hold polynomial coefficients [0..N/2)
    //   result[1]: slots[0..N/2) hold polynomial coefficients [N/2..N)
    std::vector<Ciphertext> coeff_to_slot(Ciphertext& ct, GaloisKey& g) {
        auto pair = ops.coeff_to_slot(*ct.ct, *g.gk);
        std::vector<Ciphertext> out;
        out.reserve(pair.size());
        for (auto& c : pair) {
            auto p = std::make_shared<heongpu::Ciphertext<SCHEME>>();
            *p = std::move(c);
            out.push_back(Ciphertext{p});
        }
        return out;
    }

    // slot_to_coeff: takes 2 slot-encoded cts (representing the two
    // halves of a polynomial), returns 1 coefficient-encoded ct.
    Ciphertext slot_to_coeff(Ciphertext& ct0, Ciphertext& ct1, GaloisKey& g) {
        auto out = std::make_shared<heongpu::Ciphertext<SCHEME>>();
        *out = ops.slot_to_coeff(*ct0.ct, *ct1.ct, *g.gk);
        return Ciphertext{out};
    }

    // ── NEXUS Phase 4: arbitrary-level CtoS/StoC ─────────────────
    // Precompute V/V_inv plaintext matrices for a chosen (CtoS_start,
    // StoC_start) pair, stored in a standalone transform context.
    // Lets us chain matvec → CtoS → polyval → StoC → matvec at
    // depths > 0 (the bootstrap matrices live at depth 0 only).
    void generate_encoding_transform_context(
            EncodingTransformContext& tc, double scale,
            int ctos_piece, int stoc_piece,
            int ctos_start_level, int stoc_start_level,
            bool less_key_mode) {
        heongpu::CKKSEncodingTransformConfig cfg(
            ctos_piece, stoc_piece,
            ctos_start_level, stoc_start_level, less_key_mode);
        ops.generate_encoding_transform_context(*tc.ctx, scale, cfg);
    }

    std::vector<Ciphertext> coeff_to_slot_ctx(Ciphertext& ct, GaloisKey& g,
                                              EncodingTransformContext& tc) {
        auto pair = ops.coeff_to_slot(*ct.ct, *g.gk, *tc.ctx);
        std::vector<Ciphertext> out;
        out.reserve(pair.size());
        for (auto& c : pair) {
            auto p = std::make_shared<heongpu::Ciphertext<SCHEME>>();
            *p = std::move(c);
            out.push_back(Ciphertext{p});
        }
        return out;
    }

    Ciphertext slot_to_coeff_ctx(Ciphertext& ct0, Ciphertext& ct1, GaloisKey& g,
                                 EncodingTransformContext& tc) {
        auto out = std::make_shared<heongpu::Ciphertext<SCHEME>>();
        *out = ops.slot_to_coeff(*ct0.ct, *ct1.ct, *g.gk, *tc.ctx);
        return Ciphertext{out};
    }

    // ── NEXUS Phase 3 introspection ───────────────────────────────
    // Python-side helpers need scale_boot_ and CtoS_level_ to align
    // arbitrary ciphertexts with the precomputed CtoS plaintext
    // matrices before calling coeff_to_slot.
    double bootstrapping_scale()  const { return ops.bootstrapping_scale(); }
    int    coeff_to_slot_level() const { return ops.coeff_to_slot_level(); }
    int    slot_to_coeff_level() const { return ops.slot_to_coeff_level(); }
    bool   bootstrapping_ready() const { return ops.bootstrapping_ready(); }

    // NEXUS-style escape hatch: after a `multiply_plain_inplace` we
    // sometimes want to call `coeff_to_slot` directly without first
    // rescaling (so the cipher stays at depth 0 where CtoS plaintext
    // matrices live). Rotation refuses cipher with rescale_required_,
    // so this clears the flag — the caller is responsible for ensuring
    // the next op handles the larger scale (CtoS internally manages
    // scales via its own rescale chain).
    void clear_rescale_required(Ciphertext& c) {
        c.ct->clear_rescale_required();
    }
    void set_rescale_required(Ciphertext& c) {
        c.ct->set_rescale_required();
    }

    // ── NEXUS Phase 6: batched BSGS gather inside one C++ call ──
    //
    // Replaces the Python ``gather_slots`` BSGS loop:
    //   1. pre-rotate ``ct`` by each baby shift
    //   2. for each giant g_i, accumulate Σⱼ baby_rotsⱼ ⊙ maskⱼ
    //   3. single ``rescale_inplace`` per giant (lazy rescale folds N
    //      mul_plain rescales into one), then optional giant rotate
    //   4. add giant accumulators into a single result ciphertext.
    //
    // Inputs (all parallel, CSR-style):
    //   baby_shifts        – distinct ``b`` values; ``b == 0`` skips rotation
    //   giant_shifts       – distinct ``g`` values per accumulator
    //   bucket_offsets     – size = giant_shifts.size() + 1 (CSR row ptrs)
    //   bucket_baby_idx    – per term: index into baby_shifts
    //   bucket_masks       – per term: pre-encoded mask plaintext at the
    //                        SAME chain depth as ``ct``
    //
    // Caller is responsible for:
    //   * Galois keys for every shift in ``baby_shifts ∪ giant_shifts``
    //     (excluding 0) being already generated.
    //   * Plaintext masks being mod-dropped to ct.depth() before the call.
    //
    // Output: single ciphertext at depth ``ct.depth() + 1`` (one rescale).
    Ciphertext gather_slots_bsgs(
        Ciphertext& ct,
        GaloisKey&  gk,
        const std::vector<int>&       baby_shifts,
        const std::vector<int>&       giant_shifts,
        const std::vector<int>&       bucket_offsets,
        const std::vector<int>&       bucket_baby_idx,
        const std::vector<Plaintext>& bucket_masks)
    {
        if (baby_shifts.empty() || giant_shifts.empty()) {
            throw std::invalid_argument(
                "gather_slots_bsgs: baby_shifts/giant_shifts must be non-empty");
        }
        if (bucket_offsets.size() != giant_shifts.size() + 1) {
            throw std::invalid_argument(
                "gather_slots_bsgs: bucket_offsets size must be "
                "giant_shifts.size() + 1");
        }
        const int total_terms = bucket_offsets.back();
        if ((int) bucket_baby_idx.size() != total_terms ||
            (int) bucket_masks.size()    != total_terms) {
            throw std::invalid_argument(
                "gather_slots_bsgs: bucket_baby_idx / bucket_masks length "
                "mismatch with bucket_offsets.back()");
        }

        // 1. Pre-rotate. Always copy so subsequent in-place mul_plain on
        //    a baby_rot doesn't clobber the input ciphertext.
        std::vector<heongpu::Ciphertext<SCHEME>> baby_rots;
        baby_rots.reserve(baby_shifts.size());
        for (int b : baby_shifts) {
            baby_rots.emplace_back(*ct.ct);   // copy
            if (b != 0) {
                ops.rotate_rows_inplace(baby_rots.back(), *gk.gk, b);
            }
        }

        // 2-4. Per-giant accumulate, lazy rescale, optional giant-rotate,
        //      sum across giants.
        std::shared_ptr<heongpu::Ciphertext<SCHEME>> result;

        for (size_t gi = 0; gi < giant_shifts.size(); ++gi) {
            const int start = bucket_offsets[gi];
            const int end   = bucket_offsets[gi + 1];
            if (start == end) continue;          // empty bucket — skip

            // First term initialises the accumulator (avoids encrypting 0).
            heongpu::Ciphertext<SCHEME> acc =
                baby_rots[bucket_baby_idx[start]];   // copy
            ops.multiply_plain_inplace(acc, *bucket_masks[start].pt);

            for (int j = start + 1; j < end; ++j) {
                heongpu::Ciphertext<SCHEME> term =
                    baby_rots[bucket_baby_idx[j]];   // copy
                ops.multiply_plain_inplace(term, *bucket_masks[j].pt);
                ops.add_inplace(acc, term);
            }

            // Single rescale for this giant — folds N mul_plain rescales
            // into one. All N terms shared the same scale so adds were
            // valid; after rescale the giant rotate sees a clean cipher.
            ops.rescale_inplace(acc);

            if (giant_shifts[gi] != 0) {
                ops.rotate_rows_inplace(acc, *gk.gk, giant_shifts[gi]);
            }

            if (!result) {
                result = std::make_shared<heongpu::Ciphertext<SCHEME>>(acc);
            } else {
                ops.add_inplace(*result, acc);
            }
        }

        if (!result) {
            // All buckets were empty — caller bug, but fail soft by
            // returning a copy of the input untouched.
            result = std::make_shared<heongpu::Ciphertext<SCHEME>>(*ct.ct);
        }
        return Ciphertext{result};
    }

    // -------------------------------------------------------------------------
    // Phase 7b: batched matrix-packed Halevi-Shoup matmul.
    //
    // Computes one block-cyclic Halevi-Shoup matmul:
    //
    //   result = Σ_k  perBlockRotate(ct_x, shifts[k]) ⊙ diag_pts[k]
    //
    // where perBlockRotate(ct, s) = mask_low ⊙ rotate(ct, s) +
    //                                mask_high ⊙ rotate(ct, s - block).
    //
    // For shifts[k]==0 the rotate-then-mask step is skipped; the
    // diagonal multiplies the input ct directly. The caller arranges
    // for a same-depth output via add_inplace_match (the i==0 term lands
    // 1 level shallower than the rotated terms; add_inplace_match
    // mod-drops the deeper operand on the fly).
    //
    // Pre-conditions:
    //   * ct_x already replicated within each block (caller does this).
    //   * GaloisKey contains keys for all unique non-zero `shifts[k]`
    //     AND for all `shifts[k] - block` values.
    //   * diag_pts[k] is encoded at depth = ct_x.depth() + 1
    //     (i.e. matching the post-mask, post-rescale depth of the
    //     rotated terms).
    //   * low_pts[k]/high_pts[k] are the per_block_rotate masks for
    //     shifts[k] (encoded at ct_x.depth()), or empty Plaintexts
    //     when shifts[k]==0 (slot is unused).
    //   * If with_bias is true, bias_pt is at depth ct_x.depth() + 2
    //     (i.e. final result depth).
    //
    // Result: Ciphertext at depth ct_x.depth() + 2.
    Ciphertext halevi_shoup_matvec_block(
        Ciphertext& ct_x,
        GaloisKey&  gk,
        int block,
        const std::vector<int>&       shifts,
        const std::vector<Plaintext>& diag_pts,
        const std::vector<Plaintext>& low_pts,
        const std::vector<Plaintext>& high_pts,
        bool                          with_bias,
        Plaintext&                    bias_pt)
    {
        const size_t K = shifts.size();
        if (diag_pts.size() != K || low_pts.size() != K || high_pts.size() != K) {
            throw std::invalid_argument(
                "halevi_shoup_matvec_block: shifts/diag_pts/low_pts/high_pts "
                "size mismatch");
        }
        if (K == 0) {
            throw std::invalid_argument(
                "halevi_shoup_matvec_block: shifts must be non-empty");
        }

        std::shared_ptr<heongpu::Ciphertext<SCHEME>> result;

        for (size_t k = 0; k < K; ++k) {
            const int s = shifts[k];

            heongpu::Ciphertext<SCHEME> rot_x;
            int rot_x_depth_target;
            if (s == 0) {
                // Identity branch: rot_x stays at ct_x.depth().
                rot_x = *ct_x.ct;
                rot_x_depth_target = ct_x.ct->depth();
            } else {
                // per_block_rotate_left(s).
                heongpu::Ciphertext<SCHEME> rot_left  = *ct_x.ct;
                ops.rotate_rows_inplace(rot_left, *gk.gk, s);
                ops.multiply_plain_inplace(rot_left, *low_pts[k].pt);

                heongpu::Ciphertext<SCHEME> rot_right = *ct_x.ct;
                ops.rotate_rows_inplace(rot_right, *gk.gk, s - block);
                ops.multiply_plain_inplace(rot_right, *high_pts[k].pt);

                ops.add_inplace(rot_left, rot_right);
                ops.rescale_inplace(rot_left);    // → depth d+1
                rot_x = std::move(rot_left);
                rot_x_depth_target = ct_x.ct->depth() + 1;
            }

            // Multiply by diagonal: caller pre-encoded diag_pts[k] at the
            // correct depth (depth_d for s==0 paths, depth_d+1 for s!=0).
            // We DO NOT mod-drop inside C++ — the Python wrapper builds
            // each plaintext at the exact depth needed.
            ops.multiply_plain_inplace(rot_x, *diag_pts[k].pt);
            ops.rescale_inplace(rot_x);

            if (!result) {
                result = std::make_shared<heongpu::Ciphertext<SCHEME>>(std::move(rot_x));
            } else {
                int dr = result->depth();
                int dx = rot_x.depth();
                if (dr == dx) {
                    ops.add_inplace(*result, rot_x);
                } else if (dr < dx) {
                    while (result->depth() < dx) ops.mod_drop_inplace(*result);
                    ops.add_inplace(*result, rot_x);
                } else {
                    while (rot_x.depth() < dr) ops.mod_drop_inplace(rot_x);
                    ops.add_inplace(*result, rot_x);
                }
            }
        }

        if (with_bias) {
            ops.add_plain_inplace(*result, *bias_pt.pt);
        }

        return Ciphertext{result};
    }

    // ────────────────────────────────────────────────────────────────
    // Phase 7b-BSGS: baby-step giant-step matrix-packed Halevi-Shoup.
    //
    // For n diagonals factored as n = b1 * b2 with index i = g*b1 + j:
    //
    //   y = Σ_g per_block_rot(g*b1) ( Σ_j per_block_rot(-g*b1)(diag_{g*b1+j})
    //                                  ⊙ per_block_rot(j)(x) )
    //
    // Galois keys: 2*(b1-1) baby + 2*(b2-1) giant (vs n*2 in linear HS).
    // ────────────────────────────────────────────────────────────────

    // 1) Pre-rotate ct_x by each baby shift (per_block_rotate_left).
    //    baby_shifts[0] must equal 0 (identity, returned as a copy at
    //    depth d). The remaining babies land at depth d+1.
    std::vector<Ciphertext> pre_rotate_babies(
        Ciphertext& ct_x,
        GaloisKey&  gk,
        int block,
        const std::vector<int>&       baby_shifts,
        const std::vector<Plaintext>& low_pts,
        const std::vector<Plaintext>& high_pts)
    {
        const size_t b1 = baby_shifts.size();
        if (b1 == 0 || baby_shifts[0] != 0) {
            throw std::invalid_argument(
                "pre_rotate_babies: baby_shifts must start with 0");
        }
        if (low_pts.size() != b1 || high_pts.size() != b1) {
            throw std::invalid_argument(
                "pre_rotate_babies: low_pts/high_pts size != baby_shifts");
        }

        std::vector<Ciphertext> out;
        out.reserve(b1);
        // Identity baby (j=0): copy at depth d.
        {
            auto p = std::make_shared<heongpu::Ciphertext<SCHEME>>(*ct_x.ct);
            out.emplace_back(Ciphertext{p});
        }
        for (size_t j = 1; j < b1; ++j) {
            heongpu::Ciphertext<SCHEME> rl = *ct_x.ct;
            ops.rotate_rows_inplace(rl, *gk.gk, baby_shifts[j]);
            ops.multiply_plain_inplace(rl, *low_pts[j].pt);

            heongpu::Ciphertext<SCHEME> rr = *ct_x.ct;
            ops.rotate_rows_inplace(rr, *gk.gk, baby_shifts[j] - block);
            ops.multiply_plain_inplace(rr, *high_pts[j].pt);

            ops.add_inplace(rl, rr);
            ops.rescale_inplace(rl);   // → depth d+1
            auto p = std::make_shared<heongpu::Ciphertext<SCHEME>>(std::move(rl));
            out.emplace_back(Ciphertext{p});
        }
        return out;
    }

    // 2) Per giant: inner product over b1 babies, then giant per-block rotate.
    //
    //    babies[0]      at depth d   (identity)
    //    babies[1..b1-1] at depth d+1 (post-baby-rescale)
    //
    //    For each giant g in giant_shifts:
    //      diag0_pts[g] is the j=0 diag (per-block rolled by -g*b1) at depth d.
    //      diagj_pts[g*(b1-1) + (j-1)] is the j>=1 diag at depth d+1.
    //
    //      acc_j0 = babies[0] ⊙ diag0_pts[g]                  (depth d)
    //      rescale(acc_j0)                                     (depth d+1)
    //      if b1 > 1:
    //        acc_j  = babies[1] ⊙ diagj_pts[…0]                (lazy: scale²)
    //        for j in 2..b1-1: acc_j += babies[j] ⊙ diagj_pts[…j-1]
    //        rescale(acc_j)                                    (depth d+2)
    //        acc_j0 = add_match(acc_j0, acc_j)                 (depth d+2)
    //      else acc_j0 stays at depth d+1.
    //
    //      if giant_shifts[g] != 0: per_block_rotate by g*b1
    //        using giant_low_pts[g] / giant_high_pts[g] at acc's depth.
    //        rescale → depth +1 again.
    //
    //      Sum into chunk result via add_inplace_match.
    //
    // Returns chunk result at the deepest depth produced (d+2 or d+3).
    Ciphertext bsgs_giant_chunk(
        const std::vector<Ciphertext>& babies,
        GaloisKey& gk,
        int block,
        const std::vector<int>&       giant_shifts,   // chunk subset
        const std::vector<Plaintext>& diag0_pts,      // size = giant_shifts.size()
        const std::vector<Plaintext>& diagj_pts,      // size = giant_shifts.size() * (b1-1)
        const std::vector<Plaintext>& giant_low_pts,  // size = giant_shifts.size() (idx unused if g==0)
        const std::vector<Plaintext>& giant_high_pts)
    {
        const size_t b1 = babies.size();
        const size_t G  = giant_shifts.size();
        if (G == 0) throw std::invalid_argument("bsgs_giant_chunk: empty");
        if (diag0_pts.size() != G) throw std::invalid_argument("diag0 size");
        if (diagj_pts.size() != G * (b1 - 1)) throw std::invalid_argument("diagj size");
        if (giant_low_pts.size() != G || giant_high_pts.size() != G)
            throw std::invalid_argument("giant masks size");

        std::shared_ptr<heongpu::Ciphertext<SCHEME>> result;

        for (size_t g = 0; g < G; ++g) {
            // j=0 path (depth d).
            heongpu::Ciphertext<SCHEME> acc_j0 = *babies[0].ct;
            ops.multiply_plain_inplace(acc_j0, *diag0_pts[g].pt);
            ops.rescale_inplace(acc_j0);                       // → d+1

            if (b1 > 1) {
                // Lazy-rescale inner product over j>=1.
                heongpu::Ciphertext<SCHEME> acc_j = *babies[1].ct;
                ops.multiply_plain_inplace(acc_j, *diagj_pts[g * (b1 - 1) + 0].pt);
                for (size_t j = 2; j < b1; ++j) {
                    heongpu::Ciphertext<SCHEME> term = *babies[j].ct;
                    ops.multiply_plain_inplace(term, *diagj_pts[g * (b1 - 1) + (j - 1)].pt);
                    ops.add_inplace(acc_j, term);
                }
                ops.rescale_inplace(acc_j);                    // → d+2

                // Combine: acc_j0 at d+1, acc_j at d+2.
                while (acc_j0.depth() < acc_j.depth()) ops.mod_drop_inplace(acc_j0);
                ops.add_inplace(acc_j0, acc_j);                // at d+2
            }

            // Giant per-block rotate.
            const int gs = giant_shifts[g];
            if (gs != 0) {
                heongpu::Ciphertext<SCHEME> rl = acc_j0;
                ops.rotate_rows_inplace(rl, *gk.gk, gs);
                ops.multiply_plain_inplace(rl, *giant_low_pts[g].pt);
                heongpu::Ciphertext<SCHEME> rr = acc_j0;
                ops.rotate_rows_inplace(rr, *gk.gk, gs - block);
                ops.multiply_plain_inplace(rr, *giant_high_pts[g].pt);
                ops.add_inplace(rl, rr);
                ops.rescale_inplace(rl);                       // → d+3
                acc_j0 = std::move(rl);
            }

            if (!result) {
                result = std::make_shared<heongpu::Ciphertext<SCHEME>>(std::move(acc_j0));
            } else {
                int dr = result->depth();
                int dx = acc_j0.depth();
                if (dr == dx) ops.add_inplace(*result, acc_j0);
                else if (dr < dx) {
                    while (result->depth() < dx) ops.mod_drop_inplace(*result);
                    ops.add_inplace(*result, acc_j0);
                } else {
                    while (acc_j0.depth() < dr) ops.mod_drop_inplace(acc_j0);
                    ops.add_inplace(*result, acc_j0);
                }
            }
        }

        return Ciphertext{result};
    }

    // ────────────────────────────────────────────────────────────────
    // Phase 7d-2: batched diagonal attention in C++.
    //
    // Eliminates Python overhead for the L-iteration diagonal loops in
    // enc_qk_scores_diagonal / enc_attention_apply_diagonal. Each loop
    // body has ~30 calls (rotate, mul, add, mul_plain) — at L=128 that's
    // ~4000 Python<->C++ round trips per layer per head. Moving the
    // whole loop to C++ removes GIL/dispatch overhead.
    //
    // All per-d plaintexts are pre-encoded by the Python wrapper at the
    // correct depth so we never call encode/mod_drop inside the kernel.
    // ────────────────────────────────────────────────────────────────

    // diag_qk_scores: Q@K^T in diagonal layout.
    //
    //  Inputs:
    //    Q_at, K_cyc      — already in attn layout, K_cyc replicated to 2L.
    //    gk               — galois keys covering rotations:
    //                       {-block_attn..-1, +1..+block_attn,
    //                        d*block_attn for d in [1,L)}
    //    L, block_attn, head_dim, num_slots
    //    scale_pt         — encoded mask: scale at slot 0 of each block,
    //                       0 elsewhere; at depth Q.depth() + 1 (post mul).
    //                       Actually we mul Q*K → depth +1; rescale → depth +1.
    //                       Then mul_plain by scale_pt → depth +1; rescale → +2.
    //                       So scale_pt encoded at Q.depth() + 1.
    //    col_pts[d]       — keep-slot-d mask (1.0 at slot d of each token
    //                       block, 0 elsewhere); at depth Q.depth() + 2.
    //                       (We mul_plain after the per-block-sum rescale.)
    //
    //  Algorithm per d in [0, L):
    //    K_rot = (d==0) ? K_cyc : rotate(K_cyc, d * block_attn)
    //    prod  = Q_at * K_rot                  # depth +1
    //    rescale(prod)
    //    # bare-doubling sum within block_attn (no levels consumed since
    //    # we do not rescale after add):
    //    sum = prod; for s=1; s<block_attn; s<<=1: sum += rotate(sum, s)
    //    # single mul_plain by scale-at-slot-0 mask:
    //    sum = sum * scale_pt                  # depth +1
    //    rescale(sum)                           # → depth Q+2
    //    # value lives at slot 0 of each block. Right-rotate by d to put
    //    # it at slot d:
    //    if d > 0: sum = rotate(sum, -d)
    //    # Mask only slot d of each block (zero garbage we'd accumulate):
    //    # Actually slot d is the only nonzero — others are 0 — so we
    //    # CAN skip the mask. BUT: the bare-doubling sum populated
    //    # garbage in many slots, which the slot-0 mask zeroed; after
    //    # rotate-by-(-d) the value is at slot d and everything else
    //    # is zero. No mask needed. ✓
    //    S += sum
    //
    //  Result depth = Q.depth() + 2.
    Ciphertext diag_qk_scores(
        Ciphertext& Q_at,
        Ciphertext& K_cyc,
        GaloisKey&  gk,
        RelinKey&   rk,
        int L,
        int block_attn,
        int /*head_dim*/,
        int /*num_slots*/,
        Plaintext& scale_pt)
    {
        std::shared_ptr<heongpu::Ciphertext<SCHEME>> S;

        // Pre-align K to match Q's depth (Q*K requires equal depth).
        const int target_depth = Q_at.ct->depth();
        heongpu::Ciphertext<SCHEME> K_aligned = *K_cyc.ct;
        while (K_aligned.depth() < target_depth) {
            ops.mod_drop_inplace(K_aligned);
        }
        // (If Q is shallower, we'd need to drop Q instead; assume equal in practice.)

        for (int d = 0; d < L; ++d) {
            // K_rot
            heongpu::Ciphertext<SCHEME> K_rot = K_aligned;
            if (d > 0) {
                ops.rotate_rows_inplace(K_rot, *gk.gk, d * block_attn);
            }

            // prod = Q * K_rot ; rescale.
            heongpu::Ciphertext<SCHEME> prod = *Q_at.ct;
            ops.multiply_inplace(prod, K_rot);
            ops.relinearize_inplace(prod, *rk.rk);
            ops.rescale_inplace(prod);                       // depth +1

            // Bare-doubling sum across block_attn (no rescale between adds).
            heongpu::Ciphertext<SCHEME> sum = prod;
            for (int step = 1; step < block_attn; step <<= 1) {
                heongpu::Ciphertext<SCHEME> rot = sum;
                ops.rotate_rows_inplace(rot, *gk.gk, step);
                ops.add_inplace(sum, rot);
            }

            // Single scaled-mask multiply (collapses garbage, applies scale).
            ops.multiply_plain_inplace(sum, *scale_pt.pt);
            ops.rescale_inplace(sum);                        // depth +2

            // Place at slot d via global right-rotate by d (only for d>0).
            if (d > 0) {
                ops.rotate_rows_inplace(sum, *gk.gk, -d);
            }

            if (!S) {
                S = std::make_shared<heongpu::Ciphertext<SCHEME>>(std::move(sum));
            } else {
                ops.add_inplace(*S, sum);
            }
        }

        return Ciphertext{S};
    }

    // diag_attn_apply: A@V in attn layout.
    //
    //  Inputs:
    //    A_diag           — softmax output in diagonal layout.
    //    V_cyc            — V_at replicated to 2L copies for cyclic shift.
    //    gk               — covers rotations {1..head_dim/2, -1..-head_dim/2,
    //                       +1..+L for col-d alignment, d*block_attn for d in [1,L)}
    //    L, block_attn, head_dim, num_slots
    //    col_pts[d]       — keep-slot-d-of-each-block mask (1.0 at slot d
    //                       of each token block); at depth A_diag.depth().
    //
    //  Algorithm per d in [0, L):
    //    a_d   = A_diag * col_pts[d]           # depth A+1
    //    rescale(a_d)
    //    if d>0: a_d = rotate(a_d, d)          # value moves to slot 0
    //    bcast = a_d
    //    for s=1; s<head_dim; s<<=1: bcast += rotate(bcast, -s)
    //    V_rot = (d==0) ? V_cyc : rotate(V_cyc, d * block_attn)
    //    prod  = bcast * V_rot ; rescale.       # depth A+2
    //    Out += prod
    Ciphertext diag_attn_apply(
        Ciphertext& A_diag,
        Ciphertext& V_cyc,
        GaloisKey&  gk,
        RelinKey&   rk,
        int L,
        int block_attn,
        int head_dim,
        int /*num_slots*/,
        const std::vector<Plaintext>& col_pts)
    {
        if ((int)col_pts.size() != L) {
            throw std::invalid_argument(
                "diag_attn_apply: col_pts size must equal L");
        }

        // a_d is at A.depth() + 1 (after mask + rescale). V is at its own
        // depth. We need them aligned for the mul. Pre-clone V_cyc and
        // drop to match A.depth() + 1 (V is typically shallower).
        const int target_depth = A_diag.ct->depth() + 1;
        heongpu::Ciphertext<SCHEME> V_aligned = *V_cyc.ct;
        while (V_aligned.depth() < target_depth) {
            ops.mod_drop_inplace(V_aligned);
        }

        std::shared_ptr<heongpu::Ciphertext<SCHEME>> Out;

        for (int d = 0; d < L; ++d) {
            // Pick d-column.
            heongpu::Ciphertext<SCHEME> a_d = *A_diag.ct;
            ops.multiply_plain_inplace(a_d, *col_pts[d].pt);
            ops.rescale_inplace(a_d);                        // depth +1

            // Bring to slot 0 of each block.
            if (d > 0) {
                ops.rotate_rows_inplace(a_d, *gk.gk, d);
            }

            // Bare-doubling broadcast across head_dim slots (right-rotate).
            for (int step = 1; step < head_dim; step <<= 1) {
                heongpu::Ciphertext<SCHEME> rot = a_d;
                ops.rotate_rows_inplace(rot, *gk.gk, -step);
                ops.add_inplace(a_d, rot);
            }

            // V cyclic rotate (use V_aligned which is mod-dropped to a_d depth).
            heongpu::Ciphertext<SCHEME> V_rot = V_aligned;
            if (d > 0) {
                ops.rotate_rows_inplace(V_rot, *gk.gk, d * block_attn);
            }

            // Multiply, rescale, accumulate.
            heongpu::Ciphertext<SCHEME> prod = a_d;
            ops.multiply_inplace(prod, V_rot);
            ops.relinearize_inplace(prod, *rk.rk);
            ops.rescale_inplace(prod);                       // depth +2

            if (!Out) {
                Out = std::make_shared<heongpu::Ciphertext<SCHEME>>(std::move(prod));
            } else {
                ops.add_inplace(*Out, prod);
            }
        }

        return Ciphertext{Out};
    }
};

PYBIND11_MODULE(_heongpu, m) {
    m.doc() = "HEonGPU CKKS pybind11 bindings (Phase 1 smoke surface)";

    py::class_<CKKSContext>(m, "CKKSContext")
        .def(py::init<std::size_t,
                      const std::vector<int>&,
                      const std::vector<int>&,
                      bool>(),
             py::arg("poly_modulus_degree"),
             py::arg("q_bits"),
             py::arg("p_bits"),
             py::arg("sec_none") = false)
        .def("print_parameters", &CKKSContext::print_parameters)
        .def_property_readonly("slot_count", &CKKSContext::slot_count)
        .def_readonly("poly_modulus_degree", &CKKSContext::poly_modulus_degree);

    py::class_<SecretKey>(m, "SecretKey");
    py::class_<PublicKey>(m, "PublicKey");
    py::class_<RelinKey>(m, "RelinKey");
    py::class_<GaloisKey>(m, "GaloisKey");
    py::class_<Plaintext>(m, "Plaintext")
        .def("depth", [](const Plaintext& p) { return p.pt->depth(); })
        .def("scale", [](const Plaintext& p) { return p.pt->scale(); });
    py::class_<EncodingTransformContext>(m, "EncodingTransformContext")
        .def(py::init<>())
        .def("ctos_level", &EncodingTransformContext::ctos_level)
        .def("stoc_level", &EncodingTransformContext::stoc_level)
        .def("ctos_piece", &EncodingTransformContext::ctos_piece)
        .def("stoc_piece", &EncodingTransformContext::stoc_piece)
        .def("generated", &EncodingTransformContext::generated)
        .def("key_indexs", &EncodingTransformContext::key_indexs);
    py::class_<CudaStream>(m, "CudaStream",
            "RAII handle on a cudaStream_t. Pass to *_s op variants to "
            "submit work on a non-default stream for parallel execution.")
        .def(py::init<>())
        .def("synchronize", &CudaStream::synchronize,
             "Block until all work submitted on this stream completes.");
    py::class_<Ciphertext>(m, "Ciphertext")
        .def("depth",            [](const Ciphertext& c) { return c.ct->depth(); },
             "Multiplicative depth (number of rescales applied).")
        .def("scale",            [](const Ciphertext& c) { return c.ct->scale(); },
             "Current scale factor (in linear units, not log2).")
        .def("encoding_type",    [](const Ciphertext& c) {
                 return static_cast<int>(c.ct->encoding_type());
             }, "0 = SLOT, 1 = COEFFICIENT.")
        .def("rescale_required", [](const Ciphertext& c) { return c.ct->rescale_required(); });

    py::class_<KeyGenerator>(m, "KeyGenerator")
        .def(py::init<CKKSContext&>())
        .def("generate_secret_key", &KeyGenerator::generate_secret_key)
        .def("generate_secret_key_h",
             &KeyGenerator::generate_secret_key_h,
             py::arg("ctx"), py::arg("hamming_weight"))
        .def("generate_public_key", &KeyGenerator::generate_public_key)
        .def("generate_relin_key",  &KeyGenerator::generate_relin_key)
        .def("generate_galois_key", &KeyGenerator::generate_galois_key_default,
             py::arg("ctx"), py::arg("secret_key"))
        .def("generate_galois_key", &KeyGenerator::generate_galois_key_shifts,
             py::arg("ctx"), py::arg("secret_key"), py::arg("shifts"));

    py::class_<Encoder>(m, "Encoder")
        .def(py::init<CKKSContext&>())
        .def("encode", &Encoder::encode,
             py::arg("ctx"), py::arg("values"), py::arg("scale"))
        .def("decode", &Encoder::decode)
        .def("encode_coeff", &Encoder::encode_coeff,
             py::arg("ctx"), py::arg("values"), py::arg("scale"),
             "NEXUS-style coefficient packing: encode values as polynomial "
             "coefficients (not slots).")
        .def("decode_coeff", &Encoder::decode_coeff,
             "Decode a coefficient-encoded plaintext.");

    py::class_<Encryptor>(m, "Encryptor")
        .def(py::init<CKKSContext&, PublicKey&>())
        .def("encrypt", &Encryptor::encrypt);

    py::class_<Decryptor>(m, "Decryptor")
        .def(py::init<CKKSContext&, SecretKey&>())
        .def("decrypt", &Decryptor::decrypt);

    py::class_<Operator>(m, "Operator")
        .def(py::init<CKKSContext&, Encoder&>())
        .def("add_inplace",            &Operator::add_inplace)
        .def("sub_inplace",            &Operator::sub_inplace)
        .def("multiply_inplace",       &Operator::multiply_inplace)
        .def("multiply_plain_inplace", &Operator::multiply_plain_inplace)
        .def("add_plain_inplace",      &Operator::add_plain_inplace)
        .def("relinearize_inplace",    &Operator::relinearize_inplace)
        .def("rescale_inplace",        &Operator::rescale_inplace)
        .def("mod_drop_inplace_ct",    &Operator::mod_drop_inplace_ct)
        .def("mod_drop_inplace_pt",    &Operator::mod_drop_inplace_pt)
        .def("depth",                  &Operator::depth)
        .def("depth_of_plaintext",     &Operator::depth_pt)
        .def("add_inplace_match",      &Operator::add_inplace_match)
        .def("rotate_rows_inplace",    &Operator::rotate_rows_inplace,
             py::arg("ct"), py::arg("galois_key"), py::arg("shift"))
        // ── NEXUS Phase 5: stream-aware variants ──
        .def("multiply_plain_inplace_s", &Operator::multiply_plain_inplace_s,
             py::arg("ct"), py::arg("plain"), py::arg("stream"),
             py::call_guard<py::gil_scoped_release>())
        .def("add_inplace_s",            &Operator::add_inplace_s,
             py::arg("a"), py::arg("b"), py::arg("stream"),
             py::call_guard<py::gil_scoped_release>())
        .def("rotate_rows_inplace_s",    &Operator::rotate_rows_inplace_s,
             py::arg("ct"), py::arg("galois_key"), py::arg("shift"), py::arg("stream"),
             py::call_guard<py::gil_scoped_release>())
        .def("rescale_inplace_s",        &Operator::rescale_inplace_s,
             py::arg("ct"), py::arg("stream"),
             py::call_guard<py::gil_scoped_release>())
        .def("mod_drop_inplace_ct_s",    &Operator::mod_drop_inplace_ct_s,
             py::arg("ct"), py::arg("stream"),
             py::call_guard<py::gil_scoped_release>())
        .def("mod_drop_inplace_pt_s",    &Operator::mod_drop_inplace_pt_s,
             py::arg("plain"), py::arg("stream"),
             py::call_guard<py::gil_scoped_release>())
        .def("relinearize_inplace_s",    &Operator::relinearize_inplace_s,
             py::arg("ct"), py::arg("relin_key"), py::arg("stream"),
             py::call_guard<py::gil_scoped_release>())
        .def("clone_ct_s",               &Operator::clone_ct_s,
             py::arg("ct"), py::arg("stream"))
        .def("generate_bootstrapping_params",
             &Operator::generate_bootstrapping_params,
             py::arg("scale"),
             py::arg("CtoS_piece") = 3,
             py::arg("StoC_piece") = 3,
             py::arg("taylor_number") = 11,
             py::arg("less_key_mode") = true)
        .def("bootstrapping_key_indexs", &Operator::bootstrapping_key_indexs)
        .def("regular_bootstrapping",    &Operator::regular_bootstrapping,
             py::arg("ct"), py::arg("galois_key"), py::arg("relin_key"))
        .def("clone_ct",                 &Operator::clone_ct,
             "Shallow copy preserving depth/scale/encoding metadata.")
        .def("coeff_to_slot",            &Operator::coeff_to_slot,
             py::arg("ct"), py::arg("galois_key"),
             "NEXUS Phase 3: returns 2 cts; slots of [0]/[1] hold polynomial coefficients [0..N/2) / [N/2..N).")
        .def("slot_to_coeff",            &Operator::slot_to_coeff,
             py::arg("ct0"), py::arg("ct1"), py::arg("galois_key"),
             "NEXUS Phase 3: inverse of coeff_to_slot.")
        .def("generate_encoding_transform_context",
             &Operator::generate_encoding_transform_context,
             py::arg("ctx"), py::arg("scale"),
             py::arg("ctos_piece") = 3, py::arg("stoc_piece") = 3,
             py::arg("ctos_start_level") = -1, py::arg("stoc_start_level") = -1,
             py::arg("less_key_mode") = true,
             "NEXUS Phase 4: precompute V/V_inv plaintext matrices for "
             "CtoS/StoC at arbitrary (ctos_start, stoc_start) chain levels.")
        .def("coeff_to_slot_ctx",        &Operator::coeff_to_slot_ctx,
             py::arg("ct"), py::arg("galois_key"), py::arg("transform_ctx"),
             "Phase 4: ctx-aware CtoS — input depth must equal ctx.ctos_level().")
        .def("slot_to_coeff_ctx",        &Operator::slot_to_coeff_ctx,
             py::arg("ct0"), py::arg("ct1"), py::arg("galois_key"),
             py::arg("transform_ctx"),
             "Phase 4: ctx-aware StoC — input depth must equal ctx.stoc_level().")
        .def("bootstrapping_scale",  &Operator::bootstrapping_scale,
             "Internal scale (`scale_boot_`) used by the CtoS/StoC matrices.")
        .def("coeff_to_slot_level",  &Operator::coeff_to_slot_level,
             "Chain level at which CtoS plaintext matrices live.")
        .def("slot_to_coeff_level",  &Operator::slot_to_coeff_level,
             "Chain level at which StoC plaintext matrices live.")
        .def("bootstrapping_ready",  &Operator::bootstrapping_ready,
             "True iff generate_bootstrapping_params has been called.")
        .def("clear_rescale_required", &Operator::clear_rescale_required,
             "NEXUS escape hatch: clear rescale_required_ on a ciphertext "
             "(use only when feeding into ops that handle scale alignment "
             "internally, e.g. coeff_to_slot).")
        .def("set_rescale_required", &Operator::set_rescale_required,
             "NEXUS escape hatch: force rescale_required_=true so a "
             "follow-up rescale_inplace can drop a Q prime to bring "
             "scale back to canonical (e.g. after CtoS leaves us at scale²).")
        .def("gather_slots_bsgs",   &Operator::gather_slots_bsgs,
             py::arg("ct"), py::arg("galois_key"),
             py::arg("baby_shifts"), py::arg("giant_shifts"),
             py::arg("bucket_offsets"), py::arg("bucket_baby_idx"),
             py::arg("bucket_masks"),
             py::call_guard<py::gil_scoped_release>(),
             "NEXUS Phase 6: batched BSGS gather inside one C++ call. "
             "Pre-rotates ct by each baby shift, accumulates per-giant "
             "Σ baby_rot ⊙ mask, single rescale per giant, optional "
             "giant-rotate, sums giants. Eliminates Python wrapper "
             "overhead (encoding masks must be pre-built and at "
             "ct.depth()). Result depth = ct.depth() + 1.")
        .def("halevi_shoup_matvec_block", &Operator::halevi_shoup_matvec_block,
             py::arg("ct_x"), py::arg("galois_key"),
             py::arg("block"),
             py::arg("shifts"),
             py::arg("diag_pts"),
             py::arg("low_pts"),
             py::arg("high_pts"),
             py::arg("with_bias"),
             py::arg("bias_pt"),
             py::call_guard<py::gil_scoped_release>(),
             "Phase 7b: batched matrix-packed Halevi-Shoup matmul in C++. "
             "Eliminates the Python loop over n diagonals "
             "(per_block_rotate + mul_plain + add). Caller pre-encodes all "
             "diagonal_pts at ct_x.depth()+1 and the (low,high) per_block_rotate "
             "masks at ct_x.depth(). For shifts[k]==0 the (low,high) masks are "
             "ignored; the diagonal multiplies ct_x directly. Result depth = "
             "ct_x.depth() + 2.")
        .def("encode_many_drop", &Operator::encode_many_drop,
             py::arg("vs"), py::arg("scale"), py::arg("target_depth"),
             py::call_guard<py::gil_scoped_release>(),
             "Phase 7b-BSGS: encode + mod-drop a batch of vectors in one "
             "C++ call. Eliminates Python<->C++ boundary crossings for "
             "thousands of per-matmul plaintext encodings.")
        .def("pre_rotate_babies", &Operator::pre_rotate_babies,
             py::arg("ct_x"), py::arg("galois_key"), py::arg("block"),
             py::arg("baby_shifts"), py::arg("low_pts"), py::arg("high_pts"),
             py::call_guard<py::gil_scoped_release>(),
             "Phase 7b-BSGS: pre-rotate ct_x by each baby shift "
             "(per_block_rotate_left). baby_shifts[0] must be 0; the "
             "identity baby is returned at ct_x.depth(); the rest at depth+1.")
        .def("bsgs_giant_chunk", &Operator::bsgs_giant_chunk,
             py::arg("babies"), py::arg("galois_key"), py::arg("block"),
             py::arg("giant_shifts"),
             py::arg("diag0_pts"), py::arg("diagj_pts"),
             py::arg("giant_low_pts"), py::arg("giant_high_pts"),
             py::call_guard<py::gil_scoped_release>(),
             "Phase 7b-BSGS: per-chunk BSGS giant accumulation. Inner "
             "product over b1 babies with lazy rescale, then giant "
             "per_block_rotate, summed across the chunk.")
        .def("diag_qk_scores", &Operator::diag_qk_scores,
             py::arg("Q_at"), py::arg("K_cyc"),
             py::arg("galois_key"), py::arg("relin_key"),
             py::arg("L"), py::arg("block_attn"),
             py::arg("head_dim"), py::arg("num_slots"),
             py::arg("scale_pt"),
             py::call_guard<py::gil_scoped_release>(),
             "Phase 7d-2: batched diagonal Q@K^T attention scores. "
             "Runs the full L-iteration loop in C++ with a single GIL "
             "release; eliminates per-d Python overhead. K_cyc must be "
             "the 2L-replicated K_at; scale_pt encodes the slot-0-each-block "
             "mask scaled by 1/sqrt(d), at depth Q.depth() + 1.")
        .def("diag_attn_apply", &Operator::diag_attn_apply,
             py::arg("A_diag"), py::arg("V_cyc"),
             py::arg("galois_key"), py::arg("relin_key"),
             py::arg("L"), py::arg("block_attn"),
             py::arg("head_dim"), py::arg("num_slots"),
             py::arg("col_pts"),
             py::call_guard<py::gil_scoped_release>(),
             "Phase 7d-2: batched diagonal A@V attention apply. "
             "Runs the L-iteration loop in C++ with a single GIL "
             "release. col_pts[d] is the slot-d-of-each-block mask at "
             "depth A_diag.depth().");
}
