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
    Operator(CKKSContext& c, Encoder& enc) : ops(c.ctx, enc.enc) {}

    void add_inplace(Ciphertext& a, Ciphertext& b)               { ops.add_inplace(*a.ct, *b.ct); }
    void sub_inplace(Ciphertext& a, Ciphertext& b)               { ops.sub_inplace(*a.ct, *b.ct); }
    void multiply_inplace(Ciphertext& a, Ciphertext& b)          { ops.multiply_inplace(*a.ct, *b.ct); }
    void multiply_plain_inplace(Ciphertext& a, Plaintext& p)     { ops.multiply_plain_inplace(*a.ct, *p.pt); }
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
};

// -----------------------------------------------------------------------------
// Module
// -----------------------------------------------------------------------------

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
    py::class_<Plaintext>(m, "Plaintext");
    py::class_<Ciphertext>(m, "Ciphertext");

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
        .def("relinearize_inplace",    &Operator::relinearize_inplace)
        .def("rescale_inplace",        &Operator::rescale_inplace)
        .def("mod_drop_inplace_ct",    &Operator::mod_drop_inplace_ct)
        .def("mod_drop_inplace_pt",    &Operator::mod_drop_inplace_pt)
        .def("depth",                  &Operator::depth)
        .def("depth_of_plaintext",     &Operator::depth_pt)
        .def("add_inplace_match",      &Operator::add_inplace_match)
        .def("rotate_rows_inplace",    &Operator::rotate_rows_inplace,
             py::arg("ct"), py::arg("galois_key"), py::arg("shift"))
        .def("generate_bootstrapping_params",
             &Operator::generate_bootstrapping_params,
             py::arg("scale"),
             py::arg("CtoS_piece") = 3,
             py::arg("StoC_piece") = 3,
             py::arg("taylor_number") = 11,
             py::arg("less_key_mode") = true)
        .def("bootstrapping_key_indexs", &Operator::bootstrapping_key_indexs)
        .def("regular_bootstrapping",    &Operator::regular_bootstrapping,
             py::arg("ct"), py::arg("galois_key"), py::arg("relin_key"));
}
