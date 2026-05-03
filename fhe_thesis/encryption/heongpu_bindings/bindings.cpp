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
    py::class_<Plaintext>(m, "Plaintext");
    py::class_<Ciphertext>(m, "Ciphertext");

    py::class_<KeyGenerator>(m, "KeyGenerator")
        .def(py::init<CKKSContext&>())
        .def("generate_secret_key", &KeyGenerator::generate_secret_key)
        .def("generate_public_key", &KeyGenerator::generate_public_key)
        .def("generate_relin_key",  &KeyGenerator::generate_relin_key);

    py::class_<Encoder>(m, "Encoder")
        .def(py::init<CKKSContext&>())
        .def("encode", &Encoder::encode,
             py::arg("ctx"), py::arg("values"), py::arg("scale"))
        .def("decode", &Encoder::decode);

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
        .def("depth",                  &Operator::depth);
}
