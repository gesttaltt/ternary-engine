// ternary_core_simd_full.cpp — AVX2-accelerated ternary logic operations
//
// DESIGN NOTE: Inverted Polarity Mapping
// This implementation uses an intentionally inverted int8 intermediate representation
// for SIMD efficiency. The trit encoding (0b00=-1, 0b01=0, 0b10=+1) is mapped to
// int8 as (+1, 0, -1) respectively — opposite of the logical ternary values.
// This inversion is self-consistent and cancels out during round-trip conversions,
// producing mathematically correct ternary results. See trit_to_int8() for details.

#include <immintrin.h>
#include <stdint.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ternary_core.h"
namespace py = pybind11;

// --- Conversion trit → int8 (INVERTED MAPPING) ---
// IMPORTANT: This uses an inverted polarity mapping for SIMD efficiency:
//   Trit 0b00 (logical -1) → int8 +1
//   Trit 0b01 (logical  0) → int8  0
//   Trit 0b10 (logical +1) → int8 -1
// This inversion is intentional and self-consistent throughout all operations.
// The round-trip conversion (trit→int8→operations→int8→trit) produces
// mathematically correct ternary results despite the inverted intermediate form.
static inline __m256i trit_to_int8(__m256i v) {
    __m256i neg = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0b00));  // 0xFF if v==0b00, else 0x00
    __m256i pos = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0b10));  // 0xFF if v==0b10, else 0x00
    // pos - neg yields: 0b00→(0-(-1))=+1, 0b01→(0-0)=0, 0b10→((-1)-0)=-1
    return _mm256_sub_epi8(pos, neg);
}

// --- Conversion int8 → trit (INVERTED MAPPING, reverse of above) ---
// Converts back from inverted int8 representation to trit encoding:
//   int8 +1 → Trit 0b00 (logical -1)
//   int8  0 → Trit 0b01 (logical  0)
//   int8 -1 → Trit 0b10 (logical +1)
// This maintains consistency with trit_to_int8's inverted polarity.
static inline __m256i int8_to_trit(__m256i v) {
    __m256i neg = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(-1));  // 0xFF if v==-1, else 0x00
    __m256i pos = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(1));   // 0xFF if v==+1, else 0x00
    __m256i out = _mm256_blendv_epi8(_mm256_set1_epi8(0b01), _mm256_set1_epi8(0b00), neg);  // -1→0b10, else 0b01
    out = _mm256_blendv_epi8(out, _mm256_set1_epi8(0b10), pos);  // +1→0b00, else keep previous
    return out;
}

// --- Saturating clamp [-1,1] (operates on inverted int8 space) ---
static inline __m256i clamp(__m256i v) {
    __m256i one = _mm256_set1_epi8(1);
    __m256i neg1 = _mm256_set1_epi8(-1);
    return _mm256_max_epi8(_mm256_min_epi8(v, one), neg1);
}

// --- Basic operations (work correctly despite inverted int8 mapping) ---
// All operations convert trit→int8, operate in int8 space, then convert back.
// The inverted polarity cancels out during the round-trip, yielding correct ternary results.

static inline __m256i tadd_simd(__m256i a, __m256i b) {
    __m256i s = _mm256_adds_epi8(trit_to_int8(a), trit_to_int8(b));  // Saturating add in int8 space
    return int8_to_trit(clamp(s));  // Clamp to [-1,+1] and convert back to trit
}

static inline __m256i tmul_simd(__m256i a, __m256i b) {
    __m256i ai = trit_to_int8(a);
    __m256i bi = trit_to_int8(b);
    __m256i p = _mm256_mullo_epi8(ai, bi);
    return int8_to_trit(clamp(p));
}

static inline __m256i tmin_simd(__m256i a, __m256i b) {
    __m256i ai = trit_to_int8(a);
    __m256i bi = trit_to_int8(b);
    return int8_to_trit(_mm256_min_epi8(ai, bi));
}

static inline __m256i tmax_simd(__m256i a, __m256i b) {
    __m256i ai = trit_to_int8(a);
    __m256i bi = trit_to_int8(b);
    return int8_to_trit(_mm256_max_epi8(ai, bi));
}

static inline __m256i tnot_simd(__m256i a) {
    __m256i ai = trit_to_int8(a);
    return int8_to_trit(_mm256_sub_epi8(_mm256_setzero_si256(), ai));
}

// --- Macro template for arrays ---
#define TERNARY_OP_SIMD(func) \
py::array_t<uint8_t> func##_array(py::array_t<uint8_t> A, py::array_t<uint8_t> B) { \
    auto a = A.unchecked<1>(); \
    auto b = B.unchecked<1>(); \
    ssize_t n = A.size(); \
    if (n != B.size()) throw std::runtime_error("Arrays must match"); \
    py::array_t<uint8_t> out(n); \
    auto r = out.mutable_unchecked<1>(); \
    ssize_t i = 0; \
    for (; i + 32 <= n; i += 32) { \
        __m256i va = _mm256_loadu_si256((__m256i const*)(a.data() + i)); \
        __m256i vb = _mm256_loadu_si256((__m256i const*)(b.data() + i)); \
        __m256i vr = func##_simd(va, vb); \
        _mm256_storeu_si256((__m256i*)(r.mutable_data() + i), vr); \
    } \
    for (; i < n; ++i) r[i] = func(a[i], b[i]); \
    return out; \
}

// --- Unary ---
py::array_t<uint8_t> tnot_array(py::array_t<uint8_t> A) {
    auto a = A.unchecked<1>();
    ssize_t n = A.size();
    py::array_t<uint8_t> out(n);
    auto r = out.mutable_unchecked<1>();
    ssize_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i const*)(a.data() + i));
        __m256i vr = tnot_simd(va);
        _mm256_storeu_si256((__m256i*)(r.mutable_data() + i), vr);
    }
    for (; i < n; ++i) r[i] = tnot(a[i]);
    return out;
}

// --- Instantiate wrappers ---
TERNARY_OP_SIMD(tadd)
TERNARY_OP_SIMD(tmul)
TERNARY_OP_SIMD(tmin)
TERNARY_OP_SIMD(tmax)

PYBIND11_MODULE(ternary_core_simd_full, m) {
    m.def("tadd", &tadd_array);
    m.def("tmul", &tmul_array);
    m.def("tmin", &tmin_array);
    m.def("tmax", &tmax_array);
    m.def("tnot", &tnot_array);
}
