// scalar_golden_baseline.h — Golden baseline scalar implementations for verification
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Engine Project)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// PURPOSE:
// This header provides scalar reference implementations of all SIMD operations.
// These serve as the "golden baseline" for mathematical verification of SIMD kernels.
//
// DESIGN PRINCIPLE:
// Each function here is intentionally simple, readable, and uses only the scalar
// LUT operations. These are NOT optimized - they are CORRECT BY CONSTRUCTION.
// Performance is irrelevant; semantic correctness is everything.

#ifndef SCALAR_GOLDEN_BASELINE_H
#define SCALAR_GOLDEN_BASELINE_H

#include <cstdint>
#include <cstddef>
#include "../algebra/ternary_algebra.h"

namespace ternary_reference {

// --- Scalar Reference: Binary Operations ---
// These process arrays element-by-element using the scalar LUT operations

inline void tadd_scalar_ref(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = tadd(a[i], b[i]);
    }
}

inline void tmul_scalar_ref(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = tmul(a[i], b[i]);
    }
}

inline void tmin_scalar_ref(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = tmin(a[i], b[i]);
    }
}

inline void tmax_scalar_ref(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = tmax(a[i], b[i]);
    }
}

// --- Scalar Reference: Unary Operations ---

inline void tnot_scalar_ref(const uint8_t* a, uint8_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = tnot(a[i]);
    }
}

// --- Property Testing Helpers ---
// These verify algebraic properties that must hold for all operations

// Check commutativity: f(a, b) == f(b, a)
inline bool is_commutative(uint8_t (*op)(uint8_t, uint8_t), uint8_t a, uint8_t b) {
    return op(a, b) == op(b, a);
}

// Check associativity: f(f(a, b), c) == f(a, f(b, c))
inline bool is_associative(uint8_t (*op)(uint8_t, uint8_t), uint8_t a, uint8_t b, uint8_t c) {
    return op(op(a, b), c) == op(a, op(b, c));
}

// Check identity element: f(a, identity) == a
inline bool has_identity(uint8_t (*op)(uint8_t, uint8_t), uint8_t a, uint8_t identity) {
    return op(a, identity) == a;
}

// Check involution: f(f(a)) == a (for unary operations like tnot)
inline bool is_involution(uint8_t (*op)(uint8_t), uint8_t a) {
    return op(op(a)) == a;
}

// --- Exhaustive Property Verification ---
// Test all possible trit combinations (3^2 = 9 for binary, 3 for unary)

struct PropertyTestResult {
    bool passed;
    const char* operation;
    const char* property;
    int failing_input_a;
    int failing_input_b;
    int failing_input_c;
};

// Test tadd properties
inline PropertyTestResult test_tadd_properties() {
    const uint8_t trits[3] = {0b00, 0b01, 0b10}; // -1, 0, +1
    const uint8_t identity = 0b01; // 0 is identity for addition

    // Test commutativity: tadd(a, b) == tadd(b, a)
    for (uint8_t a : trits) {
        for (uint8_t b : trits) {
            if (!is_commutative(tadd, a, b)) {
                return {false, "tadd", "commutativity",
                        trit_to_int(a), trit_to_int(b), 0};
            }
        }
    }

    // Test identity: tadd(a, 0) == a
    for (uint8_t a : trits) {
        if (!has_identity(tadd, a, identity)) {
            return {false, "tadd", "identity",
                    trit_to_int(a), trit_to_int(identity), 0};
        }
    }

    // Note: tadd is NOT associative due to saturation
    // tadd(tadd(-1, -1), -1) = tadd(-1, -1) = -1
    // tadd(-1, tadd(-1, -1)) = tadd(-1, -1) = -1
    // But: tadd(tadd(+1, +1), +1) = tadd(+1, +1) = +1
    // and: tadd(+1, tadd(+1, +1)) = tadd(+1, +1) = +1
    // Saturation breaks associativity in general

    return {true, "tadd", "all", 0, 0, 0};
}

// Test tmul properties
inline PropertyTestResult test_tmul_properties() {
    const uint8_t trits[3] = {0b00, 0b01, 0b10}; // -1, 0, +1
    const uint8_t identity = 0b10; // +1 is identity for multiplication
    const uint8_t zero = 0b01; // 0 is absorbing element

    // Test commutativity: tmul(a, b) == tmul(b, a)
    for (uint8_t a : trits) {
        for (uint8_t b : trits) {
            if (!is_commutative(tmul, a, b)) {
                return {false, "tmul", "commutativity",
                        trit_to_int(a), trit_to_int(b), 0};
            }
        }
    }

    // Test associativity: tmul(tmul(a, b), c) == tmul(a, tmul(b, c))
    for (uint8_t a : trits) {
        for (uint8_t b : trits) {
            for (uint8_t c : trits) {
                if (!is_associative(tmul, a, b, c)) {
                    return {false, "tmul", "associativity",
                            trit_to_int(a), trit_to_int(b), trit_to_int(c)};
                }
            }
        }
    }

    // Test identity: tmul(a, +1) == a
    for (uint8_t a : trits) {
        if (!has_identity(tmul, a, identity)) {
            return {false, "tmul", "identity",
                    trit_to_int(a), trit_to_int(identity), 0};
        }
    }

    // Test absorbing element: tmul(a, 0) == 0
    for (uint8_t a : trits) {
        if (tmul(a, zero) != zero) {
            return {false, "tmul", "absorbing_element",
                    trit_to_int(a), trit_to_int(zero), 0};
        }
    }

    return {true, "tmul", "all", 0, 0, 0};
}

// Test tmin properties
inline PropertyTestResult test_tmin_properties() {
    const uint8_t trits[3] = {0b00, 0b01, 0b10}; // -1, 0, +1
    const uint8_t identity = 0b10; // +1 is identity for min

    // Test commutativity: tmin(a, b) == tmin(b, a)
    for (uint8_t a : trits) {
        for (uint8_t b : trits) {
            if (!is_commutative(tmin, a, b)) {
                return {false, "tmin", "commutativity",
                        trit_to_int(a), trit_to_int(b), 0};
            }
        }
    }

    // Test associativity: tmin(tmin(a, b), c) == tmin(a, tmin(b, c))
    for (uint8_t a : trits) {
        for (uint8_t b : trits) {
            for (uint8_t c : trits) {
                if (!is_associative(tmin, a, b, c)) {
                    return {false, "tmin", "associativity",
                            trit_to_int(a), trit_to_int(b), trit_to_int(c)};
                }
            }
        }
    }

    // Test identity: tmin(a, +1) == a
    for (uint8_t a : trits) {
        if (!has_identity(tmin, a, identity)) {
            return {false, "tmin", "identity",
                    trit_to_int(a), trit_to_int(identity), 0};
        }
    }

    // Test idempotence: tmin(a, a) == a
    for (uint8_t a : trits) {
        if (tmin(a, a) != a) {
            return {false, "tmin", "idempotence",
                    trit_to_int(a), trit_to_int(a), 0};
        }
    }

    return {true, "tmin", "all", 0, 0, 0};
}

// Test tmax properties
inline PropertyTestResult test_tmax_properties() {
    const uint8_t trits[3] = {0b00, 0b01, 0b10}; // -1, 0, +1
    const uint8_t identity = 0b00; // -1 is identity for max

    // Test commutativity: tmax(a, b) == tmax(b, a)
    for (uint8_t a : trits) {
        for (uint8_t b : trits) {
            if (!is_commutative(tmax, a, b)) {
                return {false, "tmax", "commutativity",
                        trit_to_int(a), trit_to_int(b), 0};
            }
        }
    }

    // Test associativity: tmax(tmax(a, b), c) == tmax(a, tmax(b, c))
    for (uint8_t a : trits) {
        for (uint8_t b : trits) {
            for (uint8_t c : trits) {
                if (!is_associative(tmax, a, b, c)) {
                    return {false, "tmax", "associativity",
                            trit_to_int(a), trit_to_int(b), trit_to_int(c)};
                }
            }
        }
    }

    // Test identity: tmax(a, -1) == a
    for (uint8_t a : trits) {
        if (!has_identity(tmax, a, identity)) {
            return {false, "tmax", "identity",
                    trit_to_int(a), trit_to_int(identity), 0};
        }
    }

    // Test idempotence: tmax(a, a) == a
    for (uint8_t a : trits) {
        if (tmax(a, a) != a) {
            return {false, "tmax", "idempotence",
                    trit_to_int(a), trit_to_int(a), 0};
        }
    }

    return {true, "tmax", "all", 0, 0, 0};
}

// Test tnot properties
inline PropertyTestResult test_tnot_properties() {
    const uint8_t trits[3] = {0b00, 0b01, 0b10}; // -1, 0, +1

    // Test involution: tnot(tnot(a)) == a
    for (uint8_t a : trits) {
        if (!is_involution(tnot, a)) {
            return {false, "tnot", "involution",
                    trit_to_int(a), 0, 0};
        }
    }

    // Test zero fixpoint: tnot(0) == 0
    if (tnot(0b01) != 0b01) {
        return {false, "tnot", "zero_fixpoint", 0, 0, 0};
    }

    // Test sign flip: tnot(-1) == +1 and tnot(+1) == -1
    if (tnot(0b00) != 0b10 || tnot(0b10) != 0b00) {
        return {false, "tnot", "sign_flip", 0, 0, 0};
    }

    return {true, "tnot", "all", 0, 0, 0};
}

} // namespace ternary_reference

#endif // TERNARY_SCALAR_REFERENCE_H
