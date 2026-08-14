// tritnet_inference.h -- TritNet C++ inference engine (Phase 3)
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
// =============================================================================
// DESIGN
// =============================================================================
//
// Replaces the memory-bound LUT lookup (src/core/algebra/ternary_algebra.h's
// tadd/tmul/tmin/tmax/tnot, one scalar `trit` at a time) with a compute-bound
// neural network forward pass operating on a 5-trit chunk at once, per
// CLAUDE.md's TritNet vision. This is Phase 3: naive (non-SIMD) scalar C++
// inference over the real Phase 2A/2B GO checkpoints, exported to
// tritnet_weights.h. AVX2 vectorization of the forward pass is a later Phase 3
// step once this naive baseline is validated for correctness and benchmarked.
//
// Architecture per op (matches train_phase2a.py/train_phase2b.py's
// TritClassifier exactly -- see tritnet_weights.h's header comment for
// provenance):
//   h1 = ReLU(x @ W1 + b1)
//   h2 = ReLU(h1 @ W2 + b2)
//   logits = h2 @ W3 + b3, reshaped to [5, 3]
//   trit[k] = argmax(logits[k]) - 1        (class idx {0,1,2} -> value {-1,0,+1})
//
// Input/output convention: each public tritnet_* function takes/returns
// arrays of `trit` (src/core/algebra/ternary_algebra.h's 2-bit encoding,
// 0b00=-1, 0b01=0, 0b10=+1) so it's a drop-in comparison point against 5
// calls to the scalar LUT ops for the same 5-trit chunk -- see
// tests/cpp/test_tritnet_inference.cpp and
// benchmarks/cpp-native-kernels/bench_tritnet_inference.cpp.
//
// COMPILE: standalone header, no .cpp needed. Requires C++17.
//   Include path: this file expects src/core/algebra/ternary_algebra.h to be
//   reachable at "../../../src/core/algebra/ternary_algebra.h" (i.e. compiled
//   from its own directory, matching tests/cpp/test_luts.cpp's convention).

#pragma once

#include <cstdint>

#include "tritnet_weights.h"
#include "../../../src/core/algebra/ternary_algebra.h"  // trit, trit_to_int, int_to_trit

namespace tritnet {

namespace detail {

// Generic 3-layer forward pass (ReLU, ReLU, linear + argmax decode), shared
// by all 5 ops -- CLAUDE.md's "template_unification" convention: one
// implementation instead of 5 duplicated by hand. IN/HID/NOUT are compile-time
// so the compiler sees fixed-trip-count loops per op instantiation.
template<int IN, int HID, int NOUT>
inline void forward3(
    const float x[IN],
    const int8_t W1[IN][HID], const float B1[HID],
    const int8_t W2[HID][HID], const float B2[HID],
    const int8_t W3[HID][NOUT * 3], const float B3[NOUT * 3],
    int8_t out_trits[NOUT]  // plain {-1,0,+1}, NOT 2-bit `trit` encoding
) {
    float h1[HID];
    for (int j = 0; j < HID; ++j) {
        float acc = B1[j];
        for (int i = 0; i < IN; ++i) acc += x[i] * static_cast<float>(W1[i][j]);
        h1[j] = acc > 0.0f ? acc : 0.0f;
    }

    float h2[HID];
    for (int j = 0; j < HID; ++j) {
        float acc = B2[j];
        for (int i = 0; i < HID; ++i) acc += h1[i] * static_cast<float>(W2[i][j]);
        h2[j] = acc > 0.0f ? acc : 0.0f;
    }

    float logits[NOUT * 3];
    for (int j = 0; j < NOUT * 3; ++j) {
        float acc = B3[j];
        for (int i = 0; i < HID; ++i) acc += h2[i] * static_cast<float>(W3[i][j]);
        logits[j] = acc;
    }

    for (int k = 0; k < NOUT; ++k) {
        int best = 0;
        float best_val = logits[k * 3];
        for (int c = 1; c < 3; ++c) {
            if (logits[k * 3 + c] > best_val) {
                best_val = logits[k * 3 + c];
                best = c;
            }
        }
        out_trits[k] = static_cast<int8_t>(best - 1);
    }
}

inline void trits_to_floats(const trit* in, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = static_cast<float>(trit_to_int(in[i]));
}

inline void ints_to_trits(const int8_t* in, trit* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = int_to_trit(in[i]);
}

}  // namespace detail

/// TritNet inference for tnot: 1 chunk of 5 trits in, 5 trits out.
inline void tritnet_tnot(const trit in[5], trit out[5]) {
    using namespace weights::tnot_weights;
    float x[5];
    detail::trits_to_floats(in, x, 5);
    int8_t y[5];
    detail::forward3<IN_FEATURES, HIDDEN, N_OUT_TRITS>(x, W1, B1, W2, B2, W3, B3, y);
    detail::ints_to_trits(y, out, 5);
}

/// TritNet inference for tadd: 2 chunks of 5 trits in, 5 trits out.
inline void tritnet_tadd(const trit a[5], const trit b[5], trit out[5]) {
    using namespace weights::tadd_weights;
    float x[10];
    detail::trits_to_floats(a, x, 5);
    detail::trits_to_floats(b, x + 5, 5);
    int8_t y[5];
    detail::forward3<IN_FEATURES, HIDDEN, N_OUT_TRITS>(x, W1, B1, W2, B2, W3, B3, y);
    detail::ints_to_trits(y, out, 5);
}

/// TritNet inference for tmul: 2 chunks of 5 trits in, 5 trits out.
inline void tritnet_tmul(const trit a[5], const trit b[5], trit out[5]) {
    using namespace weights::tmul_weights;
    float x[10];
    detail::trits_to_floats(a, x, 5);
    detail::trits_to_floats(b, x + 5, 5);
    int8_t y[5];
    detail::forward3<IN_FEATURES, HIDDEN, N_OUT_TRITS>(x, W1, B1, W2, B2, W3, B3, y);
    detail::ints_to_trits(y, out, 5);
}

/// TritNet inference for tmin: 2 chunks of 5 trits in, 5 trits out.
inline void tritnet_tmin(const trit a[5], const trit b[5], trit out[5]) {
    using namespace weights::tmin_weights;
    float x[10];
    detail::trits_to_floats(a, x, 5);
    detail::trits_to_floats(b, x + 5, 5);
    int8_t y[5];
    detail::forward3<IN_FEATURES, HIDDEN, N_OUT_TRITS>(x, W1, B1, W2, B2, W3, B3, y);
    detail::ints_to_trits(y, out, 5);
}

/// TritNet inference for tmax: 2 chunks of 5 trits in, 5 trits out.
inline void tritnet_tmax(const trit a[5], const trit b[5], trit out[5]) {
    using namespace weights::tmax_weights;
    float x[10];
    detail::trits_to_floats(a, x, 5);
    detail::trits_to_floats(b, x + 5, 5);
    int8_t y[5];
    detail::forward3<IN_FEATURES, HIDDEN, N_OUT_TRITS>(x, W1, B1, W2, B2, W3, B3, y);
    detail::ints_to_trits(y, out, 5);
}

}  // namespace tritnet
