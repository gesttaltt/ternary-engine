// test_tritnet_inference.cpp - Validate TritNet C++ inference engine (Phase 3)
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Core Experimental Project)
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
// Iterates the FULL input space for each of the 5 TritNet operations
// (3^5 = 243 for tnot, 3^10 = 59,049 for the binary ops) and compares
// models/tritnet/inference/tritnet_inference.h's forward-pass output,
// trit-by-trit, against ternary_algebra.h's scalar LUT ops (5 calls per
// chunk = the ground truth). Reports per-op exact-match-chunk accuracy,
// which should match the Python-side result.json numbers as a
// cross-language sanity check on the export.
//
// Compile: g++ -std=c++17 -O2 test_tritnet_inference.cpp -o test_tritnet_inference && ./test_tritnet_inference

#include <cstdio>
#include <cstdint>
#include "../../models/tritnet/inference/tritnet_inference.h"

// Recorded accuracy from each op's result.json (models/tritnet/phase2a/tnot/,
// phase2b/{tadd,tmul,tmin,tmax}/), verified again here independently in C++.
struct Expected { const char* op; double best_acc; };
static const Expected EXPECTED[] = {
    {"tnot", 1.0},
    {"tadd", 1.0},
    {"tmul", 0.99491947889328},
    {"tmin", 0.9988822937011719},
    {"tmax", 0.9985097050666809},
};

static void trit_vec_from_index(int idx, trit out[5]) {
    // idx in [0, 243): base-3 digits map to {-1,0,1} in the same order
    // train_phase2a.py/train_phase2b.py's _all_trit_vectors() generates them.
    for (int i = 4; i >= 0; --i) {
        int d = idx % 3;
        out[i] = int_to_trit(d - 1);
        idx /= 3;
    }
}

static bool trits_equal(const trit a[5], const trit b[5]) {
    for (int i = 0; i < 5; ++i) if (a[i] != b[i]) return false;
    return true;
}

static bool check_unary(const char* name,
                         void (*net_fn)(const trit[5], trit[5]),
                         trit (*lut_fn)(trit),
                         double expected_acc) {
    long n_total = 243, n_match = 0;
    for (int idx = 0; idx < 243; ++idx) {
        trit a[5];
        trit_vec_from_index(idx, a);

        trit expected[5];
        for (int i = 0; i < 5; ++i) expected[i] = lut_fn(a[i]);

        trit actual[5];
        net_fn(a, actual);

        if (trits_equal(expected, actual)) n_match++;
    }

    double acc = static_cast<double>(n_match) / n_total;
    long expected_match = static_cast<long>(expected_acc * n_total + 0.5);  // round to nearest
    bool ok = (n_match == expected_match);
    printf("  %-6s cpp_acc=%.6f%%  expected=%.6f%%  n_match=%ld/%ld (expected %ld)  %s\n",
           name, acc * 100.0, expected_acc * 100.0, n_match, n_total, expected_match,
           ok ? "PASS" : "FAIL");
    return ok;
}

static bool check_binary(const char* name,
                          void (*net_fn)(const trit[5], const trit[5], trit[5]),
                          trit (*lut_fn)(trit, trit),
                          double expected_acc) {
    long n_total = 243L * 243L, n_match = 0;
    for (int ia = 0; ia < 243; ++ia) {
        trit a[5];
        trit_vec_from_index(ia, a);
        for (int ib = 0; ib < 243; ++ib) {
            trit b[5];
            trit_vec_from_index(ib, b);

            trit expected[5];
            for (int i = 0; i < 5; ++i) expected[i] = lut_fn(a[i], b[i]);

            trit actual[5];
            net_fn(a, b, actual);

            if (trits_equal(expected, actual)) n_match++;
        }
    }

    double acc = static_cast<double>(n_match) / n_total;
    long expected_match = static_cast<long>(expected_acc * n_total + 0.5);  // round to nearest
    bool ok = (n_match == expected_match);
    printf("  %-6s cpp_acc=%.6f%%  expected=%.6f%%  n_match=%ld/%ld (expected %ld)  %s\n",
           name, acc * 100.0, expected_acc * 100.0, n_match, n_total, expected_match,
           ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    printf("======================================================================\n");
    printf("TritNet C++ Inference Engine -- Full Truth Table Verification\n");
    printf("======================================================================\n\n");

    bool all_ok = true;

    all_ok &= check_unary("tnot", tritnet::tritnet_tnot, tnot, EXPECTED[0].best_acc);
    all_ok &= check_binary("tadd", tritnet::tritnet_tadd, tadd, EXPECTED[1].best_acc);
    all_ok &= check_binary("tmul", tritnet::tritnet_tmul, tmul, EXPECTED[2].best_acc);
    all_ok &= check_binary("tmin", tritnet::tritnet_tmin, tmin, EXPECTED[3].best_acc);
    all_ok &= check_binary("tmax", tritnet::tritnet_tmax, tmax, EXPECTED[4].best_acc);

    printf("\n%s\n", all_ok ? "[SUCCESS] All operations match recorded checkpoint accuracy"
                             : "[FAIL] At least one operation diverged from its recorded accuracy");
    return all_ok ? 0 : 1;
}
