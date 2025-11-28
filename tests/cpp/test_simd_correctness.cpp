// test_simd_correctness.cpp — Comprehensive SIMD verification harness
//
// Copyright (c) 2025 Jonathan Verdun (Ternary Engine Project)
//
// This implements a 3-tier SIMD testing strategy:
//   Tier 1: SIMD vs Scalar golden reference
//   Tier 2: Algebraic property verification
//   Tier 3: Fuzz testing with alignment stress

#include <iostream>
#include <iomanip>
#include <random>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

#ifdef _WIN32
#include <malloc.h>  // _aligned_malloc
#else
#include <stdlib.h>  // posix_memalign
#endif

#include "../src/core/simd/simd_avx2_32trit_ops.h"
#include "../src/core/simd/scalar_golden_baseline.h"

using namespace ternary_reference;

// ============================================================================
// Memory Management Utilities
// ============================================================================

void* aligned_alloc_portable(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return nullptr;
    return ptr;
#endif
}

void aligned_free_portable(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// RAII wrapper for aligned memory
template<typename T>
class AlignedBuffer {
public:
    AlignedBuffer(size_t count, size_t alignment = 32)
        : count_(count), alignment_(alignment) {
        size_t bytes = count * sizeof(T);
        ptr_ = static_cast<T*>(aligned_alloc_portable(alignment, bytes));
        if (!ptr_) throw std::bad_alloc();
        std::memset(ptr_, 0, bytes);
    }

    ~AlignedBuffer() {
        if (ptr_) aligned_free_portable(ptr_);
    }

    // Non-copyable
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return count_; }

    // Get pointer with offset (for misalignment testing)
    T* get_offset(size_t byte_offset) {
        return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr_) + byte_offset);
    }

private:
    T* ptr_;
    size_t count_;
    size_t alignment_;
};

// ============================================================================
// Test Framework Infrastructure
// ============================================================================

struct TestResult {
    bool passed;
    const char* test_name;
    const char* failure_reason;
    size_t failing_index;
    uint8_t expected_value;
    uint8_t actual_value;
};

class TestHarness {
public:
    TestHarness() : passed_(0), failed_(0), total_(0) {}

    void record_pass(const char* name) {
        passed_++;
        total_++;
        std::cout << "  ✓ " << name << " [PASS]\n";
    }

    void record_fail(const TestResult& result) {
        failed_++;
        total_++;
        std::cout << "  ✗ " << result.test_name << " [FAIL]\n";
        std::cout << "    Reason: " << result.failure_reason << "\n";
        if (result.failing_index != SIZE_MAX) {
            std::cout << "    Index: " << result.failing_index << "\n";
            std::cout << "    Expected: 0x" << std::hex << (int)result.expected_value
                      << " (" << trit_to_int(result.expected_value) << ")\n";
            std::cout << "    Actual:   0x" << std::hex << (int)result.actual_value
                      << " (" << trit_to_int(result.actual_value) << ")" << std::dec << "\n";
        }
    }

    void print_summary() {
        std::cout << "\n========================================\n";
        std::cout << "  SIMD Correctness Test Summary\n";
        std::cout << "========================================\n";
        std::cout << "Total:  " << total_ << "\n";
        std::cout << "Passed: " << passed_ << " ✓\n";
        std::cout << "Failed: " << failed_ << (failed_ > 0 ? " ✗" : "") << "\n";
        std::cout << "========================================\n";

        if (failed_ == 0) {
            std::cout << "\n🎉 ALL SIMD TESTS PASSED!\n";
            std::cout << "   SIMD layer is mathematically verified.\n";
        } else {
            std::cout << "\n❌ SIMD TESTS FAILED!\n";
            std::cout << "   SIMD implementation has correctness issues.\n";
            std::cout << "   DO NOT USE IN PRODUCTION until fixed.\n";
        }
    }

    int exit_code() const { return (failed_ == 0) ? 0 : 1; }

private:
    int passed_;
    int failed_;
    int total_;
};

// ============================================================================
// TIER 1: SIMD vs Scalar Golden Reference Tests
// ============================================================================

// Helper: Compare two buffers element-wise
TestResult compare_buffers(const char* test_name, const uint8_t* expected,
                          const uint8_t* actual, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        // Mask to 2-bit trit values for comparison
        uint8_t exp = expected[i] & 0x03;
        uint8_t act = actual[i] & 0x03;
        if (exp != act) {
            return {false, test_name, "SIMD output differs from scalar reference",
                    i, exp, act};
        }
    }
    return {true, test_name, nullptr, SIZE_MAX, 0, 0};
}

// SIMD array processing helpers (process full arrays with SIMD + scalar tail)
void tadd_simd_array(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n) {
    size_t i = 0;
    // Process 32 elements at a time with SIMD
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vout = tadd_simd(va, vb);
        _mm256_storeu_si256((__m256i*)(out + i), vout);
    }
    // Scalar tail
    for (; i < n; ++i) {
        out[i] = tadd(a[i], b[i]);
    }
}

void tmul_simd_array(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vout = tmul_simd(va, vb);
        _mm256_storeu_si256((__m256i*)(out + i), vout);
    }
    for (; i < n; ++i) {
        out[i] = tmul(a[i], b[i]);
    }
}

void tmin_simd_array(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vout = tmin_simd(va, vb);
        _mm256_storeu_si256((__m256i*)(out + i), vout);
    }
    for (; i < n; ++i) {
        out[i] = tmin(a[i], b[i]);
    }
}

void tmax_simd_array(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i vout = tmax_simd(va, vb);
        _mm256_storeu_si256((__m256i*)(out + i), vout);
    }
    for (; i < n; ++i) {
        out[i] = tmax(a[i], b[i]);
    }
}

void tnot_simd_array(const uint8_t* a, uint8_t* out, size_t n) {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vout = tnot_simd(va);
        _mm256_storeu_si256((__m256i*)(out + i), vout);
    }
    for (; i < n; ++i) {
        out[i] = tnot(a[i]);
    }
}

// Test a single SIMD operation against scalar reference
template<typename BinaryOp, typename BinaryOpScalar>
TestResult test_simd_vs_scalar_binary(const char* op_name, BinaryOp simd_op,
                                      BinaryOpScalar scalar_op, size_t n) {
    AlignedBuffer<uint8_t> a(n), b(n), out_simd(n), out_scalar(n);

    // Initialize with all valid trit combinations
    const uint8_t trits[3] = {0b00, 0b01, 0b10};
    for (size_t i = 0; i < n; ++i) {
        a.get()[i] = trits[i % 3];
        b.get()[i] = trits[(i / 3) % 3];
    }

    // Run both implementations
    simd_op(a.get(), b.get(), out_simd.get(), n);
    scalar_op(a.get(), b.get(), out_scalar.get(), n);

    // Compare
    return compare_buffers(op_name, out_scalar.get(), out_simd.get(), n);
}

template<typename UnaryOp, typename UnaryOpScalar>
TestResult test_simd_vs_scalar_unary(const char* op_name, UnaryOp simd_op,
                                     UnaryOpScalar scalar_op, size_t n) {
    AlignedBuffer<uint8_t> a(n), out_simd(n), out_scalar(n);

    const uint8_t trits[3] = {0b00, 0b01, 0b10};
    for (size_t i = 0; i < n; ++i) {
        a.get()[i] = trits[i % 3];
    }

    simd_op(a.get(), out_simd.get(), n);
    scalar_op(a.get(), out_scalar.get(), n);

    return compare_buffers(op_name, out_scalar.get(), out_simd.get(), n);
}

void run_tier1_tests(TestHarness& harness) {
    std::cout << "\n========================================\n";
    std::cout << "  TIER 1: SIMD vs Scalar Reference\n";
    std::cout << "========================================\n";

    // Test multiple sizes: boundary cases and typical workloads
    const size_t sizes[] = {31, 32, 33, 63, 64, 65, 100, 1000, 10000};

    for (size_t n : sizes) {
        std::cout << "\nTesting array size: " << n << " elements\n";

        auto result = test_simd_vs_scalar_binary(
            "tadd_simd_vs_scalar", tadd_simd_array, tadd_scalar_ref, n);
        if (result.passed) harness.record_pass(result.test_name);
        else harness.record_fail(result);

        result = test_simd_vs_scalar_binary(
            "tmul_simd_vs_scalar", tmul_simd_array, tmul_scalar_ref, n);
        if (result.passed) harness.record_pass(result.test_name);
        else harness.record_fail(result);

        result = test_simd_vs_scalar_binary(
            "tmin_simd_vs_scalar", tmin_simd_array, tmin_scalar_ref, n);
        if (result.passed) harness.record_pass(result.test_name);
        else harness.record_fail(result);

        result = test_simd_vs_scalar_binary(
            "tmax_simd_vs_scalar", tmax_simd_array, tmax_scalar_ref, n);
        if (result.passed) harness.record_pass(result.test_name);
        else harness.record_fail(result);

        result = test_simd_vs_scalar_unary(
            "tnot_simd_vs_scalar", tnot_simd_array, tnot_scalar_ref, n);
        if (result.passed) harness.record_pass(result.test_name);
        else harness.record_fail(result);
    }
}

// ============================================================================
// TIER 2: Algebraic Property Tests
// ============================================================================

void run_tier2_tests(TestHarness& harness) {
    std::cout << "\n========================================\n";
    std::cout << "  TIER 2: Algebraic Property Tests\n";
    std::cout << "========================================\n";

    auto result_tadd = test_tadd_properties();
    if (result_tadd.passed) {
        harness.record_pass("tadd algebraic properties");
    } else {
        TestResult fail{false, "tadd algebraic properties",
                       result_tadd.property,
                       SIZE_MAX, 0, 0};
        harness.record_fail(fail);
        std::cout << "    Failed property: " << result_tadd.property << "\n";
        std::cout << "    Inputs: a=" << result_tadd.failing_input_a
                  << ", b=" << result_tadd.failing_input_b;
        if (result_tadd.failing_input_c != 0)
            std::cout << ", c=" << result_tadd.failing_input_c;
        std::cout << "\n";
    }

    auto result_tmul = test_tmul_properties();
    if (result_tmul.passed) {
        harness.record_pass("tmul algebraic properties");
    } else {
        TestResult fail{false, "tmul algebraic properties",
                       result_tmul.property,
                       SIZE_MAX, 0, 0};
        harness.record_fail(fail);
        std::cout << "    Failed property: " << result_tmul.property << "\n";
        std::cout << "    Inputs: a=" << result_tmul.failing_input_a
                  << ", b=" << result_tmul.failing_input_b;
        if (result_tmul.failing_input_c != 0)
            std::cout << ", c=" << result_tmul.failing_input_c;
        std::cout << "\n";
    }

    auto result_tmin = test_tmin_properties();
    if (result_tmin.passed) {
        harness.record_pass("tmin algebraic properties");
    } else {
        TestResult fail{false, "tmin algebraic properties",
                       result_tmin.property,
                       SIZE_MAX, 0, 0};
        harness.record_fail(fail);
        std::cout << "    Failed property: " << result_tmin.property << "\n";
        std::cout << "    Inputs: a=" << result_tmin.failing_input_a
                  << ", b=" << result_tmin.failing_input_b;
        if (result_tmin.failing_input_c != 0)
            std::cout << ", c=" << result_tmin.failing_input_c;
        std::cout << "\n";
    }

    auto result_tmax = test_tmax_properties();
    if (result_tmax.passed) {
        harness.record_pass("tmax algebraic properties");
    } else {
        TestResult fail{false, "tmax algebraic properties",
                       result_tmax.property,
                       SIZE_MAX, 0, 0};
        harness.record_fail(fail);
        std::cout << "    Failed property: " << result_tmax.property << "\n";
        std::cout << "    Inputs: a=" << result_tmax.failing_input_a
                  << ", b=" << result_tmax.failing_input_b;
        if (result_tmax.failing_input_c != 0)
            std::cout << ", c=" << result_tmax.failing_input_c;
        std::cout << "\n";
    }

    auto result_tnot = test_tnot_properties();
    if (result_tnot.passed) {
        harness.record_pass("tnot algebraic properties");
    } else {
        TestResult fail{false, "tnot algebraic properties",
                       result_tnot.property,
                       SIZE_MAX, 0, 0};
        harness.record_fail(fail);
        std::cout << "    Failed property: " << result_tnot.property << "\n";
        std::cout << "    Inputs: a=" << result_tnot.failing_input_a << "\n";
    }
}

// ============================================================================
// TIER 3: Fuzz & Alignment Stress Tests
// ============================================================================

void run_tier3_fuzz_tests(TestHarness& harness, size_t num_trials = 10000) {
    std::cout << "\n========================================\n";
    std::cout << "  TIER 3: Fuzz & Alignment Tests\n";
    std::cout << "========================================\n";
    std::cout << "  Running " << num_trials << " randomized trials...\n\n";

    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> trit_dist(0, 2);  // 0, 1, 2 -> map to trits
    std::uniform_int_distribution<size_t> size_dist(1, 10000);
    std::uniform_int_distribution<int> align_dist(0, 31);  // 0-31 byte misalignment

    const uint8_t trit_values[3] = {0b00, 0b01, 0b10};

    size_t passed_trials = 0;
    size_t failed_trials = 0;

    for (size_t trial = 0; trial < num_trials; ++trial) {
        size_t n = size_dist(rng);
        size_t offset_a = align_dist(rng);
        size_t offset_b = align_dist(rng);
        size_t offset_out = align_dist(rng);

        // Allocate with extra space for alignment offsets
        AlignedBuffer<uint8_t> a_buf(n + 32), b_buf(n + 32),
                               out_simd_buf(n + 32), out_scalar_buf(n + 32);

        uint8_t* a = a_buf.get_offset(offset_a);
        uint8_t* b = b_buf.get_offset(offset_b);
        uint8_t* out_simd = out_simd_buf.get_offset(offset_out);
        uint8_t* out_scalar = out_scalar_buf.get();

        // Random valid trit values
        for (size_t i = 0; i < n; ++i) {
            a[i] = trit_values[trit_dist(rng)];
            b[i] = trit_values[trit_dist(rng)];
        }

        // Test all operations
        tadd_simd_array(a, b, out_simd, n);
        tadd_scalar_ref(a, b, out_scalar, n);
        auto result = compare_buffers("fuzz_tadd", out_scalar, out_simd, n);
        if (!result.passed) {
            failed_trials++;
            std::cout << "  Trial " << trial << " FAILED: tadd\n";
            std::cout << "    Size: " << n << ", Offsets: a=" << offset_a
                      << " b=" << offset_b << " out=" << offset_out << "\n";
            harness.record_fail(result);
            continue;
        }

        tmul_simd_array(a, b, out_simd, n);
        tmul_scalar_ref(a, b, out_scalar, n);
        result = compare_buffers("fuzz_tmul", out_scalar, out_simd, n);
        if (!result.passed) {
            failed_trials++;
            std::cout << "  Trial " << trial << " FAILED: tmul\n";
            harness.record_fail(result);
            continue;
        }

        tmin_simd_array(a, b, out_simd, n);
        tmin_scalar_ref(a, b, out_scalar, n);
        result = compare_buffers("fuzz_tmin", out_scalar, out_simd, n);
        if (!result.passed) {
            failed_trials++;
            std::cout << "  Trial " << trial << " FAILED: tmin\n";
            harness.record_fail(result);
            continue;
        }

        tmax_simd_array(a, b, out_simd, n);
        tmax_scalar_ref(a, b, out_scalar, n);
        result = compare_buffers("fuzz_tmax", out_scalar, out_simd, n);
        if (!result.passed) {
            failed_trials++;
            std::cout << "  Trial " << trial << " FAILED: tmax\n";
            harness.record_fail(result);
            continue;
        }

        tnot_simd_array(a, out_simd, n);
        tnot_scalar_ref(a, out_scalar, n);
        result = compare_buffers("fuzz_tnot", out_scalar, out_simd, n);
        if (!result.passed) {
            failed_trials++;
            std::cout << "  Trial " << trial << " FAILED: tnot\n";
            harness.record_fail(result);
            continue;
        }

        passed_trials++;
    }

    std::cout << "  Fuzz testing complete:\n";
    std::cout << "    Passed: " << passed_trials << " / " << num_trials << "\n";
    std::cout << "    Failed: " << failed_trials << " / " << num_trials << "\n";

    if (failed_trials == 0) {
        harness.record_pass("fuzz_testing_all_operations");
    }
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "  TERNARY SIMD CORRECTNESS TEST SUITE\n";
    std::cout << "  Comprehensive 3-Tier Verification\n";
    std::cout << "========================================\n";

    // Check AVX2 support
    std::cout << "\nSystem requirements:\n";
    std::cout << "  AVX2 support: Required (assumed present)\n";
    std::cout << "  Alignment:    32-byte\n";
    std::cout << "  Vector width: 256-bit (32 trits)\n";

    TestHarness harness;

    run_tier1_tests(harness);
    run_tier2_tests(harness);
    run_tier3_fuzz_tests(harness, 10000);

    harness.print_summary();
    return harness.exit_code();
}
