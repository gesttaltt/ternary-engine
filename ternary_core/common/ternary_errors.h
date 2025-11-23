// ternary_errors.h — Centralized exception types for ternary operations
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
// DESIGN RATIONALE
// =============================================================================
//
// Centralized error handling provides:
// - Domain-specific exception types for clear error semantics
// - Consistent error messages across scalar and SIMD operations
// - Python-friendly exception propagation via pybind11
// - Auditability through typed exceptions vs generic runtime_error
//
// Follows YAGNI principle: Only defines actually-needed exceptions.
// Expand only when real use cases emerge, not preemptively.
//
// =============================================================================

#ifndef TERNARY_ERRORS_H
#define TERNARY_ERRORS_H

#include <stdexcept>
#include <string>
#include <sstream>

// --- Base exception for all ternary operations ---
class TernaryError : public std::runtime_error {
public:
    explicit TernaryError(const std::string& message)
        : std::runtime_error(message) {}
};

// --- Array size mismatch in binary operations ---
// Thrown when binary operation inputs have different sizes
class ArraySizeMismatchError : public TernaryError {
public:
    ArraySizeMismatchError(size_t size_a, size_t size_b)
        : TernaryError(format_message(size_a, size_b)),
          size_a_(size_a),
          size_b_(size_b) {}

    size_t size_a() const { return size_a_; }
    size_t size_b() const { return size_b_; }

private:
    size_t size_a_;
    size_t size_b_;

    static std::string format_message(size_t size_a, size_t size_b) {
        std::ostringstream oss;
        oss << "Array size mismatch: array A has " << size_a
            << " elements, array B has " << size_b << " elements. "
            << "Binary operations require equal-sized arrays.";
        return oss.str();
    }
};

// --- Invalid trit value ---
// Thrown when a trit value is outside valid range (0b00, 0b01, 0b10)
// Note: Usually not needed due to sanitization (OPT-HASWELL-02)
// Provided for explicit validation scenarios
class InvalidTritError : public TernaryError {
public:
    explicit InvalidTritError(uint8_t invalid_value)
        : TernaryError(format_message(invalid_value)),
          invalid_value_(invalid_value) {}

    uint8_t invalid_value() const { return invalid_value_; }

private:
    uint8_t invalid_value_;

    static std::string format_message(uint8_t value) {
        std::ostringstream oss;
        oss << "Invalid trit value: 0b"
            << ((value >> 1) & 1) << (value & 1)
            << " (decimal " << static_cast<int>(value) << "). "
            << "Valid trit values are: 0b00 (-1), 0b01 (0), 0b10 (+1).";
        return oss.str();
    }
};

// --- Memory allocation failure ---
// Thrown when output array allocation fails
// Rare in practice, but possible with very large arrays
class AllocationError : public TernaryError {
public:
    explicit AllocationError(size_t requested_size)
        : TernaryError(format_message(requested_size)),
          requested_size_(requested_size) {}

    size_t requested_size() const { return requested_size_; }

private:
    size_t requested_size_;

    static std::string format_message(size_t size) {
        std::ostringstream oss;
        oss << "Memory allocation failed: could not allocate "
            << size << " bytes for output array. "
            << "Consider processing data in smaller chunks.";
        return oss.str();
    }
};

#endif // TERNARY_ERRORS_H
