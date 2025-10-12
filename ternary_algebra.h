// ternary_algebra.h — optimized ternary algebra core header
//
// Copyright 2025 Ternary Core Contributors
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

#ifndef TERNARY_ALGEBRA_H
#define TERNARY_ALGEBRA_H

#include <stdint.h>

// each trit occupies 2 bits → 00 = -1, 01 = 0, 10 = +1
typedef uint8_t trit;

// Platform-specific force inline
#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline)) inline
#endif

// --- conversions (kept for compatibility/reference) ---
static inline trit int_to_trit(int v) { return (v < 0) ? 0b00 : (v > 0) ? 0b10 : 0b01; }
static inline int  trit_to_int(trit t){ return (t==0b00)?-1:(t==0b10)?1:0; }

// --- Lookup tables for optimized operations (OPT-086) ---
// Index format: (a << 2) | b, where a,b are 2-bit trit values

// TADD: Saturated ternary addition
static const uint8_t TADD_LUT[16] = {
    // a=0b00 (-1): -1+-1=-1, -1+0=-1, -1+1=0, -1+xx=undefined
    0b00, 0b00, 0b01, 0b00,
    // a=0b01 (0): 0+-1=-1, 0+0=0, 0+1=+1, 0+xx=undefined
    0b00, 0b01, 0b10, 0b00,
    // a=0b10 (+1): 1+-1=0, 1+0=+1, 1+1=+1, 1+xx=undefined
    0b01, 0b10, 0b10, 0b00,
    // a=0b11 (invalid): all undefined
    0b00, 0b00, 0b00, 0b00
};

// TMUL: Ternary multiplication
static const uint8_t TMUL_LUT[16] = {
    // a=0b00 (-1): -1*-1=+1, -1*0=0, -1*+1=-1, -1*xx=undefined
    0b10, 0b01, 0b00, 0b00,
    // a=0b01 (0): 0*-1=0, 0*0=0, 0*+1=0, 0*xx=undefined
    0b01, 0b01, 0b01, 0b00,
    // a=0b10 (+1): +1*-1=-1, +1*0=0, +1*+1=+1, +1*xx=undefined
    0b00, 0b01, 0b10, 0b00,
    // a=0b11 (invalid): all undefined
    0b00, 0b00, 0b00, 0b00
};

// TMIN: Ternary minimum
static const uint8_t TMIN_LUT[16] = {
    // a=0b00 (-1): min(-1,-1)=-1, min(-1,0)=-1, min(-1,+1)=-1, min(-1,xx)=undefined
    0b00, 0b00, 0b00, 0b00,
    // a=0b01 (0): min(0,-1)=-1, min(0,0)=0, min(0,+1)=0, min(0,xx)=undefined
    0b00, 0b01, 0b01, 0b00,
    // a=0b10 (+1): min(+1,-1)=-1, min(+1,0)=0, min(+1,+1)=+1, min(+1,xx)=undefined
    0b00, 0b01, 0b10, 0b00,
    // a=0b11 (invalid): all undefined
    0b00, 0b00, 0b00, 0b00
};

// TMAX: Ternary maximum
static const uint8_t TMAX_LUT[16] = {
    // a=0b00 (-1): max(-1,-1)=-1, max(-1,0)=0, max(-1,+1)=+1, max(-1,xx)=undefined
    0b00, 0b01, 0b10, 0b00,
    // a=0b01 (0): max(0,-1)=0, max(0,0)=0, max(0,+1)=+1, max(0,xx)=undefined
    0b01, 0b01, 0b10, 0b00,
    // a=0b10 (+1): max(+1,-1)=+1, max(+1,0)=+1, max(+1,+1)=+1, max(+1,xx)=undefined
    0b10, 0b10, 0b10, 0b00,
    // a=0b11 (invalid): all undefined
    0b00, 0b00, 0b00, 0b00
};

// TNOT: Ternary negation (OPT-091)
static const uint8_t TNOT_LUT[4] = {
    0b10,  // tnot(0b00=-1) = +1 = 0b10
    0b01,  // tnot(0b01=0)  = 0  = 0b01
    0b00,  // tnot(0b10=+1) = -1 = 0b00
    0b00   // tnot(0b11=invalid) = undefined
};

// --- Optimized operations using lookup tables (OPT-051: Force inline) ---
static FORCE_INLINE trit tnot(trit a) {
    return TNOT_LUT[a & 0b11];
}

static FORCE_INLINE trit tmin(trit a, trit b) {
    return TMIN_LUT[(a << 2) | b];
}

static FORCE_INLINE trit tmax(trit a, trit b) {
    return TMAX_LUT[(a << 2) | b];
}

static FORCE_INLINE trit tadd(trit a, trit b) {
    return TADD_LUT[(a << 2) | b];
}

static FORCE_INLINE trit tmul(trit a, trit b) {
    return TMUL_LUT[(a << 2) | b];
}

// --- packing of 4 trits into 1 byte ---
static inline uint8_t pack_trits(trit t0,trit t1,trit t2,trit t3){
    return (t0) | (t1<<2) | (t2<<4) | (t3<<6);
}
static inline trit unpack_trit(uint8_t pack,int idx){
    return (pack>>(2*idx)) & 0b11;
}

#endif // TERNARY_ALGEBRA_H
