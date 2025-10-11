// ternary_core.h — optimized ternary algebra core header
#ifndef TERNARY_CORE_H
#define TERNARY_CORE_H

#include <stdint.h>

// each trit occupies 2 bits → 00 = -1, 01 = 0, 10 = +1
typedef uint8_t trit;

// --- conversions ---
static inline trit int_to_trit(int v) { return (v < 0) ? 0b00 : (v > 0) ? 0b10 : 0b01; }
static inline int  trit_to_int(trit t){ return (t==0b00)?-1:(t==0b10)?1:0; }

// --- basic logical operations ---
static inline trit tnot(trit a)   { return (a==0b00)?0b10:(a==0b10)?0b00:0b01; }
static inline trit tmin(trit a,trit b){ return (trit_to_int(a)<trit_to_int(b))?a:b; }
static inline trit tmax(trit a,trit b){ return (trit_to_int(a)>trit_to_int(b))?a:b; }
static inline trit tadd(trit a,trit b){
    int s = trit_to_int(a) + trit_to_int(b);
    if (s>1) s=1; if (s<-1) s=-1;
    return int_to_trit(s);
}
static inline trit tmul(trit a,trit b){
    return int_to_trit(trit_to_int(a)*trit_to_int(b));
}

// --- packing of 4 trits into 1 byte ---
static inline uint8_t pack_trits(trit t0,trit t1,trit t2,trit t3){
    return (t0) | (t1<<2) | (t2<<4) | (t3<<6);
}
static inline trit unpack_trit(uint8_t pack,int idx){
    return (pack>>(2*idx)) & 0b11;
}

#endif // TERNARY_CORE_H
