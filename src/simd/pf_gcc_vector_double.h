/*
 * PFFFT - GCC Vector Extensions SIMD Backend (Double Precision)
 *
 * This file provides a unified SIMD abstraction using GCC/Clang vector
 * extensions for double precision FFT operations.
 *
 * Requires: GCC 10+ or Clang 10+
 */

#ifndef PF_GCC_VECTOR_DOUBLE_H
#define PF_GCC_VECTOR_DOUBLE_H

#include <stdint.h>

/*
 * =============================================================================
 * Explicit Size Vector Types
 * =============================================================================
 *
 * Naming: v<count><type> where type is df (double float) or di (double int/long)
 */

typedef double v2df __attribute__((vector_size(16)));  /* 2 doubles, 128-bit */
typedef double v4df __attribute__((vector_size(32)));  /* 4 doubles, 256-bit */
typedef double v8df __attribute__((vector_size(64)));  /* 8 doubles, 512-bit */

typedef long long v2di __attribute__((vector_size(16)));  /* 2 longs, 128-bit (for shuffle masks) */
typedef long long v4di __attribute__((vector_size(32)));  /* 4 longs, 256-bit */
typedef long long v8di __attribute__((vector_size(64)));  /* 8 longs, 512-bit */

/*
 * =============================================================================
 * SIMD Width Detection (Double Precision)
 * =============================================================================
 *
 * For doubles, SIMD_SZ is the number of doubles per vector.
 * The algorithm requires SIMD_SZ=4 (4 doubles = 256 bits for AVX).
 */

#if !defined(SIMD_SZ) && !defined(PFFFT_SIMD_DISABLE)

#if defined(__AVX__) || defined(__AVX512F__) || defined(__SSE2__) || \
    defined(__ARM_NEON) || defined(__ARM_NEON__)
#  define SIMD_SZ 4
#  if defined(__AVX__) || defined(__AVX512F__)
#    define VARCH "AVX"
#  elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#    define VARCH "NEON"
#  else
#    define VARCH "SSE2"
#  endif
#else
/* Fallback - will use scalar implementation via pf_scalar_double.h */
#endif

#endif /* !defined(SIMD_SZ) && !defined(PFFFT_SIMD_DISABLE) */

/*
 * =============================================================================
 * Native-Width Type Aliases
 * =============================================================================
 *
 * vdouble - native-width vector of doubles (use this in algorithm code)
 * vdouble_union - for element access when needed
 */

#if defined(SIMD_SZ) && !defined(PFFFT_SIMD_DISABLE)

#if SIMD_SZ == 4
typedef v4df vdouble;
typedef v4di vlong;
#elif SIMD_SZ == 2
typedef v2df vdouble;
typedef v2di vlong;
#elif SIMD_SZ == 8
typedef v8df vdouble;
typedef v8di vlong;
#else
#error "Unsupported SIMD_SZ value for double"
#endif

typedef union vdouble_union {
    vdouble v;
    double f[SIMD_SZ];
} vdouble_union;

/* Compatibility: pffft_priv_impl.h uses vfloat and vfloat_union */
typedef vdouble vfloat;
typedef vdouble_union vfloat_union;

#define VREQUIRES_ALIGN 1

/*
 * =============================================================================
 * Shuffle Builtin Compatibility
 * =============================================================================
 */

#ifdef __clang__
#define SHUF1_4(v, i0, i1, i2, i3) \
    __builtin_shufflevector((v), (v), i0, i1, i2, i3)
#define SHUF2_4(a, b, i0, i1, i2, i3) \
    __builtin_shufflevector((a), (b), i0, i1, i2, i3)
#else /* GCC */
#define SHUF1_4(v, i0, i1, i2, i3) \
    __builtin_shuffle((v), (v4di){i0, i1, i2, i3})
#define SHUF2_4(a, b, i0, i1, i2, i3) \
    __builtin_shuffle((a), (b), (v4di){i0, i1, i2, i3})
#endif /* __clang__ */

/*
 * =============================================================================
 * Basic Arithmetic Operations
 * =============================================================================
 */

#define VZERO() ((vdouble){})
#define VMUL(a, b) ((a) * (b))
#define VADD(a, b) ((a) + (b))
#define VSUB(a, b) ((a) - (b))
#define VMADD(a, b, c) ((a) * (b) + (c))

/*
 * =============================================================================
 * Load and Broadcast Operations
 * =============================================================================
 */

#define VLOAD_ALIGNED(ptr) (*((const vdouble *)(ptr)))
#define VLOAD_UNALIGNED(ptr) (*((const vdouble *)(ptr)))
#define VALIGNED(ptr) ((((uintptr_t)(ptr)) & (sizeof(vdouble) - 1)) == 0)

#if SIMD_SZ == 4
static inline vdouble LD_PS1(double val) {
    return (vdouble){val, val, val, val};
}
#elif SIMD_SZ == 2
static inline vdouble LD_PS1(double val) {
    return (vdouble){val, val};
}
#elif SIMD_SZ == 8
static inline vdouble LD_PS1(double val) {
    return (vdouble){val, val, val, val, val, val, val, val};
}
#endif

/*
 * =============================================================================
 * Shuffle / Permutation Operations (SIMD_SZ == 4)
 * =============================================================================
 */

#if SIMD_SZ == 4

#define INTERLEAVE2(in1, in2, out1, out2) do { \
    out1 = SHUF2_4((in1), (in2), 0, 4, 1, 5); \
    out2 = SHUF2_4((in1), (in2), 2, 6, 3, 7); \
} while (0)

#define UNINTERLEAVE2(in1, in2, out1, out2) do { \
    out1 = SHUF2_4((in1), (in2), 0, 2, 4, 6); \
    out2 = SHUF2_4((in1), (in2), 1, 3, 5, 7); \
} while (0)

#define VTRANSPOSE4(r0, r1, r2, r3) do { \
    vdouble t0 = SHUF2_4((r0), (r1), 0, 4, 1, 5); \
    vdouble t1 = SHUF2_4((r0), (r1), 2, 6, 3, 7); \
    vdouble t2 = SHUF2_4((r2), (r3), 0, 4, 1, 5); \
    vdouble t3 = SHUF2_4((r2), (r3), 2, 6, 3, 7); \
    r0 = SHUF2_4(t0, t2, 0, 1, 4, 5); \
    r1 = SHUF2_4(t0, t2, 2, 3, 6, 7); \
    r2 = SHUF2_4(t1, t3, 0, 1, 4, 5); \
    r3 = SHUF2_4(t1, t3, 2, 3, 6, 7); \
} while (0)

#define VSWAPHL(a, b) SHUF2_4((b), (a), 0, 1, 6, 7)
#define VREV_S(a) SHUF1_4((a), 3, 2, 1, 0)
#define VREV_C(a) SHUF1_4((a), 2, 3, 0, 1)

#endif /* SIMD_SZ == 4 */

#endif /* defined(SIMD_SZ) && !defined(PFFFT_SIMD_DISABLE) */

#endif /* PF_GCC_VECTOR_DOUBLE_H */
