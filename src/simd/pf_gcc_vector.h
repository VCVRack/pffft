/*
 * PFFFT - GCC Vector Extensions SIMD Backend (Single Precision)
 *
 * This file provides a unified SIMD abstraction using GCC/Clang vector
 * extensions, replacing platform-specific intrinsics (SSE, AVX, NEON, etc.)
 *
 * Requires: GCC 10+ or Clang 10+
 */

#ifndef PF_GCC_VECTOR_H
#define PF_GCC_VECTOR_H

#include <stdint.h>

/*
 * =============================================================================
 * Explicit Size Vector Types
 * =============================================================================
 *
 * These are always defined regardless of the target architecture.
 * Naming: v<count><type> where type is sf (single float) or si (signed int)
 */

typedef float v4sf __attribute__((vector_size(16)));   /* 4 floats, 128-bit */
typedef float v8sf __attribute__((vector_size(32)));   /* 8 floats, 256-bit */
typedef float v16sf __attribute__((vector_size(64)));  /* 16 floats, 512-bit */

typedef int v4si __attribute__((vector_size(16)));     /* 4 ints, 128-bit (for shuffle masks) */
typedef int v8si __attribute__((vector_size(32)));     /* 8 ints, 256-bit */
typedef int v16si __attribute__((vector_size(64)));    /* 16 ints, 512-bit */

/*
 * =============================================================================
 * SIMD Width Detection
 * =============================================================================
 *
 * SIMD_SZ determines how many floats fit in a native vector register.
 * This is auto-detected from compiler feature macros.
 *
 * NOTE: SIMD_SZ > 4 (AVX, AVX-512) requires generalizing the FFT algorithm
 * in pffft_priv_impl.h which currently has hardcoded SIMD_SZ==4 assumptions.
 * Until that work is done, we limit to SIMD_SZ=4.
 */

#if !defined(SIMD_SZ) && !defined(PFFFT_SIMD_DISABLE)

#if defined(__SSE2__) || defined(__AVX__) || defined(__AVX512F__) || \
    defined(__ARM_NEON) || defined(__ARM_NEON__)
#  define SIMD_SZ 4
#  if defined(__ARM_NEON) || defined(__ARM_NEON__)
#    define VARCH "NEON"
#  else
#    define VARCH "SSE"
#  endif
#else
/* Fallback - will use scalar implementation via pf_scalar_float.h */
#endif

#endif /* !defined(SIMD_SZ) && !defined(PFFFT_SIMD_DISABLE) */

/*
 * =============================================================================
 * Native-Width Type Aliases
 * =============================================================================
 *
 * vfloat - native-width vector of floats (use this in algorithm code)
 * vfloat_union - for element access when needed
 */

#if defined(SIMD_SZ) && !defined(PFFFT_SIMD_DISABLE)

#if SIMD_SZ == 4
typedef v4sf vfloat;
typedef v4si vint;
#elif SIMD_SZ == 8
typedef v8sf vfloat;
typedef v8si vint;
#elif SIMD_SZ == 16
typedef v16sf vfloat;
typedef v16si vint;
#else
#error "Unsupported SIMD_SZ value"
#endif

typedef union vfloat_union {
    vfloat v;
    float f[SIMD_SZ];
} vfloat_union;

#define VREQUIRES_ALIGN 1

/*
 * =============================================================================
 * Shuffle Builtin Compatibility
 * =============================================================================
 *
 * GCC uses __builtin_shuffle(vec1, vec2, mask_vector)
 * Clang uses __builtin_shufflevector(vec1, vec2, idx0, idx1, ...)
 */

#ifdef __clang__
/* Clang: use __builtin_shufflevector with explicit indices */
#define SHUF1_4(v, i0, i1, i2, i3) \
    __builtin_shufflevector((v), (v), i0, i1, i2, i3)
#define SHUF2_4(a, b, i0, i1, i2, i3) \
    __builtin_shufflevector((a), (b), i0, i1, i2, i3)
#else /* GCC */
/* GCC: use __builtin_shuffle with mask vector */
#define SHUF1_4(v, i0, i1, i2, i3) \
    __builtin_shuffle((v), (v4si){i0, i1, i2, i3})
#define SHUF2_4(a, b, i0, i1, i2, i3) \
    __builtin_shuffle((a), (b), (v4si){i0, i1, i2, i3})
#endif /* __clang__ */

/*
 * =============================================================================
 * Basic Arithmetic Operations
 * =============================================================================
 */

#define VZERO() ((vfloat){})
#define VMUL(a, b) ((a) * (b))
#define VADD(a, b) ((a) + (b))
#define VSUB(a, b) ((a) - (b))
#define VMADD(a, b, c) ((a) * (b) + (c))

/*
 * =============================================================================
 * Load and Broadcast Operations
 * =============================================================================
 */

/* Load from aligned pointer */
#define VLOAD_ALIGNED(ptr) (*((const vfloat *)(ptr)))

/* Load from unaligned pointer */
#define VLOAD_UNALIGNED(ptr) (*((const vfloat *)(ptr)))

/* Check alignment */
#define VALIGNED(ptr) ((((uintptr_t)(ptr)) & (sizeof(vfloat) - 1)) == 0)

/* Broadcast scalar to all vector lanes */
#if SIMD_SZ == 4
static inline vfloat LD_PS1(float val) {
    return (vfloat){val, val, val, val};
}
#elif SIMD_SZ == 8
static inline vfloat LD_PS1(float val) {
    return (vfloat){val, val, val, val, val, val, val, val};
}
#elif SIMD_SZ == 16
static inline vfloat LD_PS1(float val) {
    return (vfloat){val, val, val, val, val, val, val, val,
                    val, val, val, val, val, val, val, val};
}
#endif

/*
 * =============================================================================
 * Shuffle / Permutation Operations (SIMD_SZ == 4)
 * =============================================================================
 */

#if SIMD_SZ == 4

/*
 * INTERLEAVE2: Interleave two vectors
 * Input:  in1 = [a0, a1, a2, a3], in2 = [b0, b1, b2, b3]
 * Output: out1 = [a0, b0, a1, b1], out2 = [a2, b2, a3, b3]
 */
#define INTERLEAVE2(in1, in2, out1, out2) do { \
    out1 = SHUF2_4((in1), (in2), 0, 4, 1, 5); \
    out2 = SHUF2_4((in1), (in2), 2, 6, 3, 7); \
} while (0)

/*
 * UNINTERLEAVE2: Deinterleave two vectors
 * Input:  in1 = [a0, a1, a2, a3], in2 = [b0, b1, b2, b3]
 * Output: out1 = [a0, a2, b0, b2], out2 = [a1, a3, b1, b3]
 */
#define UNINTERLEAVE2(in1, in2, out1, out2) do { \
    out1 = SHUF2_4((in1), (in2), 0, 2, 4, 6); \
    out2 = SHUF2_4((in1), (in2), 1, 3, 5, 7); \
} while (0)

/*
 * VTRANSPOSE4: In-place 4x4 matrix transpose
 */
#define VTRANSPOSE4(r0, r1, r2, r3) do { \
    vfloat t0 = SHUF2_4((r0), (r1), 0, 4, 1, 5); \
    vfloat t1 = SHUF2_4((r0), (r1), 2, 6, 3, 7); \
    vfloat t2 = SHUF2_4((r2), (r3), 0, 4, 1, 5); \
    vfloat t3 = SHUF2_4((r2), (r3), 2, 6, 3, 7); \
    r0 = SHUF2_4(t0, t2, 0, 1, 4, 5); \
    r1 = SHUF2_4(t0, t2, 2, 3, 6, 7); \
    r2 = SHUF2_4(t1, t3, 0, 1, 4, 5); \
    r3 = SHUF2_4(t1, t3, 2, 3, 6, 7); \
} while (0)

/*
 * VSWAPHL: Swap high and low halves
 * Input:  a = [a0, a1, a2, a3], b = [b0, b1, b2, b3]
 * Output: [b0, b1, a2, a3]
 */
#define VSWAPHL(a, b) SHUF2_4((b), (a), 0, 1, 6, 7)

/*
 * VREV_S: Reverse all scalar elements
 * Input:  a = [a0, a1, a2, a3]
 * Output: [a3, a2, a1, a0]
 */
#define VREV_S(a) SHUF1_4((a), 3, 2, 1, 0)

/*
 * VREV_C: Reverse complex pairs
 * Input:  a = [re0, im0, re1, im1]
 * Output: [re1, im1, re0, im0]
 */
#define VREV_C(a) SHUF1_4((a), 2, 3, 0, 1)

#endif /* SIMD_SZ == 4 */

/*
 * NOTE: SIMD_SZ == 8 and SIMD_SZ == 16 shuffle operations are not yet implemented.
 * They will be added when pffft_priv_impl.h is generalized for wider SIMD.
 */

#endif /* defined(SIMD_SZ) && !defined(PFFFT_SIMD_DISABLE) */

#endif /* PF_GCC_VECTOR_H */
