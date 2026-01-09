
/* Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )
   Copyright (c) 2020  Hayati Ayguen ( h_ayguen@web.de )

   Redistribution and use of the Software in source and binary forms,
   with or without modification, is permitted provided that the
   following conditions are met:

   - Neither the names of NCAR's Computational and Information Systems
   Laboratory, the University Corporation for Atmospheric Research,
   nor the names of its sponsors or contributors may be used to
   endorse or promote products derived from this Software without
   specific prior written permission.

   - Redistributions of source code must retain the above copyright
   notices, this list of conditions, and the disclaimer below.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions, and the disclaimer below in the
   documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
   HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
   SOFTWARE.
*/

#ifndef PF_SCAL_DBL_H
#define PF_SCAL_DBL_H

/*
  fallback mode(s) for situations where SSE/AVX/NEON/Altivec are not available, use scalar mode instead
*/

#if !defined(SIMD_SZ) && !defined(PFFFT_SIMD_DISABLE)
#pragma message( __FILE__ ": double SCALAR4 macros are defined" )

typedef struct {
  vsfscalar a;
  vsfscalar b;
  vsfscalar c;
  vsfscalar d;
} vfloat;

#  define SIMD_SZ 4

typedef union vfloat_union {
  vfloat v;
  vsfscalar f[SIMD_SZ];
} vfloat_union;

#  define VARCH "4xScalar"
#  define VREQUIRES_ALIGN 0

  static ALWAYS_INLINE(vfloat) VZERO() {
    vfloat r = { 0.f, 0.f, 0.f, 0.f };
    return r;
  }

  static ALWAYS_INLINE(vfloat) VMUL(vfloat A, vfloat B) {
    vfloat r = { A.a * B.a, A.b * B.b, A.c * B.c, A.d * B.d };
    return r;
  }

  static ALWAYS_INLINE(vfloat) VADD(vfloat A, vfloat B) {
    vfloat r = { A.a + B.a, A.b + B.b, A.c + B.c, A.d + B.d };
    return r;
  }

  static ALWAYS_INLINE(vfloat) VMADD(vfloat A, vfloat B, vfloat C) {
    vfloat r = { A.a * B.a + C.a, A.b * B.b + C.b, A.c * B.c + C.c, A.d * B.d + C.d };
    return r;
  }

  static ALWAYS_INLINE(vfloat) VSUB(vfloat A, vfloat B) {
    vfloat r = { A.a - B.a, A.b - B.b, A.c - B.c, A.d - B.d };
    return r;
  }

  static ALWAYS_INLINE(vfloat) LD_PS1(vsfscalar v) {
    vfloat r = { v, v, v, v };
    return r;
  }

#  define VLOAD_UNALIGNED(ptr)  (*((vfloat*)(ptr)))

#  define VLOAD_ALIGNED(ptr)    (*((vfloat*)(ptr)))

#  define VALIGNED(ptr) ((((uintptr_t)(ptr)) & (sizeof(vfloat)-1) ) == 0)


  /* INTERLEAVE2() */
  #define INTERLEAVE2( A, B, C, D) \
  do { \
    vfloat Cr = { A.a, B.a, A.b, B.b }; \
    vfloat Dr = { A.c, B.c, A.d, B.d }; \
    C = Cr; \
    D = Dr; \
  } while (0)


  /* UNINTERLEAVE2() */
  #define UNINTERLEAVE2(A, B, C, D) \
  do { \
    vfloat Cr = { A.a, A.c, B.a, B.c }; \
    vfloat Dr = { A.b, A.d, B.b, B.d }; \
    C = Cr; \
    D = Dr; \
  } while (0)


  /* VTRANSPOSE4() */
  #define VTRANSPOSE4(A, B, C, D) \
  do { \
    vfloat Ar = { A.a, B.a, C.a, D.a }; \
    vfloat Br = { A.b, B.b, C.b, D.b }; \
    vfloat Cr = { A.c, B.c, C.c, D.c }; \
    vfloat Dr = { A.d, B.d, C.d, D.d }; \
    A = Ar; \
    B = Br; \
    C = Cr; \
    D = Dr; \
  } while (0)


  /* VSWAPHL() */
  static ALWAYS_INLINE(vfloat) VSWAPHL(vfloat A, vfloat B) {
    vfloat r = { B.a, B.b, A.c, A.d };
    return r;
  }


  /* reverse/flip all floats */
  static ALWAYS_INLINE(vfloat) VREV_S(vfloat A) {
    vfloat r = { A.d, A.c, A.b, A.a };
    return r;
  }

  /* reverse/flip complex floats */
  static ALWAYS_INLINE(vfloat) VREV_C(vfloat A) {
    vfloat r = { A.c, A.d, A.a, A.b };
    return r;
  }

#else
/* #pragma message( __FILE__ ": double SCALAR4 macros are not defined" ) */
#endif


#if !defined(SIMD_SZ)
#pragma message( __FILE__ ": float SCALAR1 macros are defined" )
typedef vsfscalar vfloat;

#  define SIMD_SZ 1

typedef union vfloat_union {
  vfloat v;
  vsfscalar f[SIMD_SZ];
} vfloat_union;

#  define VARCH "Scalar"
#  define VREQUIRES_ALIGN 0
#  define VZERO() 0.0
#  define VMUL(a,b) ((a)*(b))
#  define VADD(a,b) ((a)+(b))
#  define VMADD(a,b,c) ((a)*(b)+(c))
#  define VSUB(a,b) ((a)-(b))
#  define LD_PS1(p) (p)
#  define VLOAD_UNALIGNED(ptr)  (*(ptr))
#  define VLOAD_ALIGNED(ptr)    (*(ptr))
#  define VALIGNED(ptr) ((((uintptr_t)(ptr)) & (sizeof(vsfscalar)-1) ) == 0)

#else
/* #pragma message( __FILE__ ": double SCALAR1 macros are not defined" ) */
#endif


#endif /* PF_SCAL_DBL_H */

