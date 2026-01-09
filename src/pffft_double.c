/* Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )
   Copyright (c) 2020  Hayati Ayguen ( h_ayguen@web.de )
   Copyright (c) 2020  Dario Mambro ( dario.mambro@gmail.com )

   Double precision FFT implementation.
   See pffft.c for license details.
*/

#include "pffft.h"

/* detect compiler flavour */
#if defined(_MSC_VER)
#  define COMPILER_MSVC
#elif defined(__GNUC__)
#  define COMPILER_GCC
#endif

#ifdef COMPILER_MSVC
#  define _USE_MATH_DEFINES
#  include <malloc.h>
#elif defined(__MINGW32__) || defined(__MINGW64__)
#  include <malloc.h>
#else
#  include <alloca.h>
#endif

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#if defined(COMPILER_GCC)
#  define ALWAYS_INLINE(return_type) inline return_type __attribute__ ((always_inline))
#  define NEVER_INLINE(return_type) return_type __attribute__ ((noinline))
#  define RESTRICT __restrict
#  define VLA_ARRAY_ON_STACK(type__, varname__, size__) type__ varname__[size__];
#elif defined(COMPILER_MSVC)
#  define ALWAYS_INLINE(return_type) __forceinline return_type
#  define NEVER_INLINE(return_type) __declspec(noinline) return_type
#  define RESTRICT __restrict
#  define VLA_ARRAY_ON_STACK(type__, varname__, size__) type__ *varname__ = (type__*)_alloca(size__ * sizeof(type__))
#endif

#ifdef COMPILER_MSVC
#pragma warning( disable : 4244 4305 4204 4456 )
#endif

#include "simd/pf_double.h"

#define float double
#define SETUP_STRUCT               PFFFTD_Setup
#define FUNC_NEW_SETUP             pffftd_new_setup
#define FUNC_DESTROY               pffftd_destroy_setup
#define FUNC_TRANSFORM_UNORDRD     pffftd_transform
#define FUNC_TRANSFORM_ORDERED     pffftd_transform_ordered
#define FUNC_ZREORDER              pffftd_zreorder
#define FUNC_ZCONVOLVE_ACCUMULATE  pffftd_zconvolve_accumulate
#define FUNC_ZCONVOLVE_NO_ACCU     pffftd_zconvolve_no_accu

#define FUNC_ALIGNED_MALLOC        pffftd_aligned_malloc
#define FUNC_ALIGNED_FREE          pffftd_aligned_free
#define FUNC_SIMD_SIZE             pffftd_simd_size
#define FUNC_MIN_FFT_SIZE          pffftd_min_fft_size
#define FUNC_IS_VALID_SIZE         pffftd_is_valid_size
#define FUNC_NEAREST_SIZE          pffftd_nearest_transform_size
#define FUNC_SIMD_ARCH             pffftd_simd_arch
#define FUNC_VALIDATE_SIMD_A       validate_pffftd_simd
#define FUNC_VALIDATE_SIMD_EX      validate_pffftd_simd_ex

#define FUNC_CPLX_FINALIZE         pffftd_cplx_finalize
#define FUNC_CPLX_PREPROCESS       pffftd_cplx_preprocess
#define FUNC_REAL_PREPROCESS_4X4   pffftd_real_preprocess_4x4
#define FUNC_REAL_PREPROCESS       pffftd_real_preprocess
#define FUNC_REAL_FINALIZE_4X4     pffftd_real_finalize_4x4
#define FUNC_REAL_FINALIZE         pffftd_real_finalize
#define FUNC_TRANSFORM_INTERNAL    pffftd_transform_internal

#define FUNC_COS  cos
#define FUNC_SIN  sin

#include "pffft_priv_impl.h"
