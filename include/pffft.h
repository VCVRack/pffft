/* Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )

   Based on original fortran 77 code from FFTPACKv4 from NETLIB,
   authored by Dr Paul Swarztrauber of NCAR, in 1985.

   As confirmed by the NCAR fftpack software curators, the following
   FFTPACKv5 license applies to FFTPACKv4 sources. My changes are
   released under the same terms.

   FFTPACK license:

   http://www.cisl.ucar.edu/css/software/fftpack5/ftpk.html

   Copyright (c) 2004 the University Corporation for Atmospheric
   Research ("UCAR"). All rights reserved. Developed by NCAR's
   Computational and Information Systems Laboratory, UCAR,
   www.cisl.ucar.edu.

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

/*
   PFFFT : a Pretty Fast FFT.

   This is basically an adaptation of the single precision fftpack
   (v4) as found on netlib taking advantage of SIMD instruction found
   on cpus such as intel x86 (SSE/AVX), powerpc (Altivec), and arm (NEON).

   For architectures where no SIMD instruction is available, the code
   falls back to a scalar version.

   Restrictions:

   - 1D transforms only, with 32-bit single or 64-bit double precision.

   - supports only transforms for inputs of length N of the form
   N=(2^a)*(3^b)*(5^c), a >= 5, b >=0, c >= 0 (32, 48, 64, 96, 128,
   144, 160, etc are all acceptable lengths). Performance is best for
   128<=N<=8192.

   - all pointers in the functions below are expected to have an
   "simd-compatible" alignment (16-32 bytes depending on SIMD width).

   You can allocate such buffers with the functions
   pffft_aligned_malloc / pffft_aligned_free (or with stuff like
   posix_memalign..)

*/

#ifndef PFFFT_H
#define PFFFT_H

#include <stddef.h> /* for size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Common Types and Enums                                                     */
/* ========================================================================== */

/* direction of the transform */
typedef enum { PFFFT_FORWARD, PFFFT_BACKWARD } pffft_direction_t;

/* type of transform */
typedef enum { PFFFT_REAL, PFFFT_COMPLEX } pffft_transform_t;

/* opaque structs holding internal stuff (precomputed twiddle factors)
   these structs can be shared by many threads as they contain only
   read-only data.
*/
typedef struct PFFFT_Setup PFFFT_Setup;
typedef struct PFFFTD_Setup PFFFTD_Setup;

/* ========================================================================== */
/* Single Precision (float) API                                               */
/* ========================================================================== */

/*
  prepare for performing transforms of size N -- the returned
  PFFFT_Setup structure is read-only so it can safely be shared by
  multiple concurrent threads.
*/
PFFFT_Setup *pffft_new_setup(int N, pffft_transform_t transform);
void pffft_destroy_setup(PFFFT_Setup *);

/*
   Perform a Fourier transform. The z-domain data is stored in the
   most efficient order for transforming it back, or using it for
   convolution. If you need to have its content sorted in the
   "usual" way, that is as an array of interleaved complex numbers,
   either use pffft_transform_ordered, or call pffft_zreorder after
   the forward fft, and before the backward fft.

   Transforms are not scaled: PFFFT_BACKWARD(PFFFT_FORWARD(x)) = N*x.
   Typically you will want to scale the backward transform by 1/N.

   The 'work' pointer should point to an area of N (2*N for complex
   fft) floats, properly aligned. If 'work' is NULL, then stack will
   be used instead (this is probably the best strategy for small
   FFTs, say for N < 16384).

   input and output may alias.
*/
void pffft_transform(PFFFT_Setup *setup, const float *input, float *output,
                     float *work, pffft_direction_t direction);

/*
   Similar to pffft_transform, but makes sure that the output is
   ordered as expected (interleaved complex numbers). This is
   similar to calling pffft_transform and then pffft_zreorder.

   input and output may alias.
*/
void pffft_transform_ordered(PFFFT_Setup *setup, const float *input, float *output,
                             float *work, pffft_direction_t direction);

/*
   call pffft_zreorder(.., PFFFT_FORWARD) after pffft_transform(...,
   PFFFT_FORWARD) if you want to have the frequency components in
   the correct "canonical" order, as interleaved complex numbers.

   input and output should not alias.
*/
void pffft_zreorder(PFFFT_Setup *setup, const float *input, float *output,
                    pffft_direction_t direction);

/*
   Perform a multiplication of the frequency components of dft_a and
   dft_b and accumulate them into dft_ab. The arrays should have
   been obtained with pffft_transform(.., PFFFT_FORWARD) and should
   *not* have been reordered with pffft_zreorder.

   the operation performed is: dft_ab += (dft_a * dft_b)*scaling

   The dft_a, dft_b and dft_ab pointers may alias.
*/
void pffft_zconvolve_accumulate(PFFFT_Setup *setup, const float *dft_a,
                                const float *dft_b, float *dft_ab, float scaling);

/*
   Perform a multiplication of the frequency components of dft_a and
   dft_b and put result in dft_ab.

   the operation performed is: dft_ab = (dft_a * dft_b)*scaling

   The dft_a, dft_b and dft_ab pointers may alias.
*/
void pffft_zconvolve_no_accu(PFFFT_Setup *setup, const float *dft_a,
                             const float *dft_b, float *dft_ab, float scaling);

/* return SIMD vector size (number of floats processed in parallel) */
int pffft_simd_size(void);

/* return string identifier of used architecture (SSE/AVX/NEON/..) */
const char *pffft_simd_arch(void);

/* simple helper to get minimum possible fft size */
int pffft_min_fft_size(pffft_transform_t transform);

/* simple helper to determine if size N is valid */
int pffft_is_valid_size(int N, pffft_transform_t transform);

/* determine nearest valid transform size */
int pffft_nearest_transform_size(int N, pffft_transform_t transform, int higher);

/* ========================================================================== */
/* Double Precision (double) API                                              */
/* ========================================================================== */

/*
  prepare for performing transforms of size N -- the returned
  PFFFTD_Setup structure is read-only so it can safely be shared by
  multiple concurrent threads.
*/
PFFFTD_Setup *pffftd_new_setup(int N, pffft_transform_t transform);
void pffftd_destroy_setup(PFFFTD_Setup *);

/*
   Perform a Fourier transform. See pffft_transform for details.
   Double precision version.
*/
void pffftd_transform(PFFFTD_Setup *setup, const double *input, double *output,
                      double *work, pffft_direction_t direction);

/*
   Similar to pffftd_transform, but makes sure that the output is
   ordered as expected (interleaved complex numbers).
*/
void pffftd_transform_ordered(PFFFTD_Setup *setup, const double *input, double *output,
                              double *work, pffft_direction_t direction);

/*
   Reorder frequency components to canonical order.
*/
void pffftd_zreorder(PFFFTD_Setup *setup, const double *input, double *output,
                     pffft_direction_t direction);

/*
   Frequency-domain convolution with accumulation.
   dft_ab += (dft_a * dft_b)*scaling
*/
void pffftd_zconvolve_accumulate(PFFFTD_Setup *setup, const double *dft_a,
                                 const double *dft_b, double *dft_ab, double scaling);

/*
   Frequency-domain convolution without accumulation.
   dft_ab = (dft_a * dft_b)*scaling
*/
void pffftd_zconvolve_no_accu(PFFFTD_Setup *setup, const double *dft_a,
                              const double *dft_b, double *dft_ab, double scaling);

/* return SIMD vector size (number of doubles processed in parallel) */
int pffftd_simd_size(void);

/* return string identifier of used architecture */
const char *pffftd_simd_arch(void);

/* simple helper to get minimum possible fft size */
int pffftd_min_fft_size(pffft_transform_t transform);

/* simple helper to determine if size N is valid */
int pffftd_is_valid_size(int N, pffft_transform_t transform);

/* determine nearest valid transform size */
int pffftd_nearest_transform_size(int N, pffft_transform_t transform, int higher);

/* ========================================================================== */
/* Common Utility Functions                                                   */
/* ========================================================================== */

/* simple helper to determine next power of 2 */
int pffft_next_power_of_two(int N);

/* simple helper to determine if power of 2 - returns bool */
int pffft_is_power_of_two(int N);

/*
  Allocate memory with correct alignment for SIMD operations.
  The alignment is automatically determined based on SIMD width.
*/
void *pffft_aligned_malloc(size_t nb_bytes);
void pffft_aligned_free(void *);

/* Aliases for double precision (same implementation) */
#define pffftd_next_power_of_two pffft_next_power_of_two
#define pffftd_is_power_of_two   pffft_is_power_of_two
#define pffftd_aligned_malloc    pffft_aligned_malloc
#define pffftd_aligned_free      pffft_aligned_free

#ifdef __cplusplus
}
#endif

#endif /* PFFFT_H */
