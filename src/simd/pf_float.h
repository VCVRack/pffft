
/* Copyright (c) 2013  Julien Pommier ( pommier@modartt.com )

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

#ifndef PF_FLT_H
#define PF_FLT_H

#include <assert.h>
#include <string.h>
#include <stdint.h>

typedef float vsfscalar;

/* Use unified GCC vector extensions backend */
#include "pf_gcc_vector.h"

/* Scalar fallback if SIMD is disabled or unavailable */
#include "pf_scalar_float.h"

/* shortcuts for complex multiplications (vector versions) */
#define VCPLXMUL(ar,ai,br,bi) { vfloat tmp; tmp=VMUL(ar,bi); ar=VMUL(ar,br); ar=VSUB(ar,VMUL(ai,bi)); ai=VMUL(ai,br); ai=VADD(ai,tmp); }
#define VCPLXMULCONJ(ar,ai,br,bi) { vfloat tmp; tmp=VMUL(ar,bi); ar=VMUL(ar,br); ar=VADD(ar,VMUL(ai,bi)); ai=VMUL(ai,br); ai=VSUB(ai,tmp); }

/* scalar versions for non-SIMD code paths */
#define VCPLXMUL_SCALAR(ar,ai,br,bi) { float _tmp_ = (ar)*(bi); (ar) = (ar)*(br) - (ai)*(bi); (ai) = (ai)*(br) + _tmp_; }
#define VCPLXMULCONJ_SCALAR(ar,ai,br,bi) { float _tmp_ = (ar)*(bi); (ar) = (ar)*(br) + (ai)*(bi); (ai) = (ai)*(br) - _tmp_; }

#ifndef SVMUL
/* multiply a scalar with a vector */
#define SVMUL(f,v) VMUL(LD_PS1(f),v)
#endif

#endif /* PF_FLT_H */
