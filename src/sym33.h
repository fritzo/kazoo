#ifndef KAZOO_SYM33_H
#define KAZOO_SYM33_H

#include "common.h"
#include "array.h"

/** Hard-coded functions for symmetric 3x3 matrix-vector operations
*/

namespace Sym33
{

enum Index {
  i00 = 0, i01 = 1, i02 = 2,
  i10 = 1, i11 = 3, i12 = 4,
  i20 = 2, i21 = 4, i22 = 5,
  size = 6
};

typedef Array<float,6> Sym33;
typedef Array<float, 3> float3;

Sym33 identity () { return Sym33(1,0,0,1,0,1); }

// returns |a|
inline float det (const Sym33 & a)
{
  return a[i00] * a[i11] * a[i22]
       - a[i00] * sqr(a[i12])
       - a[i11] * sqr(a[i20])
       - a[i22] * sqr(a[i01])
       + 2 * a[i01] * a[i12] * a[i20];
}

// b = 1 / a
inline void inverse (const Sym33 & a, Sym33 & b)
{
  // produced with python test/matrix_isqrt.py print-sym33-inv

  b[i00] = a[i11] * a[i22] - a[i12] * a[i21];
  b[i01] = a[i21] * a[i02] - a[i22] * a[i01];
  b[i02] = a[i01] * a[i12] - a[i02] * a[i11];
  b[i11] = a[i22] * a[i00] - a[i20] * a[i02];
  b[i12] = a[i02] * a[i10] - a[i00] * a[i12];
  b[i22] = a[i00] * a[i11] - a[i01] * a[i10];

  b *= 1.0f / det(a);
}

// c = a b
// WARNING: this assumes shared eigenspaces
inline void multiply (const Sym33 & a, const Sym33 & b, Sym33 & c)
{
  // produced with python test/matrix_isqrt.py print-sym33-mult

  c[i00] = a[i00] * b[i00] + a[i01] * b[i10] + a[i02] * b[i20];
  c[i01] = a[i00] * b[i01] + a[i01] * b[i11] + a[i02] * b[i21];
  c[i02] = a[i00] * b[i02] + a[i01] * b[i12] + a[i02] * b[i22];
  c[i11] = a[i10] * b[i01] + a[i11] * b[i11] + a[i12] * b[i21];
  c[i12] = a[i10] * b[i02] + a[i11] * b[i12] + a[i12] * b[i22];
  c[i22] = a[i20] * b[i02] + a[i21] * b[i12] + a[i22] * b[i22];
}

// y = a x
inline void multiply (const Sym33 & a, const float3 & x, float3 & y)
{
  y[0] = a[i00] * x[0] + a[i01] * x[1] + a[i02] * x[2];
  y[1] = a[i10] * x[0] + a[i11] * x[1] + a[i12] * x[2];
  y[2] = a[i20] * x[0] + a[i21] * x[1] + a[i22] * x[2];
}

// b <= pow(a, -0.5)
inline void isqrt (const Sym33 & a, Sym33 & b, size_t iters = 3)
{
  // heron's method
  Sym33 x = 0.5f * (identity() + a);
  for (size_t iter = 0; iter < iters; ++iter) {
    inverse(x, b);
    Sym33 ax;
    multiply(a, b, ax);
    x = 0.5f * (x + ax);
  }
  inverse(x, b);
}

} // namespace Sym33

#endif // KAZOO_SYM33_H
