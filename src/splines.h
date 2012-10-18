
/** Spline functions.

*/

#ifndef KAZOO_SPLINES_H
#define KAZOO_SPLINES_H

#include "common.h"
#include "vectors.h"
#include "array.h"
#include "images.h"

class Spline2DSeparable;

//----( cubic spline primitives )---------------------------------------------

inline float cubic_spline_value (float t)
{
  ASSERT((0 <= t) and (t <= 1), "spline argument out of range: " << t);
  return 2.0f * t * t * (1.5f - t);
}
inline float cubic_spline_deriv (float t)
{
  ASSERT((0 <= t) and (t <= 1), "spline argument out of range: " << t);
  return 6.0f * t * (1.0f - t);
}

struct CubicSpline : public FunctionAndDeriv
{
  virtual ~CubicSpline () {}
  virtual float value (float t) const { return cubic_spline_value(t); }
  virtual float deriv (float t) const { return cubic_spline_deriv(t); }
};

/** Interpolation functions.

  Given: approximate coord t in [0,1] and coord partition I,
  Compute: integer coord i0, weight w0 at i0, and at weight w1 at i1+1.

  Note: i1 = 1+i0, w0+w1 <= 1

*/

inline void linear_interpolate (
    float i,
    int I,
    int & restrict i0,
    float & restrict w0,
    float & restrict w1)
{
  i0 = static_cast<int>(floorf(i));

  // case: undershoot
  if (i0 < 0) {
    if (i0 + 1 < 0) {
      w0 = w1 = 0;
    } else {
      w1 = 0;
      w0 = 1 + i;
    }
    i0 = 0;
  }

  // case: overshoot
  else if (i0 + 1 >= I) {
    if (i0 >= I) {
      w0 = w1 = 0;
    } else { // i0 <= i < i0+1 = I
      w0 = 0;
      w1 = i0 + 1 - i;
    }
    i0 = I - 2;
  }

  // case: somewhere in the middle
  else {
    w0 = 1 - i + i0;
    w1 = 1.0f - w0;
  }

  ASSERT1_NONNEG(w0);
  ASSERT1_NONNEG(w1);
  ASSERT1(w0 + w1 <= 1+1e-6, "too much weight: w0 = " << w0 << ", w1 = " << w1);
}

inline void circular_interpolate (
    float i,
    int I,
    int & restrict i0,
    float & restrict w0,
    float & restrict w1)
{
  i0 = static_cast<int>(floorf(i));

  w0 = 1 - i + i0;
  w1 = 1.0f - w0;
  ASSERT_NONNEG(w0);
  ASSERT_NONNEG(w1);
  ASSERT(w0 + w1 <= 1+1e-6, "too much weight: w0 = " << w0 << ", w1 = " << w1);

  if (i0 < 0) i0 += (I-i0-1) / I * I;
  i0 %= I;
  ASSERT((0 <= i0) and (i0 < I), "i0 is out of range [0," << I << "): " << i0);
}

struct LinearInterpolate
{
  int i0, i1;
  float w0, w1;

  LinearInterpolate & operator() (float i, int I)
  {
    linear_interpolate(i, I, i0, w0, w1);
    i1 = i0 + 1;
    return * this;
  }

  LinearInterpolate () {}
  LinearInterpolate (float i, int I) { operator()(i,I); }

  float get (const float * restrict f) const
  {
    return f[i0] * w0
         + f[i1] * w1;
  }
  void iadd (float * restrict f, float df) const
  {
    f[i0] += w0 * df;
    f[i1] += w1 * df;
  }
  void imax (float * restrict f, float f0) const
  {
    ::imax(f[i0], w0 * f0);
    ::imax(f[i1], w1 * f0);
  }
};

struct BilinearInterpolate
{
  LinearInterpolate x;
  LinearInterpolate y;
  size_t Y;

  BilinearInterpolate (float i, int I, float j, int J) : x(i,I), y(j,J), Y(J) {}

  float get (const float * restrict f) const
  {
    return f[Y * x.i0 + y.i0] * x.w0 * y.w0
         + f[Y * x.i0 + y.i1] * x.w0 * y.w1
         + f[Y * x.i1 + y.i0] * x.w1 * y.w0
         + f[Y * x.i1 + y.i1] * x.w1 * y.w1;

  }
  void iadd (float * restrict f, float df) const
  {
    f[Y * x.i0 + y.i0] += x.w0 * y.w0 * df;
    f[Y * x.i0 + y.i1] += x.w0 * y.w1 * df;
    f[Y * x.i1 + y.i0] += x.w1 * y.w0 * df;
    f[Y * x.i1 + y.i1] += x.w1 * y.w1 * df;
  }
  void imax (float * restrict f, float f0) const
  {
    ::imax(f[Y * x.i0 + y.i0], x.w0 * y.w0 * f0);
    ::imax(f[Y * x.i0 + y.i1], x.w0 * y.w1 * f0);
    ::imax(f[Y * x.i1 + y.i0], x.w1 * y.w0 * f0);
    ::imax(f[Y * x.i1 + y.i1], x.w1 * y.w1 * f0);
  }
};

struct CircularInterpolate
{
  int i0, i1;
  float w0, w1;

  void operator() (float i, int I)
  {
    linear_interpolate(i, I, i0, w0, w1);
    i1 = (i0 + 1) % I;
  }

  CircularInterpolate () {}
  CircularInterpolate (float i, int I) { operator()(i,I); }
};

//----( basic spline )--------------------------------------------------------

// spline transformations preserve mass locally (but may lose mass at edges
class Spline
{
  friend class Spline2DSeparable;

  const size_t m_size_in;
  const size_t m_size_out;

  Vector<int> m_i0;
  Vector<float> m_w0;
  Vector<float> m_w1;

  Vector<float> m_scale_bwd;
  mutable Vector<float> m_scaled_e_rng;

  const float m_tolerance;

public:

  void setup (const float * fun);

  Spline (
      size_t size_in,
      size_t size_out); // identity function by default
  Spline (
      size_t size_in,
      size_t size_out,
      const float * fun);
  Spline (
      size_t size_in,
      size_t size_out,
      const Function & fun);

  // diagnostics
  size_t size_in  () const { return m_size_in; }
  size_t size_out () const { return m_size_out; }

  // these can operate in parallel
  void transform_fwd (const Vector<float> & e_dom, Vector<float> & e_rng) const;
  void transform_bwd (const Vector<float> & e_rng, Vector<float> & e_dom) const;

  //vectorized versions
  void transform_bwd (
      const Vector<float> & e_rng,
      Vector<float> & e_dom,
      size_t size) const;
};

//----( circular spline )-----------------------------------------------------

class SplineToCircle
{
  const size_t m_size_in;
  const size_t m_size_out;

  Vector<int> m_i0;
  Vector<float> m_w0;
  Vector<float> m_w1;

  Vector<float> m_scale_bwd;
  mutable Vector<float> m_scaled_e_rng;

  const float m_tolerance;

public:
  void setup (float scale, float shift);

  SplineToCircle (
      size_t size_in,
      size_t size_out,
      float index_scale = 1.0,
      float index_shift = 0.0); // linear wrapping by default

  // diagnostics
  size_t size_in  () const { return m_size_in; }
  size_t size_out () const { return m_size_out; }

  // these can operate in parallel
  void transform_fwd (const Vector<float> & e_dom, Vector<float> & e_rng) const;
  void transform_bwd (const Vector<float> & e_rng, Vector<float> & e_dom) const;
};

//----( 2d separable spline )-------------------------------------------------

class Spline2DSeparable
{
  const Spline m_spline1;
  const Spline m_spline2;

public:

  Spline2DSeparable (Rectangle shape_in, Rectangle shape_out)
    : m_spline1(shape_in.width(), shape_out.width()),
      m_spline2(shape_in.height(), shape_out.height())
  {}

  Spline2DSeparable (
      size_t size_in1,
      size_t size_in2,
      size_t size_out1,
      size_t size_out2)

    : m_spline1(size_in1, size_out1),
      m_spline2(size_in2, size_out2)
  {}

  size_t size_in1  () const { return m_spline1.size_in(); }
  size_t size_in2  () const { return m_spline2.size_in(); }
  size_t size_in   () const { return size_in1() * size_in2(); }

  size_t size_out1 () const { return m_spline1.size_out(); }
  size_t size_out2 () const { return m_spline2.size_out(); }
  size_t size_out  () const { return size_out1() * size_out2(); }

  // these can operate in parallel
  void transform_fwd (const Vector<float> & e_dom, Vector<float> & e_rng) const;
  void transform_bwd (const Vector<float> & e_rng, Vector<float> & e_dom) const;
};

//----( 2d general spline )---------------------------------------------------

/** General 2D apline for mapping regions of the plane to regions of the plane

Invariants:
(I1) Maps of constant functions should be preserved.

*/

class Spline2D
{
public:

  enum { block_size = 4 };

  typedef Image::Point Point;
  typedef Image::Transform Transform;

public:

  const size_t m_width_in;
  const size_t m_height_in;
  const size_t m_width_out;
  const size_t m_height_out;
  const size_t m_size_in;
  const size_t m_size_out;

  Vector<size_t> m_ij_out;
  Vector<float4> m_weights;

public:

  Spline2D (
      size_t width_in,
      size_t height_in,
      size_t width_out,
      size_t height_out,
      const Transform & transform,
      float min_mass = 1e-4f);

  void transform_fwd (const Vector<float> & f_in, Vector<float> & f_out) const;
};

//----( spline accumulator )--------------------------------------------------

/** A spline accumulator for linear-time blurring convolution.
*/

template<class Value>
class SplineAccumulator
{
  Vector<Value> m_accum;
public:
  SplineAccumulator (size_t size) : m_accum(size) { m_accum.zero(); }

  size_t size () const { return m_accum.size; }

  // access for [0-1] valued locations
  void add (float location, Value value);
  Value get (float location) const;
  void zero () { m_accum.zero(); }
  void scale (float factor);
  void blur (Vector<Value> & values);
};

template<class Value>
void SplineAccumulator<Value>::add (float location, Value value)
{
  float i = (0.5f + location) / size();
  int i0;
  float w0, w1;
  linear_interpolate (i, size(), i0, w0, w1);

  m_accum[i0  ] += w0 * value;
  m_accum[i0+1] += w1 * value;
}

template<class Value>
Value SplineAccumulator<Value>::get (float location) const
{
  float i = (0.5f + location) / size();
  int i0;
  float w0, w1;
  linear_interpolate (i, size(), i0, w0, w1);

  return w0 * m_accum[i0]
       + w1 * m_accum[i0+1];
}

template<class Value>
void SplineAccumulator<Value>::scale (float factor)
{
  for (size_t i = 0; i < size(); ++i) {
    m_accum[i] *= factor;
  }
}

template<class Value>
void SplineAccumulator<Value>::blur (Vector<Value> & values)
{
  zero();

  for (size_t i = 0; i < values.size; ++i) {
    float t = (0.5f + i) / values.size;
    add(t, values[i]);
  }

  for (size_t i = 0; i < values.size; ++i) {
    float t = (0.5f + i) / values.size;
    values[i] = get(t);
  }
}

#endif // KAZOO_SPLINES_H

