
#include "splines.h"
#include <algorithm>

#define LOG1(mess)

inline void safely_invert (float & t, float tol)
{
  t = (t > tol) ? 1 / t : 0;
}

//----( basic spline )--------------------------------------------------------

void Spline::setup (const float * fun)
{
  LOG1("Spline setting up forward mapping");
  for (size_t dom_i = 0; dom_i < m_size_in; ++dom_i) {
    float rng_t = fun[dom_i];
    float rng_i = rng_t * m_size_out - 0.5;

    linear_interpolate(
        rng_i,
        m_size_out,
        m_i0[dom_i],
        m_w0[dom_i],
        m_w1[dom_i]);
  }

  LOG1("Spline setting up backward mapping");
  m_scale_bwd.zero();
  for (size_t dom_i = 0; dom_i < m_size_in; ++dom_i) {
    size_t rng_i0 = m_i0[dom_i];

    m_scale_bwd[rng_i0  ] += m_w0[dom_i];
    m_scale_bwd[rng_i0+1] += m_w1[dom_i];
  }
  for (size_t rng_i = 0; rng_i < m_size_out; ++rng_i) {
    safely_invert(m_scale_bwd[rng_i], m_tolerance);
  }

  LOG1("spline inverse scale bounds = ["
      << min(m_scale_bwd) << ", "
      << max(m_scale_bwd) << "]");
}

Spline::Spline (
    size_t size_in,
    size_t size_out)

  : m_size_in(size_in),
    m_size_out(size_out),

    m_i0(size_in),
    m_w0(size_in),
    m_w1(size_in),

    m_scale_bwd(size_out),
    m_scaled_e_rng(size_out),

    m_tolerance(1e-10)
{
  ASSERTW(size_in >= size_out,
          "spline is being used : small -> large; consider reversing");
  Vector<float> vect(m_size_in);
  sample_uniform(vect);
  setup(vect);
}

Spline::Spline (
    size_t size_in,
    size_t size_out,
    const float * fun)

  : m_size_in(size_in),
    m_size_out(size_out),

    m_i0(size_in),
    m_w0(size_in),
    m_w1(size_in),

    m_scale_bwd(size_out),
    m_scaled_e_rng(size_out),

    m_tolerance(1e-10)
{
  ASSERTW(size_in >= size_out,
          "spline is being used : small -> large; consider reversing");
  setup(fun);
}

Spline::Spline (
    size_t size_in,
    size_t size_out,
    const Function & fun)

  : m_size_in(size_in),
    m_size_out(size_out),

    m_i0(size_in),
    m_w0(size_in),
    m_w1(size_in),

    m_scale_bwd(size_out),
    m_scaled_e_rng(size_out),

    m_tolerance(1e-10)
{
  ASSERTW(size_in >= size_out,
          "spline is being used : small -> large; consider reversing");
  Vector<float> vect(m_size_in);
  sample_function(fun, vect);
  setup(vect);
}

void Spline::transform_fwd (
    const Vector<float> & e_dom,
    Vector<float> & e_rng) const
{
  LOG1("Spline transforming forward");
  ASSERT_SIZE(e_dom, m_size_in);
  ASSERT_SIZE(e_rng, m_size_out);

  e_rng.zero();

  for (size_t dom_i = 0; dom_i < m_size_in; ++dom_i) {
    float e = e_dom[dom_i];
    size_t rng_i0 = m_i0[dom_i];

    e_rng[rng_i0  ] += e * m_w0[dom_i];
    e_rng[rng_i0+1] += e * m_w1[dom_i];
  }
}

void Spline::transform_bwd (
    const Vector<float> & e_rng,
    Vector<float> & e_dom) const
{
  LOG1("Spline transforming backward");
  ASSERT_SIZE(e_rng, m_size_out);
  ASSERT_SIZE(e_dom, m_size_in);

  multiply(m_scale_bwd, e_rng, m_scaled_e_rng);

  e_dom.zero();

  for (size_t dom_i = 0; dom_i < m_size_in; ++dom_i) {
    size_t rng_i0 = m_i0[dom_i];
    e_dom[dom_i] += m_scaled_e_rng[rng_i0  ] * m_w0[dom_i]
                  + m_scaled_e_rng[rng_i0+1] * m_w1[dom_i];
  }
}

void Spline::transform_bwd (
    const Vector<float> & e_rng,
    Vector<float> & e_dom,
    size_t size) const
{
  LOG1("Spline transforming vector backward");
  ASSERT_SIZE(e_rng, m_size_out * size);
  ASSERT_SIZE(e_dom, m_size_in * size);

  e_dom.zero();

  for (size_t dom_i = 0; dom_i < m_size_in; ++dom_i) {
    size_t rng_i0 = m_i0[dom_i];

    const float w0 = m_w0[dom_i] * m_scale_bwd[rng_i0];
    const float w1 = m_w1[dom_i] * m_scale_bwd[rng_i0+1];

    float * restrict ed = e_dom + size * dom_i;
    const float * restrict er0 = e_rng + size * rng_i0;
    const float * restrict er1 = er0 + size;

    for (size_t j = 0; j < size; ++j) {
      ed[j] += w0 * er0[j] + w1 * er1[j];
    }
  }
}

//----( circular spline )-----------------------------------------------------

void SplineToCircle::setup (float scale, float shift)
{
  LOG1("SplineToCircle setting up forward mapping");
  for (size_t dom_i = 0; dom_i < m_size_in; ++dom_i) {
    float rng_i = scale * dom_i + shift;

    circular_interpolate(
        rng_i,
        m_size_out,
        m_i0[dom_i],
        m_w0[dom_i],
        m_w1[dom_i]);
  }

  LOG1("Spline setting up backward mapping");
  m_scale_bwd.zero();
  for (size_t dom_i = 0; dom_i < m_size_in; ++dom_i) {
    size_t rng_i0 = m_i0[dom_i];
    size_t rng_i1 = (rng_i0+1) % m_size_out;

    m_scale_bwd[rng_i0] += m_w0[dom_i];
    m_scale_bwd[rng_i1] += m_w1[dom_i];
  }
  for (size_t rng_i = 0; rng_i < m_size_out; ++rng_i) {
    safely_invert(m_scale_bwd[rng_i], m_tolerance);
  }
}

SplineToCircle::SplineToCircle (
    size_t size_in,
    size_t size_out,
    float index_scale,
    float index_shift)

  : m_size_in(size_in),
    m_size_out(size_out),

    m_i0(size_in),
    m_w0(size_in),
    m_w1(size_in),

    m_scale_bwd(size_out),
    m_scaled_e_rng(size_out),

    m_tolerance(1e-10)
{
  setup(index_scale, index_shift);
}

void SplineToCircle::transform_fwd (
    const Vector<float> & e_dom,
    Vector<float> & e_rng) const
{
  LOG1("Spline transforming forward");
  ASSERT_SIZE(e_dom, m_size_in);
  ASSERT_SIZE(e_rng, m_size_out);

  e_rng.zero();

  for (size_t dom_i = 0; dom_i < m_size_in; ++dom_i) {
    float e = e_dom[dom_i];
    size_t rng_i0 = m_i0[dom_i];
    size_t rng_i1 = (rng_i0+1) % m_size_out;

    e_rng[rng_i0] += e * m_w0[dom_i];
    e_rng[rng_i1] += e * m_w1[dom_i];
  }
}

void SplineToCircle::transform_bwd (
    const Vector<float> & e_rng,
    Vector<float> & e_dom) const
{
  LOG1("Spline transforming backward");
  ASSERT_SIZE(e_rng, m_size_out);
  ASSERT_SIZE(e_dom, m_size_in);

  multiply(m_scale_bwd, e_rng, m_scaled_e_rng);

  e_dom.zero();

  for (size_t dom_i = 0; dom_i < m_size_in; ++dom_i) {
    size_t rng_i0 = m_i0[dom_i];
    size_t rng_i1 = (rng_i0+1) % m_size_out;

    e_dom[dom_i] += m_scaled_e_rng[rng_i0] * m_w0[dom_i]
                  + m_scaled_e_rng[rng_i1] * m_w1[dom_i];
  }
}

//----( 2d separable spline )-------------------------------------------------

void Spline2DSeparable::transform_fwd (
    const Vector<float> & e_dom,
    Vector<float> & e_rng) const
{
  ASSERT_SIZE(e_dom, size_in());
  ASSERT_SIZE(e_rng, size_out());

  const size_t dom_I = size_in1();
  const size_t dom_J = size_in2();
  const size_t rng_J = size_out2();

  const int * restrict I0 = m_spline1.m_i0;
  const int * restrict J0 = m_spline2.m_i0;

  const float * restrict W0_ = m_spline1.m_w0;
  const float * restrict W1_ = m_spline1.m_w1;
  const float * restrict W_0 = m_spline2.m_w0;
  const float * restrict W_1 = m_spline2.m_w1;

  e_rng.zero();

  for (size_t dom_i = 0; dom_i < dom_I; ++dom_i) {
    size_t rng_i0 = rng_J * (I0[dom_i] + 0);
    size_t rng_i1 = rng_J * (I0[dom_i] + 1);

    float w0_ = W0_[dom_i];
    float w1_ = W1_[dom_i];

    for (size_t dom_j = 0; dom_j < dom_J; ++dom_j) {
      size_t rng_j0 = J0[dom_j] + 0;
      size_t rng_j1 = J0[dom_j] + 1;

      float w_0 = W_0[dom_j];
      float w_1 = W_1[dom_j];

      size_t dom_ij = dom_J * dom_i + dom_j;
      float e = e_dom[dom_ij];

      e_rng[rng_i0 + rng_j0] += e * w0_ * w_0;
      e_rng[rng_i0 + rng_j1] += e * w0_ * w_1;
      e_rng[rng_i1 + rng_j0] += e * w1_ * w_0;
      e_rng[rng_i1 + rng_j1] += e * w1_ * w_1;
    }
  }
}

// XXX this has a bug whereby the last interval seems to be reversed.
//   to reproduce, try zooming in on a very small video with 1px black borders.
void Spline2DSeparable::transform_bwd (
    const Vector<float> & e_rng,
    Vector<float> & e_dom) const
{
  ASSERT_SIZE(e_rng, size_out());
  ASSERT_SIZE(e_dom, size_in());

  const size_t dom_I = size_in1();
  const size_t dom_J = size_in2();
  const size_t rng_J = size_out2();

  const int * restrict I0 = m_spline1.m_i0;
  const int * restrict J0 = m_spline2.m_i0;

  const float * restrict W0_ = m_spline1.m_w0;
  const float * restrict W1_ = m_spline1.m_w1;
  const float * restrict W_0 = m_spline2.m_w0;
  const float * restrict W_1 = m_spline2.m_w1;

  e_dom.zero();

  for (size_t dom_i = 0; dom_i < dom_I; ++dom_i) {
    size_t rng_i0 = rng_J * (I0[dom_i] + 0);
    size_t rng_i1 = rng_J * (I0[dom_i] + 1);

    float w0_ = W0_[dom_i];
    float w1_ = W1_[dom_i];

    for (size_t dom_j = 0; dom_j < dom_J; ++dom_j) {
      size_t rng_j0 = J0[dom_j] + 0;
      size_t rng_j1 = J0[dom_j] + 1;

      float w_0 = W_0[dom_j];
      float w_1 = W_1[dom_j];

      size_t dom_ij = dom_J * dom_i + dom_j;

      e_dom[dom_ij] = w0_ * ( w_0 * e_rng[rng_i0 + rng_j0]
                            + w_1 * e_rng[rng_i0 + rng_j1] )
                    + w1_ * ( w_0 * e_rng[rng_i1 + rng_j0]
                            + w_1 * e_rng[rng_i1 + rng_j1] );
    }
  }
}

//----( 2d general spline )---------------------------------------------------

Spline2D::Spline2D (
    size_t width_in,
    size_t height_in,
    size_t width_out,
    size_t height_out,
    const Transform & transform,
    float min_mass)

  : m_width_in(width_in),
    m_height_in(height_in),
    m_width_out(width_out),
    m_height_out(height_out),
    m_size_in(width_in * height_in),
    m_size_out(width_out * height_out),

    m_ij_out(m_size_in),
    m_weights(m_size_in)
{
  ASSERT_DIVIDES(block_size, m_width_in);
  ASSERT_DIVIDES(block_size, m_height_in);

  const size_t J_in = m_height_in;
  const size_t J_out = m_height_out;

  // transform via function
  for (size_t i_in = 0; i_in < m_width_in; ++i_in) {
  for (size_t j_in = 0; j_in < m_height_in; ++j_in) {

    size_t ij_in = J_in * i_in + j_in;

    int i0,j0;
    float w0_, w1_, w_0, w_1;

    float x_in = 2.0f * i_in / m_width_in - 1;
    float y_in = 2.0f * j_in / m_height_in - 1;

    Point xy(x_in, y_in);
    transform(xy);

    float i_out = m_width_out * (1 + xy.x) / 2.0f - 0.5f;
    float j_out = m_height_out * (1 + xy.y) / 2.0f - 0.5f;

    linear_interpolate(i_out, m_width_out, i0, w0_, w1_);
    linear_interpolate(j_out, m_height_out, j0, w_0, w_1);

    size_t ij_out = m_ij_out[ij_in] = J_out * i0 + j0;

    float4 & weights = m_weights[ij_out];
    weights[0] = w0_ * w_0;
    weights[1] = w0_ * w_1;
    weights[2] = w1_ * w_0;
    weights[3] = w1_ * w_1;
  }}

  // renormalize to preserve constant functions
  Vector<float> mass_in(m_size_in);
  mass_in.set(1.0f);

  Vector<float> mass_out(m_size_out);
  transform_fwd(mass_in, mass_out);

  float total_mass = 0;
  for (size_t ij = 0; ij < m_size_out; ++ij) {
    total_mass += mass_out[ij];

    mass_out[ij] = 1 / (min_mass + mass_out[ij]);
  }
  float mean_mass = total_mass / m_size_in;
  LOG("Spline2D image intersects " << mean_mass << " of range");

  for (size_t ij = 0; ij < m_size_in; ++ij) {
    size_t ij_out = m_ij_out[ij];
    float4 & weights = m_weights[ij];

    weights[0] *= mass_out[ij_out +   0   + 0];
    weights[1] *= mass_out[ij_out +   0   + 1];
    weights[2] *= mass_out[ij_out + J_out + 0];
    weights[3] *= mass_out[ij_out + J_out + 1];
  }
}

void Spline2D::transform_fwd (
    const Vector<float> & f_in,
    Vector<float> & f_out) const
{
  ASSERT_SIZE(f_in, m_size_in);
  ASSERT_SIZE(f_out, m_size_out);

  f_out.zero();

  const size_t J_in = m_height_in;
  const size_t J_out = m_height_out;

  for (size_t i_ = 0; i_ < m_width_in; i_ += block_size) {
  for (size_t j_ = 0; j_ < m_height_in; j_ += block_size) {

    for (size_t _i = 0; _i < block_size; ++_i) { size_t i = i_ + _i;
    for (size_t _j = 0; _j < block_size; ++_j) { size_t j = j_ + _j;

      size_t ij_in = J_in * i + j;
      size_t ij_out = m_ij_out[ij_in];

      float p_in = f_in[ij_in];
      float4 weights = m_weights[ij_in];
      float * restrict p_out = f_out + ij_out;

      p_out[  0   + 0] += p_in * weights[0];
      p_out[  0   + 1] += p_in * weights[1];
      p_out[J_out + 0] += p_in * weights[2];
      p_out[J_out + 1] += p_in * weights[3];
    }}
  }}
}

