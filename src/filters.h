#ifndef KAZOO_FILTERS_H
#define KAZOO_FILTERS_H

#include "common.h"
#include "array.h"
#include "vectors.h"
#include "probability.h"

#define DENOISER_SNR                    (1.5f)

namespace Filters
{

//----( autogain )------------------------------------------------------------

// TODO allow saving & loading of state.
// TODO allow freezing of state, eg, only when loaded, or after 1 minute.

class MaxGain
{
  const float m_init;
  const float m_factor;
  float m_gain;

public:

  MaxGain (float timescale, float init = 1)
    : m_init(init),
      m_factor(1 + 1 / timescale),
      m_gain(0)
  {
    ASSERT_LT(1, m_factor);
  }

  void operator= (const MaxGain & other) { m_gain = other.m_gain; }

  float timescale () const { return 1 / (m_factor - 1); }

  void clear () { m_gain = 0; }
  operator float () const { return m_gain; }
  float update (float value)
  {
    if (value > 0) {
      if (m_gain > 0) {
        m_gain = min(1 / value, m_factor * m_gain);
      } else {
        m_gain = m_init / value;
      }
    }
    return m_gain;
  }
};

class RmsGain
{
  const float m_decay;
  float m_mass;
  float m_variance;
  float m_gain;

public:

  RmsGain (float timescale)
    : m_decay(exp(-1 / timescale)),
      m_mass(0),
      m_variance(0),
      m_gain(0)
  {
    ASSERT_LT(0, m_decay);
    ASSERT_LT(m_decay, 1);
  }

  void clear () { m_mass = m_variance = m_gain = 0; }
  operator float () const { return m_gain; }
  float update (float variance, float mass = 1)
  {
    m_mass = m_decay * m_mass + mass;
    m_variance = m_decay * m_variance + variance;
    if ((m_variance > 0) and (m_mass > 0)) {
      m_gain += m_decay * (sqrtf(m_mass / m_variance) - m_gain);
    }
    return m_gain;
  }
};

class StdGain
{
  const float m_decay;
  float m_mass;
  float m_mean;
  float m_variance;

public:

  StdGain (float timescale, float init_variance = 0)
    : m_decay(exp(-1 / timescale)),
      m_mass(0),
      m_mean(0),
      m_variance(init_variance)
  {
    ASSERT_LT(0, m_decay);
    ASSERT_LT(m_decay, 1);
  }

  float mean () const { return m_mean; }
  float variance () const { return m_variance; }
  float operator() (float x) const { return (x - m_mean) / sqrtf(m_variance); }

  float update (float x, float mass = 1)
  {
    m_mass = m_decay * m_mass + mass;
    float rate = mass / m_mass;

    m_mean += rate * (x - m_mean);
    float dx = x - m_mean;
    m_variance += rate * (sqr(dx) - m_variance);

    return m_variance > 0 ? dx / sqrtf(m_variance) : 0.0f;
  }
  void clear () { m_mass = m_mean = m_variance = 0; }
  void reset () { m_mass = m_mean = m_variance = 0; }
};

//----( denoiser )------------------------------------------------------------

class Denoiser
{
  const float m_fast_decay;
  const float m_slow_decay;

  float m_lowpass_power;
  float m_min_power;
  float m_max_power;

public:

  Denoiser (
      float slow_timescale,
      float fast_timescale = 16)

    : m_fast_decay(exp(-1 / fast_timescale)),
      m_slow_decay(exp(-1 / slow_timescale)),

      m_lowpass_power(0),
      m_min_power(INFINITY),
      m_max_power(0)
  {}
  ~Denoiser ()
  {
    LOG("denoiser power in (" << m_min_power << ", " << m_max_power << ")");
  }

  float operator() (float power)
  {
    m_lowpass_power = max(power, m_lowpass_power * m_fast_decay);
    m_min_power = min(m_lowpass_power, m_min_power / m_slow_decay);
    m_max_power = max(power, m_max_power * m_slow_decay);

    float scale = (m_max_power - DENOISER_SNR * m_min_power);
    if (not (scale > 0)) return 0;

    return max(0.0f, (power - DENOISER_SNR * m_min_power) / scale);
  }
};

//----( piecewise linear filter )---------------------------------------------

/** Piecewise Linear Interpolate
*/
class Interpolate
{
  float m_pos;
  float m_vel;
  float m_age;

public:

  Interpolate (float pos = 0) : m_pos(pos), m_vel(0), m_age(0) {}

  float predict (float dt) const { return m_pos + m_vel * (dt + m_age); }

  void advance (float dt) { m_age += dt; }

  void update (float pos)
  {
    ASSERT_LT(0, m_age);
    m_vel = (pos - m_pos) / m_age;
    m_pos = pos;
    m_age = 0;
  }
};

//----( peak detector )-------------------------------------------------------

/** Peak Detector

  The peak detector finds peaks above a noise threshold,
  and masks recent peaks.
*/
class PeakDetector
{
  const float m_snr;
  const float m_mask_decay;

  float m_noise_threshold;
  float m_mask;

public:

  PeakDetector (float snr, float mask_timescale)
    : m_snr(snr),
      m_mask_decay(exp(-1 / mask_timescale)),

      m_noise_threshold(0),
      m_mask(0)
  {
    ASSERT_LT(1, m_snr);
    ASSERT_LT(0, mask_timescale);
  }
  ~PeakDetector ()
  {
    PRINT2(m_noise_threshold, m_mask);
  }

  bool detect (float value)
  {
    bool is_peak = value > m_mask + m_noise_threshold;

    m_noise_threshold = max(m_noise_threshold, value / m_snr);
    m_mask = m_mask_decay * max(m_mask, value);

    return is_peak;
  }
};

//----( moments )-------------------------------------------------------------

template<class T>
inline void update_mean (T obs, T rate, T & restrict mean)
{
  T diff = obs - mean;
  mean += rate * diff;
}

template<class T>
inline void update_mean_var (T obs, T rate, T & restrict mean, T & restrict var)
{
  T diff = obs - mean;
  mean += rate * diff;
  var += rate * sqr(diff);
}

#ifdef KAZOO_NDEBUG

template<class T>
class DebugStats
{
public:
  DebugStats () {}

  operator bool () const { return false; }
  void add (T) {}

  template<class S>
  void add (const S * restrict, size_t) {}
  template<class S>
  void add (const Vector<S> &) {}

  friend inline ostream & operator<< (ostream & o, const DebugStats<T> &)
  {
    return o << "(stats optimized out)";
  }

  T mean () const { return NAN; }
  T variance () const { return NAN; }
  T min () const { return NAN; }
  T max () const { return NAN; }
};

#else // KAZOO_NDEBUG

template<class T>
class DebugStats
{
  size_t total;
  T sum_x;
  T sum_xx;
  T min_x;
  T max_x;

public:

  DebugStats ()
    : total(0),
      sum_x(0),
      sum_xx(0),
      min_x(INFINITY),
      max_x(-INFINITY)
  {}

  operator bool () const { return total; }
  void add (T x)
  {
    ++total;
    sum_x += x;
    sum_xx += sqr(x);
    imax(max_x, x);
    imin(min_x, x);
  }

  template<class S>
  void add (const S * restrict x_data, size_t size)
  {
    T temp_sum_x = 0;
    T temp_sum_xx = 0;
    T temp_min_x = INFINITY;
    T temp_max_x = -INFINITY;

    for (size_t i = 0; i < size; ++i) {
      T x = x_data[i];

      temp_sum_x += x;
      temp_sum_xx += sqr(x);
      imax(temp_max_x, x);
      imin(temp_min_x, x);
    }

    sum_x += temp_sum_x;
    sum_xx += temp_sum_xx;
    imax(max_x, temp_max_x);
    imin(min_x, temp_min_x);
    total += size;
  }

  template<class S>
  void add (const Vector<S> & x) { add(x.data, x.size); }

  void operator+= (const DebugStats<T> & other)
  {
    total += other.total;
    sum_x += other.sum_x;
    sum_xx += other.sum_xx;
    imax(max_x, other.max_x);
    imin(min_x, other.min_x);
  }

  T mean () const { return sum_x / total; }
  T variance () const { return sum_xx / total - sqr(mean()); }
  T min () const { return min_x; }
  T max () const { return max_x; }

  friend inline ostream & operator<< (ostream & o, const DebugStats<T> & m)
  {
    return o << m.mean() << " +- " << sqrt(m.variance())
      << ", in [" << m.min() << ", " << m.max() << "]";
  }
};

#endif // KAZOO_NDEBUG

//============================================================================
/** SIMD Kalman Filters.

  These are NCV position + velocity filters and position observations.
  These filters assume all coordinates are independent,
    by omitting cross-covariance terms between coordinates.

  Notation:
    E = expectation
    P = covariance
    x = position
    y = velocity
*/

//----( NCP gaussians )-------------------------------------------------------

template<size_t size>
struct NCP : public Aligned<NCP<size> >
{
  Array<float, size> Ex;
  Array<float, size> Vxx;

  NCP () {}
  NCP (
      Array<float, size> a_Ex,
      Array<float, size> a_Vxx)

    : Ex(a_Ex),
      Vxx(a_Vxx)
  {}

  /** Gaussian likelihood ratio.
   *
   * Properties:
   *  * at fixed parameters mu,sigma,
   *    result((t-mu)/sigma) is proportional to normal density P(x;mu,sigma)
   *  * for all mu,sigma,x E[log result((x-mu)/sigma)] = 0
   */
  float likelihood (NCP<size> other) const
  {
    Array<float, size> chi2 = sqr(Ex - other.Ex) / (Vxx + other.Vxx);
    return gaussian_likelihood(sum(chi2), size);
  }
  float free_energy (NCP<size> other) const
  {
    Array<float, size> chi2 = sqr(Ex - other.Ex) / (Vxx + other.Vxx);
    return gaussian_free_energy(sum(chi2), size);
  }

  void mix (float weight, NCP<size> other) no_inline;

  const Array<float, size> & observe () const { return Ex; }
};

//----( NCV gaussians )-------------------------------------------------------

template<size_t size>
struct NCV : public NCP<size>
{
private:
  typedef NCP<size> Base;
public:
  using Base::Ex;
  using Base::Vxx;

  Array<float, size> Ey;
  Array<float, size> Vxy;
  Array<float, size> Vyy;

  NCV () {}
  NCV (
      NCP<size> x,
      Array<float, size> a_Vyy)

    : NCP<size>(x),
      Ey(0),
      Vxy(0),
      Vyy(a_Vyy)
  {}
  NCV (
      Array<float, size> a_Ex,
      Array<float, size> a_Ey,
      Array<float, size> a_Vxx,
      Array<float, size> a_Vxy,
      Array<float, size> a_Vyy)

    : NCP<size>(a_Ex, a_Vxx),
      Ey(a_Ey),
      Vxy(a_Vxy),
      Vyy(a_Vyy)
  {}

  void mix (float weight, NCV<size> other) no_inline;

  Array<float, size> predict (float time) const { return Ex + Ey * time; }

  void advance (float dt, Array<float, size> process_noise) no_inline;

  Array<float, size> update (NCP<size> observed) no_inline; // returns chi_y

  void fuse (NCV<size> other) no_inline; // other is used as temporary

private:
  inline void invert_P (); // defined only in .cpp file
};

//----( i/o )-----------------------------------------------------------------

template<size_t size>
inline ostream & operator << (ostream & o, const NCP<size> & x)
{
  return o << "NCP " << size
    << "\n Ex = " << x.Ex
    << "\n Vxx = " << x.Vxx;
}

template<size_t size>
inline ostream & operator << (ostream & o, const NCV<size> & xy)
{
  return o << "NCV " << size
    << "\n Ex = " << xy.Ex
    << "\n Ey = " << xy.Ey
    << "\n Vxx = " << xy.Vxx
    << "\n Vxy = " << xy.Vxy
    << "\n Vyy = " << xy.Vyy;
}

} // namespace Filters

#endif // KAZOO_FILTERS_H
