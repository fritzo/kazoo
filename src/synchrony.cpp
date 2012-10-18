
#include "synchrony.h"
#include "images.h"
#include "splines.h"
#include <algorithm>

#define PITCH_BEND_TIMESCALE_SEC        (0.1f)
#define PITCH_BEND_TIMESCALE_FRAMES     ( PITCH_BEND_TIMESCALE_SEC \
                                        * DEFAULT_AUDIO_FRAMERATE )

#define TEMPO_DRIFT_TIMESCALE           (20.0f)
#define LOOP_BANK_PERIOD                (16)

#define TOL                             (1e-20f)

#define LOG1(message)

//#define ASSERT2_LE(x,y) ASSERT_LE(x,y)
//#define ASSERT2_LT(x,y) ASSERT_LE(x,y)

#define ASSERT2_LE(x,y)
#define ASSERT2_LT(x,y)

namespace Synchronized
{

float safe (float x) { return (abs(x) < 1 / TOL) ? x : 0; }

inline void rotate (complex & restrict phase, float freq)
{
  complex dphase(-freq * phase.imag(), freq * phase.real());
  phase += dphase;
  phase /= abs(phase);
}

//----( oscillator tools )----------------------------------------------------

/** Bend Functions

  (E1) slow heavy + fast light
    The slow oscillator will create wells for the fast oscillator.
    The period of the fast oscillator through such a well
    should be independent of well depth.
    For a symmetric energy perturbation e(theta) with gradient f(theta),
    the fast oscillator has dynamics
    \[
      \dot \theta = \omega_0 bend(f(theta))
    \]
    with normalized period
    \[
      T \omega_0 = \int_\theta 1 / bend(f(\theta))
    \]
    For $bend(f(\theta)) = 1 / (a + b f(\theta))$,
    T is invariant under f scaling (since $\int_\theta f(\theta) = 0$).
    Matching the first two moments, we derive the period bending function
    \[
      bend(\theta) = 1 / (1 - f(theta))
    \]

  (E2) fast heavy + slow light
    The fast oscillator will create a quickly varying field which the
    slow oscillator samples wlog densely and uniformly.
    The dynamics of the slow oscillator in the adiabatic field
    should be the same as without a fast oscillator, ie via
    \[
      \dot \Theta = \Omega_0 = \Omega0 B
    \]
    where we require the bend to time-average to B = 1,
    \[
      B = \int \theta. \frac{ \omega_0 bend(f(\theta)) sin(Theta - \theta) }
                            { \dot \theta }
    \]
    For general \Theta, this is equivalent to the complex equation
    \[
      B = \int \theta. \frac{ \omega_0 bend(f(\theta)) \exp(i \theta) }
                            { \dot \theta }
    \]
    In a general force field as in (E1) a period-bending fast oscillator obeys
    \[
      1 = \int \theta. \frac { \omega_0 } { \dot \theta }
    \]
    TODO understand this example

  (E3) two oscillators at nearby frequencies

*/
inline float bend_function (float bend)
{
  return
  //1 + bend; // bend frequency
  //1 / (1 - bend); // bend timescale (has a singularity!)
  //1 + bend * (1 + bend); // bend timescale approximately
  //1 + bend * (1 + bend / 2); // bend pitch approximately
  //exp(bend); // bend pitch (slow!)
  //max(0.0f, 1 + bend); // bend frequency, but don't go backwards
  //sqr(max(0.0f, 1 + bend)); // quadratic hinge function
  max(0.1f, 1 + bend); // bend frequency, but don't stop
}

inline float approx_exp (float x)
{
  float dx = bound_to(-1.0f, 1.0f, x);
  return 1 + dx * (1 + dx / 2);
}

void random_phase (Vector<float> & re, Vector<float> & im)
{
  ASSERT_EQ(re.size, im.size);

  for (size_t i = 0, I = re.size; i < I; ++i) {
    complex z = exp_2_pi_i(random_01());
    re[i] = z.real();
    im[i] = z.imag();
  }
}

//----( individual oscillators )----------------------------------------------

complex Phasor::sample (complex force)
{
  bend = beat * cross(phase, force);
  float dphase = freq * bend_function(bend);
  rotate(phase, dphase);
  beat = beat_fun.value(phase.real());

  return phase;
}

complex Syncopator::sample (complex force)
{
  bend = beat * cross(offset * phase, force);
  rotate(phase, freq * bend_function(bend));
  beat = beat_fun.value(phase.real());

  return phase;
}

complex Shepard4::sample (complex force_in)
{
  const float p = bound_to(0.05f, 0.95f, octave / 3);
  const float q = 1 - p;
  const float weights[4] = {
    q*q*q * 1,
    q*q * 3 * p,
    q * 3 * p*p,
    1 * p*p*p
  };
  const float beat_floor = beat_fun.beat_floor;
  const float beat_scale = beat_fun.beat_scale;

  // update phase
  float new_bend = 0;
  {
    complex state = phase;
    for (size_t i = 0; i < 4; ++i) {
      float beat = max(0.0f, state.real() - beat_floor);
      new_bend += beat * weights[i] * cross(state, force_in);

      state *= state;
    }
  }
  bend = new_bend * beat_scale;
  rotate(phase, freq * bend_function(bend));

  // update force
  complex force_out(0,0);
  complex state_out(0,0);
  {
    complex state = phase;
    for (size_t i = 0; i < 4; ++i) {
      float beat = max(0.0f, state.real() - beat_floor);
      force_out += beat * weights[i] * state;
      state_out += weights[i] * state;

      state *= state;
    }
  }
  force = force_out * beat_scale;

  return state_out;
}

complex Shepard7::sample (complex force_in)
{
  const float p = bound_to(0.05f, 0.95f, octave / 6);
  const float q = 1 - p;
  const float weights[7] = {
    q*q*q*q*q*q * 1,
    q*q*q*q*q * 6 * p,
    q*q*q*q * 15 * p*p,
    q*q*q * 20 * p*p*p,
    q*q * 15 * p*p*p*p,
    q * 6 * p*p*p*p*p,
    1 * p*p*p*p*p*p
  };
  const float beat_floor = beat_fun.beat_floor;
  const float beat_scale = beat_fun.beat_scale;

  // update phase
  float new_bend = 0;
  {
    complex state = phase;
    for (size_t i = 0; i < 7; ++i) {
      float beat = max(0.0f, state.real() - beat_floor);
      new_bend += beat * weights[i] * cross(state, force_in);

      state *= state;
    }
  }
  bend = new_bend * beat_scale;
  rotate(phase, freq * bend_function(bend));

  // update force
  complex force_out(0,0);
  complex state_out(0,0);
  {
    complex state = phase;
    for (size_t i = 0; i < 7; ++i) {
      float beat = max(0.0f, state.real() - beat_floor);
      force_out += beat * weights[i] * state;
      state_out += weights[i] * state;

      state *= state;
    }
  }
  force = force_out * beat_scale;

  return state_out;
}

complex Geometric::sample (complex force)
{
  bend = dot(dbeat, force);
  rotate(phase, freq * bend_function(beat_fun.strength * bend));

  // slower
  //beat = beat_fun.value(phase);
  //dbeat = beat_fun.deriv(phase);

  // faster
  float p = beat_fun.p;
  complex z = phase;
  complex dz(-z.imag(), z.real());
  complex geom = (1 - p) / (1.0f - p * z);
  beat = geom * z;
  dbeat = sqr(geom) * dz;

  return phase;
}

complex Boltz::sample (complex force)
{
  bend = dbeat * force.real();
  float dphase = freq * bend_function(bend);
  rotate(phase, dphase);
  float b = beat_fun.value(phase.real());
  dbeat = (beat - b) / dphase;
  beat = b;

  return phase;
}

complex Phasor2::sample (complex mean_beat)
{
  bend = -dot(dbeat, mean_beat);

  rotate(phase, freq * bend_function(bend));
  complex dphase(-phase.imag(), phase.real());

  float b = beat_fun.value(phase.real());
  float db = beat_fun.deriv(phase.real(), phase.imag());
  beat = b * phase + complex(beat_fun.beat_shift,0);
  dbeat = db * phase + b * dphase;

  return phase;
}

//----( arnold tongue testing )----

template<class Oscillator>
float ArnoldTongues<Oscillator>::mean_bend (
    const typename Oscillator::BeatFun & beat_fun,
    float freq1,
    float freq2,
    size_t T)
{
  Oscillator o1(beat_fun);
  Oscillator o2(beat_fun);

  o1.mass = 1;
  o2.mass = 1;

  o1.freq = freq1;
  o2.freq = freq2;

  o1.phase = exp_2_pi_i(random_01());
  o2.phase = exp_2_pi_i(random_01());

  float bend = 0;
  for (size_t t = 0; t < T; ++t) {
    Poll poll;
    poll += o1.poll();
    poll += o2.poll();

    complex force = poll.mean();
    o1.sample(force);
    o2.sample(force);

    bend += o1.bend - o2.bend;
  }

  return bend / T;
}

template<class Oscillator>
float ArnoldTongues<Oscillator>::rms_bend (
    const typename Oscillator::BeatFun & beat_fun,
    float freq1,
    float freq2,
    float freq3,
    size_t T)
{
  Oscillator o1(beat_fun);
  Oscillator o2(beat_fun);
  Oscillator o3(beat_fun);

  o1.mass = 1;
  o2.mass = 1;
  o3.mass = 1;

  o1.freq = freq1;
  o2.freq = freq2;
  o3.freq = freq3;

  o1.phase = exp_2_pi_i(random_01());
  o2.phase = exp_2_pi_i(random_01());
  o3.phase = exp_2_pi_i(random_01());

  float bend = 0;
  for (size_t t = 0; t < T; ++t) {
    Poll poll;
    poll += o1.poll();
    poll += o2.poll();
    poll += o3.poll();

    complex force = poll.mean();
    o1.sample(force);
    o2.sample(force);
    o3.sample(force);

    bend += sqr(o1.bend) + sqr(o2.bend) + sqr(o3.bend);
  }

  return sqrt(bend / T);
}

template<class Oscillator>
void ArnoldTongues<Oscillator>::tongues (
    float acuity,
    float max_strength,
    float pitch_octaves,
    size_t num_periods)
{
  const float base_freq = 1 / 24.0f;
  const size_t num_steps = roundu(num_periods / base_freq);
  const size_t I = width();
  const size_t J = height();

  for (size_t i = 0; i < I; ++i) {
    float dfreq = powf(2.0f, (i + 0.5f) / I * pitch_octaves / 2);
    float freq1 = base_freq * dfreq;
    float freq2 = base_freq / dfreq;

    for (size_t j = 0; j < J; ++j) {
      float strength
        = (j + 0.5f) / J
        * max_strength
        * DEFAULT_SYNCHRONY_STRENGTH;

      typename Oscillator::BeatFun beat_fun(acuity, strength);

      bend[J * i + j] = mean_bend(beat_fun, freq1, freq2, num_steps);
    }

    cout << '.' << flush;
  }

  PRINT3(min(bend), rms(bend), max(bend));
}

template<class Oscillator>
void ArnoldTongues<Oscillator>::keys (
    float min_acuity,
    float max_acuity,
    float pitch_octaves,
    size_t num_periods)
{
  const float base_freq = 1 / 24.0f;
  const size_t num_steps = roundu(num_periods / base_freq);
  const size_t I = width();
  const size_t J = height();

  for (size_t i = 0; i < I; ++i) {
    float dfreq = powf(2.0f, (i + 0.5f) / I * pitch_octaves / 2);
    float freq1 = base_freq * dfreq;
    float freq2 = base_freq / dfreq;

    for (size_t j = 0; j < J; ++j) {
      float acuity = min_acuity + (max_acuity - min_acuity) * (j + 0.5f) / J;

      typename Oscillator::BeatFun beat_fun(acuity);

      bend[J * i + j] = mean_bend(beat_fun, freq1, freq2, num_steps) * acuity;
    }

    cout << '.' << flush;
  }

  PRINT3(min(bend), rms(bend), max(bend));
}

template<class Oscillator>
void ArnoldTongues<Oscillator>::islands (
    float acuity,
    float strength_scale,
    float pitch_octaves,
    size_t num_periods)
{
  const float base_freq = 1 / 24.0f;
  const size_t num_steps = roundu(num_periods / base_freq);
  const size_t I = width();
  const size_t J = height();

  typename Oscillator::BeatFun beat_fun(
      acuity,
      strength_scale * DEFAULT_SYNCHRONY_STRENGTH);

  float freq1 = base_freq;

  for (size_t i = 0; i < I; ++i) {
    float dfreq12 = powf(2.0f, (i + 0.5f) / I * pitch_octaves);
    float freq2 = base_freq * dfreq12;

    for (size_t j = 0; j < J; ++j) {
      float dfreq13 = powf(0.5f, (j + 0.5f) / J * pitch_octaves);
      float freq3 = base_freq * dfreq13;

      bend[J * i + j] = rms_bend(beat_fun, freq1, freq2, freq3, num_steps);
    }

    cout << '.' << flush;
  }

  PRINT3(min(bend), mean(bend), max(bend));
}

#define INSTANTIATE_TONGUES(O) \
  template float ArnoldTongues<O>::mean_bend ( \
      const O::BeatFun &, float, float, size_t); \
  template float ArnoldTongues<O>::rms_bend ( \
      const O::BeatFun &, float, float, float, size_t); \
  template void ArnoldTongues<O>::tongues (float, float, float, size_t); \
  template void ArnoldTongues<O>::keys (float, float, float, size_t); \
  template void ArnoldTongues<O>::islands (float, float, float, size_t);

INSTANTIATE_TONGUES(Phasor)
INSTANTIATE_TONGUES(Syncopator)
INSTANTIATE_TONGUES(Shepard4)
INSTANTIATE_TONGUES(Shepard7)
INSTANTIATE_TONGUES(Geometric)
INSTANTIATE_TONGUES(Boltz)
INSTANTIATE_TONGUES(Phasor2)

#undef INSTANTIATE_TONGUES

//----( abstract bank )-------------------------------------------------------

void Bank::init_transform (float * restrict frequency) const
{
  ASSERT_LT(0, freq0);
  ASSERT_LT(0, freq1);
  ASSERTW_LT(freq0, 1 / (2 * M_PI));
  ASSERTW_LT(freq1, 1 / (2 * M_PI));
  ASSERTW_LE(1, acuity);

  float density = size / num_tones();
  float bandwidth = fabs(log(freq1 / freq0)) / (size - 1);
  LOG("bank of " << size << " oscillators has acuity = " << acuity
      << ", density = " << density);
  LOG(" each oscillator has bandwidth "
      << (bandwidth / CRITICAL_BAND_WIDTH) << " cb = "
      << (bandwidth / logf(2)) << " octaves");
  ASSERTW_LE(1, density);
  ASSERTW_LE(bandwidth, CRITICAL_BAND_WIDTH);

  // evenly distribute background in pitch interval
  for (size_t i = 0, I = size; i < I; ++i) {
    double omega = 2 * M_PI * freq_at(i);

    frequency[i] = tan(omega);
  }
}

void Bank::init_transform (
    float * restrict trans_real,
    float * restrict trans_imag) const
{
  ASSERT_LT(0, freq0);
  ASSERT_LT(0, freq1);
  ASSERTW_LT(freq0, 1 / (2 * M_PI));
  ASSERTW_LT(freq1, 1 / (2 * M_PI));
  ASSERTW_LE(1, acuity);

  float density = size / num_tones();
  float bandwidth = fabs(log(freq1 / freq0)) / (size - 1);
  LOG("bank of " << size << " oscillators has acuity = " << acuity
      << ", density = " << density);
  LOG(" each oscillator has bandwidth "
      << (bandwidth / CRITICAL_BAND_WIDTH) << " cb = "
      << (bandwidth / logf(2)) << " octaves");
  ASSERTW_LE(1, density);
  ASSERTW_LE(bandwidth, CRITICAL_BAND_WIDTH);

  // evenly distribute background in pitch interval
  for (size_t i = 0, I = size; i < I; ++i) {
    double omega = 2 * M_PI * freq_at(i);

    trans_real[i] = cos(omega);
    trans_imag[i] = sin(omega);
  }
}

/** Exponential fourier windows.

  Let w be a rotational frequency difference, d be a decay rate,
  and E(w,d) be the energy felt at oscillator w by a signal at 0.
  Using a half-peak constraint

    E(w,d) = E(0,d) / 2

  we solve for d in terms of w.

  Case: order = 1

    E(w,d) = | int t:0,infty. exp((i w - d) t) |^2
           = | [exp((i w - d) t) / (i w - d)]_0^infty |^2
           = | 1 / (i w - d) |^2
           = 1 / (w^2 + d^2)

    whence d = w

  Case: order = 2

    E(w,d) = | int T:0,infty. exp((iw-d) t) int t:0,infty. exp((iw-d) t) |^2
           = ...
           = 1 / (w^2 + d^2)^2

    whence d = w (sqrt(2) - 1)

  Case: order = N
    ...
    d = w (2^(1/N) - 1)
*/

void Bank::init_decay (
    float * restrict decay,
    size_t order) const
{
  ASSERT_LT(0, size);
  ASSERT_DIVIDES(4, size);
  ASSERT_LT(0, freq0);
  ASSERT_LT(0, freq1);

  float nyquist_freq = 0.5f;
  PRINT(max_freq() / nyquist_freq);
  ASSERTW_LE(max_freq(), nyquist_freq);

  float damp_factor = pow(2, 1.0 / order) - 1;
  float dpitch = log(freq1 / freq0) / size;

  LOG("Bank(" << size << ", " << freq0 << ", " << freq1
      << ") has " << (log(2) / dpitch) << " steps / octave");

  if (acuity > 0.5) dpitch = log(2) / acuity;

  for (size_t i = 0, I = size; i < I; ++i) {
    double freq = 2 * M_PI * freq_at(i);
    double dfreq = fabs(dpitch * freq);

    decay[i] = exp(-damp_factor * dfreq);
  }
}

void Bank::init_decay_transform (
    float * restrict trans_real,
    float * restrict trans_imag,
    float * restrict rescale,
    size_t order,
    float min_timescale) const
{
  ASSERT_LT(0, size);
  ASSERT_DIVIDES(4, size);
  ASSERT_LT(0, freq0);
  ASSERT_LT(0, freq1);

  float nyquist_freq = 0.5f;
  PRINT(max_freq() / nyquist_freq);
  ASSERTW_LE(max_freq(), nyquist_freq);

  float damp_factor = pow(2, 1.0 / order) - 1;
  float dpitch = log(freq1 / freq0) / size;

  LOG("Bank(" << size << ", " << freq0 << ", " << freq1
      << ") has " << (log(2) / dpitch) << " steps / octave");

  if (acuity > 0.5) dpitch = log(2) / acuity;

  for (size_t i = 0, I = size; i < I; ++i) {
    double freq = 2 * M_PI * freq_at(i);
    double dfreq = 1 / (1 / fabs(dpitch * freq) + min_timescale);

    std::complex<double> omega(-damp_factor * dfreq, freq);
    std::complex<double> trans = exp(omega);
    trans_real[i] = trans.real();
    trans_imag[i] = trans.imag();

    rescale[i] = pow(dfreq, order); // = 1 / E(w,0)
  }
}

//#define BOUNDED_RETUNE

void Bank::retune (
    const float * restrict freq_bend,
    const float * restrict mass_in,
    float * restrict mass_out,
    float rate) const
{
  ASSERTW_LE(rate, 1);

  const size_t I = size;
  const float bend_scale = size / log(freq1 / freq0) * rate;

  for (size_t i = 0; i < I; ++i) {
    mass_out[i] = TOL;
  }

  for (size_t i = 0; i < I; ++i) {
    float pitch_bend = logf(1 + freq_bend[i]);

#ifdef BOUNDED_RETUNE
    float di = bound_to(-1.0f, 1.0f, bend_scale * pitch_bend);
#else // BOUNDED_RETUNE
    float di = bend_scale * pitch_bend;
#endif // BOUNDED_RETUNE

    LinearInterpolate(i + di, I).iadd(mass_out, mass_in[i]);
  }
}

void Bank::retune (
    const float * restrict freq_bend,
    const float * restrict mass_in,
    float * restrict mass_out,
    size_t vector_size,
    float rate) const
{
  ASSERTW_LE(rate, 1);

  const size_t I = size;
  const size_t J = vector_size;
  const float bend_scale = size / log(freq1 / freq0) * rate;

  for (size_t ij = 0; ij < I*J; ++ij) {
    mass_out[ij] = TOL;
  }

  for (size_t i = 0; i < I; ++i) {
    float pitch_bend = logf(1 + freq_bend[i]);

#ifdef BOUNDED_RETUNE
    float di = bound_to(-1.0f, 1.0f, bend_scale * pitch_bend);
#else // BOUNDED_RETUNE
    float di = bend_scale * pitch_bend;
#endif // BOUNDED_RETUNE

    LinearInterpolate lin(i + di, I);

    for (size_t j = 0; j < J; ++j) {
      float m = mass_in[J * i + j];

      mass_out[J * lin.i0 + j] += m * lin.w0;
      mass_out[J * lin.i1 + j] += m * lin.w1;
    }
  }
}

void Bank::retune (
    const Vector<float> & freq_bend,
    const Vector<float> & mass_in,
    Vector<float> & mass_out,
    float rate) const
{
  ASSERT_SIZE(freq_bend, size);
  ASSERT_EQ(mass_in.size, mass_out.size);
  ASSERT_DIVIDES(size, mass_in.size);

  size_t vector_size = mass_in.size / size;

  if (vector_size == 1) {
    Bank::retune(freq_bend.data, mass_in.data, mass_out.data, rate);
  } else {
    Bank::retune(
        freq_bend.data,
        mass_in.data,
        mass_out.data,
        vector_size,
        rate);
  }
}

//----( simple bank )---------------------------------------------------------

SimpleBank::SimpleBank (Bank param)

  : Bank(param),

    m_frequency(size),
    m_phase_real(size),
    m_phase_imag(size)
{
  ASSERT_DIVIDES(4, size);

  Bank::init_transform(m_frequency);

  random_phase(m_phase_real, m_phase_imag);
}

SimpleBank::~SimpleBank ()
{
  if (debug) {
    PRINT2(min(m_frequency), max(m_frequency));
    PRINT2(rms(m_phase_real), rms(m_phase_imag));
  }
}

void SimpleBank::sample_accum (
    const Vector<float> & amplitude0_in,
    const Vector<float> & damplitude_in,
    Vector<complex> & accum_out)
{
  const size_t I = size;
  const size_t T = accum_out.size;

  const float * const restrict frequency = m_frequency;
  const float * const restrict amplitude0 = amplitude0_in;
  const float * const restrict damplitude = damplitude_in;

  float * const restrict phase_real = m_phase_real;
  float * const restrict phase_imag = m_phase_imag;
  complex * const restrict accum = accum_out;

  const float tol2 = sqr(TOL);

  for (size_t t = 0; t < T; ++t) {

    // compute interpolation
    float sum_ax = 0;
    float sum_ay = 0;
    const float dt = (0.5f + t) / T;

    for (size_t i = 0; i < I; ++i) {
      float x = phase_real[i];
      float y = phase_imag[i];

      // update state
      {
        float freq = frequency[i];
        float dx = -y * freq;
        float dy =  x * freq;

        x += dx;
        y += dy;
      }

      // normalize
      {
        float r = sqrt(sqr(x) + sqr(y) + tol2);

        phase_real[i] = (x /= r);
        phase_imag[i] = (y /= r);
      }

      // accumulate
      {
        float a = amplitude0[i] + damplitude[i] * dt;

        sum_ax += a * x;
        sum_ay += a * y;
      }
    }

    accum[t] += complex(sum_ax, sum_ay);
  }
}

//----( fourier bank )--------------------------------------------------------

FourierBank::FourierBank (Bank param, float min_timescale)
  : Bank(param),

    m_rescale(size),
    m_trans_real(size),
    m_trans_imag(size),
    m_phase_real(size),
    m_phase_imag(size)
{
  Bank::init_decay_transform(
      m_trans_real,
      m_trans_imag,
      m_rescale,
      1,
      min_timescale);

  m_phase_real.zero();
  m_phase_imag.zero();
}

FourierBank::~FourierBank ()
{
  if (debug) {
    PRINT2(min(m_rescale), max(m_rescale));
    PRINT2(min(m_trans_real), max(m_trans_real));
    PRINT2(min(m_trans_imag), max(m_trans_imag));
    PRINT2(rms(m_phase_real), rms(m_phase_imag));
  }
}

void FourierBank::sample (
    const Vector<float> & time_in,
    Vector<float> & freq_out)
{
  ASSERT_SIZE(freq_out, size);

  const size_t I = size;
  const size_t T = time_in.size;

  const float * const restrict rescale = m_rescale;
  const float * const restrict trans_real = m_trans_real;
  const float * const restrict trans_imag = m_trans_imag;
  const float * const restrict time = time_in;

  float * const restrict phase_real = m_phase_real;
  float * const restrict phase_imag = m_phase_imag;
  float * const restrict freq = freq_out;

  freq_out.zero();

  for (size_t t = 0; t < T; ++t) {

    const float x = time[t];

    for (size_t i = 0; i < I; ++i) {

      float x0 = phase_real[i];
      float y0 = phase_imag[i];

      float f = trans_real[i];
      float g = trans_imag[i];

      float x1 = phase_real[i] = f * x0 - g * y0 + x;
      float y1 = phase_imag[i] = f * y0 + g * x0;

      freq[i] += sqr(x1) + sqr(y1);
    }
  }

  for (size_t i = 0; i < I; ++i) {
    freq[i] *= sqr(rescale[i]) / T;
  }
}

void FourierBank::sample (
    const Vector<complex> & time_in,
    Vector<float> & freq_out)
{
  ASSERT_SIZE(freq_out, size);

  const size_t I = size;
  const size_t T = time_in.size;

  const float * const restrict rescale = m_rescale;
  const float * const restrict trans_real = m_trans_real;
  const float * const restrict trans_imag = m_trans_imag;
  const complex * const restrict time = time_in;

  float * const restrict phase_real = m_phase_real;
  float * const restrict phase_imag = m_phase_imag;
  float * const restrict freq = freq_out;

  freq_out.zero();

  for (size_t t = 0; t < T; ++t) {

    const complex z = time[t];
    const float x = z.real();
    const float y = z.imag();

    for (size_t i = 0; i < I; ++i) {

      float x0 = phase_real[i];
      float y0 = phase_imag[i];

      float f = trans_real[i];
      float g = trans_imag[i];

      float x1 = phase_real[i] = f * x0 - g * y0 + x;
      float y1 = phase_imag[i] = f * y0 + g * x0 + y;

      freq[i] += sqr(x1) + sqr(y1);
    }
  }

  for (size_t i = 0; i < I; ++i) {
    freq[i] *= sqr(rescale[i]) / T;
  }
}

void FourierBank::resonate (Vector<complex> & time_accum)
{
  const size_t I = size;
  const size_t T = time_accum.size;

  const float * const restrict rescale = m_rescale;
  const float * const restrict trans_real = m_trans_real;
  const float * const restrict trans_imag = m_trans_imag;

  float * const restrict phase_real = m_phase_real;
  float * const restrict phase_imag = m_phase_imag;
  complex * const restrict time = time_accum;

  for (size_t t = 0; t < T; ++t) {

    const complex z = time[t];
    const float x_in = safe(z.real());
    const float y_in = safe(z.imag());

    float x_out = 0;
    float y_out = 0;

    for (size_t i = 0; i < I; ++i) {

      float x0 = phase_real[i];
      float y0 = phase_imag[i];

      float f = trans_real[i];
      float g = trans_imag[i];

      float x1 = phase_real[i] = f * x0 - g * y0 + x_in;
      float y1 = phase_imag[i] = f * y0 + g * x0 + y_in;

      float r = rescale[i];

      x_out += r * x1;
      y_out += r * y1;
    }

    time[t] += complex(x_out / I, y_out / I);
  }
}

//----( fourier bank 2 )------------------------------------------------------

FourierBank2::FourierBank2 (Bank param, float min_timescale)
  : Bank(param),

    m_rescale(size),
    m_trans_real(size),
    m_trans_imag(size),
    m_pos_real(size),
    m_pos_imag(size),
    m_vel_real(size),
    m_vel_imag(size)
{
  PRINT3(size, freq0, freq1);

  Bank::init_decay_transform(
      m_trans_real,
      m_trans_imag,
      m_rescale,
      2,
      min_timescale);

  m_pos_real.zero();
  m_pos_imag.zero();
  m_vel_real.zero();
  m_vel_imag.zero();
}

FourierBank2::~FourierBank2 ()
{
  if (debug) {
    PRINT2(min(m_rescale), max(m_rescale));
    PRINT2(min(m_trans_real), max(m_trans_real));
    PRINT2(min(m_trans_imag), max(m_trans_imag));
    PRINT2(rms(m_pos_real), rms(m_pos_imag));
    PRINT2(rms(m_vel_real), rms(m_vel_imag));
  }
}

void FourierBank2::sample (
    const Vector<float> & time_in,
    Vector<float> & freq_out)
{
  ASSERT_SIZE(freq_out, size);

  const size_t I = size;
  const size_t T = time_in.size;

  const float * const restrict rescale = m_rescale;
  const float * const restrict trans_real = m_trans_real;
  const float * const restrict trans_imag = m_trans_imag;
  const float * const restrict time = time_in;

  float * const restrict pos_real = m_pos_real;
  float * const restrict pos_imag = m_pos_imag;
  float * const restrict vel_real = m_vel_real;
  float * const restrict vel_imag = m_vel_imag;

  for (size_t t = 0; t < T; ++t) {

    const float x = time[t];

    for (size_t i = 0; i < I; ++i) {

      float x0 = pos_real[i];
      float y0 = pos_imag[i];

      float f = trans_real[i];
      float g = trans_imag[i];

      float x1 = pos_real[i] = f * x0 - g * y0 + x;
      float y1 = pos_imag[i] = f * y0 + g * x0;

      float dx0 = vel_real[i];
      float dy0 = vel_imag[i];

      vel_real[i] = f * dx0 - g * dy0 + x1;
      vel_imag[i] = f * dy0 + g * dx0 + y1;
    }
  }

  float * const restrict freq = freq_out;

  for (size_t i = 0; i < I; ++i) {
    freq[i] = sqr(rescale[i]) * (sqr(vel_real[i]) + sqr(vel_imag[i]));
  }
}

void FourierBank2::sample (
    const Vector<complex> & time_in,
    Vector<float> & freq_out)
{
  ASSERT_SIZE(freq_out, size);

  const size_t I = size;
  const size_t T = time_in.size;

  const float * const restrict rescale = m_rescale;
  const float * const restrict trans_real = m_trans_real;
  const float * const restrict trans_imag = m_trans_imag;
  const complex * const restrict time = time_in;

  float * const restrict pos_real = m_pos_real;
  float * const restrict pos_imag = m_pos_imag;
  float * const restrict vel_real = m_vel_real;
  float * const restrict vel_imag = m_vel_imag;

  for (size_t t = 0; t < T; ++t) {

    const complex z = time[t];
    const float x = z.real();
    const float y = z.imag();

    for (size_t i = 0; i < I; ++i) {

      float x0 = pos_real[i];
      float y0 = pos_imag[i];

      float f = trans_real[i];
      float g = trans_imag[i];

      float x1 = pos_real[i] = f * x0 - g * y0 + x;
      float y1 = pos_imag[i] = f * y0 + g * x0 + y;

      float dx0 = vel_real[i];
      float dy0 = vel_imag[i];

      vel_real[i] = f * dx0 - g * dy0 + x1;
      vel_imag[i] = f * dy0 + g * dx0 + y1;
    }
  }

  float * const restrict freq = freq_out;

  for (size_t i = 0; i < I; ++i) {
    freq[i] = sqr(rescale[i]) * (sqr(vel_real[i]) + sqr(vel_imag[i]));
  }
}

//----( phasor bank )---------------------------------------------------------

PhasorBank::PhasorBank (Bank param)

  : Bank(param),
    m_beat_fun(acuity),

    m_data(size * 7),

    m_mass        (size, m_data + size * 0),
    m_frequency   (size, m_data + size * 1),
    m_phase_real  (size, m_data + size * 2),
    m_phase_imag  (size, m_data + size * 3),
    m_beat        (size, m_data + size * 4),
    m_meso_bend   (size, m_data + size * 5),
    m_slow_bend   (size, m_data + size * 6),

    m_temp(m_meso_bend)
{
  ASSERT_DIVIDES(4, size);
  ASSERT_LT(0, param.acuity);
  ASSERT_LT(0, param.strength);

  Bank::init_transform(m_frequency);

  m_mass.set(TOL);

  random_phase(m_phase_real, m_phase_imag);

  // initialize beat function
  const float beat_floor = m_beat_fun.beat_floor;
  for (size_t i = 0, I = size; i < I; ++i) {
    float x = m_phase_real[i];
    m_beat[i] = max(0.0f, x - beat_floor);
  }

  m_slow_bend.zero();
}

PhasorBank::~PhasorBank ()
{
  if (debug) {
    PRINT3(min(m_mass), max(m_mass), mean(m_mass));
    PRINT2(min(m_frequency), max(m_frequency));
    PRINT2(rms(m_phase_real), rms(m_phase_imag));
    PRINT2(min(m_slow_bend), max(m_slow_bend));
    PRINT2(rms(m_slow_bend), mean(m_slow_bend));
  }
}

void PhasorBank::sample_accum (
    const Vector<float> & amplitude0_in,
    const Vector<float> & damplitude_in,
    Vector<complex> & accum_out)
{
  const size_t I = size;
  const size_t T = accum_out.size;

  const float * const restrict frequency = m_frequency;
  const float * const restrict amplitude0 = amplitude0_in;
  const float * const restrict damplitude = damplitude_in;

  float * const restrict mass = m_mass;
  float * const restrict phase_real = m_phase_real;
  float * const restrict phase_imag = m_phase_imag;
  float * const restrict beat = m_beat;
  float * const restrict meso_bend = m_meso_bend;
  float * const restrict slow_bend = m_slow_bend;
  complex * const restrict accum = accum_out;

  const float tol2 = sqr(TOL);
  const float total_mass = sum(m_mass);
  const float beat_floor = m_beat_fun.beat_floor;
  const float beat_scale2 = sqr(m_beat_fun.beat_scale) / total_mass;
  // beat_scale2 = strength * acuity / sqr(1 - beat_floor) / total_mass;

  // initialize beat
  float sum_mbx = 0;
  float sum_mby = 0;

  for (size_t i = 0; i < I; ++i) {
    float x = phase_real[i];
    float y = phase_imag[i];
    float m = mass[i];
    float b = beat[i];

    sum_mbx += b * m * x;
    sum_mby += b * m * y;

    meso_bend[i] = 0;
  }

  for (size_t t = 0; t < T; ++t) {

    // compute moments
    const float mean_bx = beat_scale2 * sum_mbx;
    const float mean_by = beat_scale2 * sum_mby;
    sum_mbx = 0;
    sum_mby = 0;

    // compute interpolation
    float sum_ax = 0;
    float sum_ay = 0;
    const float dt = (0.5f + t) / T;

    for (size_t i = 0; i < I; ++i) {
      float x = phase_real[i];
      float y = phase_imag[i];
      float dphase;

      // update phase
      {
        float mean_force = mean_by * x - mean_bx * y;
        float bend = mean_force * beat[i];
        meso_bend[i] += bend;

        dphase = frequency[i] * bend_function(bend);
        float dx = -y * dphase;
        float dy =  x * dphase;

        x += dx;
        y += dy;
      }

      // normalize
      {
        float r = sqrt(sqr(x) + sqr(y) + tol2);

        phase_real[i] = (x /= r);
        phase_imag[i] = (y /= r);
      }

      // update beat function, rescale, and accumulate
      {
        float m = mass[i];
        float b = beat[i] = max(0.0f, x - beat_floor);

        sum_mbx += b * m * x;
        sum_mby += b * m * y;
      }

      // accumulate
      {
        float a = amplitude0[i] + damplitude[i] * dt;

        sum_ax += a * x;
        sum_ay += a * y;
      }
    }

    accum[t] += complex(sum_ax, sum_ay);
  }

  // update pitch bend
  const float scale = 1.0f / T;
  const float slow_rate = 1.0f / PITCH_BEND_TIMESCALE_FRAMES;
  for (size_t i = 0; i < I; ++i) {
    slow_bend[i] += slow_rate * (scale * meso_bend[i] - slow_bend[i]);
  }
}

void PhasorBank::sample_accum (
    const Vector<float> & dmass_in,
    Vector<complex> & accum_out)
{
  const size_t I = size;
  const size_t T = accum_out.size;

  const float * const restrict frequency = m_frequency;
  const float * const restrict dmass = dmass_in;

  float * const restrict mass = m_mass;
  float * const restrict phase_real = m_phase_real;
  float * const restrict phase_imag = m_phase_imag;
  float * const restrict beat = m_beat;
  float * const restrict meso_bend = m_meso_bend;
  float * const restrict slow_bend = m_slow_bend;
  complex * const restrict accum = accum_out;

  const float tol2 = sqr(TOL);
  const float total_mass = sum(m_mass) + sum(dmass_in) / 2;
  const float beat_floor = m_beat_fun.beat_floor;
  const float beat_scale2 = sqr(m_beat_fun.beat_scale) / total_mass;

  // initialize beat function
  float sum_mbx = 0;
  float sum_mby = 0;

  // The mass used in the beat function is always a little time-skewed,
  //   but it is much more slowly varying than phase.
  for (size_t i = 0; i < I; ++i) {
    float x = phase_real[i];
    float y = phase_imag[i];
    float m = mass[i];
    float b = beat[i];

    sum_mbx += b * m * x;
    sum_mby += b * m * y;

    meso_bend[i] = 0;
  }

  for (size_t t = 0; t < T; ++t) {

    // compute moments
    const float mean_bx = beat_scale2 * sum_mbx;
    const float mean_by = beat_scale2 * sum_mby;
    sum_mbx = 0;
    sum_mby = 0;

    // compute interpolation
    const float dt = (0.5f + t) / T;
    float sum_mx = 0;
    float sum_my = 0;

    for (size_t i = 0; i < I; ++i) {
      float x = phase_real[i];
      float y = phase_imag[i];
      float dphase;

      // update phase
      {
        float mean_force = mean_by * x - mean_bx * y;
        float bend = mean_force * beat[i];
        meso_bend[i] += bend;

        dphase = frequency[i] * bend_function(bend);
        float dx = -y * dphase;
        float dy =  x * dphase;

        x += dx;
        y += dy;
      }

      // normalize
      {
        float r = sqrt(sqr(x) + sqr(y) + tol2);

        phase_real[i] = (x /= r);
        phase_imag[i] = (y /= r);
      }

      // update beat function, rescale, and accumulate
      {
        float m = mass[i] + dmass[i] * dt;
        float b = beat[i] = max(0.0f, x - beat_floor);

        float mx = m * x;
        float my = m * y;

        sum_mx += mx;
        sum_my += my;

        sum_mbx += b * mx;
        sum_mby += b * my;
      }
    }

    accum[t] += complex(sum_mx, sum_my);
  }

  // update mass & pitch bend
  const float scale = 1.0f / T;
  const float slow_rate = 1.0f / PITCH_BEND_TIMESCALE_FRAMES;
  for (size_t i = 0; i < I; ++i) {
    slow_bend[i] += slow_rate * (scale * meso_bend[i] - slow_bend[i]);
    mass[i] = max(TOL, mass[i] + dmass[i]);
  }
}

void PhasorBank::debias_slow_bend ()
{
  float blur_radius = size / num_octaves();
  Image::exp_blur_1d_zero(size, blur_radius, m_slow_bend, m_temp);
  m_slow_bend -= m_temp;
}

void PhasorBank::retune_particles (float rate)
{
  const size_t I = size;

  float * const restrict freq = m_frequency;
  float * const restrict bend = m_slow_bend;

  float max_bend = 0.2f;
  for (size_t i = 0; i < I; ++i) {
    float df = bound_to(-max_bend, max_bend, rate * bend[i]);
    freq[i] *= 1 + df;
    bend[i] -= df;
  }
}

//----( syncopator bank )-----------------------------------------------------

SyncopatorBank::SyncopatorBank (Bank param)

  : Bank(param),
    m_beat_fun(acuity),

    m_data(size * 8),

    m_mass        (size, m_data + size * 0),
    m_frequency   (size, m_data + size * 1),
    m_offset_real (size, m_data + size * 2),
    m_offset_imag (size, m_data + size * 3),
    m_phase_real  (size, m_data + size * 4),
    m_phase_imag  (size, m_data + size * 5),
    m_beat        (size, m_data + size * 6),
    m_temp        (size, m_data + size * 7)
{
  ASSERT_DIVIDES(4, size);
  ASSERT_LT(0, param.acuity);
  ASSERT_LT(0, param.strength);

  m_mass.set(TOL);

  Bank::init_transform(m_frequency);

  m_offset_real.set(1.0f);
  m_offset_imag.set(0.0f);
  m_phase_real.set(01.0f);
  m_phase_imag.set(00.0f);
}

SyncopatorBank::~SyncopatorBank ()
{
  if (debug) {
    PRINT3(min(m_mass), max(m_mass), mean(m_mass));
    PRINT2(min(m_frequency), max(m_frequency));
    PRINT(norm_squared(m_offset_real) + norm_squared(m_offset_imag));
    PRINT(norm_squared(m_phase_real) + norm_squared(m_phase_imag));
  }
}

//----( data manipulation )----

void SyncopatorBank::set (const BoundedMap<Id, State> & states)
{
  ASSERT_LE(states.size, size);
  ASSERT_LE(size, states.capacity);

  Image::transpose(size, state_dim, states.values[0], m_data);

  for (size_t i = states.size; i < size; ++i) {
    m_mass[i] = 0;
  }
}

void SyncopatorBank::get (BoundedMap<Id, State> & states) const
{
  ASSERT_LE(states.size, size);
  ASSERT_LE(size, states.capacity);

  Image::transpose(state_dim, size, m_data, states.values[0]);
}

//----( synchronization primitives )----

void SyncopatorBank::advance (float dt)
{
  const size_t I = size;

  const float * const restrict mass = m_mass;
  const float * const restrict frequency = m_frequency;
  const float * const restrict offset_real = m_offset_real;
  const float * const restrict offset_imag = m_offset_imag;

  float * const restrict phase_real = m_phase_real;
  float * const restrict phase_imag = m_phase_imag;
  float * const restrict beat = m_beat;

  const float tol2 = (TOL);
  const float total_mass = sum(m_mass);
  const float beat_floor = m_beat_fun.beat_floor;
  const float beat_scale2 = sqr(m_beat_fun.beat_scale) / total_mass;

  // update beat function
  float sum_mbx = 0;
  float sum_mby = 0;

  for (size_t i = 0; i < size; ++i) {

    float u = phase_real[i];
    float v = phase_imag[i];

    float x = offset_real[i] * u - offset_imag[i] * v;
    float y = offset_real[i] * v + offset_imag[i] * u;

    float m = mass[i];
    float b = beat[i] = max(0.0f, u - beat_floor);

    sum_mbx += b * m * x;
    sum_mby += b * m * y;
  }

  const float force_x = beat_scale2 * sum_mbx;
  const float force_y = beat_scale2 * sum_mby;

  for (size_t i = 0; i < I; ++i) {

    float force_u = offset_real[i] * force_x + offset_imag[i] * force_y;
    float force_v = offset_real[i] * force_y - offset_imag[i] * force_x;

    float u = phase_real[i];
    float v = phase_imag[i];

    // update state
    {
      float force = force_v * u - force_u * v;
      float bend = beat[i] * force;
      float bent = frequency[i] * bend_function(bend);
      float du = -v * bent;
      float dv =  u * bent;

      u += du;
      v += dv;
    }

    // normalize
    {
      float r = sqrt(sqr(u) + sqr(v) + tol2);

      phase_real[i] = u / r;
      phase_imag[i] = v / r;
    }
  }
}

//----( geometric bank )------------------------------------------------------

GeomBank::GeomBank (Bank param)
  : Bank(param),

    m_data(size * 12),

    m_mass        (size, m_data + size * 0),
    m_frequency   (size, m_data + size * 1),
    m_phase_real  (size, m_data + size * 2),
    m_phase_imag  (size, m_data + size * 3),
    m_beat_real   (size, m_data + size * 4),
    m_beat_imag   (size, m_data + size * 5),
    m_dbeat_real  (size, m_data + size * 6),
    m_dbeat_imag  (size, m_data + size * 7),
    m_meso_energy (size, m_data + size * 8),
    m_slow_energy (size, m_data + size * 9),
    m_meso_bend   (size, m_data + size * 10),
    m_slow_bend   (size, m_data + size * 11),

    m_temp(m_meso_bend),

    m_force_gain(4.0f / min_freq()),
    m_energy_gain(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_SAMPLE_RATE),

    m_force_snapshot(0,0)
{
  ASSERT_DIVIDES(4, size);
  ASSERT_LT(0, param.acuity);
  ASSERT_LT(0, param.strength);

  Bank::init_transform(m_frequency);

  m_mass.set(TOL);

  random_phase(m_phase_real, m_phase_imag);

  // initialize beat function
  float p = 1 - 1 / acuity;
  for (size_t i = 0, I = size; i < I; ++i) {
    complex z(m_phase_real[i], m_phase_imag[i]);

    complex geom = (1.0f - p) / (1.0f - p * z);
    complex b = z * geom;
    m_beat_real[i] = b.real();
    m_beat_imag[i] = b.imag();

    complex db_over_i = (1.0f - p + p * z) * sqr(geom);
    m_dbeat_real[i] = -db_over_i.imag();
    m_dbeat_imag[i] = db_over_i.real();
  }

  m_slow_bend.zero();
  m_slow_energy.zero();

  // check for timescale separation
  float safe_num_periods = 3.0f;
  ASSERT_LE(safe_num_periods * min_freq(), m_force_gain.timescale());
}

GeomBank::~GeomBank ()
{
  if (debug) {
    PRINT3(min(m_mass), max(m_mass), mean(m_mass));
    PRINT2(min(m_frequency), max(m_frequency));
    PRINT2(rms(m_phase_real), rms(m_phase_imag));
    PRINT2(min(m_slow_bend), max(m_slow_bend));
    PRINT2(rms(m_slow_bend), mean(m_slow_bend));
    PRINT3(m_force_gain, m_energy_gain.mean(), m_energy_gain.variance());
    PRINT(m_force_snapshot);
  }
}

void GeomBank::sample_accum (
    const Vector<float> & dmass_in,
    Vector<complex> & accum_out)
{
  const size_t I = size;
  const size_t T = accum_out.size;

  const float * const restrict frequency = m_frequency;
  const float * const restrict dmass = dmass_in;

  float * const restrict mass = m_mass;
  float * const restrict phase_real = m_phase_real;
  float * const restrict phase_imag = m_phase_imag;
  float * const restrict beat_real = m_beat_real;
  float * const restrict beat_imag = m_beat_imag;
  float * const restrict dbeat_real = m_dbeat_real;
  float * const restrict dbeat_imag = m_dbeat_imag;
  float * const restrict meso_energy = m_meso_energy;
  float * const restrict slow_energy = m_slow_energy;
  float * const restrict meso_bend = m_meso_bend;
  float * const restrict slow_bend = m_slow_bend;
  complex * const restrict accum = accum_out;

  const float p = 1 - 1 / acuity;

  // initialize beat function
  // The mass used in the beat function is always a little time-skewed,
  //   but it is much more slowly varying than phase.
  complex sum_mb = 0.0f;
  for (size_t i = 0; i < I; ++i) {
    float m = mass[i];
    complex b(beat_real[i], beat_imag[i]);

    sum_mb += m * b;

    meso_energy[i] = 0;
    meso_bend[i] = 0;
  }

  for (size_t t = 0; t < T; ++t) {

    // compute moments
    float abs_sum_mb = abs(sum_mb);
    const complex mean_force = m_energy_gain.update(abs_sum_mb) * sum_mb;
    const complex mean_bend = m_force_gain.update(abs_sum_mb) * sum_mb;
    sum_mb = 0.0f;

    // compute interpolation
    const float dt = (0.5f + t) / T;

    for (size_t i = 0; i < I; ++i) {
      complex z(phase_real[i], phase_imag[i]);

      // update phase
      {
        complex b(beat_real[i], beat_imag[i]);
        float energy = -dot(b, mean_force);
        meso_energy[i] += energy;

        complex db(dbeat_real[i], dbeat_imag[i]);
        float bend = dot(db, mean_bend);
        meso_bend[i] += bend;

        z += z * complex(0, frequency[i] * (1 + bend));
        z /= abs(z);
        phase_real[i] = z.real();
        phase_imag[i] = z.imag();
      }

      // update beat function, rescale, and accumulate
      {
        complex geom = (1.0f - p) / (1.0f - p * z);
        complex b = z * geom;
        beat_real[i] = b.real();
        beat_imag[i] = b.imag();

        complex db_over_i = (1.0f - p + p * z) * sqr(geom);
        dbeat_real[i] = -db_over_i.imag();
        dbeat_imag[i] = db_over_i.real();

        float m = mass[i] + dmass[i] * dt;
        sum_mb += m * b;
      }
    }

    accum[t] += sum_mb;
  }

  m_force_snapshot = sum_mb;

  // update mass & pitch bend
  const float scale = 1.0f / T;
  const float slow_rate = 1.0f / PITCH_BEND_TIMESCALE_FRAMES;
  for (size_t i = 0; i < I; ++i) {
    mass[i] = max(TOL, mass[i] + dmass[i]);
    slow_energy[i] += slow_rate * (scale * meso_energy[i] - slow_energy[i]);
    slow_bend[i] += slow_rate * (scale * meso_bend[i] - slow_bend[i]);
  }
}

void GeomBank::retune_zeromean (
    const Vector<float> & mass_in,
    Vector<float> & reass_out,
    float rate) const
{
  float mean_bend = mean_wrt(m_slow_bend, m_mass);

  add(-mean_bend, m_slow_bend, m_temp);

  Bank::retune(m_temp, mass_in, reass_out, rate);
}

//----( geometric set )-------------------------------------------------------

GeomSet::GeomSet (
    size_t size,
    float min_freq,
    float max_freq,
    float min_duration)

  : m_size(size),
    m_min_mass(1e-8f),
    m_min_omega(2 * M_PI * min_freq),
    m_max_omega(2 * M_PI * max_freq),
    m_min_duration(max(min_freq / max_freq, min_duration)),
    m_timescale(4.0f / min_freq),

    m_data(size * 12),

    m_mass        (size, m_data + size * 0),
    m_duration    (size, m_data + size * 1),
    m_omega0      (size, m_data + size * 2),
    m_omega       (size, m_data + size * 3),
    m_phase_real  (size, m_data + size * 4),
    m_phase_imag  (size, m_data + size * 5),
    m_beat_real   (size, m_data + size * 6),
    m_beat_imag   (size, m_data + size * 7),
    m_dbeat_real  (size, m_data + size * 8),
    m_dbeat_imag  (size, m_data + size * 9),

    m_norm_beat(0),

    m_force_gain(m_timescale)
{
  ASSERT_DIVIDES(4, m_size);
  ASSERT_LT(min_freq, max_freq);
  ASSERT_LT(max_freq, 0.1f);
  ASSERT_LT(0, m_min_duration);
  ASSERT_LT(m_min_duration, 1);

  /*
                ^
                |
        freq    |\
  log(--------) |xx\  <-- area in param space
      min_freq  |xxxx\
                |xxxxx|\
                |xxxxx|  \
                +-----+---+->                1                   max_freq
                     a0  a1    a0 = log(------------),  a1 = log(--------) 
                                        min_duration             min_freq
                log(acuity)
  */

  float a0 = log(1 / m_min_duration);
  float a1 = log(max_freq / min_freq);
  float area_in_param_space = sqr(a1) / 2 - sqr(a1 - a0) / 2;
  float density = m_size / area_in_param_space;
  LOG("GeomSet of size " << m_size << " has density " << density);

  for (size_t i = 0; i < m_size; ++i) init_random(i);
  //TODO do Voronoi relaxation via Loyd's algorithm
  // http://en.wikipedia.org/wiki/Lloyd%27s_algorithm

  update_beat_function();
}

GeomSet::GeomSet (const GeomSet & other)

  : m_size(other.size()),
    m_min_mass(1e-8f),
    m_min_omega(other.m_min_omega),
    m_max_omega(other.m_max_omega),
    m_min_duration(other.m_min_duration),
    m_timescale(4.0f / other.min_freq()),

    m_data(m_size * 10),

    m_mass        (m_size, m_data + m_size * 0),
    m_duration    (m_size, m_data + m_size * 1),
    m_omega0      (m_size, m_data + m_size * 2),
    m_omega       (m_size, m_data + m_size * 3),
    m_phase_real  (m_size, m_data + m_size * 4),
    m_phase_imag  (m_size, m_data + m_size * 5),
    m_beat_real   (m_size, m_data + m_size * 6),
    m_beat_imag   (m_size, m_data + m_size * 7),
    m_dbeat_real  (m_size, m_data + m_size * 8),
    m_dbeat_imag  (m_size, m_data + m_size * 9),

    m_norm_beat(0),

    m_force_gain(m_timescale),

    debug(false)
{
  operator=(other);
}

void GeomSet::operator= (const GeomSet & other)
{
  ASSERT_EQ(m_size, other.m_size);
  ASSERT_EQ(m_min_duration, other.m_min_duration);
  ASSERT_EQ(m_min_omega, other.m_min_omega);
  ASSERT_EQ(m_max_omega, other.m_max_omega);

  m_data = other.m_data;
  m_norm_beat = other.m_norm_beat;
  m_force_gain = other.m_force_gain;
}

GeomSet::~GeomSet ()
{
  if (debug) {
    float total_mass = sum(m_mass);
    PRINT(total_mass);
    PRINT2(min(m_duration), max(m_duration));
    PRINT2(min(m_omega), max(m_omega));
    PRINT2(rms(m_phase_real), rms(m_phase_imag));
    PRINT(m_force_gain);
  }
}

void GeomSet::update_beat_function ()
{
  const float * restrict duration = m_duration;
  const float * restrict phase_real = m_phase_real;
  const float * restrict phase_imag = m_phase_imag;

  float * restrict beat_real = m_beat_real;
  float * restrict beat_imag = m_beat_imag;
  float * restrict dbeat_real = m_dbeat_real;
  float * restrict dbeat_imag = m_dbeat_imag;

  float norm_beat = 0;
  for (size_t i = 0, I = m_size; i < I; ++i) {
    complex z(phase_real[i], phase_imag[i]);

    float p = 1.0f - duration[i];
    complex geom = 1.0f / (1.0f - p * z);
    //float scale = (1.0f - p); // unit maximum (infty-norm)
    float scale = (1.0f - sqr(p)); // unit variance (2-norm)

    complex beat = scale * z * geom;
    beat_real[i] = beat.real();
    beat_imag[i] = beat.imag();

    complex dbeat_over_i = beat * geom;
    dbeat_real[i] = -dbeat_over_i.imag();
    dbeat_imag[i] =  dbeat_over_i.real();

    norm_beat += norm(beat);
  }

  m_norm_beat = norm_beat;
}

/** Beat Optimization.
  see ideas/music.text (2011:05-19-25) (Q2.A1)

  The energy-based beat induction strategy is to do stochastic gradient descent
  of an energy function capturing beat + bend
  in omega x phase metric space.

    Minimize: beat_energy + bend_energy
    By Varying: phase,
    Subject To:
      min_omega <= omega <= omega / duration
    WRT Metric:
                  dphase^2        domega^2 
      ds^2 = ------------------ + --------
             duration^2 omega^2    omega^2
  Letting

    z = phase (complex, on unit circle)
    w0 = omega0 = natural omega
    w = bent omega
    d = duration
                    d z            d z
    b = beat = ------------- = -----------
               1 - (1 - d) z   1 - z + d z
    B = total_beat = sum i. m_i b_i
    E = E_beat + E_bend
    E_beat = beat energy = adiabatic average of -<B|b>
    E_bend = log(w/w0)^2/2
 
  The beat portion of the gradient descent steps is thus
 
    -d/dw E_beat = -d/dtheta E      # when synchronizing
                 = <B | db/dtheta>
                 = <B|dbeat>
*/

void GeomSet::advance ()
{
  const float * restrict mass = m_mass;
  const float * restrict beat_real = m_beat_real;
  const float * restrict beat_imag = m_beat_imag;

  complex total_beat = 0;
  float total_mass = 0;
  for (size_t i = 0, I = m_size; i < I; ++i) {
    float m = mass[i];
    total_mass += m;

    complex beat(beat_real[i], beat_imag[i]);
    total_beat += m * beat;
  }

  float max_force = norm(total_beat) / total_mass / sqr(m_min_duration);
  total_beat *= m_force_gain.update(max_force);

  const float * restrict duration = m_duration;
  const float * restrict omega0 = m_omega0;
  const float * restrict dbeat_real = m_dbeat_real;
  const float * restrict dbeat_imag = m_dbeat_imag;

  float * restrict omega = m_omega;
  float * restrict phase_real = m_phase_real;
  float * restrict phase_imag = m_phase_imag;

  const float decay_rate = 1.0f / m_timescale;

  const float min_omega = m_min_omega * 0.5f; // allow a little slower
  const float max_omega = m_max_omega * 2.0f; // allow a little faster

  for (size_t i = 0, I = m_size; i < I; ++i) {

    complex dbeat(dbeat_real[i], dbeat_imag[i]);

    float omega_accel = dot(total_beat, dbeat);
    float w0 = omega0[i];
    float w = omega[i];
    float max_omega_i = max_omega * duration[i];
    omega[i] = bound_to(min_omega, max_omega_i,
        w * (1 + omega_accel) + (w0 - w) * decay_rate);

    float phase_accel = omega_accel * duration[i];
    complex z(phase_real[i], phase_imag[i]);
    complex dz(-z.imag(), z.real());
    z += w * (1 + phase_accel) * dz;
    z /= abs(z);
    phase_real[i] = z.real();
    phase_imag[i] = z.imag();
  }

  update_beat_function();
}

float GeomSet::learn (float observed_value, Vector<float> & mass_io) const
{
  const float norm_beat = m_norm_beat;
  const float predicted_value = predict_value(mass_io);
  const float surprise = observed_value - predicted_value;
  const float dmass_dbeat = surprise / norm_beat;
  const float min_mass = m_min_mass;

  const float * restrict omega = m_omega;
  const float * restrict beat_real = m_beat_real;
  const float * restrict beat_imag = m_beat_imag;

  float * restrict mass = mass_io;

  for (size_t i = 0, I = m_size; i < I; ++i) {
    complex beat(beat_real[i], beat_imag[i]);

    //                        norm(beat) beat.real()
    // dmass = omega surprise ---------- -----------
    //                        norm_beat   abs(beat)
    float dmass = omega[i] * dmass_dbeat * abs(beat) * beat.real();

    mass[i] = max(min_mass, mass[i] + dmass);
  }

  return sqr(surprise);
}

void GeomSet::init_random (size_t i)
{
  ASSERT2_LT(i, m_size);

  m_mass[i] = m_min_mass;

  // randomly sample uniformly in log(omega),log(duration) space
  // constrained to
  //   min_omega <= omega <= omega / duration <= max_omega
  //   min_duration <= duration
  const float f0 = m_min_omega;
  const float f1 = m_max_omega;
  const float d0 = m_min_duration;
  float omega, duration;
  do {
    float u = random_01(), v = random_01();
    omega = affine_prod(f0, f1, min(u,v));
    duration = affine_prod(f0 / f1, 1.0f, max(u,v));
  } while (duration < d0);
  ASSERT_LE(m_min_omega, omega);
  ASSERT_LE(omega, omega / duration);
  ASSERT_LE(omega / duration, m_max_omega);
  ASSERT_LE(m_min_duration, duration);

  m_omega[i] = m_omega0[i] = omega;
  m_duration[i] = duration;

  complex phase = exp_2_pi_i(random_01());
  m_phase_real[i] = phase.real();
  m_phase_imag[i] = phase.imag();
}

//----( boltzmann bank )------------------------------------------------------

BoltzBank::BoltzBank (Bank param)
  : Bank(param),
    m_beat_fun(acuity),

    m_data(size * 10),

    m_mass        (size, m_data + size * 0),
    m_frequency   (size, m_data + size * 1),
    m_phase_real  (size, m_data + size * 2),
    m_phase_imag  (size, m_data + size * 3),
    m_beat        (size, m_data + size * 4),
    m_dbeat       (size, m_data + size * 5),
    m_meso_energy (size, m_data + size * 6),
    m_slow_energy (size, m_data + size * 7),
    m_meso_bend   (size, m_data + size * 8),
    m_slow_bend   (size, m_data + size * 9),

    m_temp(m_meso_bend),

    m_force_gain(4.0f / min_freq()),
    m_energy_gain(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_SAMPLE_RATE),

    m_force_snapshot(0,0)
{
  ASSERT_DIVIDES(4, size);
  ASSERT_LT(0, param.acuity);
  ASSERT_LT(0, param.strength);

  Bank::init_transform(m_frequency);

  m_mass.set(TOL);

  random_phase(m_phase_real, m_phase_imag);

  // initialize beat function
  for (size_t i = 0, I = size; i < I; ++i) {
    float x = m_phase_real[i];
    float y = m_phase_imag[i];
    m_beat[i] = m_beat_fun.value(x);
    m_dbeat[i] = m_beat_fun.deriv(x,y);
  }

  m_slow_bend.zero();
  m_slow_energy.zero();

  // check for timescale separation
  float safe_num_periods = 3.0f;
  ASSERT_LE(safe_num_periods * min_freq(), m_force_gain.timescale());
}

BoltzBank::~BoltzBank ()
{
  if (debug) {
    PRINT3(min(m_mass), max(m_mass), mean(m_mass));
    PRINT2(min(m_frequency), max(m_frequency));
    PRINT2(rms(m_phase_real), rms(m_phase_imag));
    PRINT2(min(m_slow_bend), max(m_slow_bend));
    PRINT2(rms(m_slow_bend), mean(m_slow_bend));
    PRINT3(m_force_gain, m_energy_gain.mean(), m_energy_gain.variance());
    PRINT(m_force_snapshot);
  }
}

void BoltzBank::sample_accum (
    const Vector<float> & dmass_in,
    Vector<complex> & accum_out)
{
  const size_t I = size;
  const size_t T = accum_out.size;

  const float * const restrict frequency = m_frequency;
  const float * const restrict dmass = dmass_in;

  float * const restrict mass = m_mass;
  float * const restrict phase_real = m_phase_real;
  float * const restrict phase_imag = m_phase_imag;
  float * const restrict beat = m_beat;
  float * const restrict dbeat = m_dbeat;
  float * const restrict meso_energy = m_meso_energy;
  float * const restrict slow_energy = m_slow_energy;
  float * const restrict meso_bend = m_meso_bend;
  float * const restrict slow_bend = m_slow_bend;
  complex * const restrict accum = accum_out;

  const StdBeatFun beat_fun = m_beat_fun;
  const float max_dbeat = beat_fun.max_deriv();

  // initialize beat function
  // The mass used in the beat function is always a little time-skewed,
  //   but it is much more slowly varying than phase.
  float sum_mb = 0;
  for (size_t i = 0; i < I; ++i) {
    sum_mb += beat[i] * mass[i];

    meso_energy[i] = 0;
    meso_bend[i] = 0;
  }

  for (size_t t = 0; t < T; ++t) {

    // compute moments
    const float mean_force = -m_energy_gain.update(fabsf(sum_mb)) * sum_mb;
    const float mean_bend = m_force_gain.update(fabsf(sum_mb)) * sum_mb
                         / max_dbeat;
    sum_mb = 0;

    // compute interpolation
    const float dt = (0.5f + t) / T;
    float sum_mx = 0;
    float sum_my = 0;

    for (size_t i = 0; i < I; ++i) {
      float x = phase_real[i];
      float y = phase_imag[i];
      float dphase;

      // update phase
      {
        float energy = beat[i] * mean_force;
        meso_energy[i] += energy;

        float bend = dbeat[i] * mean_bend;
        meso_bend[i] += bend;

        dphase = frequency[i] * bend_function(bend);
        float dx = -y * dphase;
        float dy =  x * dphase;

        x += dx;
        y += dy;
      }

      // normalize
      {
        float r = sqrt(sqr(x) + sqr(y));

        phase_real[i] = (x /= r);
        phase_imag[i] = (y /= r);
      }

      // update beat function, rescale, and accumulate
      {
        float m = mass[i] + dmass[i] * dt;
        float b = beat_fun.value(x);
        //dbeat[i] = beat_fun.deriv(x,y); // XXX not sse-optimized
        dbeat[i] = (b - beat[i]) / dphase;
        beat[i] = b;

        sum_mb += b * m;
        m *= sqr(max(0.0f, beat_fun.beat_floor - x)); // silent while beating
        sum_mx += m * x;
        sum_my += m * y;
      }
    }

    accum[t] += complex(sum_mx, sum_my);
  }

  // update mass & pitch bend
  const float scale = 1.0f / T;
  const float slow_rate = 1.0f / PITCH_BEND_TIMESCALE_FRAMES;
  for (size_t i = 0; i < I; ++i) {
    mass[i] = max(TOL, mass[i] + dmass[i]);
    slow_energy[i] += slow_rate * (scale * meso_energy[i] - slow_energy[i]);
    slow_bend[i] += slow_rate * (scale * meso_bend[i] - slow_bend[i]);
  }

  m_force_snapshot = complex(m_force_gain * sum_mb, 0.0f);
}

void BoltzBank::retune_zeromean (
    const Vector<float> & mass_in,
    Vector<float> & reass_out,
    float rate) const
{
  float mean_bend = mean_wrt(m_slow_bend, m_mass);

  add(-mean_bend, m_slow_bend, m_temp);

  Bank::retune(m_temp, mass_in, reass_out, rate);
}

void BoltzBank::retune_particles_zeromean (
    Vector<float> & pitch_table,
    const Vector<float> & mass_in,
    float rate)
{
  size_t I = size;
  size_t J = pitch_table.size;
  ASSERT_EQ(pitch_table.size, mass_in.size);

  const float * restrict mass = mass_in;
  const float * restrict bend = m_slow_bend;
  float * restrict pitch = pitch_table;

  const float bend_scale = rate / logf(freq1 / freq0);

  float sum_m = 0;
  float sum_mdp = 0;
  for (size_t j = 0; j < J; ++j) {
    float p = pitch[j];
    float dp = LinearInterpolate(p * I, I).get(bend) * bend_scale;
    pitch[j] = p + dp;

    float m = mass[j];
    sum_m += m;
    sum_mdp += m * dp;
  }

  const float bias = sum_m > 0 ? sum_mdp / sum_m : 0.0f;

  for (size_t j = 0; j < J; ++j) {
    pitch[j] = bound_to(0.0f, 1.0f, pitch[j] - bias);
  }
}

//----( loop bank )-----------------------------------------------------------

LoopBank::LoopBank (Bank param, bool coalesce, float expected_dt)
  : Bank(param),
    m_beat_fun(acuity),

    period(LOOP_BANK_PERIOD),
    max_dt(1.0 / (max_freq() * period)),

    m_drift_rate( size
                / log(freq1 / freq0)
                / period
                / TEMPO_DRIFT_TIMESCALE
                ),
    m_coalesce(coalesce),

    m_freq_p(size),
    m_cos_t(period),
    m_sin_t(period),
    m_beat_t(period),

    m_mass_tp(period * size),
    m_temp_tp(period * size),

    mass_now(size, m_mass_tp)
{
  LOG("LoopBank:")
  PRINT4(size, period, freq0, freq1);
  PRINT3(acuity, strength, m_drift_rate);

  LOG1("initialize freq");
  ASSERT_LE(expected_dt, max_dt / 2);
  for (size_t i = 0; i < size; ++i) {
    m_freq_p[i] = freq_at(i) * period;
  }
  PRINT2(expected_dt, max_dt / 2);
  PRINT2(min(m_freq_p), max(m_freq_p));

  LOG1("initialize beat");
  float beat_floor = m_beat_fun.beat_floor;
  float beat_scale = m_beat_fun.beat_scale;
  for (size_t t = 0; t < period; ++t) {
    float angle = 2 * M_PI * t / period;
    float x = cos(angle);
    float y = sin(angle);
    float beat = beat_scale * max(0.0f, x - beat_floor);

    m_beat_t[t] = beat;
    m_cos_t[t] = x;
    m_sin_t[t] = y;
  }

  m_mass_tp.set(TOL);
}

LoopBank::~LoopBank ()
{
  if (debug) {
    LOG("LoopBank stats:");
    if (m_mass_stats) PRINT(m_mass_stats);
    if (m_force_x_stats) PRINT(m_force_x_stats);
    if (m_force_y_stats) PRINT(m_force_y_stats);
    if (m_bend_stats) PRINT(m_bend_stats);
    PRINT3(min(mass_now), rms(mass_now), max(mass_now));
  }
}

void LoopBank::advance (float dt, float decay, float add)
{
  ASSERTW_LT(dt, max_dt);
  imax(dt, TOL);

  const size_t T = period;
  const size_t I = size;

  const float * restrict frequency = m_freq_p;
  const float * restrict cos = m_cos_t;
  const float * restrict sin = m_sin_t;
  const float * restrict beat = m_beat_t;

  const float drift_rate = m_drift_rate;

  // accumulate beat
  float total_mass = 0;
  float force_x = 0;
  float force_y = 0;
  for (size_t t = 0; t < T; ++t) {
    float m = sum(m_mass_tp.block(I, t));
    total_mass += m;

    float b = beat[t];
    float x = cos[t];
    float y = sin[t];
    force_x += b * m * x;
    force_y += b * m * y;
  }
  imax(total_mass, TOL);
  force_x /= total_mass; ASSERT_FINITE(force_x);
  force_y /= total_mass; ASSERT_FINITE(force_y);

  m_mass_stats.add(total_mass);
  m_force_x_stats.add(force_x);
  m_force_y_stats.add(force_y);

  // shift by freq + bent
  m_temp_tp.zero();
  for (size_t t = 0; t < T; ++t) {

    float * restrict mass  = m_mass_tp + I * t;
    float * restrict temp0 = m_temp_tp + I * t;
    float * restrict temp1 = m_temp_tp + I * ((1+t) % T);

    if (beat[t] < TOL) {

      // advance, bend phase
      for (size_t i = 0; i < I; ++i) {
        float freq = frequency[i] * dt;

        float w1 = min(1.0f, freq);
        float w0 = 1 - w1;

        float m = mass[i];

        temp0[i] += w0 * m;
        temp1[i] += w1 * m;
      }

    } else {

      float force = force_y * cos[t] - force_x * sin[t];
      float bend = beat[t] * force;
      m_bend_stats.add(bend);

      // advance, bend phase, drift pitch
      { size_t i = 0;
        size_t i1 = i;
        size_t i2 = i + 1;

        float freq = frequency[i] * dt;
        float bent = freq * bend_function(bend); ASSERT2_LE(0, bent);
        float drift = drift_rate * (bent - freq);

        float w1_ = bound_to(0.0f, 1.0f, bent);
        float w0_ = 1 - w1_;

        float w_2 = bound_to(0.0f, 1.0f, drift);
        float w_1 = 1 - w_2;

        float m = mass[i];
        w0_ *= m;
        w1_ *= m;

        temp0[i1] += w0_ * w_1;
        temp0[i2] += w0_ * w_2;
        temp1[i1] += w1_ * w_1;
        temp1[i2] += w1_ * w_2;
      }

      for (size_t i = 1; i < I - 1; ++i) {
        size_t i0 = i - 1;
        size_t i1 = i;
        size_t i2 = i + 1;

        float freq = frequency[i] * dt;
        float bent = freq * bend_function(bend); ASSERT2_LE(0, bent);
        float drift = drift_rate * (bent - freq);

        float w1_ = bound_to(0.0f, 1.0f, bent);
        float w0_ = 1 - w1_;

        float w_0 = bound_to(0.0f, 1.0f, -drift);
        float w_2 = bound_to(0.0f, 1.0f, drift);
        float w_1 = 1 - (w_0 + w_2);

        float m = mass[i];
        w0_ *= m;
        w1_ *= m;

        temp0[i0] += w0_ * w_0;
        temp0[i1] += w0_ * w_1;
        temp0[i2] += w0_ * w_2;
        temp1[i0] += w1_ * w_0;
        temp1[i1] += w1_ * w_1;
        temp1[i2] += w1_ * w_2;
      }

      { size_t i = I - 1;
        size_t i0 = i - 1;
        size_t i1 = i;

        float freq = frequency[i] * dt;
        float bent = freq * bend_function(bend); ASSERT2_LE(0, bent);
        float drift = drift_rate * (bent - freq);

        float w1_ = bound_to(0.0f, 1.0f, bent);
        float w0_ = 1 - w1_;

        float w_0 = bound_to(0.0f, 1.0f, -drift);
        float w_1 = 1 - w_0;

        float m = mass[i];
        w0_ *= m;
        w1_ *= m;

        temp0[i0] += w0_ * w_0;
        temp0[i1] += w0_ * w_1;
        temp1[i0] += w1_ * w_0;
        temp1[i1] += w1_ * w_1;
      }
    }
  }

  // reassign mass
  if (m_coalesce) {
    // bit slower but nicer
    Image::reassign_wrap_repeat(T, I, m_temp_tp, m_mass_tp, decay, add);

    // bit faster but uglier
    //m_mass_tp = m_temp_tp;
    //Image::reassign_wrap_repeat_xy(T, I, m_mass_tp, m_temp_tp, decay);
  } else {
    Image::reassign_wrap_x(T, I, m_temp_tp, m_mass_tp, decay, add);
  }

  // this should happen automatically in reassignment
  //imax(m_mass_tp, TOL);
}

//----( vector loop bank )----------------------------------------------------

VectorLoopBank::VectorLoopBank (Bank param, size_t w)
  : Bank(param),
    m_beat_fun(acuity),

    width(w),
    height(size),
    period(roundu(ceil(3 * acuity))),

    m_freq_y(height),
    m_phase_y(height),

    m_mass_ytx(height * period * width),
    m_temp_mass_tx(period * width) // overestimated
{
  LOG("VectorLoopBank of size " << height << " x " << period << " x " << width
      << " has range " << freq0 << "..." << freq1);

  m_phase_y.zero();
  for (size_t i = 0; i < height; ++i) {
    m_freq_y[i] = freq_at(i);
  }

  m_mass_ytx.zero();
}

VectorLoopBank::~VectorLoopBank ()
{
  if (debug) {
    PRINT2(mean(m_mass_ytx), rms(m_mass_ytx));
    if (m_bend_stats) PRINT(m_bend_stats);
  }
}

void VectorLoopBank::get_mass (Vector<float> & mass_yx) const
{
  ASSERT_SIZE(mass_yx, width * height);

  const size_t I = height;
  const size_t J = width;
  const size_t T = period;

  for (size_t i = 0; i < I; ++i) {

    CircularInterpolate circ(m_phase_y[i] * T, T);

    float * restrict mass = mass_yx.block(J, i);
    const float * restrict mass0 = m_mass_ytx.block(J, T * i + circ.i0);
    const float * restrict mass1 = m_mass_ytx.block(J, T * i + circ.i1);

    for (size_t j = 0; j < J; ++j) {
      mass[j] = circ.w0 * mass0[j]
              + circ.w1 * mass1[j];
    }
  }
}

void VectorLoopBank::add_mass (const Vector<float> & dmass_yx, float dt)
{
  ASSERT_SIZE(dmass_yx, width * size);

  const size_t I = height;
  const size_t J = width;
  const size_t T = period;

  for (size_t i = 0; i < I; ++i) {

    CircularInterpolate circ(m_phase_y[i] * T, T);

    float dphase = m_freq_y[i] * dt;

    float add_scale = T * dphase;
    float add0 = add_scale * circ.w0;
    float add1 = add_scale * circ.w1;

    const float * restrict dmass = dmass_yx.block(J, i);
    float * restrict mass0 = m_mass_ytx.block(J, T * i + circ.i0);
    float * restrict mass1 = m_mass_ytx.block(J, T * i + circ.i1);

    for (size_t j = 0; j < J; ++j) {
      mass0[j] += add0 * dmass[j];
      mass1[j] += add1 * dmass[j];
    }
  }
}

void VectorLoopBank::decay_add_mass (const Vector<float> & dmass_yx, float dt)
{
  ASSERT_SIZE(dmass_yx, width * height);

  const size_t I = height;
  const size_t J = width;
  const size_t T = period;

  for (size_t i = 0; i < I; ++i) {

    CircularInterpolate circ(m_phase_y[i] * T, T);

    float dphase = m_freq_y[i] * dt;

    float decay_scale = exp(-dphase);
    float decay0 = (1 - circ.w0) + circ.w0 * decay_scale;
    float decay1 = (1 - circ.w1) + circ.w1 * decay_scale;

    float add_scale = T * dphase;
    float add0 = add_scale * circ.w0;
    float add1 = add_scale * circ.w1;

    const float * restrict dmass = dmass_yx.block(J, i);
    float * restrict mass0 = m_mass_ytx.block(J, T * i + circ.i0);
    float * restrict mass1 = m_mass_ytx.block(J, T * i + circ.i1);

    for (size_t j = 0; j < J; ++j) {
      mass0[j] = decay0 * mass0[j] + add0 * dmass[j];
      mass1[j] = decay1 * mass1[j] + add1 * dmass[j];
    }
  }
}

void VectorLoopBank::scale_add_mass (
    const Vector<float> & scale_x,
    const Vector<float> & scale_y,
    const Vector<float> & dmass_yx,
    float dt)
{
  ASSERT_SIZE(scale_x, width);
  ASSERT_SIZE(scale_y, height);
  ASSERT_SIZE(dmass_yx, width * height);

  const size_t I = height;
  const size_t J = width;
  const size_t T = period;

  for (size_t i = 0; i < I; ++i) {

    CircularInterpolate circ(m_phase_y[i] * T, T);

    float dphase = m_freq_y[i] * dt;

    float add_scale = T * dphase;
    float add0 = add_scale * circ.w0;
    float add1 = add_scale * circ.w1;

    const float sy = scale_y[i];

    const float * restrict dmass = dmass_yx.block(J, i);
    const float * restrict sx = scale_x;
    float * restrict mass0 = m_mass_ytx.block(J, J * i + circ.i0);
    float * restrict mass1 = m_mass_ytx.block(J, J * i + circ.i1);

    for (size_t j = 0; j < J; ++j) {
      float s = sy * sx[j] - 1;
      float scale0 = (1 - circ.w0) + circ.w0 * s;
      float scale1 = (1 - circ.w1) + circ.w1 * s;

      mass0[j] = scale0 * mass0[j] + add0 * dmass[j];
      mass1[j] = scale1 * mass1[j] + add1 * dmass[j];
    }
  }
}

void VectorLoopBank::synchronize (float dt)
{
  const size_t I = height;
  const size_t J = width;
  const size_t T = period;

  const float beat_floor = cos(M_PI / acuity);

  // integrate beat
  float sum_m = 0;
  float sum_mbx = 0;
  float sum_mby = 0;

  for (size_t i = 0; i < I; ++i) {
    float phase = m_phase_y[i];
    for (size_t t = 0; t < T; ++t) {
      float angle = 2 * M_PI * (phase + t);
      Vector<float> mass = m_mass_ytx.block(J, T * i + t);
      float m = sum(mass);
      float x = cos(angle);
      float y = sin(angle);
      float b = max(0.0f, x - beat_floor);

      sum_m += m;
      sum_mbx += m * b * x;
      sum_mby += m * b * y;
    }
  }

  if (sum_m < TOL) return;
  const float beat_scale2 = sqr(m_beat_fun.beat_scale) / sum_m;
  const float force_x = beat_scale2 * sum_mbx;
  const float force_y = beat_scale2 * sum_mby;

  // synchronize
  for (size_t i = 0; i < I; ++i) {
    float phase = m_phase_y[i];

    float * restrict mass = m_mass_ytx.block(J, T * i);
    float * restrict temp = m_temp_mass_tx;

    // move shifting mass to temp
    for (size_t t = 0; t < T; ++t) {
      float angle = 2 * M_PI * (phase + t);
      float x = cos(angle);
      if (x < beat_floor) continue;

      float * restrict mass_t = mass + J * t;
      float * restrict temp_t = temp + J * t;

      for (size_t j = 0; j < J; ++j) {
        temp_t[j] = mass_t[j];
        mass_t[j] = 0;
      }
    }

    // shift temp back to mass
    for (size_t t = 0; t < T; ++t) {
      float angle = 2 * M_PI * (phase + t);
      float x = cos(angle);
      if (x < beat_floor) continue;

      float y = sin(angle);
      float b = max(0.0f, x - beat_floor);
      float force = force_y * x - force_x * y;
      float dphase = m_freq_y[i] * dt;
      float bend = b * force * dphase;

      m_bend_stats.add(bend * T);

      CircularInterpolate circ(t + bend * T, T);

      float * restrict mass_t0 = mass + J * circ.i0;
      float * restrict mass_t1 = mass + J * circ.i1;
      float * restrict temp_t = temp + J * t;

      for (size_t j = 0; j < J; ++j) {
        mass_t0[j] += circ.w0 * temp_t[j];
        mass_t1[j] += circ.w1 * temp_t[j];
      }
    }
  }
}

void VectorLoopBank::sample (Vector<float> & amplitude_x, float dt)
{
  ASSERT_SIZE(amplitude_x, width);

  const size_t I = height;
  const size_t J = width;
  const size_t T = period;

  amplitude_x.zero();
  float * restrict amp = amplitude_x;

  for (size_t i = 0; i < I; ++i) {

    CircularInterpolate circ(m_phase_y[i] * T, T);

    const float * restrict mass0 = m_mass_ytx.block(J, T * i + circ.i0);
    const float * restrict mass1 = m_mass_ytx.block(J, T * i + circ.i1);

    for (size_t j = 0; j < J; ++j) {
      amp[j] += circ.w0 * mass0[j]
              + circ.w1 * mass1[j];
    }

    // rotate
    float dphase = m_freq_y[i] * dt;
    m_phase_y[i] = wrap(m_phase_y[i] - dphase);
  }

  for (size_t j = 0; j < J; ++j) {
    amp[j] = max(0.0f, amp[j]);
  }
}

//====( transform wrappers )==================================================

//----( phasogram )-----------------------------------------------------------

Phasogram::Phasogram (
    size_t block_size,
    Bank param)
  : PhasorBank(param),

    m_block_size(block_size),

    m_amplitude0(size),
    m_damplitude(size)
{
  m_amplitude0.zero();
  m_damplitude.zero();
}

void Phasogram::transform (
    const Vector<float> & mass_in,
    const Vector<float> & amplitude_in,
    Vector<complex> & sound_out,
    float timescale,
    bool do_retune)
{
  ASSERT_SIZE(mass_in, size_in());
  ASSERT_SIZE(amplitude_in, size_in());
  ASSERT_SIZE(sound_out, m_block_size);

  float old_part = exp(-1/timescale);
  accumulate_step(old_part, PhasorBank::m_mass, mass_in);

  float * restrict a0 = m_amplitude0;
  float * restrict da = m_damplitude;
  const float * restrict a1 = amplitude_in;

  for (size_t i = 0, I = size_in(); i < I; ++i) {
    a0[i] += da[i];
    da[i] = a1[i] - a0[i];
  }

  sound_out.zero();

  sample_accum(m_amplitude0, m_damplitude, sound_out);

  if (do_retune) retune();
}

} // namespace Synchronized

