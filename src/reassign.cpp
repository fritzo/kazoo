
#include "reassign.h"

#define LOG1(mess)

//----( super-resolution spectrum )-------------------------------------------

SuperSpectrum::SuperSpectrum (
    float sample_rate_hz,
    float min_freq_hz)

  : m_min_freq(min_freq_hz / (sample_rate_hz / 2)),
    m_log_min_freq(log(m_min_freq)),
    m_scale(1 / (2 - m_min_freq)),

    m_tolerance(1e-6)
{
#ifndef KAZOO_NDEBUG
  LOG("validating SuperSpectrum(" << sample_rate_hz << ", "
                                  << min_freq_hz << ")");
  for (size_t i = 0, I = 100; i < I; ++i) {
    float s = (0.5f + i) / I;

    float w = super_to_freq(s);
    float s2 = freq_to_super(w);
    ASSERT(sqr(s2 - s) < m_tolerance,
           "SuperSpectrum mapped " << s << " (super) |-> "
                                   << w << " (freq) |-> "
                                   << s2 << " (super)");

    float p = super_to_pitch(s);
    float s3 = pitch_to_super(p);
    ASSERT(sqr(s3 - s) < m_tolerance,
           "SuperSpectrum mapped " << s << " (super) |-> "
                                   << p << " (pitch) |-> "
                                   << s3 << " (super)");
  }
#endif // KAZOO_NDEBUG
}

float SuperSpectrum::super_to_freq (float super) const
{
  // bisection search: slow but easy to debug...
  float LB = m_min_freq, UB = 1;
  while (UB - LB > m_tolerance) {
    float x = 0.5f * (UB + LB);
    (freq_to_super(x) < super ? LB : UB) = x;
  }
  return 0.5f * (UB + LB);
}

float SuperSpectrum::super_to_pitch (float super) const
{
  // bisection search: slow but easy to debug...
  float LB = 0, UB = 1;
  while (UB - LB > m_tolerance) {
    float x = 0.5f * (UB + LB);
    (pitch_to_super(x) < super ? LB : UB) = x;
  }
  return 0.5f * (UB + LB);
}

//----( supersampled reassignment )-------------------------------------------

Supergram::Supergram (
    size_t size_exponent,
    size_t time_exponent,
    size_t freq_exponent,
    float sample_rate)

  : m_size(1 << size_exponent),
    m_time_factor(1 << time_exponent),
    m_freq_factor(1 << freq_exponent),
    m_small_size(m_size / m_time_factor),
    m_large_size(m_size * m_freq_factor),
    m_energy_size(m_large_size / 2),
    m_super_size(m_energy_size),
    m_sample_rate(sample_rate),

    m_tolerance(1e-10),

    m_fft(size_exponent),

    m_h(2 * m_size),
    m_th(2 * m_size),
    m_dh(2 * m_size),
    m_synth(2 * m_size),
    m_time_in(2 * m_size),

    m_h_old(m_h.block(m_size, 0)),
    m_h_new(m_h.block(m_size, 1)),
    m_th_old(m_th.block(m_size, 0)),
    m_th_new(m_th.block(m_size, 1)),
    m_dh_old(m_dh.block(m_size, 0)),
    m_dh_new(m_dh.block(m_size, 1)),
    m_time_old(m_time_in.block(m_size, 0)),
    m_time_new(m_time_in.block(m_size, 1)),

    m_freq_h(m_size),
    m_freq_th(m_size),
    m_freq_dh(m_size),

    m_spectrum(m_sample_rate),
    m_accumulator(m_time_factor, m_super_size),

    // XXX segfault here

    m_energy_in(m_large_size / 2),
    m_freq_large(m_large_size / 2),
    m_blurred(m_large_size / 2),
    m_freq_blur(m_large_size / 2, 0.5),
    m_time_out(2 * m_size)
{
  LOG("building Supergram(" << size_exponent << ", "
                            << time_exponent << ", "
                            << freq_exponent << ", "
                            << sample_rate << ")");

  ASSERT_LT(0, time_exponent);
  ASSERT_LT(0, freq_exponent);
  ASSERT_LT(time_exponent, size_exponent);
  ASSERT_LT(freq_exponent, size_exponent);
  ASSERT_LE(size_exponent + time_exponent, 24);
  ASSERT_LE(size_exponent + freq_exponent, 24);

  m_time_in.zero();
  m_time_out.zero();
  m_freq_large.zero();

  // WARNING new_FreqScale is a method: keep this out of the initializer list!
  m_super_to_freq = new_FreqScale(m_energy_size);

  //HannWindow h_window;
  //NarrowWindow wide_window(1);
  BlackmanNuttallWindow bn_window;
  NarrowWindow narrow_window(1.0 / m_time_factor);

  LOG1("building analysis and synthesis windows");

  FunctionAndDeriv & anal_window = bn_window;
  FunctionAndDeriv & synth_window = narrow_window;

  for (size_t i = 0; i < 2 * m_size; ++i) {
    float t = (0.5f + i - m_size) / m_size;

    m_h[i] = anal_window(t);
    m_th[i] = t * anal_window(t);
    m_dh[i] = anal_window.deriv(t);
    m_synth[i] = synth_window(t);
  }

  // renormalize to preserve energy
  float h_scale = pow(norm_squared(m_h), -0.5);
  m_h *= h_scale;
  m_th *= h_scale;
  m_dh *= h_scale;
  float synth_scale = pow(norm_squared(m_synth), -0.5);
  m_synth *= synth_scale;
}

/** The reassignment primitive.
  See (R1), or equivalently (R2) equations (64) and (65).

  Notes:
  (N1) The extra factor of 1/(2 pi) in (65) is because
    we are using discretized units of 2 pi.

  (N2) (R1)'s signs for equations (64),(65) are -,+, resp,
    whereas results look best when using the opposite +,-.
    Theoretically also, I get + for (64) by this calculation:
    Assume impulses $\delta(t-\tau)$ are perfectly localized.
    Then energy must shift by time $\tau$,
    which relates to the H,TH integrals via
    \begin{align*}
         H &:= \int \delta(t-\tau) h(t) \, dt = h(\tau)        \\
        TH &:= \int \delta(t-\tau) t h(t) \, dt = \tau h(\tau) \\
      \tau &=  \frac {TH} {H}                                  \\
           &=  \frac {TH\;H} {H^2}
    \end{align*}

  (N3) the supergram combines pos+neg frequencies
    only after aligning by each supersample frequency offset
*/
void Supergram::transform_fwd (
    const Vector<complex> & time_in,
    Vector<float> & super_out)
{
  LOG1("Supergram transforming forward");

  ASSERT_SIZE(time_in, m_small_size);
  ASSERT_SIZE(super_out, m_super_size);
  ASSERT1_LE(norm_squared(time_in), 2 * m_small_size);
  ASSERT1_LE(max_norm_squared(time_in), 2);

  LOG1("shifting time window");
  for (size_t f = 0; f+1 < 2*m_time_factor; ++f) {
    m_time_in.block(m_small_size, f) = m_time_in.block(m_small_size, f+1);
  }
  m_time_in.block(m_small_size, 2 * m_time_factor - 1) = time_in;
  m_accumulator.shift_front();

  for (size_t f = 0; f < m_freq_factor; ++f) {

    LOG1("setting up reassignment ffts");

    // compute energy
    linear_combine(m_h_old, m_time_old,
                   m_h_new, m_time_new, m_fft.time_in);
    m_fft.transform_fwd();
    m_freq_h = m_fft.freq_out;

    // compute frequency shifts
    linear_combine(m_th_old, m_time_old,
                   m_th_new, m_time_new, m_fft.time_in);
    m_fft.transform_fwd();
    m_freq_th = m_fft.freq_out;

    // compute time shifts
    linear_combine(m_dh_old, m_time_old,
                   m_dh_new, m_time_new, m_fft.time_in);
    m_fft.transform_fwd();
    m_freq_dh = m_fft.freq_out;

    const complex * restrict h  = m_freq_h;
    const complex * restrict th = m_freq_th;
    const complex * restrict dh = m_freq_dh;

    LOG1("accumulating reassigned fft");
    for (size_t i = 0; i < m_size; ++i) {

      float energy = norm(h[i]);
      if (not (energy > m_tolerance)) continue;
      ASSERT1_LE(energy, m_energy_size);

      // eqn (64) in (R2)
      float d_time = (th[i] * conj(h[i])).real() / energy;

      // eqn (65) in (R2)
      float d_freq = -(dh[i] * conj(h[i])).imag() / (2 * M_PI * energy);

      // rescale by factors
      float e = energy / (m_time_factor * m_freq_factor);
      float t = wrap( m_time_factor * d_time
                   + m_time_factor / 2
                   + 0.5f / m_time_factor,
                   m_time_factor);
      float w = wrap( m_freq_factor * (d_freq + i)
                   + f,
                   m_large_size);

      // combine pos+neg frequencies
      if (w > m_energy_size) {
        w = m_large_size - w;
      }

      // convert to super scale
      float s = m_spectrum.freq_to_super(w / m_energy_size) * m_super_size;

      if (not safe_isfinite(t)) {
        //WARN("t is not finite (" << t << "); skipping point");
        continue;
      }
      if (not safe_isfinite(s)) {
        //WARN("s is not finite (" << s << "); skipping point");
        continue;
      }

      int t0, s0;
      float a0_, a1_, a_0, a_1;
      linear_interpolate(t, m_time_factor, t0, a0_, a1_);
      linear_interpolate(s, m_super_size,  s0, a_0, a_1);

      m_accumulator[t0  ][s0  ] += e * a0_ * a_0;
      m_accumulator[t0  ][s0+1] += e * a0_ * a_1;
      m_accumulator[t0+1][s0  ] += e * a1_ * a_0;
      m_accumulator[t0+1][s0+1] += e * a1_ * a_1;
    }

    for (size_t i = 0; i < 2*m_size; ++i) {
      m_time_in[i] *= exp_2_pi_i( -(float) i / m_large_size );
    }
  }
  for (size_t i = 0; i < 2*m_size; ++i) {
    m_time_in[i] *= exp_2_pi_i( +(float) i / m_size );
  }

  super_out = m_accumulator[0];

  // TODO("this assertion fails, figure out why");
  ASSERT1_LE(max_norm_squared(super_out), 2 * m_large_size * m_time_factor);
}

inline void add_with_phase_twist (
    const Vector<float> & window,
    const Vector<complex> & restrict time_in,
    Vector<complex> & restrict time_out,
    float twist,
    int size)
{
  for (int i = 0; i < 2*size; ++i) {
    float phase = twist * (float)(i - size) / size;
    time_out[i] += time_in[i%size] * window[i] * exp_2_pi_i(phase);
  }
}

/** Reconstruct a complex signal from a superresolutoin spectrogram.

  To match phase upon reconstruction,
   phase is matched to the most probable previous tone source,
   averaged WRT energy over - m_time_factor + m_freq_factor frequencies away.
*/

void Supergram::transform_bwd (
    const Vector<float> & super_in,
    Vector<complex> & time_out)
{
  LOG1("Supergram transforming backward");

  ASSERT_SIZE(super_in, m_super_size);
  ASSERT_SIZE(time_out, m_small_size);

  LOG1("shifting time window");
  for (size_t f_src = 1,f_dst = 0; f_src < 2*m_time_factor; ++f_src,++f_dst) {
    m_time_out.block(m_small_size, f_dst)
      = m_time_out.block(m_small_size, f_src);
  }
  m_time_out.block(m_small_size, 2 * m_time_factor - 1).zero();

  LOG1("transform from super spectrum to large_size scale");
  m_super_to_freq->transform_fwd(super_in, m_energy_in);

  LOG1("updating previous phase");
  const size_t factor = m_freq_factor * m_time_factor;
  for (size_t i = 0; i < m_energy_size; ++i) {
    float phase = (float)(i % factor) / factor;
    m_freq_large[i] *= exp_2_pi_i(phase);
  }

#define BLUR_PHASE_MEMORY
#ifdef BLUR_PHASE_MEMORY
  m_freq_blur.transform(m_freq_large, m_blurred);
  m_freq_large = m_blurred;
#endif // BLUR_PHASE_MEMORY

  LOG1("combining phase and magnitude");
  for (size_t i = 0; i < m_energy_size; ++i) {

    float magnitude = sqrtf(m_energy_in[i]);
    complex estimate = m_freq_large[i];
    float norm_estimate = norm(estimate);

    if (norm_estimate > m_tolerance) {
      complex phase = estimate / sqrtf(norm_estimate);
      m_freq_large[i] = magnitude * phase;
    } else {
      m_freq_large[i] = magnitude;
    }
  }

  LOG1("running ffts for each frequency offset");
  for (size_t f = 0; f < m_freq_factor; ++f) {

    m_fft.freq_in.zero();
    for (size_t i = 0; i < m_size/2; ++i) {
      m_fft.freq_in[i] = m_freq_large[f + m_freq_factor * i];
    }

    m_fft.transform_bwd();

    add_with_phase_twist(m_synth,
                         m_fft.time_out,
                         m_time_out,
                         (float) f / m_freq_factor,
                         m_size);
  }

  time_out = m_time_out.block(m_small_size);
  soft_clip(time_out);

  ASSERT1_LE(max_norm_squared(time_out), 2 + m_tolerance);
}

Spline * Supergram::new_FreqScale (size_t size_out) const
{
  if (size_out == 0) size_out = pitch_size();

  size_t size_in = m_super_size;
  Vector<float> super_to_freq(size_in);
  sample_function(SuperToFreq(m_spectrum), super_to_freq);
  return new Spline(size_in,
                    size_out,
                    super_to_freq);
}

Spline * Supergram::new_PitchScale (size_t size_out) const
{
  if (size_out == 0) size_out = freq_size();

  size_t size_in = m_super_size;
  Vector<float> super_to_pitch(size_in);
  sample_function(SuperToPitch(m_spectrum), super_to_pitch);
  return new Spline(size_in,
                    size_out,
                    super_to_pitch);
}

