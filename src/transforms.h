
/** Invertible (mostly frequency-domain) transforms.

  TODO implement Wiener filter for Correlations
  TODO normalize Correlogram by variance (or rename to Covarigram)
  TODO implement in-place full watershed sharpener
*/

#ifndef KAZOO_TRANSFORMS_H
#define KAZOO_TRANSFORMS_H

#include "common.h"
#include "vectors.h"
#include "window.h"
#include "fft.h"
#include "splines.h"
#include "threads.h"

//----( highpass/lowpass splitter )-------------------------------------------

/** The HiLoSplitter multiplicatively splits a channel
  into high+low frequency components.

  Data types: full should be in [0,1];
    high and low should be nonnegative and usually not much more than 1
*/

class HiLoSplitter
{
  const size_t m_size;
  const size_t m_size_lowpass;

  Spline m_spline;

  const float m_exponent;

//#define HILO_MAP_BEFORE_SPLINE
#ifdef HILO_MAP_BEFORE_SPLINE
  float map_fwd (float x) const { return expf(m_exponent * x) - 1; }
  float map_bwd (float x) const { return logf(x + 1) / m_exponent; }
#else // HILO_MAP_BEFORE_SPLINE
  float map_fwd (float x) const { return x; }
  float map_bwd (float x) const { return x; }
#endif // HILO_MAP_BEFORE_SPLINE

public:
  HiLoSplitter (size_t size,
                size_t size_lowpass);

  // diagnostics
  size_t size          () const { return m_size; }
  size_t size_lowpass  () const { return m_size; }
  size_t size_highpass () const { return m_size_lowpass; }

  // these can operate concurrently
  void transform_fwd (const Vector<float> & full_in,
                      Vector<float> & high_out,
                      Vector<float> & low_out) const;
  void transform_bwd (const Vector<float> & high_in,
                      Vector<float> & low_in,
                      Vector<float> & full_out) const;

  // testing
  float test () const;
};

//----( multiscale splitter )-------------------------------------------------

/** The MultiScale splitter splits a large spectrum into two arbitrary spectra
*/
class MultiScale
{
  const size_t m_size_super;
  const size_t m_size_fst;
  const size_t m_size_snd;

  const Spline * const m_super_to_fst;
  const Spline * const m_super_to_snd;

  Vector<float> m_super_fst;
  Vector<float> m_super_snd;

public:
  MultiScale (const Spline * super_to_fst,
              const Spline * super_to_snd);

  // diagnostics
  size_t size_super () const { return m_size_super; }
  size_t size_fst   () const { return m_size_fst; }
  size_t size_snd   () const { return m_size_snd; }

  // these can operate concurrently
  void transform_fwd (const Vector<float> & super_in,
                      Vector<float> & fst_out,
                      Vector<float> & snd_out);
  void transform_bwd (Vector<float> & fst_io,
                      Vector<float> & snd_io,
                      Vector<float> & super_out);
};

//----( shepard scale )-------------------------------------------------------

/** The Shepard scale transforms pitch to an octave-periodic 12-tone scale,
   while automatically fitting a subtone alignment parameter.
*/

class Shepard
{
  enum ScaleSizes
  {
    e_size_out = 12,      // twelve tone scale
    e_align_factor = 5,   // just enough to estimate phase
    e_size_mid = e_size_out * e_align_factor
  };
  const size_t m_size_in;

  Vector<float> m_mid_fwd;
  Vector<float> m_mid_bwd;
  SplineToCircle m_wrap;

  // tone alignment
  float m_tone_alignment;
  SplineToCircle m_align;

  void align (float alignment_rate = 2e-3);

  Mutex m_alignment_mutex;

public:
  Shepard (size_t size_in,
           float min_freq_hz,
           float max_freq_hz);
  ~Shepard () { SAVE("Shepard.alignment", m_tone_alignment); }

  // diagnostics
  size_t size_in () const { return m_size_in; }
  size_t size_out () const { return e_size_out; }

  // these can operate in parallel
  void transform_fwd (const Vector<float> & pitch_in,
                      Vector<float> & tone_out);
  void transform_bwd (const Vector<float> & tone_in,
                      Vector<float> & pitch_out);
};

//----( sharpener )-----------------------------------------------------------

/** Sharpening filter via 1-step watershed method
*/

class Sharpener
{
  const size_t m_size;
public:
  Sharpener (size_t size);

  // diagnostics
  size_t size () const { return m_size; }

  void transform (const Vector<float> & data_in,
                  Vector<float> & data_out);
};

//----( spline sharpener )----------------------------------------------------

/** Moment-based frequency sharpener.
*/

class Sharpen1D
{
  const size_t m_size;
  const size_t m_radius;
  const float m_min_mass;

  Vector<float> m_mass_blur;
  Vector<float> m_freq_blur;
  Vector<float> m_temp_blur;

public:
  Sharpen1D (size_t size, size_t radius);
  virtual ~Sharpen1D () {}

  // diagnostics
  size_t size () const { return m_size; }
  size_t radius () const { return m_radius; }

  void transform (const Vector<float> & mass_in,
                  const Vector<float> & freq_in,
                  Vector<float> & freq_out);
  void transform (const Vector<float> & freq_in,
                  Vector<float> & freq_out)
  {
    transform(freq_in, freq_in, freq_out);
  }
};

//----( spline sharpener )----------------------------------------------------

/** Moment-based time-frequency sharpener.
*/

class Sharpen2D
{
  const size_t m_size;
  const size_t m_radius;
  const size_t m_length;
  const float m_min_mass;

  Vector<float> m_mass_in;

  Vector<float> m_mass_blur;
  Vector<float> m_time_blur;
  Vector<float> m_freq_blur;
  Vector<float> m_temp_blur;

  Vector<float> m_mass_out;

public:
  Sharpen2D (size_t size, size_t radius);
  virtual ~Sharpen2D () {}

  // diagnostics
  size_t size () const { return m_size; }
  size_t radius () const { return m_radius; }

  void transform (const Vector<float> & freq_in,
                  Vector<float> & freq_out);
};

//----( blur )----------------------------------------------------------------

/** 1-step frequency blurring
*/

template<class T>
class Blur
{
  const size_t m_size;
  const float m_old_part;
  const float m_new_part;

public:

  Blur (size_t size, float strength)
    : m_size(size),
      m_old_part(1 - strength * 2 / 3),
      m_new_part(strength / 3)
  {
    ASSERT_LT(2, size);
    ASSERT_LT(0, strength);
    ASSERT_LE(strength, 1);
  }

  // diagnostics
  size_t size () const { return m_size; }

  void transform (const Vector<T> & data_in,
                  Vector<T> & data_out) const
  {
    ASSERT_SIZE(data_in, m_size);
    ASSERT_SIZE(data_out, m_size);

    data_out[0] = (m_old_part + m_new_part) * data_in[0]
                + m_new_part * data_in[1];
    for (size_t i = 1; i < m_size - 1; ++i) {
      data_out[i] = m_old_part * data_in[i]
                  + m_new_part * (data_in[i-1] + data_in[i+1]);
    }
    data_out[m_size-1] = (m_old_part + m_new_part) * data_in[m_size-1]
                       + m_new_part * data_in[m_size-2];
  }
};

//----( auto correlation )----------------------------------------------------

class AutoCorrelation
{
  const size_t m_size;
  mutable FFT_R2C m_fft;

public:
  AutoCorrelation (size_t exponent);

  // diagnostics
  size_t size  () const { return m_size; }

  void transform (const Vector<float> & f_in,
                  Vector<float> & ff_out);
};

//----( cross correlation )---------------------------------------------------

class CrossCorrelation
{
  const size_t m_size_in;
  const size_t m_size_out;
  mutable FFT_R2C m_fft_fwd;
  mutable FFT_R2C m_fft_bwd;

  Vector<complex> m_freq_fwd;
  Vector<complex> m_freq_bwd;

public:
  CrossCorrelation (size_t exponent_in);

  // diagnostics
  size_t size_in  () const { return m_size_in; }
  size_t size_out () const { return m_size_out; }

  // helpers to place real values within zero padding
protected:
  void fft_fwd_in (const Vector<float> & time_in);
  void fft_fwd_out (Vector<float> & time_out) const;
public:

  // these can operate in parallel
  void transform_fwd (const Vector<float> & f_in,
                      const Vector<float> & g_in,
                      Vector<float> & fg_out);
  void transform_bwd (const Vector<float> & fg_in,
                      const Vector<float> & f_in,
                      Vector<float> & g_out);
};

//----( laplace filterbank )--------------------------------------------------

class LaplaceFilterBank
{
  const size_t m_vect_size;
  const size_t m_bank_size;
  Vector<float> m_factors;
  Vector<float> m_bank;
public:
  LaplaceFilterBank (size_t vect_size,
                     size_t bank_size,
                     float min_time_scale,
                     float max_time_scale);

  // the setter
  void accumulate (const Vector<float> & sample);

  // getters
  Vector<float> operator[] (size_t b) { return m_bank.block(m_vect_size, b); }
};

//----( Laplace-Fourier transform )-------------------------------------------

class LaplaceFourier
{
  const size_t m_size_in;
  const size_t m_num_freqs;

  float      m_info;
  Vector<float>     m_coeff_new;
  Vector<complex> m_coeff_old;

  Vector<complex> m_history_fwd;
  Vector<complex> m_history_bwd;

  void accumulate (Vector<complex> & history, const Vector<complex> & data);
public:
  LaplaceFourier (size_t size_in,
                  size_t num_freqs,
                  complex min_freq,
                  complex max_freq);

  // diagnostics
  size_t size_in   () const { return m_size_in; }
  size_t num_freqs () const { return m_num_freqs; }
  size_t size_out  () const { return m_size_in * m_num_freqs; }

  // these can operate in parallel
  void transform_fwd (const Vector<complex> & data_in,
                      Vector<float> & energy_out);
  void transform_bwd (const Vector<float> & energy_in,
                      Vector<complex> & data_out);
};

//----( melodigram )----------------------------------------------------------

/** A transformation from spectrum to melody.

  The melodigram
   accumulates pitch loudness in a Laplace transform filterbank, and
   correlates current sound with each of the transformed sounds.
*/

class Melodigram
{
  const size_t m_size;
  const size_t m_size_corr;
  const size_t m_num_filters;

  LaplaceFilterBank m_filters_fwd;
  LaplaceFilterBank m_filters_bwd;
  CrossCorrelation  m_correlation;

public:
  Melodigram (size_t exponent,
              size_t num_filters,
              float frame_rate,
              float min_time_scale_sec = 0.2,
              float max_time_scale_sec = 15);

  // diagnostics
  size_t size        () const { return m_size; }
  size_t size_corr   () const { return m_size_corr; }
  size_t num_filters () const { return m_num_filters; }
  size_t size_out    () const { return m_size_corr * m_num_filters; }

  // these can operate in parallel
  void transform_fwd (const Vector<float> & pitch_in,
                      Vector<float> & corr_out);

  void transform_bwd (const Vector<float> & prev_pitch_in,
                      const Vector<float> & corr_in,
                      Vector<float> & pitch_out);
};

//----( harmonigram )---------------------------------------------------------

class FreqToPitch : public Function
{
  const float m_min_freq;
  const float m_max_freq;
public:
  FreqToPitch (float min_freq, float max_freq)
    : m_min_freq(min_freq),
      m_max_freq(max_freq)
  {}
  virtual float value (float freq) const
  {
    if (freq < m_min_freq) return -1;
    return logf(freq / m_min_freq) / logf(m_max_freq / m_min_freq);
  }
};

//----( harmonics )-----------------------------------------------------------

/** Harmonic
  Ensures each moment of sound is relatively harmonic,
  by adjusting overtones to lie on exact integer multiples.

  Let e(w) be an energy spectrum.
  The harmonizer adjusts
    e(w) |--> sqrt(...) + delta(0,w)
         |--> autocorr(...)
         |--> ???
*/

class Harmonic
{
  const size_t m_size;

  FFT_R2C m_fft;

  float map_fwd (float x) const { return sqrtf(x); }
  //float map_bwd (float x) const { return expf(x) - 1.0f; }

public:

  Harmonic (size_t exponent);

  void adjust (Vector<float> & space_in, Vector<float> & dspace_out);
};

//----( rhythmgram )----------------------------------------------------------

/** A multichannel transformation from time to tempo.

  The Rhythmgram looks for periodic behavior in each channel of input by
  computing the Fourier transform of exponentially-windowed autocorrelation.
  The r'gram accumulates producs of each channel f(t) with a basis functions
    g(t,w) = exp(2 pi i w t) / (1 + 1/num_beats)^(w t)
*/

class Rhythmgram
{
  const size_t m_size_in;
  const size_t m_size_factor;

  Vector<float>     m_coeff_new;
  Vector<complex> m_coeff_old;
  Vector<complex> m_history_fwd;
  Vector<complex> m_history_bwd;

public:
  Rhythmgram (size_t size_in,
              size_t size_factor,
              float num_beats = 4.0f);

  // diagnostics
  size_t size_in     () const { return m_size_in; }
  size_t size_factor () const { return m_size_factor; }
  size_t size_out    () const { return sqr(m_size_in) * m_size_factor; }

  // these can operate in parallel
  void transform_fwd (const Vector<float> & value_in,
                      Vector<float> & tempo_out);
  void transform_bwd (const Vector<float> & tempo_in,
                      Vector<float> & value_out);
};

//----( correlogram )---------------------------------------------------------

class Correlogram
{
  const size_t m_size_in;
  const size_t m_size_out;

  const float m_decay_factor;

  Vector<float> m_history_fwd;
  Vector<float> m_history_bwd;

  CrossCorrelation m_correlation;

public:
  Correlogram (size_t exponent_in,
               float decay_factor);

  // diagnostics
  size_t size_in  () const { return m_size_in; }
  size_t size_out () const { return m_size_out; }
  float decay_factor () const { return m_decay_factor; }

  // these can operate in parallel
  void transform_fwd (const Vector<float> & freq_in,
                      Vector<float> & corr_out);
  void transform_bwd (const Vector<float> & prev_freq_in,
                      const Vector<float> & corr_in,
                      Vector<float> & freq_out);
};

//----( pitch shifting )------------------------------------------------------

class OctaveLower
{
  const size_t m_size;

  Vector<float> m_window;
  Vector<complex> m_sound_in;
  Vector<complex> m_sound_out;
public:
  OctaveLower (size_t size);

  size_t size () const { return m_size; }

  void transform (const Vector<complex> & sound_in,
                  Vector<complex> & sound_out);
};

/* OLD
class PitchShift
{
  const float m_factor;

  Supergram m_supergram;
  SuperSpectrum::SuperToFreq m_conj;
  AffineFunction m_fun;
  ConjugateFunction m_conj_fun;
  Spline m_spline;

  Vector<float> m_super_in;
  Vector<float> m_super_out;
public:
  PitchShift (size_t size_exponent,
              size_t factor_exponent,
              float halftone_shift)
    : m_factor(pow(2.0, 12.0 * halftone_shift)),

      m_supergram(size_exponent, factor_exponent),
      m_conj(m_supergram.spectrum()),
      m_fun(m_factor),
      m_conj_fun(m_conj, m_fun),
      m_spline(size(), size(), m_conj_fun),

      m_super_in(m_supergram.super_size()),
      m_super_out(m_supergram.super_size())
  {}

  // diagnostics
  size_t size () const { return m_supergram.small_size(); }
  float factor () const { return m_factor; }

  void transform (const Vector<complex> & sound_in,
                  Vector<complex> & sound_out);
};
*/

//----( looping )-------------------------------------------------------------

class Loop
{
  const size_t m_size;
  const size_t m_length;
  const float m_decay;

  Vector<float> m_memory;
  size_t m_phase;

public:

  Loop (size_t size, size_t length, float timescale);
  ~Loop () {}

  // diagnostics
  size_t size () const { return m_size; }
  size_t length () const { return m_length; }

  void transform (const Vector<float> & energy_in,
                  Vector<float> & energy_out);
};

//----( chorus )--------------------------------------------------------------

/** Time-blur while pitch sharpening.
*/

class Chorus
{
  const float m_loud_decay;
  const float m_pitch_decay;

  Sharpen1D m_sharpen;

  Vector<float> m_mass;
  Vector<float> m_state;

public:

  Chorus (size_t size,
          size_t radius = 32,
          float loud_timescale = 128,
          float pitch_timescale = 1)
    : m_loud_decay(1 / (1 + loud_timescale)),
      m_pitch_decay(1 / (1 + pitch_timescale)),
      m_sharpen(size, radius),
      m_mass(size),
      m_state(size)
  {}
  ~Chorus () {}

  size_t size () const { return m_sharpen.size(); }
  size_t radius () const { return m_sharpen.radius(); }

  void transform (const Vector<float> & sound_in, Vector<float> & sound_out);
};

#endif // KAZOO_TRANSFORMS_H

