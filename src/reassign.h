#ifndef KAZOO_REASSIGN_H
#define KAZOO_REASSIGN_H

/** Invertible Time-Frequency Reassigned Spectrogram Stream.

  References on Time-Frequency Reassignment:
  (R1) http://en.wikipedia.org/wiki/Reassignment_method
  (R2) "A unified theory of time-frequency reassignment"
    -Kelly Fitz, Sean Fulop
    http://arxiv.org/abs/0903.3080
  (R3) "Time-frequency reassignment: From principles to algorithms"
    -P.Flandrin, F.Auger, E.Chassande-Mottin
    http://perso.ens-lyon.fr/patrick.flandrin/0065-Ch05.pdf

  TODO fix supergram.transform_bwd(-,-)
*/

#include "common.h"
#include "fft.h"
#include "vectors.h"
#include "window.h"
#include "splines.h"
#include "transforms.h"

//----( cyclic buffer of vectors )--------------------------------------------

template<class T>
class CyclicBuffer
{
  const size_t m_cycles;
  const size_t m_size;
  Vector<Vector<T> *> m_blocks;
  Vector<Vector<T> *> m_shifted;

  Vector<T> data (size_t i) { return m_blocks.block(m_size, i); }

public:

  CyclicBuffer (size_t cycles, size_t size)
    : m_cycles(cycles),
      m_size(size),
      m_blocks(cycles),
      m_shifted(cycles)
  {
    for (size_t i = 0; i < m_cycles; ++i) {
      Vector<T> * block = new Vector<T>(size);
      block->zero();
      m_blocks[i] = block;
      m_shifted[i] = block;
    }
  }
  ~CyclicBuffer ()
  {
    for (size_t i = 0; i < m_cycles; ++i) {
      delete m_blocks[i];
    }
  }

  size_t cycles () const { return m_cycles; }
  size_t size () const { return m_size; }
  Vector<T> & operator[] (size_t i) { return *(m_shifted[i]); }

  void shift_front ()
  {
    Vector<T> * front = m_shifted[0];
    for (size_t i = 0; i+1 < m_cycles; ++i) {
      m_shifted[i] = m_shifted[i+1];
    }
    m_shifted[m_cycles-1] = front;
    front->zero();
  }
  void shift_back ()
  {
    Vector<T> * back = m_shifted[m_cycles-1];
    for (size_t i = m_cycles-1; i>0; --i) {
      m_shifted[i] = m_shifted[i-1];
    }
    m_shifted[0] = back;
    back->zero();
  }
};

//----( super-resolution spectrum )-------------------------------------------

/** Super-resolution spectrum for high log- and linear- resolution.

  The purpose is to retain high resolution during reassignment,
  and later convert to either log or linear frequency scale.
  The super scale is similar to the bark scale (R1) in this regard.

  Consider two variables: pitch p:[0,1] and frequency w:[0,1],
  related by a minimum frequency w0 << 1

                             log w             -dw
    w = w0^(1-p),    p = 1 - ------ ,    dp = --------
                             log w0           w log w0

  We want to define a super variable s:[0,1] whose metric dominates the others

    ds >> dp,dw

  We therefore require for some scaling factor A:(.5,1)(hopefully closer to 1)

                                   1                     log w
    ds = A (dp + dw)   = A (1 - -------- ) dw   = A (w - ------) + C
                                w log w0                 log w0

  Applying the boundary conditions s(0) = w0, s(1) = 1, we find

          1              1 - w0          1 + w - w0 - log w/log w0
    A = ------ ,     C = ------ ,    s = -------------------------
        2 - w0           2 - w0                   2 - w0

        1 + w - w0 - log w/log w0     p + w0^(1-p) - w0
    s = -------------------------   = -----------------
                 2 - w0                    2 - w0

  For typical w0 << 1, each of p,w has resolution A ~ 1/2 of s;
  Therefore we split the superspectrum s of size 2^n
  into two spectra p,w each of size 2^(n-1)

  References:
  (R1) "Bark and ERB Bilinear Transforms" -J.O. Smith III, J.S.Abel
*/

class SuperSpectrum
{
  const float m_min_freq;        // w0
  const float m_log_min_freq;    // log(w0)
  const float m_scale;           // 1 / (2 - w0)

  const float m_tolerance;

public:

  SuperSpectrum (
      float sample_rate_hz = DEFAULT_SAMPLE_RATE,
      float min_freq_hz = 20);

  // wicked fast
  float freq_to_super (float freq) const
  {
    // ASSERT_LT(0, freq);
    // ASSERT_LE(freq, 1);
    return m_scale * (1 + (freq - m_min_freq) - logf(freq) / m_log_min_freq);
  }
  float pitch_to_super (float pitch) const
  {
    return m_scale * (pitch + powf(m_min_freq, 1-pitch) - m_min_freq);
  }

  // wicked slow
  float super_to_freq (float super) const;
  float super_to_pitch (float super) const;
};

class SuperToFreq : public FunctionAndInverse
{
  const SuperSpectrum & m_spectrum;
public:
  SuperToFreq (const SuperSpectrum & spectrum) : m_spectrum(spectrum) {}
  virtual float value (float s) const { return m_spectrum.super_to_freq(s); }
  virtual float inverse (float f) const { return m_spectrum.freq_to_super(f); }
};

class SuperToPitch : public FunctionAndInverse
{
  const SuperSpectrum & m_spectrum;
public:
  SuperToPitch (const SuperSpectrum & spectrum) : m_spectrum(spectrum) {}
  virtual float value (float s) const { return m_spectrum.super_to_pitch(s); }
  virtual float inverse (float p) const { return m_spectrum.pitch_to_super(p); }
};

//----( supersampled reassignment )-------------------------------------------

/** Supersampled Time-Frequency Reassignment.

  Sample m_freq_factor FFTs of size m_size,
   using m_time_factor windows of size m_size / m_time_factor = m_small_size.
  Accumulate these into a superresolution reassignment representation
   with m_size * m_freq_factor = m_large_size frequency samples
   at each of m_time_factor time samples.
*/

class Supergram
{
  const size_t m_size;
  const size_t m_time_factor;
  const size_t m_freq_factor;
  const size_t m_small_size;
  const size_t m_large_size;
  const size_t m_energy_size;
  const size_t m_super_size;
  const float m_sample_rate;

  const float m_tolerance;

  FFT_C2C m_fft;

  // windowing functions & variants, 2x window size
  Vector<float> m_h;      // basic window (Hann or Blackman-Nuttall)
  Vector<float> m_th;     // time * basic
  Vector<float> m_dh;     // d/dt basic
  Vector<float> m_synth;  // narrow window
  Vector<complex> m_time_in;

  // blocks aliasing the above
  Vector<float> m_h_old;
  Vector<float> m_h_new;
  Vector<float> m_th_old;
  Vector<float> m_th_new;
  Vector<float> m_dh_old;
  Vector<float> m_dh_new;
  Vector<complex> m_time_old;
  Vector<complex> m_time_new;

  // multiple ffts
  Vector<complex> m_freq_h;
  Vector<complex> m_freq_th;
  Vector<complex> m_freq_dh;

  // energy accumulation
  SuperSpectrum m_spectrum;
  CyclicBuffer<float> m_accumulator;
  const Spline * m_super_to_freq;

  // phase matching
  Vector<float> m_energy_in;
  Vector<complex> m_freq_large;
  Vector<complex> m_blurred;
  Blur<complex> m_freq_blur;
  Vector<complex> m_time_out;

public:

  Supergram (
      size_t size_exponent = 10,
      size_t time_exponent = 2,
      size_t freq_exponent = 2,
      float sample_rate = DEFAULT_SAMPLE_RATE);

  ~Supergram () { delete m_super_to_freq; }

  // diagnostics
  size_t size () const { return m_size; }
  size_t time_factor() const { return m_time_factor; }
  size_t freq_factor() const { return m_freq_factor; }
  size_t small_size () const { return m_small_size; }
  size_t large_size () const { return m_large_size; }
  size_t energy_size() const { return m_energy_size; }
  size_t super_size () const { return m_super_size; }
  size_t pitch_size () const { return m_super_size / 2; }
  size_t freq_size  () const { return m_super_size / 2; }
  float sample_rate    () const { return m_sample_rate; }
  float frame_rate     () const { return m_sample_rate / m_small_size; }
  const Vector<float> &  weights () const { return m_h; }
  const Vector<float> & tweights () const { return m_th; }
  const Vector<float> & dweights () const { return m_dh; }
  const Vector<float> & synth    () const { return m_synth; }
  const SuperSpectrum & spectrum () const { return m_spectrum; }

  // these can operate concurrently
  void transform_fwd (const Vector<complex> & time_in,
                      Vector<float> & super_out);
  void transform_bwd (const Vector<float> & super_in,
                      Vector<complex> & time_out);

  // conversions to other scales
  Spline * new_FreqScale (size_t size_out = 0) const;
  Spline * new_PitchScale (size_t size_out = 0) const;
};

#endif // KAZOO_REASSIGN_H

