#ifndef KAZOO_SPECTROGRAM_H
#define KAZOO_SPECTROGRAM_H

/** Invertible Spectrogram Stream.

  Notes:
  (N1) We use a Hann window function.
    see http://en.wikipedia.org/wiki/Window_function
  (N2) Input is stereo-as-complex stream.

*/

#include "common.h"
#include "fft.h"
#include "vectors.h"
#include "window.h"

class Spectrogram
{
  const size_t m_size;

  const float m_scale;
  FFT_C2C m_fft;

  Vector<float> m_weights;
  Vector<complex> m_old_time_in;
  Vector<complex> m_old_time_out;
  Vector<complex> m_freq_in;
  Vector<complex> m_freq_out;

public:

  Spectrogram (size_t exponent = 10);

  // diagnostics
  size_t size () const { return m_size; }
  const Vector<float> & weights () const { return m_weights; }

  // short-time fourier transforms
  // these can operate concurrently
  void stft_fwd (const Vector<complex> & time_in,
                 Vector<complex> & freq_out);
  void stft_bwd (const Vector<complex> & freq_in,
                 Vector<complex> & time_out);

  // these can operate concurrently
  void transform_fwd (const Vector<complex> & time_in,
                      Vector<float> & freq_out);
  void transform_bwd (const Vector<float> & freq_in,
                      Vector<complex> & time_out);
};

#endif // KAZOO_SPECTROGRAM_H

