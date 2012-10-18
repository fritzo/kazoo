
/** Invertible FFT objects wrapping FFWT3.

  FFTW3 = fastest fourier transform in the west, version 3.
        @ http://www.fftw.org

  Note on Precision:
    Single-precision (23-24 bits) is plenty for 16-bit audio data.
    This requires linking with -lfftw3f instesd of -lfftw3,
    and prepending fftwf_ instead of fftw_.
    See fftw manual ss. 4.3.2, pp. 21-22 for details.

  References:
  (R1) http://www.fftw.org/fftw3_doc/
  (R2) example spectrogram code at
    http://www.captain.at/howto-fftw-spectrograph.php

  TODO implement simplex fft for convolver (everything's duplex now)
  TODO make convolution output entire 2*size spectrum, not just middle size
*/

#ifndef KAZOO_FFT_H
#define KAZOO_FFT_H

#include "common.h"
#include "vectors.h"
#include <fftw3.h>

//----( misc functions )------------------------------------------------------

/** Conversions between energy <--> value as functions of frequency.
  Note:
  (N1) phase is dropped
  (N2) positive and negative frequencies are combined, so
    value_size == 2 * energy_size
  (N3) on output, positive and negative values are equal
*/

void value_to_energy (const Vector<complex> & value,
                      Vector<float> & energy);

void energy_to_value (const Vector<float> & energy,
                      Vector<complex> & value);

//----( fast fourier transform classes )--------------------------------------

class FFT_C2C
{
  const size_t m_exponent;
  const size_t m_size;

public:
  Vector<complex> time_in, time_out;
  Vector<complex> freq_in, freq_out;

private:
  fftwf_plan m_fwd_plan;
  fftwf_plan m_bwd_plan;

public:
  FFT_C2C (size_t exponent);
  ~FFT_C2C ();

  // diagnostics
  size_t size_in  () const { return m_size; }
  size_t size_out () const { return m_size; }

  // these can work concurrently
  void transform_fwd (void) { fftwf_execute(m_fwd_plan); }
  void transform_bwd (void) { fftwf_execute(m_bwd_plan); }
};

class FFT_R2C
{
  const size_t m_exponent;
  const size_t m_size;

public:
  Vector<float> time_in, time_out;
  Vector<complex> freq_in, freq_out;

private:
  fftwf_plan m_fwd_plan;
  fftwf_plan m_bwd_plan;

public:
  FFT_R2C (size_t exponent);
  ~FFT_R2C ();

  // diagnostics
  size_t size_in  () const { return m_size; }
  size_t size_out () const { return m_size/2 + 1; }

  // these can work concurrently
  void transform_fwd (void) { fftwf_execute(m_fwd_plan); }
  void transform_bwd (void) { fftwf_execute(m_bwd_plan); }
};

class FFT_CC2CC
{
  const size_t m_exponent0;
  const size_t m_exponent1;
  const size_t m_size0;
  const size_t m_size1;
  const size_t m_size;

public:
  Vector<complex> time_in, time_out;
  Vector<complex> freq_in, freq_out;

private:
  fftwf_plan m_fwd_plan;
  fftwf_plan m_bwd_plan;

public:
  FFT_CC2CC (size_t exponent0,
             size_t exponent1);
  ~FFT_CC2CC ();

  // diagnostics
  size_t size_in  () const { return m_size0 * m_size1; }
  size_t size_out () const { return m_size0 * m_size1; }

  // these can work concurrently
  void transform_fwd (void) { fftwf_execute(m_fwd_plan); }
  void transform_bwd (void) { fftwf_execute(m_bwd_plan); }
};

#endif // KAZOO_FFT_H

