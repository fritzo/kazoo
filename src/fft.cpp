
#include "fft.h"
#include "vectors.h"
#include <cstring>

#define LOG1(mess)

//----( misc functions )------------------------------------------------------

/** Conversions between energy <--> value as functions of frequency.
  Note:
  (N1) phase is dropped
  (N2) positive and negative frequencies are combined, so
    value_size == 2 * energy_size
  (N3) on output, all energy is positive, so
    stereo information is lost
*/

void value_to_energy (const Vector<complex> & value,
                      Vector<float> & energy)
{
  ASSERT_SIZE(value, 2 * energy.size);

  energy[0] = 0;
  for (int i = 1; i<static_cast < int>(energy.size); ++i) {
    energy[i] = norm(value[i]) + norm(value[value.size - i]);
  }
}

void energy_to_value (const Vector<float> & energy,
                      Vector<complex> & value)
{
  ASSERT_SIZE(value, 2 * energy.size);

  value.zero();
  for (size_t i = 1; i < energy.size; ++i) {
    value[i] = sqrtf(energy[i]);
  }
}

//----( fftw wrappers )-------------------------------------------------------

fftwf_plan make_plan (size_t size, complex * input,
                                     complex * output, int sign)
{
  return fftwf_plan_dft_1d(size,
                           reinterpret_cast<fftwf_complex*>(input),
                           reinterpret_cast<fftwf_complex*>(output),
                           sign,
                           FFTW_MEASURE);
}

fftwf_plan make_plan (size_t size, float * input, complex * output)
{
  return fftwf_plan_dft_r2c_1d(size,
                               input,
                               reinterpret_cast<fftwf_complex*>(output),
                               FFTW_MEASURE);
}

fftwf_plan make_plan (size_t size, complex * input, float * output)
{
  return fftwf_plan_dft_c2r_1d(size,
                               reinterpret_cast<fftwf_complex*>(input),
                               output,
                               FFTW_MEASURE);
}

fftwf_plan make_plan (size_t size0,
                      size_t size1,
                      complex * input,
                      complex * output,
                      int sign)
{
  return fftwf_plan_dft_2d(size0, size1,
                           reinterpret_cast<fftwf_complex*>(input),
                           reinterpret_cast<fftwf_complex*>(output),
                           sign,
                           FFTW_MEASURE);
}

//----( fast fourier transform classes )--------------------------------------

FFT_C2C::FFT_C2C (size_t exponent)
  : m_exponent(exponent),
    m_size(1 << exponent),
    time_in(m_size),
    time_out(m_size),
    freq_in(m_size),
    freq_out(m_size),
    m_fwd_plan(make_plan(m_size, time_in, freq_out, FFTW_FORWARD)),
    m_bwd_plan(make_plan(m_size, freq_in, time_out, FFTW_BACKWARD))
{
  LOG1("initializing fft transform");
}

FFT_C2C::~FFT_C2C ()
{
  fftwf_destroy_plan(m_fwd_plan);
  fftwf_destroy_plan(m_bwd_plan);
}

FFT_R2C::FFT_R2C (size_t exponent)
  : m_exponent(exponent),
    m_size(1 << exponent),
    time_in(m_size),
    time_out(m_size),
    freq_in(m_size/2 + 1),
    freq_out(m_size/2 + 1),
    m_fwd_plan(make_plan(m_size, time_in, freq_out)),
    m_bwd_plan(make_plan(m_size, freq_in, time_out))
{}

FFT_R2C::~FFT_R2C ()
{
  fftwf_destroy_plan(m_fwd_plan);
  fftwf_destroy_plan(m_bwd_plan);
}

FFT_CC2CC::FFT_CC2CC (size_t exponent0,
                      size_t exponent1)
  : m_exponent0(exponent0),
    m_exponent1(exponent1),
    m_size0(1 << exponent0),
    m_size1(1 << exponent1),
    m_size(m_size0 * m_size1),
    time_in(m_size),
    time_out(m_size),
    freq_in(m_size),
    freq_out(m_size),
    m_fwd_plan(make_plan(m_size0, m_size1, time_in, freq_out, FFTW_FORWARD)),
    m_bwd_plan(make_plan(m_size0, m_size1, freq_in, time_out, FFTW_BACKWARD))
{
  LOG1("initializing fft transform");
}

FFT_CC2CC::~FFT_CC2CC ()
{
  fftwf_destroy_plan(m_fwd_plan);
  fftwf_destroy_plan(m_bwd_plan);
}

