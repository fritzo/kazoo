
#include "spectrogram.h"
#include <cstring>
#include <algorithm>

#define LOG1(mess)

//#define NO_BLEND

Spectrogram::Spectrogram (size_t exponent)
  : m_size(1 << exponent),
    m_scale(pow(m_size, -0.5)),
    m_fft(exponent),
    m_weights(m_size),
    m_old_time_in(m_size),
    m_old_time_out(m_size),
    m_freq_in(m_size),
    m_freq_out(m_size)
{
  // extrapolate to history of zeros
  m_old_time_in.zero();
  m_old_time_out.zero();

  // define weights on unit interval
  for (size_t i = 0; i < m_size; ++i) {
    float t = (0.5f + i) / m_size;
    m_weights[i] = window_Hann(t);
  }
}

void Spectrogram::stft_fwd (const Vector<complex> & time_in,
                                  Vector<complex> & freq_out)
{
  LOG1("transforming forward");

  // affine-combine new + old data; save data for later; transform

#ifdef NO_BLEND
  m_fft.time_in = time_in;
#else // NO_BLEND
  affine_combine(m_weights, time_in, m_old_time_in, m_fft.time_in);
  m_old_time_in = time_in;
#endif // NO_BLEND

  m_fft.transform_fwd();

  multiply(m_scale, m_fft.freq_out, freq_out);
}

void Spectrogram::stft_bwd (const Vector<complex> & freq_in,
                                  Vector<complex> & time_out)
{
  LOG1("transforming backward");

  // transform; affine-combine with old data; save transformed data for later

  multiply(m_scale, freq_in, m_fft.freq_in);
  m_fft.transform_bwd();

#ifdef NO_BLEND
  time_out = m_fft.time_out;
#else // NO_BLEND
  affine_combine(m_weights, m_old_time_out, m_fft.time_out, time_out);
  m_old_time_out = m_fft.time_out;
#endif // NO_BLEND

  soft_clip(time_out);
}

void Spectrogram::transform_fwd (const Vector<complex> & time_in,
                                       Vector<float> & freq_out)
{
  ASSERT_SIZE(time_in, m_size);
  ASSERT_SIZE(freq_out, m_size / 2);

  stft_fwd(time_in, m_freq_out);
  value_to_energy(m_freq_out, freq_out);
}

void Spectrogram::transform_bwd (const Vector<float> & freq_in,
                                       Vector<complex> & time_out)
{
  ASSERT_SIZE(freq_in, m_size / 2);
  ASSERT_SIZE(time_out, m_size);

  energy_to_value(freq_in, m_freq_in);
  stft_bwd(m_freq_in, time_out);
}

