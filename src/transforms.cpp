
#include "transforms.h"

#define LOG1(mess)

//----( highpass/lowpass splitter )-------------------------------------------

HiLoSplitter::HiLoSplitter (size_t size,
                            size_t size_lowpass)
  : m_size(size),
    m_size_lowpass(size_lowpass),

    m_spline(m_size, m_size_lowpass),

    m_exponent(80) // should be < 88 for single precision floats
{}

void HiLoSplitter::transform_fwd (const Vector<float> & full_in,
                                  Vector<float> & high_out,
                                  Vector<float> & low_out) const
{
  ASSERT_SIZE(full_in, m_size);
  ASSERT_SIZE(high_out, m_size);
  ASSERT_SIZE(low_out, m_size_lowpass);

  for (size_t i = 0; i < m_size; ++i) {
    high_out[i] = map_fwd(full_in[i]);
  }
  m_spline.transform_fwd(high_out, low_out);
  m_spline.transform_bwd(low_out, high_out);
  for (size_t i = 0; i < m_size_lowpass; ++i) {
    low_out[i] = map_bwd(low_out[i]);
  }
  for (size_t i = 0; i < m_size; ++i) {
    high_out[i] = safe_div(full_in[i], map_bwd(high_out[i]));
  }
}

void HiLoSplitter::transform_bwd (const Vector<float> & high_in,
                                  Vector<float> & low_in,
                                  Vector<float> & full_out) const
{
  ASSERT_SIZE(high_in, m_size);
  ASSERT_SIZE(low_in, m_size_lowpass);
  ASSERT_SIZE(full_out, m_size);

  for (size_t i = 0; i < m_size_lowpass; ++i) {
    low_in[i] = map_fwd(low_in[i]);
  }
  m_spline.transform_bwd(low_in, full_out);
  for (size_t i = 0; i < m_size; ++i) {
    full_out[i] = high_in[i] * map_bwd(full_out[i]);
  }
}

float HiLoSplitter::test () const
{
  LOG("Testing HiLoSplitter(" << m_size << ", " << m_size_lowpass << ")");

  Vector<float> full_in(m_size);
  Vector<float> full_out(m_size);
  Vector<float> high(m_size);
  Vector<float> low(m_size_lowpass);

  for (size_t i = 0; i < m_size; ++i) {
    float t = (0.5f + i) / m_size;
    full_in[i] = sqr(random_std() + 8 * sin(2 * M_PI * t));
  }
  transform_fwd(full_in, high, low);
  transform_bwd(high, low, full_out);

  float RMS_error = rms_error(full_in, full_out);
  LOG("RMS inversion error = " << RMS_error << " (should be tiny)");
  return RMS_error;
}

//----( multiscale splitter )-------------------------------------------------

MultiScale::MultiScale (const Spline * super_to_fst,
                        const Spline * super_to_snd)
  : m_size_super(super_to_fst->size_in()),
    m_size_fst(super_to_fst->size_out()),
    m_size_snd(super_to_snd->size_out()),

    m_super_to_fst(super_to_fst),
    m_super_to_snd(super_to_fst),

    m_super_fst(m_size_super),
    m_super_snd(m_size_super)
{
  ASSERT(super_to_fst->size_in() == super_to_snd->size_in(),
         "super scale sizes disagree: "
         << super_to_fst->size_in() << " != " << super_to_snd->size_in());
}

void MultiScale::transform_fwd (const Vector<float> & super_in,
                                Vector<float> & fst_out,
                                Vector<float> & snd_out)
{
  ASSERT_SIZE(super_in, m_size_super);
  ASSERT_SIZE(fst_out, m_size_fst);
  ASSERT_SIZE(snd_out, m_size_snd);

  m_super_to_fst->transform_fwd(super_in, fst_out);
  m_super_to_snd->transform_fwd(super_in, snd_out);
}

void MultiScale::transform_bwd (Vector<float> & fst_io,
                                Vector<float> & snd_io,
                                Vector<float> & super_out)
{
  //TODO("estimate or input covariances and fuse via Wiener filter");
  ASSERT_SIZE(fst_io, m_size_fst);
  ASSERT_SIZE(snd_io, m_size_snd);
  ASSERT_SIZE(super_out, m_size_super);

  // fuse via averaging
  m_super_to_fst->transform_bwd(fst_io, m_super_fst);
  m_super_to_snd->transform_bwd(snd_io, m_super_snd);

  for (size_t i = 0; i < m_size_super; ++i) {
    super_out[i] = (m_super_fst[i] + m_super_snd[i]) / 2;
  }

  // propagate fused info back to sources
  m_super_to_fst->transform_fwd(super_out, fst_io);
  m_super_to_snd->transform_fwd(super_out, snd_io);
}

//----( shepard scale )-------------------------------------------------------

inline float range_in_halftones (float min_freq_hz, float max_freq_hz)
{
  return 12 * log(max_freq_hz / min_freq_hz) / log(2);
}

inline float freq_to_halftone (float freq)
{
  return 12 * log(freq) / log(2);
}

Shepard::Shepard (size_t size_in,
                  float min_freq_hz,
                  float max_freq_hz)
  : m_size_in(size_in),

    m_mid_fwd(e_size_mid),
    m_mid_bwd(e_size_mid),
    m_wrap(m_size_in,
           e_size_mid,
           range_in_halftones(min_freq_hz, max_freq_hz) * e_align_factor
                                                        / size_in),

    m_tone_alignment(0),
    m_align(e_size_mid, e_size_out, 1.0 / e_align_factor)
{
  // initialize alignment to a440
  float dtone = freq_to_halftone(max_freq_hz / min_freq_hz) / size_in;
  float tone0 = freq_to_halftone(min_freq_hz) + 0.5f * dtone;
  float center = freq_to_halftone(440);
  m_tone_alignment = fmod((center - tone0) / dtone, 1.0);
  LOG("tone alignment = " << m_tone_alignment);
}

void Shepard::align (float alignment_rate)
{
  complex observed_phase = 0;
  for (size_t i = 0; i < e_size_mid; ++i) {
    float theta = static_cast<float>(i % e_align_factor) / e_align_factor;
    observed_phase += m_mid_fwd[i] * exp_2_pi_i(theta);
  }
  float observed_norm = norm(observed_phase);
  if (not (observed_norm > 0)) return;
  observed_phase *= alignment_rate / sqrtf(observed_norm);

  complex predicted_phase = exp_2_pi_i(m_tone_alignment);
  m_tone_alignment = arg(predicted_phase + observed_phase) / (2 * M_PI);

//#define LOG_ALIGNMENT
#ifdef LOG_ALIGNMENT
  static size_t whether_to_log = 0;
  if (not (whether_to_log = (whether_to_log+1) % 128)) {
    LOG("alignment = " << m_tone_alignment);
  }
#endif // LOG_ALIGNMENT

  m_alignment_mutex.lock();
  m_align.setup(1.0 / e_align_factor,
                -m_tone_alignment / e_align_factor);
  m_alignment_mutex.unlock();
}

void Shepard::transform_fwd (const Vector<float> & pitch_in,
                             Vector<float> & tone_out)
{
  ASSERT_SIZE(pitch_in, m_size_in);
  ASSERT_SIZE(tone_out, e_size_out);

  m_wrap.transform_fwd(pitch_in, m_mid_fwd);

  align();

  m_align.transform_fwd(m_mid_fwd, tone_out);
}

void Shepard::transform_bwd (const Vector<float> & tone_in,
                             Vector<float> & pitch_out)
{
  ASSERT_SIZE(tone_in, e_size_out);
  ASSERT_SIZE(pitch_out, m_size_in);

  m_alignment_mutex.lock();
  m_align.transform_bwd(tone_in, m_mid_bwd);
  m_alignment_mutex.unlock();

  m_wrap.transform_bwd(m_mid_bwd, pitch_out);
}

//----( sharpener )-----------------------------------------------------------

Sharpener::Sharpener (size_t size)
  : m_size(size)
{
  ASSERT(size >= 2,
         "Sharpener is too small: size = " << size);
}

void Sharpener::transform (const Vector<float> & data_in,
                           Vector<float> & data_out)
{
  ASSERT_SIZE(data_in, m_size);
  ASSERT_SIZE(data_out, m_size);

  data_out.zero();

  // decide where to put mass at zero
  {
    float center = data_in[0];
    float right  = data_in[1];

    if (right > center) {
      data_out[1] += center;
    } else {
      data_out[0] += center;
    }
  }

  // decide where to put mass at i
  for (size_t i = 1; i+1 < m_size; ++i) {
    float left   = data_in[i-1];
    float center = data_in[i];
    float right  = data_in[i+1];

    if (left > center) {
      if (right > center) {
        if (left > right) {
          data_out[i-1] = center;
        } else {
          data_out[i+1] = center;
        }
      } else {
        data_out[i-1] += center;
      }
    } else {
      if (right > center) {
        data_out[i+1] += center;
      } else {
        data_out[i] += center;
      }
    }
  }

  // decide where to put mass at m_size-1
  {
    float left   = data_in[m_size-2];
    float center = data_in[m_size-1];

    if (left > center) {
      data_out[m_size-2] += center;
    } else {
      data_out[m_size-1] += center;
    }
  }
}

//----( spline sharpener )----------------------------------------------------

Sharpen1D::Sharpen1D (size_t size, size_t radius)
  : m_size(size),
    m_radius(radius),
    m_min_mass(1e-6),

    m_mass_blur(size),
    m_freq_blur(size),
    m_temp_blur(size)
{
  ASSERT_LT(0, radius);
}

/** Running-sum method of rectangular convolution
*/

void convolve_block (Vector<float> & x, Vector<float> & temp, size_t radius)
{
  ASSERT_EQ(x.size, temp.size);
  ASSERT_LT(radius, x.size);

  double running_sum = 0;
  for (size_t i = 0; i < radius; ++i) {
    running_sum += x[i];
  }
  for (size_t i = 0; i < x.size; ++i) {
    if (i >= radius)         running_sum -= x[i - radius];
    if (i + radius < x.size) running_sum += x[i + radius];
    temp[i] = running_sum;
  }
  x = temp;
}

void Sharpen1D::transform (const Vector<float> & mass_in,
                           const Vector<float> & freq_in,
                           Vector<float> & freq_out)
{
  // compute moments
  m_mass_blur.zero();
  m_freq_blur.zero();
  for (size_t i = 0; i < m_size; ++i) {
    float mass = mass_in[i];

    m_mass_blur[i] += mass;
    m_freq_blur[i] += mass * i;
  }
  convolve_block(m_mass_blur, m_temp_blur, m_radius);
  convolve_block(m_freq_blur, m_temp_blur, m_radius);

  // distribute mass
  freq_out.zero();
  for (size_t i = 0; i < m_size; ++i) {
    float m = m_mass_blur[i];
    if (m < m_min_mass) continue;

    float f = m_freq_blur[i] / m;

    int i0;
    float w0, w1;
    linear_interpolate(f, m_size,   i0,w0,w1);
    size_t i1 = i0 + 1;

    float mass = freq_in[i];

    freq_out[i0] += mass * w0;
    freq_out[i1] += mass * w1;
  }
}

//----( spline sharpener )----------------------------------------------------

Sharpen2D::Sharpen2D (size_t size, size_t radius)
  : m_size(size),
    m_radius(radius),
    m_length(2 * m_radius + 1),
    m_min_mass(1e-6),

    m_mass_in(size * m_length),

    m_mass_blur(size),
    m_time_blur(size),
    m_freq_blur(size),
    m_temp_blur(size),

    m_mass_out(size * m_length)
{
  ASSERT_LT(0, radius);

  m_mass_in.zero();
  m_mass_out.zero();
}

void Sharpen2D::transform (const Vector<float> & freq_in,
                           Vector<float> & freq_out)
{
  // push new frame
  for (size_t i = 0, j = 1; j < m_length; ++i, ++j) {
    m_mass_in.block(m_size, i) = m_mass_in.block(m_size, j);
    m_mass_out.block(m_size, i) = m_mass_out.block(m_size, j);
  }
  m_mass_in.block(m_size, m_length - 1) = freq_in;
  m_mass_out.block(m_size, m_length - 1).zero();

  // compute moments
  m_mass_blur.zero();
  m_time_blur.zero();
  m_freq_blur.zero();
  for (size_t i = 0; i < m_size; ++i) {
    for (size_t j = 0; j < m_length; ++j) {
      size_t ij = i + j * m_size;
      float mass = m_mass_in[ij];

      m_mass_blur[i] += mass;
      m_time_blur[i] += mass * j;
      m_freq_blur[i] += mass * i;
    }
  }
  convolve_block(m_mass_blur, m_temp_blur, m_radius);
  convolve_block(m_time_blur, m_temp_blur, m_radius);
  convolve_block(m_freq_blur, m_temp_blur, m_radius);

  // distribute mass from center
  Vector<float> center = m_mass_in.block(m_size, m_radius);
  for (size_t i = 0; i < m_size; ++i) {
    float m = m_mass_blur[i];
    if (m < m_min_mass) continue;

    float t = m_time_blur[i] / m;
    float f = m_freq_blur[i] / m;

    int i0_;
    int i_0;

    float w0_, w1_;
    float w_0, w_1;

    linear_interpolate(f, m_size,   i0_,w0_,w1_);
    linear_interpolate(t, m_length, i_0,w_0,w_1);

    size_t i1_ = i0_ + 1;
    size_t i_1 = i_0 + 1;

    float mass = center[i];

    m_mass_out[i0_ + i_0 * m_size] += mass * w0_ * w_0;
    m_mass_out[i0_ + i_1 * m_size] += mass * w0_ * w_1;
    m_mass_out[i1_ + i_0 * m_size] += mass * w1_ * w_0;
    m_mass_out[i1_ + i_1 * m_size] += mass * w1_ * w_1;
  }

  // pop oldest frame
  freq_out = m_mass_out.block(m_size);
}

//----( auto correlation )----------------------------------------------------

AutoCorrelation::AutoCorrelation (size_t exponent)
  : m_size(1 << exponent),
    m_fft(1 + exponent)
{}

void AutoCorrelation::transform (const Vector<float> & f_in,
                                 Vector<float> & ff_out)
{
  ASSERT_SIZE(f_in, m_size);
  ASSERT_SIZE(ff_out, m_size);

  m_fft.time_in.zero();
  m_fft.time_in.block(m_size / 2, 0) = f_in.block(m_size / 2, 0);
  m_fft.time_in.block(m_size / 2, 3) = f_in.block(m_size / 2, 1);
  m_fft.transform_fwd();

  for (size_t i = 0; i < m_fft.freq_out.size; ++i) {
    m_fft.freq_out[i] = norm(m_fft.freq_out[i]);
  }

  m_fft.transform_bwd();

  ff_out = m_fft.time_out.block(m_size);
}

//----( cross correlation )---------------------------------------------------

CrossCorrelation::CrossCorrelation (size_t exponent_in)
  : m_size_in(1 << exponent_in),
    m_size_out(1 << (1 + exponent_in)),
    m_fft_fwd(1 + exponent_in),
    m_fft_bwd(1 + exponent_in),
    m_freq_fwd(m_size_in + 1),
    m_freq_bwd(m_size_in + 1)
{}

void CrossCorrelation::fft_fwd_in (const Vector<float> & time_in)
{
  m_fft_fwd.time_in.zero();
  m_fft_fwd.time_in.block(m_size_in / 2, 0) = time_in.block(m_size_in / 2, 0);
  m_fft_fwd.time_in.block(m_size_in / 2, 3) = time_in.block(m_size_in / 2, 1);
  m_fft_fwd.transform_fwd();
}

void CrossCorrelation::fft_fwd_out (Vector<float> & time_out) const
{
  m_fft_fwd.transform_bwd();
  time_out = m_fft_fwd.time_out;
}

void CrossCorrelation::transform_fwd (const Vector<float> & f_in,
                                      const Vector<float> & g_in,
                                      Vector<float> & fg_out)
{
  ASSERT_SIZE(f_in, m_size_in);
  ASSERT_SIZE(g_in, m_size_in);
  ASSERT_SIZE(fg_out, m_size_out); // is this right?

  fft_fwd_in(f_in);
  multiply(pow(m_size_out,-2), m_fft_fwd.freq_out, m_freq_fwd);
  fft_fwd_in(g_in);
  multiply_conj(m_freq_fwd, m_fft_fwd.freq_out, m_fft_fwd.freq_in);
  fft_fwd_out(fg_out);
}

/** Iterative deconvolution.
  Goal: solve the inverse problem
    fg ~ f*g
     f ~ f0
     g = (given)
  Method: Wiener filter
    ???
*/
void CrossCorrelation::transform_bwd (const Vector<float> & fg_in,
                                      const Vector<float> & f_in,
                                      Vector<float> & g_out)
{
  ASSERT_SIZE(fg_in, m_size_out);
  ASSERT_SIZE(f_in, m_size_in);
  ASSERT_SIZE(g_out, m_size_in);

  TODO("deal with deconvolution ill-posedness");
}

//----( laplace filterbank )--------------------------------------------------

LaplaceFilterBank::LaplaceFilterBank (size_t vect_size,
                                      size_t bank_size,
                                      float min_time_scale,
                                      float max_time_scale)
  : m_vect_size(vect_size),
    m_bank_size(bank_size),
    m_factors(bank_size),
    m_bank(bank_size * vect_size)
{
  float scale_step = pow(
      max_time_scale / min_time_scale,
      1.0 / (bank_size - 1));
  for (size_t b = 0; b < m_bank_size; ++b) {
    float time_scale = min_time_scale * pow(scale_step, b);
    m_factors[b] = exp(-1 / time_scale);
  }
  m_bank.zero();
}

void LaplaceFilterBank::accumulate (const Vector<float> & sample)
{
  ASSERT_SIZE(sample, m_vect_size);

  for (size_t b = 0; b < m_bank_size; ++b) {
    Vector<float> bank_b = m_bank.block(m_vect_size, b);
    accumulate_step(m_factors[b], bank_b, sample);
  }
}

//----( Laplace-Fourier transform )-------------------------------------------

LaplaceFourier::LaplaceFourier (size_t size_in,
                                size_t num_freqs,
                                complex min_freq,
                                complex max_freq)
  : m_size_in(size_in),
    m_num_freqs(num_freqs),

    m_coeff_new(num_freqs),
    m_coeff_old(num_freqs),

    m_history_fwd(size_in * num_freqs),
    m_history_bwd(size_in * num_freqs)
{
  // define coefficients
  m_info = 0;
  for (size_t i = 0; i < num_freqs; ++i) {
    float t = i / (num_freqs-1.0f);
    complex freq = min_freq * pow(max_freq / min_freq, t);
    float scale = abs(complex(1,0)-sqr(freq)); // normalizes geometric series

    m_info += scale;
    m_coeff_new[i] = scale;
    m_coeff_old[i] = freq;
  }

  m_info = 1.0 / m_info;
}

void  LaplaceFourier::accumulate (
    Vector<complex> & history,
    const Vector<complex> & data)
{
  for (size_t i = 0; i < m_num_freqs; ++i) {
    complex coeff_new = m_coeff_new[i];
    complex coeff_old = m_coeff_old[i];
    for (size_t j = 0; j < m_size_in; ++j) {
      size_t ij = m_size_in * i + j;
      complex & h = history[ij];
      h = coeff_old * h + coeff_new * data[j];
    }
  }
}

void LaplaceFourier::transform_fwd (const Vector<complex> & data_in,
                                    Vector<float> & energy_out)
{
  ASSERT_SIZE(data_in, m_size_in);
  ASSERT_SIZE(energy_out, size_out());

  accumulate(m_history_fwd, data_in);

  for (size_t i = 0; i < size_out(); ++i) {
    energy_out[i] = norm(m_history_fwd[i]);
  }
}

void LaplaceFourier::transform_bwd (const Vector<float> & energy_in,
                                    Vector<complex> & data_out)
{
  ASSERT_SIZE(energy_in, size_out());
  ASSERT_SIZE(data_out, m_size_in);

  data_out.zero();

  for (size_t i = 0; i < m_num_freqs; ++i) {
    complex coeff_new = m_coeff_new[i];
    complex coeff_old = m_coeff_old[i];
    for (size_t j = 0; j < m_size_in; ++j) {
      size_t ij = m_size_in * i + j;

      // find min-energy phase
      complex val = m_history_bwd[ij] * coeff_old;
      float norm_val = norm(val); if (not (norm_val > 0)) continue;
      float scale = m_info * (sqrtf(energy_in[ij]) - sqrtf(norm_val));

      data_out[j] += scale * val;
    }
  }

  accumulate(m_history_bwd, data_out);
}

//----( melodigram )----------------------------------------------------------

Melodigram::Melodigram (size_t exponent,
                        size_t num_filters,
                        float frame_rate,
                        float min_time_scale_sec,
                        float max_time_scale_sec)
  : m_size(1 << exponent),
    m_size_corr(2 << exponent),
    m_num_filters(num_filters),

    m_filters_fwd(m_size,
                  num_filters,
                  min_time_scale_sec * frame_rate,
                  max_time_scale_sec * frame_rate),
    m_filters_bwd(m_size,
                  num_filters,
                  min_time_scale_sec * frame_rate,
                  max_time_scale_sec * frame_rate),
    m_correlation(exponent)
{}

void Melodigram::transform_fwd (const Vector<float> & pitch_in,
                                Vector<float> & corr_out)
{
  ASSERT_SIZE(pitch_in, m_size);
  ASSERT_SIZE(corr_out, m_size_corr);

  for (size_t b = 0; b < m_num_filters; ++b) {

    //TODO("maybe swap to get sign right");
    Vector<float> block = corr_out.block(m_size_corr, b);
    m_correlation.transform_fwd(m_filters_fwd[b],
                                pitch_in,
                                block);
  }

  m_filters_fwd.accumulate(pitch_in);
}

void Melodigram::transform_bwd (const Vector<float> & prev_pitch_in,
                                const Vector<float> & corr_in,
                                Vector<float> & pitch_out)
{
  ASSERT_SIZE(corr_in, m_size_corr);
  ASSERT_SIZE(pitch_out, m_size);

  m_filters_bwd.accumulate(prev_pitch_in);

  TODO("compute estimate & covariance");
}

//----( harmonics )-----------------------------------------------------------

Harmonic::Harmonic (size_t exponent)
  : m_size(1 << exponent),
    m_fft(1 + exponent)
{
  LOG("building Harmonic(" << exponent << ")");
}

void Harmonic::adjust (Vector<float> & space_in, Vector<float> & dspace_out)
{
  ASSERT_SIZE(space_in, m_size);
  ASSERT_SIZE(dspace_out, m_size);

  size_t size_in = m_fft.size_in();
  size_t size_out = m_fft.size_out();

  LOG1("compute autocorrelation");
  float total = 0;
  m_fft.time_in.zero();
  for (size_t i = 1; i < m_size; ++i) {
    total
      += m_fft.time_in[i]
      = m_fft.time_in[size_in - i]
      = map_fwd(space_in[i]);
  }
  m_fft.time_in[0] = total;
  m_fft.transform_fwd();

  LOG1("compute adjustment");
  for (size_t i = 0; i < size_out; ++i) {
    complex prev = m_fft.freq_out[(i + size_out - 1) % size_out];
    complex here = m_fft.freq_out[i];
    complex next = m_fft.freq_out[(i + 1) % size_out];
    m_fft.freq_in[i] = 2.0f * here - prev - next;
  }
  m_fft.transform_bwd();

  for (size_t i = 1; i < m_size; ++i) {
    dspace_out[i] = m_fft.time_out[i];
  }
  dspace_out[0] = 0;
}

//----( rhythmgram )----------------------------------------------------------

Rhythmgram::Rhythmgram (size_t size_in,
                        size_t size_factor,
                        float num_beats)
  : m_size_in(size_in),
    m_size_factor(size_factor),

    m_coeff_new(size_factor),
    m_coeff_old(size_factor),
    m_history_fwd(size_in * size_factor),
    m_history_bwd(size_in * size_factor)
{
  ASSERT((0.5 <= num_beats) and (num_beats <= 32),
         "time scale out of range: " << num_beats << " not in [1/2, 32]");

  // initialize decay factors
  for (size_t i = 0; i < m_size_factor; ++i) {
    float freq = (1.0f + i) / (2.0f * size_factor);  // freq in (zero, nyquist]
    float old_part = pow(1 + 1/num_beats, -freq);
    float new_part = 1 - old_part;

    m_coeff_new[i] = new_part;
    m_coeff_old[i] = old_part * exp_2_pi_i(freq);
  }
}

void Rhythmgram::transform_fwd (const Vector<float> & value_in,
                                Vector<float> & tempo_out)
{
  ASSERT_SIZE(value_in, m_size_in);
  ASSERT_SIZE(tempo_out, size_out());

  const float    * restrict value     = value_in;
        float    * restrict tempo     = tempo_out;
  const float    * restrict coeff_new = m_coeff_new;
  const complex * restrict coeff_old = m_coeff_old;
        complex * restrict history   = m_history_fwd;
  const size_t I = m_size_in;
  const size_t J = m_size_factor;

  for (size_t i = 0; i < I; ++i) {
    for (size_t j = 0; j < J; ++j) {

      complex & h = history[J * i + j];

      h = coeff_new[j] * value[i] + coeff_old[j] * h;
    }
  }

  for (size_t i1 = 0; i1 < I; ++i1) {
    for (size_t i2 = 0; i2 < I; ++i2) {
      for (size_t j = 0; j < J; ++j) {

        complex h1 = history[J * i1 + j];
        complex h2 = history[J * i1 + j];
        float & t = tempo[J * (i1 + J * (i2 + j))];

        t = h1.real() * h2.real() - h1.imag() * h2.imag();
      }
    }
  }
}

void Rhythmgram::transform_bwd (const Vector<float> & tempo_in,
                                Vector<float> & value_out)
{
  ASSERT_SIZE(tempo_in, size_out());
  ASSERT_SIZE(value_out, m_size_in);

  TODO("deconvolve");

  /*
  const float    * restrict tempo     = tempo_in;
        float    * restrict value     = value_out;
  const float    * restrict coeff_new = m_coeff_new;
  const complex * restrict coeff_old = m_coeff_old;
        complex * restrict history   = m_history_bwd;
  */
}

//----( correlogram )---------------------------------------------------------

Correlogram::Correlogram (size_t exponent_in,
                          float decay_factor)
  : m_size_in(1 << exponent_in),
    m_size_out(2 << exponent_in),
    m_decay_factor(decay_factor),

    m_history_fwd(m_size_in),
    m_history_bwd(m_size_in),

    m_correlation(exponent_in)
{
  ASSERT((0 < decay_factor) and (decay_factor < 1),
         "decay factor out of range: " << decay_factor);

  m_history_fwd.set(sqrt(1.0 / m_size_in));
  m_history_bwd.set(sqrt(1.0 / m_size_in));
}

void Correlogram::transform_fwd (const Vector<float> & freq_in,
                                 Vector<float> & corr_out)
{
  ASSERT_SIZE(freq_in, m_size_in);
  ASSERT_SIZE(corr_out, m_size_out);

  m_correlation.transform_fwd(m_history_fwd, freq_in, corr_out);

  accumulate_step(m_decay_factor, m_history_fwd, freq_in);
}

void Correlogram::transform_bwd (const Vector<float> & prev_freq_in,
                                 const Vector<float> & corr_in,
                                 Vector<float> & freq_out)
{
  ASSERT_SIZE(corr_in, m_size_out);
  ASSERT_SIZE(freq_out, m_size_in);

  accumulate_step(m_decay_factor, m_history_bwd, prev_freq_in);

  m_correlation.transform_bwd(m_history_bwd, corr_in, freq_out);
}

//----( pitch shifting )------------------------------------------------------

OctaveLower::OctaveLower (size_t size)
  : m_size(size),
    m_window(2 * size),
    m_sound_in(2 * size),
    m_sound_out(2 * size)
{
  for (size_t i = 0; i < 2*m_size; ++i) {
    float t = (0.5f + i) / m_size - 1.0f;
    m_window[i] = window_Hann(t);
  }
  m_sound_in.zero();
  m_sound_out.zero();
}

void OctaveLower::transform (const Vector<complex> & sound_in,
                             Vector<complex> & sound_out)
{
  ASSERT_SIZE(sound_in, m_size);
  ASSERT_SIZE(sound_out, m_size);

  // shift window
  m_sound_out.block(m_size, 0) = m_sound_out.block(m_size, 1);
  m_sound_out.block(m_size, 1).zero();

  // expand sound by factor of 2
  m_sound_in[0] = 0.75f * sound_in[0];
  for (size_t i = 1; i < m_size; ++i) {
    complex x0 = sound_in[i];
    complex x1 = sound_in[i+1];

    m_sound_in[2*i-1] = 0.75f * x0 + 0.25f * x1;
    m_sound_in[2*i  ] = 0.25f * x0 + 0.75f * x1;
  }
  m_sound_in[2*m_size-1] = 0.75f * sound_in[m_size-1];

#define FIND_OPTIMAL_PHASE
#ifdef FIND_OPTIMAL_PHASE
  // find optimal phase
  complex phase = 0;
  for (size_t i = 0; i < m_size; ++i) {
    phase += m_window[i] * conj(m_sound_in[i]) * m_sound_out[i];
  }
  float norm_phase = norm(phase);
  if (norm_phase > 0) {
    phase *= pow(norm_phase, -0.5);
    for (size_t i = 0; i < 2*m_size; ++i) {
      m_sound_in[i] *= phase;
    }
  }
#endif // FIND_OPTIMAL_PHASE

  // window and add to result
  multiply_add(m_window, m_sound_in, m_sound_out);
  sound_out = m_sound_out.block(m_size);
}

/* OLD
void PitchShift::transform (const Vector<complex> & sound_in,
                            Vector<complex> & sound_out)
{
  ASSERT_SIZE(sound_in, size());
  ASSERT_SIZE(sound_out, size());

  m_supergram.transform_fwd(sound_in, m_super_in);
  m_spline.transform_fwd(m_super_in, m_super_out);
  m_supergram.transform_bwd(m_super_out, sound_out);
}
*/

//----( looping )-------------------------------------------------------------

Loop::Loop (size_t size, size_t length, float timescale)
  : m_size(size),
    m_length(length),
    m_decay(pow(0.5, -1 / timescale)),
    m_memory(length * size),
    m_phase(0)
{
  ASSERT_LT(0, length);

  m_memory.zero();
}

void Loop::transform (const Vector<float> & energy_in,
                      Vector<float> & energy_out)
{
  ASSERT_SIZE(energy_in, size());
  ASSERT_SIZE(energy_out, size());

  Vector<float> memory = m_memory.block(length(), m_phase);
  accumulate_step(m_decay, memory, energy_in);

  m_phase = (m_phase + 1) % length(); // timestep here to counteract latency

  energy_out = m_memory.block(length(), m_phase);
}

//----( chorus )--------------------------------------------------------------

void Chorus::transform (const Vector<float> & sound_in,
                        Vector<float> & sound_out)
{
  affine_combine(m_pitch_decay, m_state, sound_in, m_mass);
  accumulate_step(m_loud_decay, m_state, sound_in);
  m_sharpen.transform(m_mass, m_state, sound_out);
  m_state = sound_out;
}

