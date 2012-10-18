
#include "raster_audio.h"

namespace RasterAudio
{

void random_phase (Vector<float> & re, Vector<float> & im)
{
  ASSERT_EQ(re.size, im.size);

  for (size_t i = 0, I = re.size; i < I; ++i) {
    complex z = exp_2_pi_i(random_01());
    re[i] = z.real();
    im[i] = z.imag();
  }
}

//----( spectrum param )------------------------------------------------------

SpectrumParam::SpectrumParam (size_t s, float f0, float f1, float t)
  : size(s),
    min_freq_hz(f0),
    max_freq_hz(f1),
    max_timescale_sec(t)
{
  ASSERT_LT(0, size);
  ASSERT_LT(0, min_freq_hz);
  ASSERT_LT(min_freq_hz, max_freq_hz);
  ASSERT_LT(1, max_timescale_sec * min_freq_hz);

  double nyquist_freq = 0.5 * DEFAULT_SAMPLE_RATE;
  ASSERTW_LE(max_freq_hz, nyquist_freq);

  const float infinity = 1e9f;
  if (max_timescale_sec < infinity) {

    cout << "solving for spectral parameters";

    float base_freq = 0;
    for (float stepsize = INFINITY, tol = 1e-8f; stepsize > tol;) {
      float freq_ratio = (max_freq_hz + base_freq) / (min_freq_hz + base_freq);
      float update = size / (max_timescale_sec * logf(freq_ratio));
      stepsize = fabsf(update - base_freq);
      base_freq = update;
      cout << '.' << flush;
    }
    cout << endl;

    m_pitch_scale = 1.0f / (base_freq * max_timescale_sec);
    m_freq_scale = 2 * M_PI * (min_freq_hz + base_freq);
    m_freq_shift = - 2 * M_PI * base_freq;

  } else {

    m_pitch_scale = logf(max_freq_hz / min_freq_hz) / size;
    m_freq_scale = 2 * M_PI * min_freq_hz;
    m_freq_shift = 0;
  }
}

//----( pitch analyzer )------------------------------------------------------

PitchAnalyzer::PitchAnalyzer (const SpectrumParam & param)
  : m_param(param),

    m_rescale(param.size),
    m_trans_real(param.size),
    m_trans_imag(param.size),
    m_stage1_real(param.size),
    m_stage1_imag(param.size),
    m_stage2_real(param.size),
    m_stage2_imag(param.size),

    m_signal_in(),
    m_energy_out(param.size),

    debug(true),

    signal_in("PitchAnalyzer.signal_in", m_signal_in.size),
    energy_out("PitchAnalyzer.energy_out", param.size)
{
  // this was adapted from Synchronized::Bank::init_decay_transform

  if (debug) {
    PRINT(param.size);
    PRINT2(param.min_freq_hz, param.max_freq_hz);
    PRINT(param.max_timescale_sec);
    //PRINT(param.max_freq_hz / nyquist_freq);
  }

  const size_t I = param.size;

  const double damp_factor_2nd_order = sqrt(2) - 1;
  const double leakage_overcounting_correction = 3.0;

  for (size_t i = 0; i < I; ++i) {

    double freq = param.get_omega_at(i) / DEFAULT_SAMPLE_RATE;
    double dfreq = param.get_domega_at(i) / DEFAULT_SAMPLE_RATE;

    std::complex<double> omega(-damp_factor_2nd_order * dfreq, freq);
    std::complex<double> trans = exp(omega);
    m_trans_real[i] = trans.real();
    m_trans_imag[i] = trans.imag();

    m_rescale[i] = pow(dfreq, 4) / leakage_overcounting_correction;
  }

  m_stage1_real.zero();
  m_stage1_imag.zero();
  m_stage2_real.zero();
  m_stage2_imag.zero();
}

PitchAnalyzer::~PitchAnalyzer ()
{
  if (debug) {
    if (energy_out) {
      PRINT3(min(m_energy_out), mean(m_energy_out), max(m_energy_out));
    }
    if (signal_in) {
      PRINT(rms(m_signal_in));
    }
  }
}

void PitchAnalyzer::transform (
    const MonoAudioFrame & signal_in,
    Vector<float> & energy_out)
{
  const size_t I = m_param.size;
  const size_t T = signal_in.size;

  const float * restrict rescale = m_rescale;
  const float * restrict trans_real = m_trans_real;
  const float * restrict trans_imag = m_trans_imag;
  const float * restrict signal = signal_in;

  float * restrict stage1_real = m_stage1_real;
  float * restrict stage1_imag = m_stage1_imag;
  float * restrict stage2_real = m_stage2_real;
  float * restrict stage2_imag = m_stage2_imag;
  float * restrict energy = energy_out;

  energy_out.zero();

  for (size_t t = 0; t < T; ++t) {

    const float x = signal[t];

    for (size_t i = 0; i < I; ++i) {

      const float f = trans_real[i];
      const float g = trans_imag[i];

      float x0 = stage1_real[i];
      float y0 = stage1_imag[i];

      float x1 = f * x0 - g * y0 + x;
      float y1 = f * y0 + g * x0;

      stage1_real[i] = x1;
      stage1_imag[i] = y1;

      x0 = stage2_real[i];
      y0 = stage2_imag[i];

      x1 = f * x0 - g * y0 + x1;
      y1 = f * y0 + g * x0 + y1;

      stage2_real[i] = x1;
      stage2_imag[i] = y1;

      energy[i] += sqr(x1) + sqr(y1);
    }
  }

  const float one_over_T = 1.0f / T;

  for (size_t i = 0; i < I; ++i) {
    energy[i] *= rescale[i] * one_over_T;
  }
}

void PitchAnalyzer::pull (Seconds time, Vector<float> & energy_out)
{
  ASSERT_SIZE(energy_out, m_param.size);

  signal_in.pull(time, m_signal_in);
  transform(m_signal_in, energy_out);
}

void PitchAnalyzer::push (Seconds time, const MonoAudioFrame & signal_in)
{
  transform(signal_in, m_energy_out);
  energy_out.push(time, m_energy_out);
}

//----( pitch reassigner )----------------------------------------------------

PitchReassigner::PitchReassigner (const SpectrumParam & param)
  : m_param(param),

    m_frames_have_been_processed(false),

    m_times(),
    m_frames(),

    m_image(NULL),

    m_frame_pos(0),

    out("PitchReassigner.out", param.size)
{}

PitchReassigner::~PitchReassigner ()
{
  delete_all(m_frames.begin(), m_frames.end());
  if (m_image) delete m_image;
}

void PitchReassigner::push (Seconds time, const Vector<float> & energy_in)
{
  ASSERT_SIZE(energy_in, m_param.size);
  ASSERT(not m_frames_have_been_processed,
      "tried to PitchReassigner::push after processing frames");

  m_times.push_back(time);
  m_frames.push_back(new Vector<float>(energy_in));
}

void PitchReassigner::pull (Seconds time, Vector<float> & reassigned_out)
{
  ASSERT_SIZE(reassigned_out, m_param.size);
  ASSERT(m_frames_have_been_processed,
      "tried to PitchReassigner::pull before processing frames");

  if (m_frame_pos < m_frames.size()) {
    reassigned_out = m_image->block(m_param.size, m_frame_pos);
    ++m_frame_pos;
  } else {
    reassigned_out.zero();
  }
}

void PitchReassigner::run ()
{
  if (not out) return;

  ASSERT(m_frames_have_been_processed,
      "tried to PitchReassigner::run before processing frames");

  LOG("pushing " << m_frames.size() << " audio frames");

  const size_t I = m_frames.size();
  const size_t J = m_param.size;

  for (size_t i = 0; i < I; ++i) {
    Vector<float> frame = m_image->block(J, i);
    out.push(m_times[i], frame);
  }

  m_times.clear();
}

void PitchReassigner::process ()
{
  ASSERT(not m_frames_have_been_processed,
      "tried to PitchReassigner::process a second time");

  LOG("Reassigning " << m_frames.size() << " audio frames");

  if (not m_frames.size()) {
    WARN("PitchReassigner had nothing to process");
    m_frames_have_been_processed = true;
    return;
  }

  const size_t I = m_frames.size();
  const size_t J = m_param.size;

  m_image = new Vector<float>(I * J);
  Vector<float> & image = * m_image;

  for (size_t i = 0; i < I; ++i) {
    Vector<float> frame = image.block(J, i);
    frame = * m_frames[i];
    delete m_frames[i];
  }
  m_frames.clear();

  TODO("reassign time & frequency simultaneously");

  m_frames_have_been_processed = true;
}

//----( pitch synthesizer )---------------------------------------------------

PitchSynthesizer::PitchSynthesizer (const SpectrumParam & param)
  : m_param(param),

    m_trans_real(param.size),
    m_trans_imag(param.size),
    m_phase_real(param.size),
    m_phase_imag(param.size),
    m_amplitude1(param.size),
    m_damplitude(param.size),

    debug(true),

    amplitude_in("PitchSynthesizer.amplitude_in", param.size)
{
  if (debug) {
    PRINT(param.size);
    PRINT2(param.min_freq_hz, param.max_freq_hz);
    PRINT(param.max_timescale_sec);
    //PRINT(param.max_freq_hz / nyquist_freq);
  }

  const size_t I = param.size;

  for (size_t i = 0; i < I; ++i) {

    double freq = param.get_omega_at(i) / DEFAULT_SAMPLE_RATE;

    std::complex<double> omega(0, freq);
    std::complex<double> trans = exp(omega);
    m_trans_real[i] = trans.real();
    m_trans_imag[i] = trans.imag();
  }

  m_amplitude1.zero();
  random_phase(m_phase_real, m_phase_imag);
}

PitchSynthesizer::~PitchSynthesizer ()
{
  if (debug) {
    PRINT3(min(m_amplitude1), mean(m_amplitude1), max(m_amplitude1));
  }
}

void PitchSynthesizer::transform (StereoAudioFrame & signal_out)
{
  const size_t I = m_param.size;
  const size_t T = signal_out.size;

  const float * restrict trans_real = m_trans_real;
  const float * restrict trans_imag = m_trans_imag;
  const float * restrict amplitude1 = m_amplitude1;
  const float * restrict damplitude = m_damplitude;

  float * restrict phase_real = m_phase_real;
  float * restrict phase_imag = m_phase_imag;
  complex * restrict signal = signal_out;

  for (size_t t = 0; t < T; ++t) {

    const float dt = (T - t - 1.0f) / T;

    float x = 0;
    float y = 0;

    for (size_t i = 0; i < I; ++i) {

      const float f = trans_real[i];
      const float g = trans_imag[i];

      float x0 = phase_real[i];
      float y0 = phase_imag[i];

      float x1 = f * x0 - g * y0;
      float y1 = f * y0 + g * x0;

      float r = sqrt(sqr(x1) + sqr(y1) + 1e-16f);
      x1 /= r;
      y1 /= r;

      phase_real[i] = x1;
      phase_imag[i] = y1;

      float amp = amplitude1[i] + damplitude[i] * dt;
      x += x1 * amp;
      y += y1 * amp;
    }

    signal[t] = complex(x,y);
  }
}

void PitchSynthesizer::transform (
    const Vector<float> amplitude_in,
    StereoAudioFrame & signal_out)
{
  subtract(m_amplitude1, amplitude_in, m_damplitude);
  m_amplitude1 = amplitude_in;

  transform(signal_out);
}

void PitchSynthesizer::pull (Seconds time, StereoAudioFrame & signal_out)
{
  m_damplitude = m_amplitude1;
  amplitude_in.pull(time, m_amplitude1);
  m_damplitude -= m_amplitude1;

  transform(signal_out);
}

} // namespace RasterAudio

