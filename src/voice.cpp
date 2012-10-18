
#include "voice.h"
#include "images.h"

namespace Voice
{

//----( voice model )---------------------------------------------------------

FeatureProcessor::FeatureProcessor (const char * config_filename)
  : m_config(config_filename),

    m_large_size(m_config("spectrum_large_size", SPECTRUM_LARGE_SIZE)),
    m_medium_size(m_config("spectrum_medium_size", SPECTRUM_MEDIUM_SIZE)),
    m_small_size(m_config("spectrum_small_size", SPECTRUM_SMALL_SIZE)),
    m_history_length(m_config("history_length", HISTORY_LENGTH)),
    m_feature_size(m_medium_size + m_small_size * m_history_length),
    m_batch_size(m_config("feature_batch_size", Voice::FEATURE_BATCH_SIZE)),

    m_timestep(1.0f / DEFAULT_VIDEO_FRAMERATE),
    m_feature_timescale(
        m_config("feature_timescale_sec", FEATURE_TIMESCALE_SEC)),
    m_feature_rate(1.0f / m_feature_timescale / DEFAULT_AUDIO_FRAMERATE),
    m_logamp_mean(m_config("logamp_mean", LOGAMP_MEAN)),
    m_logamp_sigma(m_config("logamp_sigma", LOGAMP_SIGMA)),
    m_history_density(m_config("history_density", HISTORY_DENSITY)),

    m_spectrum_param(
        m_large_size,
        m_config("min_freq_hz", RasterAudio::SPECTRUM_MIN_FREQ_HZ),
        m_config("max_freq_hz", RasterAudio::SPECTRUM_MAX_FREQ_HZ),
        m_config("max_timescale_sec", RasterAudio::SPECTRUM_MAX_TIMESCALE_SEC)),

    m_history(m_small_size, 1 + m_history_length, m_history_density),

    m_large_to_medium(m_large_size, m_medium_size),
    m_medium_to_small(m_medium_size, m_small_size),

    m_features_real(m_feature_size),
    m_medium_energy(m_medium_size, m_features_real.begin()),
    m_small_history(
        m_small_size * m_history_length,
        m_medium_energy.end()),

    m_small_energy(
        m_small_size,
        m_small_size == m_medium_size ? m_medium_energy.data : NULL),

    m_features(m_feature_size)
{
  ASSERT_LE(m_medium_size, m_large_size);
  ASSERT_LE(m_small_size, m_medium_size);
}

void FeatureProcessor::update_history ()
{
  if (m_history_length) {

    if (m_small_size != m_medium_size) {
      m_medium_to_small.transform_fwd(m_medium_energy, m_small_energy);
      m_small_energy *= float(m_small_size) / m_medium_size;
    }

    m_history.add(m_small_energy);
    m_history.get_after(1, m_small_history);
  }
}

void FeatureProcessor::update_history (Vector<uint8_t> & features)
{
  ASSERT_SIZE(features, m_features.size);

  if (m_history_length == 0) return;

  uchar_to_real(features, m_features_real);

  update_history();

  Vector<uint8_t> history(
      m_small_history.size,
      features.data + (m_small_history.data - m_features_real.data));
  real_to_uchar(m_small_history, history);
}

} // namespace Voice

namespace Streaming
{

//----( analyzer )------------------------------------------------------------

VoiceAnalyzer::VoiceAnalyzer (const char * config_filename)
  : Voice::FeatureProcessor(config_filename),

    m_analyzer(m_spectrum_param),

    m_large_frame(m_large_size),
    m_medium_frame(
        m_medium_size,
        m_medium_size == m_large_size ? m_large_frame.data : NULL),
    m_small_frame(
        m_small_size,
        m_small_size == m_medium_size ? m_medium_frame.data : NULL),

    m_medium_accum(m_medium_size),
    m_accum_count(0),

    m_initialized(false),
    m_time(),

    large_monitor("VoiceAnalyzer.large_monitor", m_large_size),
    medium_monitor("VoiceAnalyzer.medium_monitor", m_medium_size),
    small_monitor("VoiceAnalyzer.small_monitor", m_small_size),
    features_out("VoiceAnalyzer.features_out", m_feature_size),
    debug_out("VoiceAnalyzer.debug_out", m_feature_size)
{
  m_medium_accum.zero();
  m_accum_count = 0;
}

VoiceAnalyzer::~VoiceAnalyzer ()
{
  if (m_initialized) {

    PRINT(m_real_stats);
    PRINT(m_int_stats);

    float logamp_mean = m_real_stats.mean() * m_logamp_sigma + m_logamp_mean;
    float logamp_sigma = sqrtf(m_real_stats.variance()) * m_logamp_sigma;
    LOG("Suggested logamp_mean = " << logamp_mean
        << ", logamp_sigma = " << logamp_sigma);
  }
}

void VoiceAnalyzer::push (Seconds time, const MonoAudioFrame & frame)
{
  m_analyzer.transform(frame, m_large_frame);

  if (m_medium_size != m_large_size) {
    m_large_to_medium.transform_fwd(m_large_frame, m_medium_frame);
    m_medium_frame *= float(m_medium_size) / m_large_size;
  }
  if (small_monitor and m_small_size != m_medium_size) {
    m_medium_to_small.transform_fwd(m_medium_frame, m_small_frame);
    m_small_frame *= float(m_small_size) / m_medium_size;
  }

  if (large_monitor) large_monitor.push(time, m_large_frame);
  if (medium_monitor) medium_monitor.push(time, m_medium_frame);
  if (small_monitor) small_monitor.push(time, m_small_frame);

  m_medium_accum += m_medium_frame;
  m_accum_count += 1;

  if (not m_initialized) {

    m_time = time;
    m_initialized = true;

  } else if (time > m_time + m_timestep) {

    m_time += m_timestep; // this introduces a little jitter

    push_features();
  }
}

void VoiceAnalyzer::push_features ()
{
  multiply(1.0f / m_accum_count, m_medium_accum, m_medium_energy);
  m_accum_count = 0;
  m_medium_accum.zero();

  // add history

  update_history();

  // map energy -> log(amplitude)

  { const size_t I = m_feature_size;

    const float logenergy_sigma = 2 * m_logamp_sigma;
    const float logenergy_mean = 2 * m_logamp_mean;
    const float scale = 1 / logenergy_sigma;
    const float shift = -scale * logenergy_mean;

    float * restrict features = m_features_real;

    for (size_t i = 0; i < I; ++i) {
      features[i] = shift + scale * logf(features[i] + 1e-16f);
    }
  }

  real_to_uchar(m_features_real, m_features);

  m_real_stats.add(m_medium_energy);
  m_int_stats.add(m_features.data, m_medium_size);

  if (features_out and (m_history_length == 0 or m_history.full())) {
    features_out.push(m_time, m_features);
  }
  if (debug_out) debug_out.push(m_time, m_features);
}

//----( synthesizer )---------------------------------------------------------

VoiceSynthesizer::VoiceSynthesizer (const char * config_filename)
  : Voice::FeatureProcessor(config_filename),

    m_synthesizer(m_spectrum_param),

    m_large_energy(
        m_large_size,
        m_large_size == m_medium_size ? m_medium_energy.data : NULL),
    m_large_amp(m_large_size),
    m_large_frame(m_large_size)
{
}

VoiceSynthesizer::~VoiceSynthesizer ()
{
}

void VoiceSynthesizer::push (Seconds time, const Vector<uint8_t> & features)
{
  // TODO use entire history of feature to interpolate synthesized signal
  uchar_to_real(features, m_features_real);

  // map log(amplitude) -> energy

  { const size_t I = m_medium_size;

    float * restrict energy = m_medium_energy; // aliased into features_real

    const float scale = 2 * m_logamp_sigma;
    const float shift = 2 * m_logamp_mean;

    for (size_t i = 0; i < I; ++i) {
      energy[i] = expf(shift + scale * energy[i]);
    }
  }

  if (m_large_size != m_medium_size) {
    m_large_to_medium.transform_bwd(m_medium_energy, m_large_energy);
    m_large_energy *= float(m_large_size) / m_medium_size;
  }

  // map energy -> amplitude

  m_mutex.lock();
  {
    const size_t I = m_large_size;

    const float * restrict energy = m_large_energy;
    float * restrict amp = m_large_amp;

    for (size_t i = 0; i < I; ++i) {
      amp[i] = sqrtf(energy[i]);
    }
  }
  m_mutex.unlock();
}

void VoiceSynthesizer::pull (Seconds time, StereoAudioFrame & frame)
{
  m_mutex.lock();
  m_large_frame = m_large_amp;
  m_mutex.unlock();

  // TODO do some sort of interpolation of history/shingles here

  m_synthesizer.transform(m_large_frame, frame);
}

//----( frame builder )-------------------------------------------------------

VoiceFeatureBuffer::VoiceFeatureBuffer (const Voice::FeatureProcessor & model)
  : m_feature_size(model.get_feature_size()),
    m_batch_size(model.get_batch_size()),

    m_buffer(m_feature_size * m_batch_size),
    m_buffer_pos(0),

    out("VoiceFeatureBuffer", Rectangle(m_batch_size, m_feature_size))
{
}

void VoiceFeatureBuffer::push (Seconds time, const Vector<uint8_t> & features)
{
  if (m_buffer_pos == 0) {
    m_time = time;
  }

  m_buffer.block(m_feature_size, m_buffer_pos++) = features;

  if (m_buffer_pos == m_batch_size) {
    out.push(m_time, m_buffer);
    m_buffer_pos = 0;
  }
}

void VoiceFeatureBuffer::flush ()
{
  if (not m_buffer_pos) return;

  while (m_buffer_pos) {
    m_buffer.block(m_feature_size, m_buffer_pos++).zero();
  }

  out.push(m_time, m_buffer);
  m_buffer_pos = 0;
}

} // namespace Streaming

