#ifndef KAZOO_VOICE_H
#define KAZOO_VOICE_H

#include "common.h"
#include "streaming.h"
#include "raster_audio.h"
#include "psycho.h"
#include "splines.h"
#include "filters.h"
#include "config.h"

namespace Voice
{

static const int SPECTRUM_LARGE_SIZE = 384;
static const int SPECTRUM_MEDIUM_SIZE = 384;
static const int SPECTRUM_SMALL_SIZE = 96;
static const int HISTORY_LENGTH = 4;

static const float FEATURE_TIMESCALE_SEC = 0.04f;
static const float LOGAMP_MEAN = -3.0f;
static const float LOGAMP_SIGMA = 0.8f;
static const float HISTORY_DENSITY = 2.0f;

static const int FEATURE_BATCH_SIZE = 128;

//----( voice model )---------------------------------------------------------

// TODO do TF-lowpass partial autogain when creating features
//   to account for the ear's autogain adjustment

class FeatureProcessor
{
  ConfigParser m_config;

protected:

  const size_t m_large_size;
  const size_t m_medium_size;
  const size_t m_small_size;
  const size_t m_history_length;
  const size_t m_feature_size;
  const size_t m_batch_size;

  const float m_timestep;
  const float m_feature_timescale;
  const float m_feature_rate;
  const float m_logamp_mean;
  const float m_logamp_sigma;
  const float m_history_density;

  RasterAudio::SpectrumParam m_spectrum_param;

  Psycho::History m_history;

  Spline m_large_to_medium;
  Spline m_medium_to_small;

  Vector<float> m_features_real;
  // these are aliases into m_features_real
  Vector<float> m_medium_energy;
  Vector<float> m_small_history;

  Vector<float> m_small_energy;

  Vector<uint8_t> m_features;

public:

  FeatureProcessor (const char * config_filename = "config/default.voice.conf");
  virtual ~FeatureProcessor () {}

  size_t get_feature_size () const { return m_feature_size; }
  size_t get_batch_size () const { return m_batch_size; }

  void update_history ();
  void update_history (Vector<uint8_t> & features);
};

} // namespace Voice

namespace Streaming
{

//----( analyzer )------------------------------------------------------------

class VoiceAnalyzer
  : public Voice::FeatureProcessor,
    public Pushed<MonoAudioFrame>
{
  RasterAudio::PitchAnalyzer m_analyzer;

  Vector<float> m_large_frame;
  Vector<float> m_medium_frame;
  Vector<float> m_small_frame;

  Vector<float> m_medium_accum;
  size_t m_accum_count;

  bool m_initialized;
  Seconds m_time;

  typedef Filters::DebugStats<float> DebugStats;
  DebugStats m_real_stats;
  DebugStats m_int_stats;

public:

  SizedPort<Pushed<Vector<float> > > large_monitor;
  SizedPort<Pushed<Vector<float> > > medium_monitor;
  SizedPort<Pushed<Vector<float> > > small_monitor;
  SizedPort<Pushed<Vector<uint8_t> > > features_out;
  SizedPort<Pushed<Vector<uint8_t> > > debug_out;

  VoiceAnalyzer (const char * config_filename = "config/default.voice.conf");
  virtual ~VoiceAnalyzer ();

  virtual void push (Seconds time, const MonoAudioFrame & frame);

protected:

  void push_features ();
};

//----( synthesizer )---------------------------------------------------------

class VoiceSynthesizer
  : public Voice::FeatureProcessor,
    public Pushed<Vector<uint8_t> >,
    public Pulled<StereoAudioFrame>
{
  RasterAudio::PitchSynthesizer m_synthesizer;

  Vector<float> m_large_energy;
  Vector<float> m_large_amp;
  Vector<float> m_large_frame;

  Mutex m_mutex;

public:

  VoiceSynthesizer (const char * config_filename = "config/default.voice.conf");
  virtual ~VoiceSynthesizer ();

  virtual void push (Seconds time, const Vector<uint8_t> & features);
  virtual void pull (Seconds time, StereoAudioFrame & frame);
};

//----( frame builder )-------------------------------------------------------

class VoiceFeatureBuffer : public Pushed<Vector<uint8_t> >
{
  const size_t m_feature_size;
  const size_t m_batch_size;

  Mono8Image m_buffer;
  size_t m_buffer_pos;

  Seconds m_time;

public:

  RectangularPort<Pushed<Mono8Image> > out;

  VoiceFeatureBuffer (const Voice::FeatureProcessor & model);
  virtual ~VoiceFeatureBuffer () {}

  virtual void push (Seconds time, const Vector<uint8_t> & features);
  void flush ();
};

} // namespace Streaming

#endif // KAZOO_VOICE_H
