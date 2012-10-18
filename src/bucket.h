
#ifndef KAZOO_BUCKET_H
#define KAZOO_BUCKET_H

#include "common.h"
#include "streaming_audio.h"
#include "streaming_video.h"
#include "streaming_camera.h"
#include "streaming_devices.h"
#include "synthesis.h"
#include "synchrony.h"
#include "tracker.h"
#include "filters.h"
#include "hash_map.h"

#define BUCKET_FINGER_CAPACITY          (8)

namespace Streaming
{

//----( bucket components )---------------------------------------------------

class Bucket
{
protected:

  DiskThread m_camera;
  ShrinkBy4 m_shrink;
  EnhancePoints m_fingers;
  DiskMask m_mask;

public:

  RectangularPort<Pushed<MonoImage> > & out;

  Bucket ()
    : m_camera(),
      m_shrink(m_camera.out),
      m_fingers(m_shrink.out),
      m_mask(m_fingers.out),

      out(m_mask.out)
  {
    m_camera.out - m_shrink;
    m_shrink.out - m_fingers;
    m_fingers.out - m_mask;
  }
};

class FingersBucket : protected Bucket
{
  PeakDetector m_peaks;
  Tracking::Tracker m_tracker;
  ImpactDistributor m_distributor;
  PixToPolar m_calibrate;

  Denoiser m_denoiser;
  Shared<float> m_impact;

public:

  Pushed<float> & power_in;
  Pulled<BoundedMap<Id, Gestures::Finger> > & fingers_out;

  FingersBucket (
      bool deaf = false,
      size_t finger_capacity = BUCKET_FINGER_CAPACITY)

    : Bucket(),

      m_peaks(Bucket::out, 2 * finger_capacity),
      m_tracker(2 * finger_capacity),
      m_distributor(),
      m_calibrate(Bucket::out),
      m_denoiser(),
      m_impact(0),

      power_in(m_denoiser),
      fingers_out(m_calibrate)
  {
    Bucket::out - m_peaks;
    m_peaks.out - m_tracker;
    m_distributor.fingers_in - m_tracker;
    m_calibrate.in - m_distributor;

    if (not deaf) {
      m_denoiser.out - m_impact;
      m_distributor.impact_in - m_impact;
    }
  }
};

//----( controllers )---------------------------------------------------------

struct VoiceControl
  : public Rectangle,
    public Pushed<MonoImage>
{
  const size_t m_circ;

  Filters::MaxGain m_impedance_gain;

  Vector<float> m_x2;
  Vector<float> m_y2;
  Vector<unsigned char> m_angle;

  Vector<float> m_temp;

public:

  struct State
  {
    Vector<float> impedance;
    float pitch;
    float sustain;

    State (size_t size, float init_impedance = 0)
      : impedance(size),
        pitch(0),
        sustain(0)
    {
      impedance.set(init_impedance);
    }
  };

protected:

  State m_state;

public:

  Streaming::Port<Streaming::Pushed<State> > out;

  VoiceControl (Rectangle shape);
  virtual ~VoiceControl () {}

  size_t circ () const { return m_circ; }

  virtual void push (Seconds time, const MonoImage & image);
};

class SpinControl
  : public Rectangle,
    public Pushed<MonoImage>
{
  Vector<float> m_x;
  Vector<float> m_y;

public:

  struct State
  {
    float phase;
    float rate;
    float amplitude;

    State () : phase(0), rate(1), amplitude(0) {}
  };

  Streaming::Port<Streaming::Pushed<State> > out;

  SpinControl (Rectangle shape);
  virtual ~SpinControl () {}

  virtual void push (Seconds time, const MonoImage & image);
};

//----( synthesis )-----------------------------------------------------------

class FormantSynth : public Pulled<StereoAudioFrame>
{
  const float m_pitch_center;

  Synthesis::Glottis m_actuate1;
  Synthesis::ExpSine m_actuate2;
  Synthesis::LoopResonator m_resonate;
  Synthesis::Lowpass<complex> m_lowpass;

  VoiceControl::State m_state;

public:

  Shared<VoiceControl::State, size_t> voice_in;
  Shared<float> power_in;

  FormantSynth (size_t circ, float pitch_shift = 0);
  virtual ~FormantSynth ();

  virtual void pull (Seconds time, StereoAudioFrame & sound_accum);
};

class LoopSynth : public Pulled<StereoAudioFrame>
{
  Synthesis::VariableSpeedLoop m_loop;
  Synthesis::VariableSpeedLoop::State m_loop_state;

  SpinControl::State m_state;

public:

  Shared<SpinControl::State> spin_in;
  Shared<bool> impact_in;

  LoopSynth (StereoAudioFrame & loop);
  virtual ~LoopSynth ();

  virtual void pull (Seconds time, StereoAudioFrame & sound);

  static StereoAudioFrame load_loop (
      const char * filename,
      float duration_sec,
      float begin_sec);
};

// Formant -> Loop
class FormantLoopSynth : public Pulled<StereoAudioFrame>
{
  Synthesis::Glottis m_actuate1;
  Synthesis::ExpSine m_actuate2;
  Synthesis::LoopResonator m_resonate;
  Synthesis::Lowpass<complex> m_lowpass;

  StereoAudioFrame m_loop_data;
  Synthesis::VariableSpeedLoop m_loop;
  Synthesis::VariableSpeedLoop::State m_loop_state;

  float m_decay0;
  float m_decay1;
  StereoAudioFrame m_sound;

  VoiceControl::State m_voice_state;
  SpinControl::State m_spin_state;

public:

  Shared<VoiceControl::State, size_t> voice_in;
  Shared<float> power_in;

  Shared<SpinControl::State> spin_in;
  Shared<bool> impact_in;

  FormantLoopSynth (size_t circ, size_t loop_size);
  virtual ~FormantLoopSynth ();

  virtual void pull (Seconds time, StereoAudioFrame & sound);
};

class SyncopatedSynth : public Pulled<StereoAudioFrame>
{
  typedef Synchronized::SyncopatorBank Bank;
  Bank m_bank;

  BoundedMap<Id, float3> m_polar;
  BoundedMap<Id, Bank::State> m_states;
  std::hash_map<Id, size_t> m_ids;

  Seconds m_time;

public:

  SizedPort<Pulled<BoundedMap<Id, float3> > > polar_in;

  SyncopatedSynth (size_t capacity);
  virtual ~SyncopatedSynth () {}

  virtual void pull (Seconds time, StereoAudioFrame & sound);
};

//----( bucket instruments )--------------------------------------------------

class FormantBucket : public Pulled<StereoAudioFrame>
{
  Bucket m_bucket;
  VoiceControl m_voice;
  Denoiser m_denoiser;
  FormantSynth m_synth;

public:

  Pushed<float> & power_in;

  FormantBucket (float pitch_shift = 0)
    : m_bucket(),
      m_voice(m_bucket.out),
      m_denoiser(),
      m_synth(m_voice.circ(), pitch_shift),
      power_in(m_denoiser)
  {
    m_bucket.out - m_voice;
    m_voice.out - m_synth.voice_in;
    m_denoiser.out - m_synth.power_in;
  }
  virtual ~FormantBucket () {}

  virtual void pull (Seconds time, StereoAudioFrame & sound)
  {
    m_synth.pull(time, sound);
  }
};

class LoopBucket : public Pulled<StereoAudioFrame>
{
  Bucket m_bucket;
  SpinControl m_spin;
  ImpactDetector m_impact;
  LoopSynth m_synth;

public:

  Pushed<float> & power_in;

  LoopBucket (StereoAudioFrame & loop)
    : m_bucket(),
      m_spin(m_bucket.out),
      m_impact(),
      m_synth(loop),
      power_in(m_impact)
  {
    m_bucket.out - m_spin;
    m_spin.out - m_synth.spin_in;
    m_impact.out - m_synth.impact_in;
  }
  virtual ~LoopBucket () {}

  virtual void pull (Seconds time, StereoAudioFrame & sound)
  {
    m_synth.pull(time, sound);
  }
};

class FormantLoopBuckets : public Pulled<StereoAudioFrame>
{
  Bucket m_formant_bucket;
  VoiceControl m_voice;
  Denoiser m_denoiser;

  Bucket m_loop_bucket;
  SpinControl m_spin;
  ImpactDetector m_impact;

  PushedPair<float, bool> m_pair;
  FormantLoopSynth m_synth;

public:

  Pushed<float> & formant_power_in;
  Pushed<float> & loop_power_in;

  FormantLoopBuckets (float duration_sec = 4.0f)
    : m_formant_bucket(),
      m_voice(m_formant_bucket.out),
      m_denoiser(),

      m_loop_bucket(),
      m_spin(m_loop_bucket.out),
      m_impact(),

      m_synth(
          m_voice.circ(),
          roundu(duration_sec * DEFAULT_SAMPLE_RATE)),

     formant_power_in(m_denoiser),
     loop_power_in(m_impact)
  {
    m_formant_bucket.out - m_voice;
    m_voice.out - m_synth.voice_in;
    m_denoiser.out - m_synth.power_in;

    m_loop_bucket.out - m_spin;
    m_spin.out - m_synth.spin_in;
    m_impact.out - m_synth.impact_in;
  }
  virtual ~FormantLoopBuckets () {}

  virtual void pull (Seconds time, StereoAudioFrame & sound)
  {
    m_synth.pull(time, sound);
  }
};

} // namespace Streaming

#endif // KAZOO_BUCKET_H

