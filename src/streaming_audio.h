
#ifndef KAZOO_STREAMING_AUDIO_H
#define KAZOO_STREAMING_AUDIO_H

#include "common.h"
#include "streaming.h"
#include "synthesis.h"
#include "synchrony.h"
#include "psycho.h"
#include "filters.h"

namespace Streaming
{

class Silence : public Pulled<StereoAudioFrame>
{
public:

  virtual ~Silence () {}
  virtual void pull (Seconds time, StereoAudioFrame & sound) {}
};

class AudioWire : public Bounced<StereoAudioFrame>
{
public:
  virtual ~AudioWire () {}

  virtual void bounce (
      Seconds time,
      const StereoAudioFrame & in,
      StereoAudioFrame & out)
  {
    out = in;
  }
};

class AudioLowpass
  : public Pulled<StereoAudioFrame>
{
  Synthesis::Lowpass<complex> m_lowpass;

public:

  Port<Pulled<StereoAudioFrame> > in;

  AudioLowpass (float timescale_frames)
    : m_lowpass(timescale_frames),
      in("AudioLowpass.in")
  {}
  virtual ~AudioLowpass() {}

  void pull (Seconds time, StereoAudioFrame & sound)
  {
    m_lowpass.sample(sound);
  }
};

class PowerMeter : public Pushed<StereoAudioFrame>
{
public:

  Port<Pushed<float> > out;

  PowerMeter ();
  virtual ~PowerMeter () {}

  virtual void push (Seconds time, const StereoAudioFrame & sound);
};

class PowerSplitter : public Pushed<StereoAudioFrame>
{
public:

  Port<Pushed<float> > out1;
  Port<Pushed<float> > out2;

  PowerSplitter ();
  virtual ~PowerSplitter () {}

  virtual void push (Seconds time, const StereoAudioFrame & sound);
};

class Denoiser : public Pushed<float>
{
  Filters::Denoiser m_denoiser;

public:

  Port<Pushed<float> > out;

  Denoiser ();
  virtual ~Denoiser () {}

  virtual void push (Seconds time, const float & value);
};

class ImpactDetector : public Pushed<float>
{
  Filters::PeakDetector m_detector;

public:

  Port<Pushed<bool> > out;

  ImpactDetector ();
  virtual ~ImpactDetector () {}

  virtual void push (Seconds time, const float & value);
};

class MicGain : public Pushed<StereoAudioFrame>
{
  Filters::MaxGain m_gain;

public:

  Port<Pushed<StereoAudioFrame> > out;

  MicGain ();
  virtual ~MicGain () { LOG("mic gain = " << m_gain); }

  virtual void push (Seconds time, const StereoAudioFrame & sound);
};

class SpeakerGain : public Pulled<StereoAudioFrame>
{
  Filters::MaxGain m_gain;

public:

  Port<Pulled<StereoAudioFrame> > in;

  SpeakerGain ();
  virtual ~SpeakerGain () { LOG("speaker gain = " << m_gain); }

  virtual void pull (Seconds time, StereoAudioFrame & sound);
};

class AudioMixer : public Pulled<StereoAudioFrame>
{
  StereoAudioFrame m_sound;

public:

  Port<Pulled<StereoAudioFrame> > in1;
  Port<Pulled<StereoAudioFrame> > in2;

  AudioMixer ();
  virtual ~AudioMixer () {}

  virtual void pull (Seconds time, StereoAudioFrame & sound);
};

class EnergyToBeat
  : public Pushed<float>,
    public Pulled<StereoAudioFrame>
{
  Psycho::EnergyToBeat m_energy_to_beat;
  float m_energy;

public:

  Port<Pushed<complex> > beat_monitor;

  EnergyToBeat ();
  virtual ~EnergyToBeat () { PRINT(m_energy); }

  virtual void push (Seconds time, const float & power);
  virtual void pull (Seconds time, StereoAudioFrame & sound);
};

template<class Spec>
class Spectrogram_ : public Pushed<StereoAudioFrame>
{
  Spec m_spec;

  Vector<float> m_loudness;

public:

  SizedPort<Pushed<Vector<float> > > out;

  Spectrogram_ (size_t size, float acuity = 7 * 2)
    : m_spec(Synchronized::Bank(
          size,
          PSYCHO_MIN_PITCH_HZ / DEFAULT_SAMPLE_RATE,
          PSYCHO_MAX_PITCH_HZ / DEFAULT_SAMPLE_RATE,
          acuity)),
      m_loudness(size),

      out("Spectrogram.out", size)
  {}
  virtual ~Spectrogram_ () {}

  virtual void push (Seconds time, const StereoAudioFrame & sound)
  {
    m_spec.sample(sound, m_loudness);

    float * restrict loudness = m_loudness;
    for (size_t i = 0, I = m_loudness.size; i < I; ++i) {
      loudness[i] = powf(loudness[i], 0.33333f);
    }

    out.push(time, m_loudness);
  }
};

typedef Spectrogram_<Synchronized::FourierBank> Spectrogram;
typedef Spectrogram_<Synchronized::FourierBank2> Spectrogram2;
typedef Spectrogram_<Psycho::Masker> Maskogram;

class Psychogram : public Pushed<StereoAudioFrame>
{
  Psycho::Psychogram m_psychogram;

  Vector<float> m_loudness;

public:

  SizedPort<Pushed<Vector<float> > > out;

  Psychogram (size_t size)
    : m_psychogram(size),
      m_loudness(size),

      out("Psychogram.out", size)
  {}
  virtual ~Psychogram () {}

  virtual void push (Seconds time, const StereoAudioFrame & sound)
  {
    m_psychogram.transform_fwd(sound, m_loudness);
    out.push(time, m_loudness);
  }
};

class HearVector : public Pulled<StereoAudioFrame>
{
  Synchronized::SimpleBank m_bank;

  Vector<float> m_amplitude0;
  Vector<float> m_amplitude1;

public:

  SizedPort<Pulled<Vector<float> > > in;

  HearVector (size_t size);
  virtual ~HearVector ();

  virtual void pull (Seconds time, StereoAudioFrame & sound);
};

} // namespace Streaming

#endif // KAZOO_STREAMING_AUDIO_H

