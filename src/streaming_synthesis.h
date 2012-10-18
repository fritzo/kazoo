
#ifndef KAZOO_STREAMING_SYNTHESIS_H
#define KAZOO_STREAMING_SYNTHESIS_H

#include "common.h"
#include "synthesis.h"
#include "streaming.h"
#include "bounded.h"
#include "threads.h"
#include "synchrony.h"
#include "gestures.h"
#include "image_types.h"
#include "hash_map.h"
#include <map>
#include <vector>

#define SYNTHESIS_HIGHPASS_CUTOFF_HZ    (8.0f)

#define DEFAULT_REVERB_BANK_SIZE        (512)
#define DEFAULT_REVERB_BANK_FREQ0       (18.0f)
#define DEFAULT_REVERB_BANK_FREQ1       (18e3f)

#define VOICE_FADOUT_TIMESCALE          (0.1f)

#define DEFAULT_CHORUS_CAPACITY         (16)

#define COUPLED_HATS_TEMPO_HZ           (1.0f)
#define COUPLED_HATS_FREQ_HZ            (2e3f)
#define COUPLED_HATS_NUM_OCTAVES        (5)

#define WOBBLER_TONE_FREQ_HZ            (60.0f)
#define WOBBLER_SHARPNESS               (6.0f)

namespace Streaming
{

using Gestures::Finger;

//----( effects )-------------------------------------------------------------

class Reverb : public Pulled<StereoAudioFrame>
{
  Synthesis::Reverberator m_reverb;

public:

  Port<Pulled<StereoAudioFrame> > in;

  Reverb () : in("Reverb.in") {}
  virtual ~Reverb () {}

  virtual void pull (Seconds time, StereoAudioFrame & sound)
  {
    in.pull(time, sound);
    m_reverb.transform(sound);
  }
};

class ReverbBank : public Pulled<StereoAudioFrame>
{
  Synchronized::FourierBank m_reverb;

public:

  Port<Pulled<StereoAudioFrame> > in;

  ReverbBank ()
    : m_reverb(Synchronized::Bank(
        DEFAULT_REVERB_BANK_SIZE,
        DEFAULT_REVERB_BANK_FREQ0 / DEFAULT_SAMPLE_RATE,
        DEFAULT_REVERB_BANK_FREQ1 / DEFAULT_SAMPLE_RATE)),
      in("ReverbBank.in")
  {}
  virtual ~ReverbBank () {}

  virtual void pull (Seconds time, StereoAudioFrame & sound)
  {
    in.pull(time, sound);
    m_reverb.resonate(sound);
  }
};

//----( vocoder )-------------------------------------------------------------

class Vocoder
  : public Synthesis::Vocoder,
    public Pulled<StereoAudioFrame>,
    public Pulled<RgbImage>
{
  Synthesis::Vocoder::Timbre m_timbre;

  Filters::MaxGain m_mass_gain;
  Filters::MaxGain m_bend_gain;

  Mutex m_mutex;

public:

  Port<Pulled<Timbre> > in;

  Vocoder ();
  virtual ~Vocoder () {}

  virtual void pull (Seconds time, StereoAudioFrame & sound_accum);
  virtual void pull (Seconds time, RgbImage & image);
};

class VocoSliders
  : public Pushed<Sliders>,
    public Pulled<Vocoder::Timbre>
{
  Vocoder::Timbre m_timbre;

  Mutex m_mutex;

public:

  virtual ~VocoSliders () {}

  virtual void push (Seconds time, const Sliders & sliders)
  {
    m_mutex.lock();
    m_timbre = sliders.mass;
    m_mutex.unlock();
  }

  virtual void pull (Seconds time, Vocoder::Timbre & timbre)
  {
    m_mutex.lock();
    timbre = m_timbre;
    m_mutex.unlock();
  }
};

//----( vocoder testing )----

class SimVocoderChirp : public Pulled<Vocoder::Timbre>
{
  const size_t m_num_harmonics;
  const float m_rate;
  float m_state;
  Seconds m_time;

public:

  SimVocoderChirp (
      float period_sec = 4.0f,
      size_t num_harmonics = 1)
    : m_num_harmonics(num_harmonics),
      m_rate(1 / period_sec),
      m_state(0),
      m_time(Seconds::now())
  {}
  virtual ~SimVocoderChirp () {}

  virtual void pull (Seconds time, Vocoder::Timbre & timbre);
};

class SimVocoderChord : public Pulled<Vocoder::Timbre>
{
  const size_t m_num_tones;
  const float m_rate;
  float m_state;
  Seconds m_time;

public:

  SimVocoderChord (
      float period_sec = 4.0f,
      size_t num_tones = 1)
    : m_num_tones(num_tones),
      m_rate(1 / period_sec),
      m_state(0),
      m_time(Seconds::now())
  {}
  virtual ~SimVocoderChord () {}

  virtual void pull (Seconds time, Vocoder::Timbre & timbre);
};

class SimVocoderDrone : public Pulled<Vocoder::Timbre>
{
  const size_t m_num_tones;
  const float m_rate;
  float m_state;
  Seconds m_time;

public:

  SimVocoderDrone (
      float period_sec = 4.0f,
      size_t num_tones = 1)
    : m_num_tones(num_tones),
      m_rate(1 / period_sec),
      m_state(0),
      m_time(Seconds::now())
  {}
  virtual ~SimVocoderDrone () {}

  virtual void pull (Seconds time, Vocoder::Timbre & timbre);
};

class SimVocoderNoiseBand : public Pulled<Vocoder::Timbre>
{
  size_t m_ratio;
  const float m_rate;
  float m_state;
  Seconds m_time;

public:

  SimVocoderNoiseBand (float period_sec = 4.0f, size_t ratio = 1)
    : m_ratio(ratio),
      m_rate(1 / period_sec),
      m_state(0),
      m_time(Seconds::now())
  {}
  virtual ~SimVocoderNoiseBand () {}

  virtual void pull (Seconds time, Vocoder::Timbre & timbre);
};

//----( beater )--------------------------------------------------------------

class Beater
  : public Synthesis::Beater,
    public Pulled<Synthesis::Beater::Timbre>,
    public Pulled<BoundedMap<Id, Gestures::Finger> >,
    public Pulled<MonoImage>
{
  Mutex m_mutex;
  Timbre m_beat;
  float m_power;

public:

  const Rectangle shape;

  Port<Pulled<Synthesis::Beater::Timbre> > beat_in;
  Port<Pulled<BoundedMap<Id, Gestures::Finger> > > fingers_in;
  Port<Pulled<float> > power_in;

  Beater (bool coalesce = true, float blur_factor = 1.0f);
  virtual ~Beater () {}

  virtual void pull (Seconds time, Timbre & amplitude);
  virtual void pull (Seconds time, BoundedMap<Id, Gestures::Finger> & fingers);
  virtual void pull (Seconds time, MonoImage & image);
};

//----( soloist )-------------------------------------------------------------

template<class Voice, class Sound = StereoAudioFrame>
class Soloist : public Pulled<Sound>
{
  Voice m_voice;

  Finger m_finger;

public:

  typedef typename Voice::Timbre Timbre;

  Port<Pulled<Finger> > in;

  Soloist () : in("Soloist.in") {}
  virtual ~Soloist () {}

  virtual void pull (Seconds time, Sound & sound_accum)
  {
    in.pull(time, m_finger);

    Timbre timbre = Voice::layout(m_finger);

    m_voice.sample(timbre, sound_accum);
  }
};

//----( chorus )--------------------------------------------------------------

template<class Descriptor, class Voice, class Sound = StereoAudioFrame>
class Chorus : public Pulled<Sound>
{
  BoundedMap<Id, Descriptor> m_descriptors;

  typedef std::hash_map<Id, Voice *> Voices;
  Voices m_voices;
  Voices m_ended;

public:

  typedef typename Voice::Timbre Timbre;

  Port<Pulled<BoundedMap<Id, Descriptor> > > in;

  Chorus (size_t capacity)
    : m_descriptors(capacity),
      in("Chorus.in")
  {}
  virtual ~Chorus () { clear(); }

  virtual void pull (Seconds time, Sound & sound);

  void clear ();
};

template<class Descriptor, class Voice, class Sound>
void Chorus<Descriptor, Voice, Sound>::pull (Seconds time, Sound & sound_accum)
{
  std::swap(m_voices, m_ended);

  in.pull(time, m_descriptors);

  // sample continuing voices & fade in new voices
  typedef typename Voices::iterator Auto;
  for (size_t i = 0; i < m_descriptors.size; ++i) {
    Id id = m_descriptors.keys[i];
    const Descriptor & descriptor = m_descriptors.values[i];

    Timbre timbre = Voice::layout(descriptor);
    Auto v = m_ended.find(id);
    if (v != m_ended.end()) {
      Voice * voice = v->second;
      voice->sample(timbre, sound_accum);

      m_voices.insert(*v);
      m_ended.erase(v);
    } else {
      Voice * voice = new Voice(timbre);
      voice->fadein(sound_accum);

      m_voices.insert(std::make_pair(id, voice));
    }
  }

  // fade out ended voices
  for (Auto v = m_ended.begin(); v != m_ended.end(); ++v) {
    Voice * voice = v->second;
    voice->fadeout(sound_accum);

    if (voice->active()) m_voices.insert(*v); else delete voice;
  }
  m_ended.clear();
}

template<class Descriptor, class Voice, class Sound>
void Chorus<Descriptor, Voice, Sound>::clear ()
{
  typedef typename Voices::iterator Auto;
  for (Auto v = m_voices.begin(); v != m_voices.end(); ++v)
  {
    delete v->second;
  }
  m_voices.clear();
}

//============================================================================

//----( piano )---------------------------------------------------------------

class Piano
  : public Pushed<Vector<float> >,
    public Pulled<StereoAudioFrame>
{
  const size_t m_size;
  const float m_sustain;
  const float m_release;

  Vector<float> m_freq;
  Vector<float> m_attack;
  StereoAudioFrame m_rate;
  StereoAudioFrame m_phase;

  Mutex m_mutex;

public:

  Shared<float> impact_in;

  Piano (
      size_t range      = DEFAULT_SYNTHESIS_RANGE,
      float mid_freq_hz  = DEFAULT_SYNTHESIS_MID_FREQ_HZ,
      float pitch_step   = DEFAULT_SYNTHESIS_PITCH_STEP / 12,
      float sustain_sec  = DEFAULT_SYNTHESIS_SUSTAIN_SEC,
      float release_sec  = DEFAULT_SYNTHESIS_RELEASE_SEC);
  virtual ~Piano () {}

  size_t size () const { return m_size; }

  virtual void push (Seconds time, const Vector<float> & tones);
  virtual void pull (Seconds time, StereoAudioFrame & sound_accum);
};

//----( actuators )-----------------------------------------------------------

class Actuators
  : public Pushed<Vector<float> >,
    public Pulled<StereoAudioFrame>
{
  const size_t m_size;
  const float m_sustain;
  const float m_release;

  Vector<int> m_freq;
  Vector<float> m_decay;
  Vector<float> m_attack;
  Vector<float> m_diffuse;
  Vector<float> m_power;
  Vector<int> m_sawtooth;

  Mutex m_mutex;

public:

  Shared<float> impact_in;

  Actuators (
      size_t range      = DEFAULT_SYNTHESIS_RANGE,
      float mid_freq_hz  = DEFAULT_SYNTHESIS_MID_FREQ_HZ,
      float pitch_step   = DEFAULT_SYNTHESIS_PITCH_STEP / 12,
      float sustain_sec  = DEFAULT_SYNTHESIS_SUSTAIN_SEC,
      float release_sec  = DEFAULT_SYNTHESIS_RELEASE_SEC);
  virtual ~Actuators () {}

  size_t size () const { return m_size; }

  virtual void push (Seconds time, const Vector<float> & tones);
  virtual void pull (Seconds time, StereoAudioFrame & sound_accum);
};

//----( resonators )----------------------------------------------------------

class Resonators
  : public Pushed<Vector<float> >,
    public Bounced<StereoAudioFrame, StereoAudioFrame, size_t>
{
  const size_t m_size;

  StereoAudioFrame m_rate;
  Vector<float> m_power;
  StereoAudioFrame m_phase;

  Mutex m_mutex;

public:

  Resonators (
      size_t range      = DEFAULT_SYNTHESIS_RANGE,
      float mid_freq_hz  = DEFAULT_SYNTHESIS_MID_FREQ_HZ,
      float pitch_step   = DEFAULT_SYNTHESIS_PITCH_STEP / 12);
  virtual ~Resonators () {}

  size_t size () const { return m_size; }

  virtual void push (Seconds time, const Vector<float> & tones);
  virtual void bounce (
      Seconds time,
      const StereoAudioFrame & sound_in,
      StereoAudioFrame & sound_out);
};

//============================================================================

//----( vectorized chorus )---------------------------------------------------

template<class V>
class VectorizedChorus : public Pulled<StereoAudioFrame>
{
public:

  typedef V Voice;

private:

  BoundedMap<Id, Finger> m_fingers;
  typedef std::map<Id, Voice> Voices;
  Voices m_voices;

protected:

  std::vector<Voice> m_voice_vector;

public:

  Port<Pulled<BoundedMap<Id, Finger> > > in;

  VectorizedChorus (size_t capacity = DEFAULT_CHORUS_CAPACITY)
    : m_fingers(capacity),
      in("VectorizedChorus.in")
  {}
  virtual ~VectorizedChorus () {}

  virtual void pull (Seconds time, StereoAudioFrame & sound_accum);

protected:

  virtual void sample (StereoAudioFrame & sound_accum) = 0;
};

template<class Voice>
class SimpleVectorizedChorus : public VectorizedChorus<Voice>
{
public:

  SimpleVectorizedChorus (size_t capacity = DEFAULT_CHORUS_CAPACITY)
    : VectorizedChorus<Voice>(capacity)
  {}
  virtual ~SimpleVectorizedChorus () {}

protected:

  virtual void sample (StereoAudioFrame & sound_accum);
};

typedef SimpleVectorizedChorus<Synthesis::Coupled::Sine> CoupledSines;
typedef SimpleVectorizedChorus<Synthesis::Coupled::Shepard4> Shepard4s;
typedef SimpleVectorizedChorus<Synthesis::Coupled::Shepard7> Shepard7s;
typedef SimpleVectorizedChorus<Synthesis::Coupled::SitarString> SitarStrings;

//----( wideband )------------------------------------------------------------

class Wideband : public VectorizedChorus<Synthesis::Coupled::Sine>
{
  Synthesis::Highpass<float> m_highpass;

public:

  Wideband (size_t capacity = DEFAULT_CHORUS_CAPACITY)
    : VectorizedChorus<Synthesis::Coupled::Sine>(capacity),
      m_highpass(DEFAULT_SAMPLE_RATE / SYNTHESIS_HIGHPASS_CUTOFF_HZ)
  {}
  virtual ~Wideband () {}

protected:

  virtual void sample (StereoAudioFrame & sound_accum);
};

//----( splitband )-----------------------------------------------------------

template<class Voice>
class Splitband : public VectorizedChorus<Voice>
{
  StereoAudioFrame m_sound;

public:

  Splitband (size_t capacity = DEFAULT_CHORUS_CAPACITY)
    : VectorizedChorus<Voice>(capacity)
  {}
  virtual ~Splitband () {}

private:

  virtual void sample (StereoAudioFrame & sound_accum);
};

typedef Splitband<Synthesis::Coupled::BeatingSine> Splitband1;
typedef Splitband<Synthesis::Coupled::SyncoSine> Splitband2;
typedef Splitband<Synthesis::Coupled::BeatingPlucked> Splitband3;
typedef Splitband<Synthesis::Coupled::SyncoPlucked> Splitband4;

//----( sitar )---------------------------------------------------------------

// the Sitar is intended for use with SitarStrings
class Sitar : public Pulled<BoundedMap<Id, Gestures::Finger> >
{
  BoundedMap<Id, Gestures::Finger> m_hands;

  const float m_timescale;
  Filters::MaxGain m_hand_energy_gain;
  Filters::MaxGain m_finger_energy_gain;
  Filters::RmsGain m_finger_pos_gain;
  Filters::RmsGain m_finger_vel_gain;

public:

  Port<Pulled<BoundedMap<Id, Gestures::Finger> > > hands_in;
  Port<Pulled<BoundedMap<Id, Gestures::Finger> > > fingers_in;

  Sitar (size_t hand_capacity);
  virtual ~Sitar ();

  virtual void pull (Seconds time, BoundedMap<Id, Gestures::Finger> & plectra);
};

//============================================================================

//----( coupled band )--------------------------------------------------------

template<class Sound>
class CoupledBand : public Pulled<Sound>
{
public:

  struct Member
    : public Pulled<Synchronized::Poll>,
      public Bounced<complex, Sound, void>
  {
    virtual ~Member () {}
  };

private:

  std::vector<Member *> m_members;

public:

  CoupledBand () {}
  virtual ~CoupledBand () {}

  void add (Member & member) { m_members.push_back(& member); }
  void clear () { m_members.clear(); }

  // serially pull & synchronize all members
  virtual void pull (Seconds time, Sound & sound_accum);
};

template<class Sound>
void CoupledBand<Sound>::pull (Seconds time, Sound & sound_accum)
{
  ASSERT(not m_members.empty(), "an empty band cannot play :(");

  Synchronized::Poll poll;
  for (size_t i = 0; i < m_members.size(); ++i) {
    m_members[i]->pull(time, poll);
  }

  complex force = poll.mean();

  for (size_t i = 0; i < m_members.size(); ++i) {
    m_members[i]->bounce(time, force, sound_accum);
  }
}

//----( hats )----------------------------------------------------------------

class CoupledHats : public CoupledBand<StereoAudioFrame>::Member
{
  Synchronized::Phasor m_tempo;

  const float m_freq;
  const complex m_trans;
  complex m_state;

  BoundedMap<Id, Finger> m_fingers;
  float m_mass;

public:

  Port<Pulled<BoundedMap<Id, Finger> > > in;

  typedef StereoAudioFrame Sound;

  CoupledHats (
      size_t capacity = DEFAULT_CHORUS_CAPACITY,
      float tempo_hz = COUPLED_HATS_TEMPO_HZ,
      float freq_hz = COUPLED_HATS_FREQ_HZ);

  virtual void pull (Seconds time, Synchronized::Poll & poll);
  virtual void bounce (
      Seconds time,
      const complex & force,
      Sound & sound_accum);
};

//----( coupled chorus )------------------------------------------------------

template<class Voice>
class CoupledChorus
  : public CoupledBand<typename Voice::Sound>::Member
{
public:

  typedef typename Voice::Sound Sound;
  typedef Synchronized::Poll Poll;

private:

  BoundedMap<Id, Finger> m_fingers;

protected:

  typedef std::map<Id, Voice> Voices;
  Voices m_voices;

public:

  Port<Pulled<BoundedMap<Id, Finger> > > in;

  CoupledChorus (size_t capacity = DEFAULT_CHORUS_CAPACITY)
    : m_fingers(capacity),
      in("CoupledChorus.in")
  {}
  virtual ~CoupledChorus () {}

  virtual void pull (Seconds time, Poll & poll);
  virtual void bounce (Seconds time, const complex & force, Sound & sound);
};

template<class Voice>
void CoupledChorus<Voice>::pull (
    Seconds time,
    typename CoupledChorus<Voice>::Poll & poll)
{
  // advance existing active voices, limiting to oldest 2 * capacity voices
  int max_voices = 2 * m_fingers.capacity;
  int num_voices = m_voices.size();
  size_t num_ending = max(num_voices - max_voices, 0);
  size_t n = 0;
  typedef typename Voices::iterator Auto;
  for (Auto i = m_voices.begin(); i != m_voices.end();) {
    Voice & voice = i->second;

    if ((n >= num_ending) and voice.active()) {
      voice.advance();
      ++i;
    } else {
      m_voices.erase(i++);
    }
    ++n;
  }

  in.pull(time, m_fingers);

  // continue & initialize new voices
  // note: only active voices are polled
  for (size_t i = 0; i < m_fingers.size; ++i) {
    Id id = m_fingers.keys[i];
    Voice & voice = m_voices[id];

    voice.set_timbre(m_fingers.values[i]);
    poll += voice.poll();
  }
}

template<class Voice>
void CoupledChorus<Voice>::bounce (
    Seconds time,
    const complex & force,
    CoupledChorus<Voice>::Sound & sound)
{
  // advance & fuse signals
  typedef typename Voices::iterator Auto;
  for (Auto i = m_voices.begin(); i != m_voices.end(); ++i) {
    i->second.sample(force, sound);
  }
}

typedef CoupledChorus<Synthesis::Coupled::SyncoBlob> SyncoBlobs;
typedef CoupledChorus<Synthesis::Coupled::SyncoPoint> SyncoPoints;
typedef CoupledChorus<Synthesis::Coupled::SyncoString> SyncoStrings;

//----( syncopipes )----------------------------------------------------------

class SyncoPipes : public CoupledChorus<Synthesis::Coupled::SyncoPipe>
{
  float m_pitch_shift;
  Synthesis::Lowpass<complex> m_lowpass;
  StereoAudioFrame m_sound;
  std::vector<complex> m_phases;

public:

  typedef Synthesis::Coupled::SyncoPipe Voice;

  Port<Pushed<std::vector<complex> > > phases_monitor;

  SyncoPipes (
      size_t capacity = DEFAULT_CHORUS_CAPACITY,
      float pitch_shift = 0)
    : CoupledChorus<Voice>(capacity),
      m_pitch_shift(pitch_shift),
      m_lowpass(4),
      phases_monitor("SyncoPipes.phases_monitor")
  {}
  virtual ~SyncoPipes () {}

  virtual void bounce (
      Seconds time,
      const complex & force,
      CoupledChorus<Voice>::Sound & sound);
};

//----( wobbler )-------------------------------------------------------------

class Wobbler : public CoupledBand<StereoAudioFrame>::Member
{
  typedef Synthesis::Coupled::SyncoWobble Voice;

  CoupledChorus<Voice> m_chorus;
  complex m_envelope;
  const complex m_trans;
  complex m_phase;

public:


  Port<Pulled<BoundedMap<Id, Finger> > > & in;

  Wobbler (
      size_t capacity = DEFAULT_CHORUS_CAPACITY,
      float tone_freq_hz = WOBBLER_TONE_FREQ_HZ);
  virtual ~Wobbler () {}

  virtual void pull (Seconds time, Synchronized::Poll & poll);

  virtual void bounce (
      Seconds time,
      const complex & force,
      StereoAudioFrame & sound_accum);
};

} // namespace Streaming

#endif // KAZOO_STREAMING_SYNTHESIS_H

