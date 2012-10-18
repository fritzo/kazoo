
#include "streaming_audio.h"

#define AUDIO_PEAK_SNR                  (5.0f)
#define AUDIO_PEAK_TIMESCALE_SEC        (0.1f)
#define AUDIO_PEAK_TIMESCALE            ( AUDIO_PEAK_TIMESCALE_SEC \
                                        / DEFAULT_AUDIO_FRAMERATE )

#define HEAR_VECTOR_FREQ0               (4e0f / DEFAULT_SAMPLE_RATE)
#define HEAR_VECTOR_FREQ1               (4e3f / DEFAULT_SAMPLE_RATE)

namespace Streaming
{

//----( power meter )---------------------------------------------------------

PowerMeter::PowerMeter ()
  : out("PowerMeter.out")
{}

void PowerMeter::push (Seconds time, const StereoAudioFrame & sound)
{
  float power = sqrtf(max_norm_squared(sound));
  out.push(time, power);
}

PowerSplitter::PowerSplitter ()
  : out1("PowerSplitter.out1"),
    out2("PowerSplitter.out2")
{}

void PowerSplitter::push (Seconds time, const StereoAudioFrame & sound)
{
  const float * restrict data = (const float *) sound.data;
  float power1 = 0;
  float power2 = 0;
  for (size_t i = 0, I = sound.size; i < I; ++i) {
    float x = data[2 * i + 0];
    float y = data[2 * i + 1];

    imax(power1, max(x,-x));
    imax(power2, max(y,-y));
  }

  out1.push(time, power1);
  out2.push(time, power2);
}

//----( denoiser )------------------------------------------------------------

Denoiser::Denoiser ()
  : m_denoiser(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_AUDIO_FRAMERATE),
    out("Denoiser.out")
{}

void Denoiser::push (Seconds time, const float & value)
{
  out.push(time, m_denoiser(value));
}

//----( impact detector )-----------------------------------------------------

ImpactDetector::ImpactDetector ()
  : m_detector(AUDIO_PEAK_SNR, AUDIO_PEAK_TIMESCALE),
    out("ImpactDetector.out")
{}

void ImpactDetector::push (Seconds time, const float & value)
{
  bool impact = m_detector.detect(value);

  if (impact) cout << " x " << endl;

  out.push(time, impact);
}

//----( autogain speaker )----------------------------------------------------

MicGain::MicGain ()
  : m_gain(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_AUDIO_FRAMERATE),

    out("MicGain.out")
{}

void MicGain::push (Seconds time, const StereoAudioFrame & const_sound)
{
  // WARNING HACK this uses input as working memory
  StereoAudioFrame & sound = const_cast<StereoAudioFrame &>(const_sound);

  float power_out = rms(sound);
  sound *= m_gain.update(power_out);
  soft_clip(sound);

  out.push(time, sound);
}

SpeakerGain::SpeakerGain ()
  : m_gain(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_AUDIO_FRAMERATE),

    in("SpeakerGain.in")
{}

void SpeakerGain::pull (Seconds time, StereoAudioFrame & sound)
{
  in.pull(time, sound);

  float power_out = rms(sound);
  sound *= m_gain.update(power_out);
  soft_clip(sound);
}

//----( audio mixer )---------------------------------------------------------

AudioMixer::AudioMixer ()
  : in1("AudioMixer.in1"),
    in2("AudioMixer.in2")
{}

void AudioMixer::pull (Seconds time, StereoAudioFrame & sound)
{
  in1.pull(time, sound);
  in2.pull(time, m_sound);

  sound += m_sound;
}

//----( beat perception )-----------------------------------------------------

EnergyToBeat::EnergyToBeat ()
  : m_energy(0),
    beat_monitor("EnergyToBeat.beat_monitor")
{}

void EnergyToBeat::push (Seconds time, const float & power)
{
  complex beat = m_energy_to_beat.transform_fwd(time, power);
  m_energy = m_energy_to_beat.transform_bwd(beat);

  if (beat_monitor) beat_monitor.push(time, beat);
}

void EnergyToBeat::pull (Seconds time, StereoAudioFrame & sound)
{
  float energy = m_energy;

  for (size_t t = 0, T = sound.size; t < T; ++t) {
    sound[t] = energy * random_normal_complex();
  }
}

//----( heaaring vectors )----------------------------------------------------

HearVector::HearVector (size_t size)
  : m_bank(Synchronized::Bank(
        size,
        HEAR_VECTOR_FREQ0,
        HEAR_VECTOR_FREQ1)),
    m_amplitude0(size),
    m_amplitude1(size),
    in("HearVector.in", size)
{
  m_amplitude0.zero();
}

HearVector::~HearVector ()
{
  PRINT3(min(m_amplitude0), mean(m_amplitude0), max(m_amplitude0));
  PRINT3(min(m_amplitude1), mean(m_amplitude1), max(m_amplitude1));
}

void HearVector::pull (Seconds time, StereoAudioFrame & sound)
{
  in.pull(time, m_amplitude1);

  Vector<float> & amp0 = m_amplitude0;
  Vector<float> & damp = m_amplitude1;

  imax(amp0, 0.0f);
  imax(damp, 0.0f);

  damp -= amp0;

  sound.zero();
  m_bank.sample_accum(amp0, damp, sound);

  amp0 += damp;
}

} // namespace Streaming

