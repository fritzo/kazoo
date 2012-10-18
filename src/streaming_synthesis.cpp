
#include "streaming_synthesis.h"
#include "splines.h"
#include <algorithm>
#include <limits.h>

#define TOL (1e-8f)

namespace Streaming
{

//----( vocoder )-------------------------------------------------------------

Vocoder::Vocoder ()
  : m_timbre(),

    m_mass_gain(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_SCREEN_FRAMERATE),
    m_bend_gain(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_SCREEN_FRAMERATE),

    in("Vocoder.in")
{}

void Vocoder::pull (Seconds time, StereoAudioFrame & sound_accum)
{
  m_timbre.zero();
  in.pull(time, m_timbre);

  m_mutex.lock();
  sample(m_timbre, sound_accum);
  m_mutex.unlock();
}

void Vocoder::pull (Seconds time, RgbImage & image)
{
  ASSERT_SIZE(image.red, size);

  Vector<float> & mass = image.blue;
  Vector<float> & bend_up = image.red;
  Vector<float> & bend_down = image.green;

  m_mutex.lock();
  mass = m_mass;
  bend_up = m_slow_bend;
  m_mutex.unlock();

  const size_t I = mass.size;

  float * restrict m = mass;
  float * restrict b_up = bend_up;
  float * restrict b_down = bend_down;

  float max_m = 0;
  float max_b = 0;

  for (size_t i = 0; i < I; ++i) {
    float b = b_up[i];
    imax(max_b, b_up[i] = max(0.0f, b));
    imax(max_b, b_down[i] = max(0.0f, -b));
    imax(max_m, m[i]);
  }

  float bend_scale = m_bend_gain.update(max_b);
  float mass_scale = m_mass_gain.update(max_m);

  mass *= mass_scale;
  bend_up *= bend_scale;
  bend_down *= bend_scale;
}

//----( vocoder testing )----

void SimVocoderChirp::pull (Seconds time, Vocoder::Timbre & timbre)
{
  float dt = max(1e-8f, time - m_time);
  m_state = wrap(m_state + m_rate * dt);
  m_time = time;

  static const float dpitch
    = log( SYNTHESIS_VOCODER_MAX_FREQ_HZ
         / SYNTHESIS_VOCODER_MIN_FREQ_HZ )
    / SYNTHESIS_VOCODER_SIZE;

  timbre.zero();
  for (size_t n = 1; n <= m_num_harmonics; ++n) {
    float state = m_state + log(n) / dpitch / timbre.size;
    if (state > 1) break;

    LinearInterpolate(state * timbre.size, timbre.size).iadd(timbre, 1.0f);
  }
}

void SimVocoderChord::pull (Seconds time, Vocoder::Timbre & timbre)
{
  float dt = max(1e-8f, time - m_time);
  m_state = wrap(m_state + m_rate * dt);
  m_time = time;

  timbre.zero();
  for (size_t n = 1; n <= m_num_tones; ++n) {
    float pitch_01 = 0.5f - 0.4f * cos(2 * M_PI * n * m_state);

    LinearInterpolate(pitch_01 * timbre.size, timbre.size).iadd(timbre, 1.0f);
  }
}

void SimVocoderDrone::pull (Seconds time, Vocoder::Timbre & timbre)
{
  float dt = max(1e-8f, time - m_time);
  m_state = wrap(m_state + m_rate * dt);
  m_time = time;

  timbre.set(1.0f / timbre.size);
  float amp = 1 + cos(2 * M_PI * m_state);
  for (size_t n = 0; n < m_num_tones; ++n) {
    timbre[timbre.size / (n+2)] = amp;
  }
}

void SimVocoderNoiseBand::pull (Seconds time, Vocoder::Timbre & timbre)
{
  float dt = max(1e-8f, time - m_time);
  m_state = wrap(m_state + m_rate * dt);
  m_time = time;

  size_t I = timbre.size;
  float LB = I * (0.5f - 0.4f * cos(2 * M_PI * m_state));
  float UB = I * (0.5f - 0.4f * cos(2 * M_PI * m_state * m_ratio));
  float center = (UB + LB) / 2;
  float radius = max(1.0f, fabsf(UB - LB) / 2);

  for (size_t i = 0; i < I; ++i) {
    timbre[i] = exp(-sqr((i - center) / radius) / 2);
  }
}

//----( beater )--------------------------------------------------------------

Beater::Beater (bool coalesce, float blur_factor)
  : Synthesis::Beater(coalesce, blur_factor),
    shape(period, size),
    beat_in("Beater.beat_in"),
    fingers_in("Beater.fingers_in"),
    power_in("Beater.power_in")
{
  m_beat.zero();
  m_beat += 1.0f;
}

void Beater::pull (Seconds time, Timbre & amplitude)
{
  m_mutex.lock();

  advance();

  power_in.pull(time, m_power);
  if (beat_in) {
    beat_in.pull(time, m_beat);
    multiply_add(m_power, m_beat, mass_now);
  } else {
    mass_now += m_power;
  }

  m_mutex.unlock();

  sample(amplitude);
}

void Beater::pull (Seconds time, BoundedMap<Id, Gestures::Finger> & fingers)
{
  m_mutex.lock();

  advance();

  power_in.pull(time, m_power);
  fingers_in.pull(time, fingers);

  for (size_t i = 0; i < fingers.size; ++i) {
    const Finger & finger = fingers.values[i];

    float mass = finger.get_z();
    float tempo = bound_to(0.0f, 1.0f, finger.get_y() / GRID_SIZE_Y + 0.5f);
    LinearInterpolate(tempo * size, size).iadd(mass_now, m_power * mass);
  }

  m_mutex.unlock();

  sample(m_beat);

  for (size_t i = 0; i < fingers.size; ++i) {
    Finger & finger = fingers.values[i];

    float mass = finger.get_z();
    float tempo = bound_to(0.0f, 1.0f, finger.get_y() / GRID_SIZE_Y + 0.5f);
    float beat = LinearInterpolate(tempo * size, size).get(m_beat);

    finger.set_energy(mass * beat);
  }
}

void Beater::pull (Seconds time, MonoImage & image)
{
  ASSERT_SIZE(image, shape.size());

  m_mutex.lock();
  image = LoopBank::m_mass_tp;
  m_mutex.unlock();
}

//----( piano )---------------------------------------------------------------

Piano::Piano (
    size_t range,
    float mid_freq_hz,
    float pitch_step,
    float sustain_sec,
    float release_sec)

  : m_size(range),
    m_sustain(sustain_sec * DEFAULT_SAMPLE_RATE),
    m_release(release_sec * DEFAULT_SAMPLE_RATE),

    m_freq(range),
    m_attack(range),
    m_rate(range),
    m_phase(range),

    impact_in(0)
{
  m_rate.zero();
  m_phase.zero();
  m_attack.zero();

  float mid_freq = mid_freq_hz / DEFAULT_SAMPLE_RATE;
  float min_pitch = log(mid_freq) - pitch_step * (range / 2);
  for (size_t i = 0; i < range; ++i) {
    float freq = expf(min_pitch + pitch_step * i);
    m_freq[i] = 2 * M_PI * freq;
  }
}

void Piano::push (Seconds time, const Vector<float> & tones)
{
  ASSERT_SIZE(tones, size());

  float attack_scale = 1 / sqrt(norm_squared(tones));

  m_mutex.lock(); //--------

  for (size_t i = 0; i < size(); ++i) {
    float timescale = m_release + m_sustain * tones[i];
    m_rate[i] = exp(complex(-1 / timescale, m_freq[i]));
    m_attack[i] = attack_scale * tones[i];
  }

  m_mutex.unlock(); //--------
}

void Piano::pull (Seconds time, StereoAudioFrame & sound_accum)
{
  float impact;
  impact_in.pull(time, impact);

  m_mutex.lock(); //--------

  for (size_t t = 0, T = sound_accum.size; t < T; ++t) {
    complex accumulator = 0;

    for (size_t i = 0, I = size(); i < I; ++i) {

      complex & phase = m_phase[i];
      complex rate = m_rate[i];
      float energy = norm(phase);
      complex attack = impact
                     * m_attack[i]
                     * (energy > TOL ? phase * powf(energy, -0.5)
                                     : complex(1,0));

      phase += attack * impact;
      phase *= rate;
      accumulator += phase;
    }
    sound_accum[t] += accumulator;
  }

  m_mutex.unlock(); //--------
}

//----( actuators )-----------------------------------------------------------

Actuators::Actuators (
    size_t range,
    float mid_freq_hz,
    float pitch_step,
    float sustain_sec,
    float release_sec)

 :  m_size(range),
    m_sustain(sustain_sec * DEFAULT_SAMPLE_RATE),
    m_release(release_sec * DEFAULT_SAMPLE_RATE),

    m_freq(range),
    m_decay(range),
    m_attack(range),
    m_diffuse(range),
    m_power(range),
    m_sawtooth(range),

    impact_in(0)
{
  m_decay.zero();
  m_attack.zero();
  m_diffuse.zero();
  m_power.zero();
  for (size_t i = 0, I = range; i < I; ++i) {
    m_sawtooth[i] = 0;
  }

  float mid_freq = mid_freq_hz / DEFAULT_SAMPLE_RATE;
  float min_pitch = log(mid_freq) - pitch_step * (range / 2);
  for (size_t i = 0, I = range; i < I; ++i) {
    float freq = expf(min_pitch + pitch_step * i);
    m_freq[i] = roundi(freq * UINT_MAX);
  }
  ASSERTW(1000 < m_freq[0],
          "actuator tones have low precision: " << (1.0f / m_freq[0]));
}

void Actuators::push (Seconds time, const Vector<float> & tones)
{
  ASSERT_SIZE(tones, size());

  float attack_scale = safe_div(1.0f, sqrtf(norm_squared(tones)));

  m_mutex.lock(); //--------

  for (size_t i = 0, I = size(); i < I; ++i) {
    float timescale = m_release + m_sustain * tones[i];
    m_decay[i] = timescale > TOL ? exp(-1 / timescale) : 0;
    m_attack[i] = attack_scale * tones[i] / UINT_MAX;
  }

  for (size_t i = 0, I = size() - 1; i < I; ++i) {
    m_diffuse[i] = (tones[i+1] - tones[i])
                 / (tones[i+1] + tones[i] + TOL)
                 * DEFAULT_VIDEO_FRAMERATE
                 / DEFAULT_SAMPLE_RATE;
  }

  m_mutex.unlock(); //--------
}

void Actuators::pull (Seconds time, StereoAudioFrame & sound_accum)
{
  float impact;
  impact_in.pull(time, impact);

  m_mutex.lock(); //--------

  for (size_t t = 0, T = sound_accum.size; t < T; ++t) {
    float accumulator = 0;

    for (size_t i = 0, I = size(); i < I; ++i) {
      m_sawtooth[i] += m_freq[i];
      m_power[i] *= m_decay[i];
      m_power[i] += impact * m_attack[i];
      accumulator += m_power[i] * m_sawtooth[i];
    }

    for (size_t i = 0, I = size() - 1; i < I; ++i) {
      float diffuse = m_diffuse[i];
      float power = diffuse * (diffuse > 0 ? m_power[i] : m_power[i+1]);

      m_power[i] -= power;
      m_power[i+1] += power;
    }

    sound_accum[t] += accumulator;
  }

  m_mutex.unlock(); //--------
}

//----( resonators )----------------------------------------------------------

#define SYNTHESIS_RESONANCE_Q_FACTOR    (0.99f)
#define SYNTHESIS_RESONANCE_LEAK        (1 - SYNTHESIS_RESONANCE_Q_FACTOR)

Resonators::Resonators (
    size_t range,
    float mid_freq_hz,
    float pitch_step)

  : Bounced<StereoAudioFrame, StereoAudioFrame, size_t>(
      DEFAULT_FRAMES_PER_BUFFER),

    m_size(range),
    m_rate(range),
    m_power(range),
    m_phase(range)
{
  m_power.zero();
  m_phase.zero();

  float mid_freq = mid_freq_hz / DEFAULT_SAMPLE_RATE;
  float min_pitch = log(mid_freq) - pitch_step * (range / 2);

  for (size_t i = 0; i < range; ++i) {
    float freq = expf(min_pitch + pitch_step * i);
    m_rate[i] = SYNTHESIS_RESONANCE_Q_FACTOR * exp_2_pi_i(freq);
  }
}

void Resonators::push (Seconds time, const Vector<float> & tones)
{
  ASSERT_SIZE(tones, size());

  float scale = 1 / sqrt(norm_squared(tones));

  m_mutex.lock(); //--------

  for (size_t i = 0, I = size(); i < I; ++i) {
    m_power[i] = scale * tones[i];
  }

  m_mutex.unlock(); //--------
}

void Resonators::bounce (
    Seconds time,
    const StereoAudioFrame & sound_in,
    StereoAudioFrame & sound_out)
{
  ASSERT_EQ(sound_in.size, sound_out.size);

  m_mutex.lock(); //--------

  for (size_t t = 0, T = sound_in.size; t < T; ++t) {
    complex actuate = sound_in[t];
    complex accumulator = 0;

    for (size_t i = 0, I = size(); i < I; ++i) {

      complex & phase = m_phase[i];
      complex rate = m_rate[i];

      phase = rate * phase + SYNTHESIS_RESONANCE_LEAK * actuate;
      accumulator += m_power[i] * phase;
    }
    sound_out[t] = accumulator;
  }

  m_mutex.unlock(); //--------
}

//----( vectorized chorus )---------------------------------------------------

template<class Voice>
void VectorizedChorus<Voice>::pull (
    Seconds time,
    StereoAudioFrame & sound_accum)
{
  // advance existing active voices
  // WARNING Voice is responsible for preventing many short tracks from
  //   staying active after advancing.  Two example strategies are:
  //   (1) make all voices inactve soon after mass is set to zero.
  //   (2) only allow post-death activity proportional to preceding lifetime.
  typedef typename Voices::iterator Auto;
  for (Auto i = m_voices.begin(); i != m_voices.end();) {
    Voice & voice = i->second;

    if (voice.active()) {
      voice.advance();
      ++i;
    } else {
      m_voices.erase(i++);
    }
  }

  in.pull(time, m_fingers);

  // initialize new voices
  for (size_t i = 0; i < m_fingers.size; ++i) {
    Id id = m_fingers.keys[i];
    m_voices[id].set_timbre(m_fingers.values[i]);
  }

  // vectorize voices
  m_voice_vector.resize(m_voices.size());
  {
    size_t j = 0;
    for (Auto i = m_voices.begin(); i != m_voices.end(); ++i) {
      m_voice_vector[j++] = i->second;
    }
  }

  sample(sound_accum);

  // de-vectorize voices
  {
    size_t j = 0;
    for (Auto i = m_voices.begin(); i != m_voices.end(); ++i) {
      i->second = m_voice_vector[j++];
    }
  }
  m_voice_vector.clear();
}

#define INSTANTIATE_TEMPLATES(T) \
  template void VectorizedChorus<T>::pull (Seconds, StereoAudioFrame &);

INSTANTIATE_TEMPLATES(Synthesis::Coupled::BeatingSine)
INSTANTIATE_TEMPLATES(Synthesis::Coupled::BeatingPlucked)
INSTANTIATE_TEMPLATES(Synthesis::Coupled::SyncoSine)
INSTANTIATE_TEMPLATES(Synthesis::Coupled::SyncoPlucked)
INSTANTIATE_TEMPLATES(Synthesis::Coupled::Shepard4)
INSTANTIATE_TEMPLATES(Synthesis::Coupled::Shepard7)
INSTANTIATE_TEMPLATES(Synthesis::Coupled::SitarString)
INSTANTIATE_TEMPLATES(Synthesis::Coupled::Sine)

#undef INSTANTIATE_TEMPLATES

template<class Voice>
void SimpleVectorizedChorus<Voice>::sample (StereoAudioFrame & sound_accum)
{
  const size_t I = VectorizedChorus<Voice>::m_voice_vector.size();
  const size_t T = sound_accum.size;
  complex * restrict sound = sound_accum;
  Voice * restrict voices = &(VectorizedChorus<Voice>::m_voice_vector[0]);

  for (size_t t = 0; t < T; ++t) {
    float time_01 = (t + 0.5f) / T;

    // compute mass & coupling force
    Synchronized::Poll poll;
    for (size_t i = 0; i < I; ++i) {
      voices[i].interpolate(time_01);
      poll += voices[i].poll();
    }
    complex force = poll.mean();

    // advance & fuse signals
    complex mix = 0;
    for (size_t i = 0; i < I; ++i) {
      mix += voices[i].sample(force);
    }

    sound[t] += mix;
  }
}

#define INSTANTIATE_TEMPLATES(T) \
  template void SimpleVectorizedChorus<T>::sample (StereoAudioFrame &);

INSTANTIATE_TEMPLATES(Synthesis::Coupled::Sine)
INSTANTIATE_TEMPLATES(Synthesis::Coupled::Shepard4)
INSTANTIATE_TEMPLATES(Synthesis::Coupled::Shepard7)
INSTANTIATE_TEMPLATES(Synthesis::Coupled::SitarString)

#undef INSTANTIATE_TEMPLATES

//----( wideband )------------------------------------------------------------

void Wideband::sample (StereoAudioFrame & sound_accum)
{
  const size_t I = m_voice_vector.size();
  const size_t T = sound_accum.size;
  complex * restrict sound = sound_accum;
  Voice * restrict voices = &(m_voice_vector[0]);

  for (size_t t = 0; t < T; ++t) {
    float time_01 = (t + 0.5f) / T;

    // compute mass & coupling force
    Synchronized::Poll poll;
    for (size_t i = 0; i < I; ++i) {
      voices[i].interpolate(time_01);
      poll += voices[i].poll();
    }
    complex force = poll.mean();

    // advance & fuse signals
    float mix = 0;
    for (size_t i = 0; i < I; ++i) {
      mix += voices[i].sample(force).real();
    }
    float product = expf(mix) - 1;

    sound[t] += complex(1,1) * m_highpass(product);
  }
}

//----( splitband )-----------------------------------------------------------

template<class Voice>
void Splitband<Voice>::sample (StereoAudioFrame & sound_accum)
{
  const size_t I = VectorizedChorus<Voice>::m_voice_vector.size();
  const size_t T = m_sound.size;
  complex * restrict sound = m_sound;
  Voice * restrict voices = &(VectorizedChorus<Voice>::m_voice_vector[0]);

  // couple slow time scale
  Synchronized::Poll poll;
  for (size_t i = 0; i < I; ++i) {
    poll += voices[i].poll_slow();
  }
  complex force = poll.mean();
  for (size_t i = 0; i < I; ++i) {
    voices[i].sample_slow(force);
  }

  // couple fast time scale
  for (size_t t = 0; t < T; ++t) {
    float time_01 = (t + 0.5f) / T;

    // compute mass & coupling force
    Synchronized::Poll poll;
    for (size_t i = 0; i < I; ++i) {
      voices[i].interpolate(time_01);
      poll += voices[i].poll_fast();
    }
    complex force = poll.mean();

    // advance & fuse signals
    complex mix = 0.0f;
    for (size_t i = 0; i < I; ++i) {
      mix += voices[i].sample_fast(force);
    }

    sound[t] = mix;
  }

  // increases apparent loudness
  m_sound *= SPLITBAND_SHARPNESS;
  soft_clip(m_sound);
  sound_accum += m_sound;
}

template void Splitband<Synthesis::Coupled::BeatingSine>::sample (
    StereoAudioFrame & sound);
template void Splitband<Synthesis::Coupled::BeatingPlucked>::sample (
    StereoAudioFrame & sound);
template void Splitband<Synthesis::Coupled::SyncoSine>::sample (
    StereoAudioFrame & sound);
template void Splitband<Synthesis::Coupled::SyncoPlucked>::sample (
    StereoAudioFrame & sound);

//----( sitar )---------------------------------------------------------------

Sitar::Sitar (size_t hand_capacity)
  : m_hands(hand_capacity),

    m_timescale(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_AUDIO_FRAMERATE),
    m_hand_energy_gain(m_timescale),
    m_finger_energy_gain(m_timescale),
    m_finger_pos_gain(m_timescale),
    m_finger_vel_gain(m_timescale),

    hands_in("Sitar.hands_in"),
    fingers_in("Sitar.fingers_in")
{}

Sitar::~Sitar ()
{
  PRINT(m_hand_energy_gain);
  PRINT2(m_finger_pos_gain, m_finger_vel_gain);
}

void Sitar::pull (Seconds time, BoundedMap<Id, Gestures::Finger> & plectra)
{
  hands_in.pull(time, m_hands);
  fingers_in.pull(time, plectra);

  if (not m_hands.empty()) {

    const size_t H = m_hands.size;
    Finger * restrict hands = m_hands.values;
    Id * restrict ids = m_hands.keys;

    float max_energy = 0;
    for (size_t h = 0; h < H; ++h) {
      ids[h] = 2 * ids[h]; // shuffle even
      imax(max_energy, hands[h].get_energy());
    }

    //float energy_gain = m_hand_energy_gain.update(max_energy);
    for (size_t h = 0; h < H; ++h) {
      Finger & hand = hands[h];

      //hand.set_energy(energy_gain * hand.get_energy() * hand.get_z());
      hand.set_x(hand.get_x() / 4 - 1); // compress & shift pitch
      //hand.set_y(GRID_SIZE_Y * (hand.get_energy() * energy_gain - 0.5f));
    }
  }

  if (not plectra.empty()) {

    const size_t F = plectra.size;
    Finger * restrict fingers = plectra.values;
    Id * restrict ids = plectra.keys;

    float max_energy = 0;
    float sum_z = 0;
    float sum_pos2 = 0;
    float sum_vel2 = 0;
    for (size_t f = 0; f < F; ++f) {
      ids[f] = 2 * ids[f] + 1; // shuffle odd
      const Finger & finger = fingers[f];
      float z = finger.get_z();

      imax(max_energy, fingers[f].get_energy());
      sum_z += z;
      sum_pos2 += z * (sqr(finger.get_x()) + sqr(finger.get_y()));
      sum_vel2 += z * (sqr(finger.get_x_t()) + sqr(finger.get_y_t()));
    }

    float energy_gain = m_finger_energy_gain.update(max_energy);
    float pos_gain = 3.0f * m_finger_pos_gain.update(sum_pos2, sum_z);
    float vel_gain = 0.5f * m_finger_vel_gain.update(sum_vel2, sum_z);
    for (size_t f = 0; f < F; ++f) {
      Finger & finger = fingers[f];

      float x = pos_gain * finger.get_x() + vel_gain * finger.get_x_t();
      float y = pos_gain * finger.get_y() + vel_gain * finger.get_y_t();

      finger.energy() *= energy_gain;
      finger.set_x(x);
      finger.set_y(y);
    }
  }

  ASSERT_LE(m_hands.size + plectra.size, plectra.capacity);
  for (size_t h = 0; h < m_hands.size; ++h) {
    plectra.add(m_hands.keys[h], m_hands.values[h]);
  }
}

//============================================================================

//----( hats )----------------------------------------------------------------

CoupledHats::CoupledHats (size_t capacity, float tempo_hz, float freq_hz)
  : m_tempo(Synthesis::Coupled::g_synco_param),
    m_freq(freq_hz / DEFAULT_SAMPLE_RATE),
    m_trans(exp( complex(-m_freq, 2 * M_PI * m_freq))),
    m_state(0.0f,0.0f),
    m_fingers(capacity),
    m_mass(0),
    in("CoupledHats.in")
{
  m_tempo.freq = tempo_hz * 2 * M_PI / DEFAULT_AUDIO_FRAMERATE;
}

void CoupledHats::pull (Seconds time, Synchronized::Poll & poll)
{
  poll += m_tempo.poll();
}

void CoupledHats::bounce (
    Seconds time,
    const complex & force,
    Sound & sound_accum)
{
  in.pull(time, m_fingers);

  complex phase = m_tempo.sample(force);
  float angle = wrap(arg(phase) / (2 * M_PI));

  m_mass = 0;
  float amplitude = 0;
  for (size_t i = 0; i < m_fingers.size; ++i) {
    const Finger & finger = m_fingers.values[i];

    float mass = finger.get_z();
    float beat = wrap(angle - finger.get_y()); // counter-clockwise
    float rate = 2 * powf(2, COUPLED_HATS_NUM_OCTAVES * finger.get_x());

    imax(m_mass, mass);
    amplitude += mass * rate * expf(-rate * beat);
  }

  const size_t T = sound_accum.size;
  complex * restrict sound = sound_accum;
  const float freq = m_freq;
  const complex trans = m_trans;
  complex state = m_state;

  for (size_t t = 0; t < T; ++t) {
    complex actuate = amplitude * random_normal_complex();
    complex raw = state = trans * state + freq * actuate;
    sound[t] += raw / (1 + abs(raw));
  }

  m_state = state;
}

//----( syncopipes )----------------------------------------------------------

void SyncoPipes::bounce (
    Seconds time,
    const complex & force,
    CoupledChorus<Voice>::Sound & sound_accum)
{
  typedef Voices::iterator Auto;
  for (Auto i = m_voices.begin(); i != m_voices.end(); ++i) {
    i->second.shift_pitch(m_pitch_shift);
  }

  m_sound.zero();
  CoupledChorus<Voice>::bounce(time, force, m_sound);

  m_lowpass.sample(m_sound);
  sound_accum += m_sound;

  if (phases_monitor) {
    m_phases.clear();
    for (Auto i = m_voices.begin(); i != m_voices.end(); ++i) {
      m_phases.push_back(i->second.as_complex());
    }
    phases_monitor.push(time, m_phases);
  }
}

//----( wobbler )-------------------------------------------------------------

Wobbler::Wobbler (size_t capacity, float tone_freq_hz)
  : m_chorus(capacity),
    m_envelope(0,0),
    m_trans(exp_2_pi_i(tone_freq_hz / DEFAULT_SAMPLE_RATE)),
    m_phase(1.0f, 0.0f),
    in(m_chorus.in)
{
}

void Wobbler::pull (Seconds time, Synchronized::Poll & poll)
{
  m_chorus.pull(time, poll);
}

void Wobbler::bounce (
    Seconds time,
    const complex & force,
    StereoAudioFrame & sound_accum)
{
  const complex envelope0 = m_envelope;
  m_envelope = 0;
  m_chorus.bounce(time, force, m_envelope);
  const complex denvelope = m_envelope - envelope0;

  const size_t T = DEFAULT_FRAMES_PER_BUFFER;
  complex * restrict sound = sound_accum;
  const complex trans = m_trans;
  complex phase = m_phase;

  for (size_t t = 0; t < T; ++t) {
    float dt = (t + 1.0f) / T;
    complex envelope = envelope0 + denvelope * dt;
    complex overtones = exp(WOBBLER_SHARPNESS * envelope * phase) - 1.0f;
    sound[t] += overtones / (1 + abs(overtones));
    phase *= trans;
  }

  m_phase = phase * (1.0f / abs(phase));
}

} // namespace Streaming

