
#include "bucket.h"
#include "splines.h"
#include "images.h"

#define BUCKET_IMPEDANCE_RATIO          (32.0f)
#define BUCKET_PITCH_SCALE              (1.0f)
#define BUCKET_PITCH_CENTER             (-1.5f)
#define BUCKET_SUSTAIN_POWER            (1.0f)
#define BUCKET_ATTACK_POWER             (2.0f)
#define BUCKET_ATTACK_ROUGHNESS         (2.0f)
#define BUCKET_ATTACK_DECAY_TIME        (0.2f)
#define BUCKET_LOWPASS_TIMESCALE        (64.0f)

#define BUCKET_PEAK_SNR                 (5.0f)
#define BUCKET_PEAK_TIMESCALE_SEC       (0.1f)
#define LOOP_RELEASE_TIMESCALE          (0.5f)
#define LOOP_SUSTAIN_TIMESCALE          (16.0f)

#define TOL                             (1e-20f)

#define LOG1(message)

namespace Streaming
{

//----( components )----------------------------------------------------------

//----( voice control )----

VoiceControl::VoiceControl (Rectangle shape)
  : Rectangle(shape),
    m_circ(SYNTHESIS_LOOP_RESONATOR_SIZE),

    m_impedance_gain(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_VIDEO_FRAMERATE),

    m_x2(width()),
    m_y2(height()),
    m_angle(size()),

    m_temp(m_circ),
    m_state(m_circ, 0),

    out("VoiceControl.out")
{
  ASSERT_LE(m_circ, 256);

  const size_t I = m_width;
  const size_t J = m_height;
  const size_t A = m_circ;

  // x,y range in (-1,1)
  for (size_t i = 0; i < I; ++i) {
    float x = (i + 0.5f) / I * 2 - 1;
    m_x2[i] = sqr(x);
  }

  for (size_t j = 0; j < J; ++j) {
    float y = (j + 0.5f) / J * 2 - 1;
    m_y2[j] = sqr(y);
  }

  for (size_t i = 0; i < I; ++i) {
    float x = (i + 0.5f) / I * 2 - 1;

    for (size_t j = 0; j < J; ++j) {
      float y = (j + 0.5f) / J * 2 - 1;

      m_angle[J * i + j] = roundu(atan2(y,x) / (2 * M_PI) * A + A) % A;
    }
  }
}

void VoiceControl::push (Seconds time, const MonoImage & image)
{
  ASSERT_SIZE(image, size());

  const size_t I = m_width;
  const size_t J = m_height;
  const size_t A = m_circ;

  // extract radial moment, and accumulate x position

  m_state.impedance.zero();

  const float * restrict x2 = m_x2;
  const float * restrict y2 = m_y2;
  float * restrict imp = m_state.impedance;
  float * restrict temp = m_temp;

  float max_m = 0;
  float sum_m = 0;
  float sum_mx2 = 0;
  float sum_my2 = 0;

  for (size_t i = 0; i < I; ++i) {
    const float * restrict im = image.data + J * i;
    const unsigned char * restrict angle = m_angle + J * i;

    for (size_t j = 0; j < J; ++j) {
      imp[angle[j]] += im[j];
    }

    float sum_m_i = 0;
    for (size_t j = 0; j < J; ++j) {

      float m = im[j];

      imax(max_m, m);
      sum_m_i += m;
      sum_my2 += m * y2[j];
    }

    sum_m += sum_m_i;
    sum_mx2 += sum_m_i * x2[i];
  }

  imax(max_m, TOL);

  float prior_at_zero = max_m;
  sum_m += prior_at_zero;
  float Er2 = (sum_mx2 + sum_my2) / sum_m;

  // blur
  for (size_t a = 0; a < A; ++a) {
    size_t a0 = a;
    size_t a1 = (a + 1) % A;
    temp[a] = imp[a0] + imp[a1];
  }
  for (size_t a = 0; a < A; ++a) {
    size_t a0 = a;
    size_t a1 = (a + 1) % A;
    imp[a] = temp[a0] + temp[a1];
  }

  float gain = m_impedance_gain.update(max(m_state.impedance));
  float shift = 1 / BUCKET_IMPEDANCE_RATIO;
  float scale = gain * (1 - shift);

  for (size_t a = 0; a < A; ++a) {
    imp[a] = scale * imp[a] + shift;
  }

  //--------

  m_state.sustain = sqrt(max_m);
  m_state.pitch = 2 * sqrt(Er2 + TOL) - 1;
  out.push(time, m_state);
}

//----( spin control )----

SpinControl::SpinControl (Rectangle shape)
  : Rectangle(shape),

    m_x(width()),
    m_y(height()),

    out("SpinControl.out")
{
  const size_t I = m_width;
  const size_t J = m_height;

  // x,y range in (-1,1)
  for (size_t i = 0; i < I; ++i) m_x[i] = (i + 0.5f) / I * 2 - 1;
  for (size_t j = 0; j < J; ++j) m_y[j] = (j + 0.5f) / J * 2 - 1;
}

void SpinControl::push (Seconds time, const MonoImage & image)
{
  const size_t I = m_width;
  const size_t J = m_height;

  const float * restrict x = m_x;
  const float * restrict y = m_y;
  const float * restrict im = image;

  float max_m = 0;
  float sum_m = 0;
  float sum_mx = 0;
  float sum_my = 0;
  float sum_mx2 = 0;
  float sum_my2 = 0;

  for (size_t i = 0; i < I; ++i) {
    float sum_m_i = 0;

    for (size_t j = 0; j < J; ++j) {
      float m = im[J * i + j];

      imax(max_m, m);
      sum_m_i += m;
      sum_my += m * y[j];
      sum_my2 += m * sqr(y[j]);
    }

    sum_m += sum_m_i;
    sum_mx += sum_m_i * x[i];
    sum_mx2 += sum_m_i * sqr(x[i]);
  }

  imax(sum_m, TOL);
  float Ex = sum_mx / sum_m;
  float Ey = sum_my / sum_m;
  float Er2 = (sum_mx2 + sum_my2) / sum_m;
  float radius = powf(Er2 + TOL, 0.7f);
  float r2 = sqr(Ex) + sqr(Ey);
  float angle = r2 > 0 ? atan2(Ey, Ex) : 0;

  //--------

  State state;
  state.phase = wrap(0.5f + angle / (2 * M_PI));
  state.rate = pow(4.0f, 1 - 2 * radius);
  state.amplitude = sqrt(max_m + TOL);

  out.push(time, state);
}

//----( formant synth )----

FormantSynth::FormantSynth (size_t circ, float pitch_shift)
  : m_pitch_center(BUCKET_PITCH_CENTER + pitch_shift),

    m_actuate1(),
    m_actuate2(),
    m_resonate(),
    m_lowpass(BUCKET_LOWPASS_TIMESCALE),

    m_state(circ),

    voice_in(circ),
    power_in(0)
{
  voice_in.unsafe_access().impedance.set(1.0f / BUCKET_IMPEDANCE_RATIO);
}

FormantSynth::~FormantSynth ()
{
  PRINT2(m_state.sustain, m_state.pitch);
  Vector<float> & impedance = m_state.impedance;
  PRINT3(min(impedance), mean(impedance), max(impedance));

  PRINT(m_actuate1);
  PRINT(m_actuate2);
}

void FormantSynth::pull (Seconds time, StereoAudioFrame & sound_accum)
{
  voice_in.pull(time, m_state);
  float pitch = m_state.pitch;
  float sustain = m_state.sustain;

  float power;
  power_in.pull(time, power);

  pitch = (BUCKET_PITCH_SCALE * pitch + m_pitch_center) / logf(2);

  power /= BUCKET_SUSTAIN_POWER + BUCKET_ATTACK_POWER;
  float power1 = power * BUCKET_SUSTAIN_POWER;
  float power2 = power * BUCKET_ATTACK_POWER;
  Synthesis::Glottis::Timbre timbre1(power1, sustain, pitch);
  Synthesis::ExpSine::Timbre timbre2(
      power2,
      0,
      pitch,
      BUCKET_ATTACK_ROUGHNESS);

  m_actuate1.sample(timbre1, sound_accum);
  m_resonate.sample(m_state.impedance, sound_accum);
  m_lowpass.sample(sound_accum);
  m_actuate2.sample(timbre2, sound_accum);
  soft_clip(sound_accum);
}

//----( loop synth )----

LoopSynth::LoopSynth (StereoAudioFrame & loop)
  : m_loop(loop),
    m_loop_state(),

    spin_in(SpinControl::State()),
    impact_in(false)
{
}

LoopSynth::~LoopSynth ()
{
  PRINT3(m_state.phase, m_state.rate, m_state.amplitude);
}

void LoopSynth::pull (Seconds time, StereoAudioFrame & sound_accum)
{
  spin_in.pull(time, m_state);

  float phase = m_state.phase;
  float rate = m_state.rate;
  float amp = m_state.amplitude;

  bool jump;
  impact_in.pull(time, jump);

  m_loop_state.set_rate_amp(rate, amp);
  if (jump) m_loop_state.jump_to_phase(phase);

  m_loop.sample(m_loop_state, sound_accum);
  soft_clip(sound_accum);
}

StereoAudioFrame LoopSynth::load_loop (
    const char * filename,
    float duration_sec,
    float begin_sec)
{
  size_t loop_size = roundu(duration_sec * DEFAULT_SAMPLE_RATE);
  size_t loop_begin = roundu(begin_sec * DEFAULT_SAMPLE_RATE);
  PRINT2(loop_size, loop_begin);

  complex * restrict loop_data = malloc_complex(loop_size);
  StereoAudioFrame loop(loop_size, loop_data);
  bool good = read_audio_sample(filename, loop, loop_begin);
  ASSERT(good, "failed to read sample from file " << filename);

  return loop;
}

//----( formant -> loop synth )----

FormantLoopSynth::FormantLoopSynth (
    size_t circ,
    size_t loop_size)

  : m_actuate1(),
    m_actuate2(),
    m_resonate(),
    m_lowpass(BUCKET_LOWPASS_TIMESCALE),

    m_loop_data(loop_size, malloc_complex(loop_size)),
    m_loop(m_loop_data),
    m_loop_state(),

    m_decay0(0),
    m_decay1(0),
    m_sound(),

    m_voice_state(circ),
    m_spin_state(),

    voice_in(circ),
    power_in(0),

    spin_in(SpinControl::State()),
    impact_in(false)
{
  m_loop_data.zero();
  voice_in.unsafe_access().impedance.set(1.0f / BUCKET_IMPEDANCE_RATIO);
}

FormantLoopSynth::~FormantLoopSynth ()
{
  PRINT2(m_voice_state.sustain, m_voice_state.pitch);
  Vector<float> & impedance = m_voice_state.impedance;
  PRINT3(min(impedance), mean(impedance), max(impedance));

  PRINT(m_actuate1);
  PRINT(m_actuate2);

  SpinControl::State & s2 = m_spin_state;
  PRINT3(s2.phase, s2.rate, s2.amplitude);
}

void FormantLoopSynth::pull (Seconds time, StereoAudioFrame & sound_accum)
{
  float power;
  power_in.pull(time, power);

  bool jump;
  impact_in.pull(time, jump);

  voice_in.pull(time, m_voice_state);
  float pitch = m_voice_state.pitch;
  float sustain = m_voice_state.sustain;

  //--------

  pitch = (BUCKET_PITCH_SCALE * pitch + BUCKET_PITCH_CENTER) / logf(2);

  power /= BUCKET_SUSTAIN_POWER + BUCKET_ATTACK_POWER;
  float power1 = power * BUCKET_SUSTAIN_POWER;
  float power2 = power * BUCKET_ATTACK_POWER;
  Synthesis::Glottis::Timbre timbre1(power1, sustain, pitch);
  Synthesis::ExpSine::Timbre timbre2(
      power2,
      0,
      pitch,
      BUCKET_ATTACK_ROUGHNESS);

  m_sound.zero();
  m_actuate1.sample(timbre1, m_sound);
  m_resonate.sample(m_voice_state.impedance, m_sound);
  m_lowpass.sample(m_sound);
  m_actuate2.sample(timbre2, m_sound);

  //--------

  spin_in.pull(time, m_spin_state);
  float phase = m_spin_state.phase;
  float rate = m_spin_state.rate;
  float amp = m_spin_state.amplitude;

  //--------

  m_loop_state.set_rate_amp(rate, amp);
  if (jump) m_loop_state.jump_to_phase(phase);

  float timescale = LOOP_RELEASE_TIMESCALE
                 + LOOP_SUSTAIN_TIMESCALE * amp;
  m_decay0 = m_decay1;
  m_decay1 = expf(-1 / timescale);

  m_loop.sample(m_loop_state, m_sound, sound_accum, m_decay0, m_decay1);
  soft_clip(sound_accum);
}

//----( syncopated synth )----

SyncopatedSynth::SyncopatedSynth (size_t capacity)
  : m_bank(Synchronized::Bank(
        capacity,
        0.5f,2.0f, // bogus frequencies
        COUPLED_TEMPO_ACUITY)),
    m_polar(capacity),
    m_states(capacity),
    m_time(Seconds::now()),

    polar_in("SyncopatedSynth.polar_in", capacity)
{
  m_bank.get(m_states);
}

void SyncopatedSynth::pull (Seconds time, StereoAudioFrame & sound)
{
  float dt = max(1e-8f, time - m_time);
  m_time = time;

  LOG1("advance prior states");
  m_bank.set(m_states);
  m_bank.advance(dt);
  m_bank.get(m_states);

  TODO("deal with new vs old states");
  for (size_t i = 0; i < m_states.size; ++i) {
    m_ids[m_states.keys[i]] = i;
  }

  polar_in.pull(time, m_polar);

  TODO("sample states via voices");

  m_ids.clear();
}

} // namespace Streaming

