
#include "psycho.h"
#include <algorithm>

#define LOG1(mess)

namespace Psycho
{

//----( loudness transform )--------------------------------------------------

Loudness::Loudness (
    size_t size,
    float frame_rate,
    float time_scale,
    float ss_factor)

  : m_size(size),
    m_frame_rate(frame_rate),
    m_time_scale(time_scale),
    m_ss_factor(ss_factor),
    m_tolerance(1e-10),

    m_decay_factor_fast(exp(-2.0f / (frame_rate * time_scale))),
    m_decay_factor_slow(exp(-0.5f / (frame_rate * time_scale))),
    m_max_loudness(1.0),

    m_current_fwd(size),
    m_history_fwd(size),
    m_current_bwd(size),
    m_history_bwd(size)
{
  m_history_fwd.zero();
  m_history_bwd.zero();
}

inline float expand_loudness   (float x) { return 1 - sqr(1 - x); }
inline float contract_loudness (float x) { return 1 - sqrt(1 - x); }

void Loudness::transform_fwd (
    Vector<float> & energy_in,
    Vector<float> & loudness_out)
{
  ASSERT_SIZE(energy_in, m_size);
  ASSERT_SIZE(loudness_out, m_size);

  // check that energy is in [0,inf)
  float min_energy = min(energy_in);
  float max_energy = max(energy_in);
  ASSERTW(min_energy >= 0, "min energy is too small: " << min_energy);

  // update max loudness
  m_max_loudness_mutex.lock();
  imax(m_max_loudness, powf(max_energy, 1.0f/3));
  const float max_loudness = m_max_loudness + m_tolerance;
  m_max_loudness *= m_decay_factor_slow;
  m_max_loudness_mutex.unlock();

  loudness_out[0] = loudness_out[m_size-1] = 0;
  for (size_t i = 1; i < m_size-1; ++i) {

    ASSERT1_FINITE(energy_in[i]);
    float energy = energy_in[i];
    if (energy < m_tolerance) {
      m_current_fwd[i] = 0;
      loudness_out[i] = 0;
      continue;
    }

    // apply e^1/3 power law
    float loudness_new = powf(energy, 1.0f/3);

    // rescale to [0,1]
    loudness_new /= max_loudness;

    // highpass filter
    m_current_fwd[i] = loudness_new;
    float loudness_old = m_history_fwd[i];
    float loudness_diff = max(0.0f, loudness_new - loudness_old);
    loudness_new =    m_ss_factor  * loudness_new
                 + (1-m_ss_factor) * loudness_diff;

    ASSERT1_LE(0, loudness_new);
    ASSERT1_LE(loudness_new, 1);

    // expand
    loudness_out[i] = 1 - sqr(1 - loudness_new);
    ASSERT1_FINITE(loudness_out[i]);
  }

  // update history
  float a = m_decay_factor_fast;
  float b = 1-a;

  for (size_t i = 1; i < m_size-1; ++i) {
    float & h = m_history_fwd[i];

    h = a * h
      + b * max (0.5f * m_current_fwd[i-1],
            max (       m_current_fwd[i],
                 0.5f * m_current_fwd[i+1]));
  }
}

void Loudness::transform_bwd (
    Vector<float> & loudness_in,
    Vector<float> & energy_out)
{
  ASSERT_SIZE(loudness_in, m_size);
  ASSERT_SIZE(energy_out, m_size);

  // ensure loudness_in is in [0,1]
  float min_in = min(loudness_in);
  float max_in = max(loudness_in);

  ASSERTW(min_in >= 0, "min loudness is too small: " << min_in);
  ASSERTW(max_in <= 1, "max loudness is too large: " << max_in);

  if (min_in < m_tolerance) {
    max_in += min_in + m_tolerance;
    loudness_in += min_in + m_tolerance;
    min_in = m_tolerance;
  }
  if (max_in > 1 - m_tolerance) {
    loudness_in *= (1 - m_tolerance) / max_in;
    min_in *= (1 - m_tolerance) / max_in;
    max_in = 1 - m_tolerance;
  }

  m_max_loudness_mutex.lock();
  const float max_loudness = m_max_loudness - m_tolerance;
  m_max_loudness_mutex.unlock();

  energy_out[0] = energy_out[m_size-1] = 0;
  for (size_t i = 1; i < m_size-1; ++i) {

    ASSERT1_FINITE(loudness_in[i]);

    // contract
    float loudness_new = contract_loudness(loudness_in[i]);

    // approximate lowpass filter
    float loudness_int = m_history_bwd[i];
    loudness_new += (1-m_ss_factor) * loudness_int;
    m_current_bwd[i] = loudness_new;

    // XXX loudness_new may no longer lie in [0,1]

    // rescale from [0,1]
    loudness_new *= max_loudness;

    // apply e^1/3 power law
    energy_out[i] = powf(loudness_new, 3);
    ASSERT1_FINITE(energy_out[i]);
  }

  // update history
  float a = m_decay_factor_fast;
  float b = 1-a;

  for (size_t i = 1; i < m_size-1; ++i) {
    float & h = m_history_bwd[i];

    h = a * h
      + b * max (0.0f, m_current_bwd[i]
                     - 0.5f * ( m_current_bwd[i-1]
                              + m_current_bwd[i+1]));
  }
}

//----( beat perception )-----------------------------------------------------

EnergyToBeat::EnergyToBeat (
    const char * config_filename,
    float min_tempo_hz,
    float max_tempo_hz)

  : m_config(config_filename),

    m_min_tempo_hz(min_tempo_hz),
    m_max_tempo_hz(max_tempo_hz),

    m_time(Seconds::now()),

    m_pos(0),
    m_vel(0),
    m_acc(0),
    m_norm(0),

    m_pos_mean(m_config("energy_to_beat.pos_mean", 10.0f)),
    m_pos_variance(m_config("energy_to_beat.pos_variance", 100.0f)),
    m_acc_variance(m_config("energy_to_beat.acc_variance", 1000.0f)),

    m_lag(0)
{
  ASSERT_LT(1/60.0f, min_tempo_hz);
  ASSERT_LT(min_tempo_hz, max_tempo_hz);
  ASSERT_LT(max_tempo_hz, 60.0f);
}

EnergyToBeat::~EnergyToBeat ()
{
  LOG("EnergyToBeat:");
  PRINT3(m_pos, m_vel, m_acc);
  PRINT(m_norm);
  PRINT2(m_pos_mean, m_pos_variance);
}

complex EnergyToBeat::transform_fwd (Seconds time, float energy)
{
  float dt = max(1e-8f, time - m_time);
  m_time = time;
  float fast_rate = 1.0f - expf(-m_max_tempo_hz * dt);
  float slow_rate = 1.0f - expf(-m_min_tempo_hz * dt);

  float old_pos_mean = m_pos_mean;
  float old_pos = m_pos;
  float new_pos = powf(max(0.0f, energy), -1/3.0f);
  float pos_mean = old_pos_mean + slow_rate * (new_pos - old_pos_mean);
  float pos = old_pos + fast_rate * (new_pos - old_pos_mean - old_pos);
  m_pos = pos;

  float old_vel = m_vel;
  float new_vel = (pos - old_pos) / dt;
  float vel = old_vel + fast_rate * (new_vel - old_vel);
  m_vel = vel;

  float old_acc = m_acc;
  float new_acc = (vel - old_vel) / dt;
  float acc = old_acc + fast_rate * (new_acc - old_acc);
  m_acc = acc;

  // Version 1: exact for sinusoids
  //float beat_real_absval = sqrtf(max(0.0f, -pos * acc));
  // Version 2: continuous
  float beat_real_absval = sqrtf(fabsf(pos * acc));

  float beat_real_sign = ( pos / sqrtf(m_pos_variance)
                        - acc / sqrtf(m_acc_variance)
                        ) / 2;
  complex beat(beat_real_absval * beat_real_sign, -vel);

  float old_norm = m_norm;
  float new_norm = norm(beat);
  float norm = old_norm + fast_rate * (new_norm - old_norm);
  m_norm = norm;

  beat /= sqrtf(norm);

  m_mutex.lock();
  m_pos_mean = pos_mean;
  m_pos_variance += slow_rate * (sqr(pos - m_pos_mean) - m_pos_variance);
  m_acc_variance += slow_rate * (sqr(acc) - m_acc_variance);
  m_mutex.unlock();

  return beat;
}

float EnergyToBeat::transform_bwd (complex beat)
{
  m_mutex.lock();
  float pos_mean = m_pos_mean;
  float pos_variance = m_pos_variance;
  m_mutex.unlock();

  float pos = max(0.0f, pos_mean + sqrtf(pos_variance) * beat.real());

  return powf(pos, 3);
}

//----( logarithmic history )-------------------------------------------------

History::History (
    size_t size,
    size_t length,
    size_t density)

  : m_size(size),
    m_length(length ? length : size),
    m_density(density ? density : m_length / 4),
    m_tau(m_density * log(2)),

    m_frames(NULL),
    m_free_frames(NULL),
    m_num_cropped(0),

    m_spline_weights(m_length)
{}

History::~History ()
{
  if (m_frames) delete m_frames;
  if (m_free_frames) delete m_free_frames;
}

void History::Frame::init (const Vector<float> & present)
{
  ASSERT_SIZE(present, data.size);
  data = present;
  time = 0;
  rank = 0;
}

void History::add_frame (const Vector<float> & present)
{
  LOG1("adding frame");

  Frame * new_frame = m_free_frames;
  if (new_frame) {
    m_free_frames = new_frame->next;
  } else {
    new_frame = new Frame(m_size);
  }
  new_frame->next = m_frames;
  m_frames = new_frame;

  new_frame->init(present);
}

void History::crop_to_frame (Frame * frame)
{
  LOG1("cropping to frame");

  Frame * old_frame = frame->next;
  ASSERT(old_frame != NULL, "nothing to pop");
  ASSERT(old_frame->next == NULL, "too much to pop");

  frame->next = NULL;
  old_frame->next = m_free_frames;
  m_free_frames = old_frame;

  m_num_cropped += 1;
}

void History::merge_frames (Frame * frame)
{
  LOG1("merging frames");

  Frame * old_frame = frame->next;
  ASSERT(old_frame, "nothing to merge with");
  ASSERT(frame->rank == old_frame->rank, "merge rank mismatch");

  frame->data += old_frame->data;
  frame->time = 0.5 * (frame->time + old_frame->time);
  frame->rank += 1;

  frame->next = old_frame->next;
  old_frame->next = m_free_frames;
  m_free_frames = old_frame;
}

History & History::add (const Vector<float> & present)
{
  ASSERT_SIZE(present, m_size);

  // advance all frames
  for (Frame * f = m_frames; f; f = f->next) {
    f->time += 1;
  }

  // merge redundant frames of same rank
  for (Frame * f = m_frames; f and f->next; f = f->next) {
    Frame * g = f->next;

    if (g->rank != f->rank) {
      ASSERTW(g->rank == f->rank + 1,
          "found gap in history frames; try using higher density");
      continue;
    }

    while (g->next and (g->next->rank == g->rank)) {
      f = g;
      g = g->next;
    }

    const float resolution = 0.25f;
    if (log_time(g->time) - log_time(f->time) < resolution) {
      merge_frames(f);
    }
  }

  // pop obsolete frames
  for (Frame * f = m_frames; f and f->next; f = f->next) {
    const float padding = 1;
    if (log_time(f->next->time) >= m_length + padding) {
      crop_to_frame(f);
    }
  }

  // add new frame
  add_frame(present);

  return * this;
}

History & History::get (Vector<float> & past)
{
  ASSERT_SIZE(past, size_out());

  past.zero();
  m_spline_weights.zero();

  for (Frame * f = m_frames; f; f = f->next) {
    float i = log_time(f->time);

    LinearInterpolate lin(i, m_length);
    m_spline_weights[lin.i0] += lin.w0;
    m_spline_weights[lin.i1] += lin.w1;

    float scale = 1.0 / f->num_terms();
    Vector<float> past0 = past.block(m_size, lin.i0);
    Vector<float> past1 = past.block(m_size, lin.i1);
    if (lin.w0 > 0) multiply_add(lin.w0 * scale, f->data, past0);
    if (lin.w1 > 0) multiply_add(lin.w1 * scale, f->data, past1);
  }

  for (size_t i = 0; i < m_length; ++i) {
    Vector<float> past_i = past.block(m_size, i);
    if (m_spline_weights[i] > 0) {
      float normalize_factor = 1.0 / m_spline_weights[i];
      LOG1("normalize_factor = " << normalize_factor);
      for (size_t j = 0; j < m_size; ++j) {
        past_i[j] *= normalize_factor;
      }
    }
  }

  return * this;
}

History & History::get_after (float delay, Vector<float> & past)
{
  ASSERT_DIVIDES(m_size, past.size);
  const size_t length = past.size / m_size;
  ASSERT_LE(length, m_length);

  past.zero();
  m_spline_weights.zero();

  for (Frame * f = m_frames; f; f = f->next) {
    float i = log_time(f->time) - delay;

    LinearInterpolate lin(i, length);
    m_spline_weights[lin.i0] += lin.w0;
    m_spline_weights[lin.i1] += lin.w1;

    float scale = 1.0 / f->num_terms();
    Vector<float> past0 = past.block(m_size, lin.i0);
    Vector<float> past1 = past.block(m_size, lin.i1);
    if (lin.w0 > 0) multiply_add(lin.w0 * scale, f->data, past0);
    if (lin.w1 > 0) multiply_add(lin.w1 * scale, f->data, past1);
  }

  for (size_t i = 0; i < length; ++i) {
    Vector<float> past_i = past.block(m_size, i);
    if (m_spline_weights[i] > 0) {
      float normalize_factor = 1.0 / m_spline_weights[i];
      LOG1("normalize_factor = " << normalize_factor);
      for (size_t j = 0; j < m_size; ++j) {
        past_i[j] *= normalize_factor;
      }
    }
  }

  return * this;
}

History & History::at (float time, Vector<float> & moment)
{
  Frame * f = m_frames;

  if (not f) {
    WARN("no history to get at");
    moment.zero();
    return * this;
  }

  if (time <= f->time) {
    moment = f->data;
    return * this;
  }

  while (f->next and f->next->time < time) {
    f = f->next;
  }
  if (not f->next) {
    moment = f->data;
    return * this;
  }

  LOG1("linearly interpolate f and f->next");
  Frame * g = f->next;
  ASSERT(f->time <= time and time <= g->time,
         "times are out of order: "
         << f->time << " - " << time << " - " << g->time);

  float tau = log_time(time);
  float tau_f = log_time(f->time);
  float tau_g = log_time(g->time);

  float weight_f = (tau_g - tau) / (tau_g - tau_f);
  ASSERT_LE(0, weight_f);
  ASSERT_LE(weight_f, 1);
  affine_combine(weight_f, f->data, g->data, moment);

  return * this;
}

//----( masking )-------------------------------------------------------------

Masker::Masker (Synchronized::Bank param)

  : Synchronized::FourierBank2(param),

    m_radius(tone_size() / 2),
    m_mask_dt(size),

    m_energy(size),
    m_time(size)
{
  Bank::init_decay(m_mask_dt);
  for (size_t i = 0; i < size; ++i) {
    m_mask_dt[i] = 1 - m_mask_dt[i];
  }

  m_time.zero();
}

Masker::~Masker ()
{
  PRINT3(min(m_time), mean(m_time), max(m_time));
}

void Masker::sample (const Vector<float> & time, Vector<float> & masked)
{
  FourierBank2::sample(time, m_energy);
  sample(masked);
}

void Masker::sample (const Vector<complex> & time, Vector<float> & masked)
{
  FourierBank2::sample(time, m_energy);
  sample(masked);
}

void Masker::sample (Vector<float> & masked)
{
  const float * restrict dt = m_mask_dt;
  const float * restrict e = m_energy;
  float * restrict bg = m_time;
  float * restrict m = masked;

  Image::exp_blur_1d_zero(size, m_radius, m_energy, masked);

  for (size_t i = 0, I = size; i < I; ++i) {
    bg[i] += (m[i] - bg[i]) * dt[i];
    m[i] = sqr(e[i]) / (e[i] + bg[i]);
  }
}

//----( psychogram )----------------------------------------------------------

Psychogram::Psychogram (
    size_t size,
    float min_freq_hz,
    float max_freq_hz)
  : m_bank(Synchronized::Bank(
      size,
      min_freq_hz / DEFAULT_SAMPLE_RATE,
      max_freq_hz / DEFAULT_SAMPLE_RATE)),
    m_loudness(size, DEFAULT_AUDIO_FRAMERATE),

    m_energy(size)
{
  ASSERT_LT(0, min_freq_hz);
  ASSERT_LT(min_freq_hz, max_freq_hz);
}

void Psychogram::transform_fwd (
    const StereoAudioFrame & sound_in,
    Vector<float> & loudness_out)
{
  const size_t I = m_bank.size;

  ASSERT_SIZE(loudness_out, I);

  // analyze on pitch scale
  m_bank.sample(sound_in, m_energy);

  // mask pitch by reassigning (twice)
  Image::reassign_repeat_x(I, 1, m_energy, loudness_out);
  Image::reassign_repeat_x(I, 1, loudness_out, m_energy);

  // apply loudness transform
  m_loudness.transform_fwd(m_energy, loudness_out);
}

//----( harmony )-------------------------------------------------------------

Harmony::Harmony (
    size_t size,
    float acuity,
    float min_freq_hz,
    float max_freq_hz,
    float min_timescale_sec)

  : Bank(
      size,
      min_freq_hz / DEFAULT_SAMPLE_RATE,
      max_freq_hz / DEFAULT_SAMPLE_RATE,
      acuity),

    m_anal_bank(* this, min_timescale_sec * DEFAULT_SAMPLE_RATE),
    m_synth_bank(* this),

    m_retune_rate( logf(2)
                 / PSYCHO_PITCH_ACUITY
                 / PSYCHO_MIN_PITCH_HZ
                 / DEFAULT_AUDIO_FRAMERATE ),

    m_temp(size)
{
}

void Harmony::analyze (
    const MonoAudioFrame & sound_in,
    Vector<float> & mass_out)
{
  Vector<float> & energy = m_temp;
  m_anal_bank.sample(sound_in, energy);

  float * restrict mass = energy;
  for (size_t i = 0, I = size; i < I; ++i) {
    mass[i] = sqrtf(mass[i] + 1e-20f);
  }

  // retune to simulate listening bands
  m_synth_bank.retune(energy, mass_out);
}

void Harmony::analyze (
    const StereoAudioFrame & sound_in,
    Vector<float> & mass_out)
{
  Vector<float> & energy = m_temp;
  m_anal_bank.sample(sound_in, energy);

  float * restrict mass = energy;
  for (size_t i = 0, I = size; i < I; ++i) {
    mass[i] = sqrtf(mass[i] + 1e-20f);
  }

  // retune to simulate listening bands
  m_synth_bank.retune(energy, mass_out);
}

void Harmony::synthesize (Vector<float> & mass_io, StereoAudioFrame & sound_out)
{
  Vector<float> & dmass = m_temp;
  subtract(mass_io, m_synth_bank.get_mass(), dmass);
  m_synth_bank.sample_accum(dmass, sound_out);

  Vector<float> & old_mass = m_temp;
  old_mass = mass_io;
  m_synth_bank.retune_zeromean(old_mass, mass_io, m_retune_rate);
}

void Harmony::synthesize_mix (
    const Vector<float> & weights,
    Vector<float> & masses_io,
    StereoAudioFrame & sound_out)
{
  ASSERT_DIVIDES(size, masses_io.size);
  const size_t voice_count = masses_io.size / size;

  Vector<float> & dmass = m_temp;
  multiply(-1.0f, m_synth_bank.get_mass(), dmass);
  for (size_t v = 0; v < voice_count; ++v) {
    Vector<float> voice_mass = masses_io.block(size, v);
    multiply_add(weights[v], voice_mass, dmass);
  }
  m_synth_bank.sample_accum(dmass, sound_out);

  Vector<float> & old_mass = m_temp;
  for (size_t v = 0; v < voice_count; ++v) {
    Vector<float> voice_mass = masses_io.block(size, v);
    old_mass = voice_mass;
    m_synth_bank.retune_zeromean(old_mass, voice_mass, m_retune_rate);
  }
}

//----( rhythm )--------------------------------------------------------------

Rhythm::Rhythm (
    size_t size,
    float min_freq_hz,
    float max_freq_hz,
    float duration)

  : m_num_observations(0),
    m_total_residual(0.0f),

    m_set(
        size,
        min_freq_hz / DEFAULT_AUDIO_FRAMERATE,
        max_freq_hz / DEFAULT_AUDIO_FRAMERATE,
        duration)
{}

Rhythm::Rhythm (const Rhythm & other)
  : m_num_observations(other.m_num_observations),
    m_total_residual(other.m_total_residual),
    m_set(other.m_set)
{}

float Rhythm::sample ()
{
  m_set.advance();

  return m_set.predict_value();
}

float Rhythm::learn_and_sample (float value)
{
  m_set.advance();

  m_num_observations += 1;
  m_total_residual += m_set.learn(value);

  return m_set.predict_value();
}

float Rhythm::predict (size_t num_steps)
{
  for (size_t t = 0; t < num_steps; ++t) {
    m_set.advance();
  }

  return m_set.predict_value();
}

//----( polyrhythm )----------------------------------------------------------

Polyrhythm::Polyrhythm (
    size_t voice_count,
    size_t tempo_size,
    float min_tempo_hz,
    float max_tempo_hz,
    float duration)

  : m_set(
        tempo_size,
        min_tempo_hz / DEFAULT_AUDIO_FRAMERATE,
        max_tempo_hz / DEFAULT_AUDIO_FRAMERATE,
        duration),

    m_voice_count(voice_count),

    m_masses(voice_count * tempo_size),
    m_mass(m_set.get_mass()),
    m_values(tempo_size)
{
  ASSERT_LT(0, voice_count);
  LOG("creating Polyrhythm with " << voice_count << " voices");

  m_masses.zero();
}

void Polyrhythm::project_mass ()
{
  m_mass = m_masses.block(m_set.size(), 0);
  for (size_t v = 1; v < m_voice_count; ++v) {
    m_mass += m_masses.block(m_set.size(), v);
  }
}

void Polyrhythm::learn_one (size_t voice, float value)
{
  ASSERT_LT(voice, m_voice_count);

  Vector<float> voice_mass = m_masses.block(m_set.size(), voice);
  m_set.learn(value, voice_mass);
  project_mass();
}

void Polyrhythm::learn_all (const Vector<float> & values)
{
  for (size_t v = 0; v < m_voice_count; ++v) {
    Vector<float> voice_mass = m_masses.block(m_set.size(), v);
    m_set.learn(values[v], voice_mass);
  }
  project_mass();
}

void Polyrhythm::sample (Vector<float> & values_out)
{
  m_set.advance();

  for (size_t v = 0; v < m_voice_count; ++v) {
    Vector<float> voice_mass = m_masses.block(m_set.size(), v);
    values_out[v] = m_set.predict_value(voice_mass);
  }
}

} // namespace Psycho

