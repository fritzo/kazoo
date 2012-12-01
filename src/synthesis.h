#ifndef KAZOO_SYNTHESIS_H
#define KAZOO_SYNTHESIS_H

/** Audio Synthesis Algorithms.

  TODO treat non-detection events as intensity-0 events in tracker,
    so that finger.get_y() in [0,1]
*/

#include "common.h"
#include "vectors.h"
#include "array.h"
#include "gestures.h"
#include "audio_types.h"
#include "cyclic_time.h"
#include "synchrony.h"
#include "threads.h"
#include <vector>

#define DEFAULT_SYNTHESIS_RANGE         (64)
#define DEFAULT_SYNTHESIS_MID_FREQ_HZ   (CONCERT_A_HZ / 2.0)
#define DEFAULT_SYNTHESIS_PITCH_STEP    (logf(2))
#define DEFAULT_SYNTHESIS_RELEASE_SEC   (0.1f)
#define DEFAULT_SYNTHESIS_SUSTAIN_SEC   (2.0f)

#define SHEPARD_FINE_SPEED_UNIT         (2.0f / 12)

// see test/synchronize.py tongues
#define COUPLED_PITCH_ACUITY            (7)
#define COUPLED_TEMPO_ACUITY            (3)
#define COUPLED_SYNCO_ACUITY            (2)

#define DEFAULT_ECHOBOX_DECAY           (0.9f)
#define DEFAULT_ECHOBOX_LOWPASS         (0.5f)
#define DEFAULT_REVERB_INTERVAL         GOLDEN_RATIO
#define DEFAULT_REVERB_BASE_SIZE        (128)

#define SYNTHESIS_LOOP_RESONATOR_SIZE   (48)
#define SYNTHESIS_FORMANT_TIMESCALE     (16.0)

#define SYNTHESIS_LOOP_MAX_RATE         (10.0f)
#define SYNTHESIS_LOOP_MIN_RATE         (0.1f)

#define SYNTHESIS_NOISE_PITCH_PRECISION     (0.1f)
#define SYNTHESIS_RESONATE_PITCH_PRECISION  (0.3f)
#define SYNTHESIS_RESONATE_Q_FACTOR         (0.99f)
#define SYNTHESIS_RESONATE_LEAK             (1 - SYNTHESIS_RESONATE_Q_FACTOR)

#define SYNTHESIS_VOCODER_SIZE          (512)
#define SYNTHESIS_VOCODER_MIN_FREQ_HZ   (20.0f)
#define SYNTHESIS_VOCODER_MAX_FREQ_HZ   (7e3f)
#define SYNTHESIS_VOCODER_ACUITY        (3.0f)

#define SYNTHESIS_BEATER_SIZE           (128)
#define SYNTHESIS_BEATER_MIN_FREQ       (0.5f)
#define SYNTHESIS_BEATER_MAX_FREQ       (8.0f)
#define SYNTHESIS_BEATER_ACUITY         (3.0f)
#define SYNTHESIS_BEATER_TIMESCALE      (30.0f)

#define BANDS_SUSTAIN_SEC           (4.0f)

#define WIDEBAND_MIN_HZ             (1/2.0f)
#define WIDEBAND_MAX_HZ             (5e3f)
#define WIDEBAND_MID_HZ             (sqrtf(WIDEBAND_MIN_HZ * WIDEBAND_MAX_HZ))
#define WIDEBAND_PITCH_RANGE        (logf(WIDEBAND_MAX_HZ / WIDEBAND_MIN_HZ))
#define WIDEBAND_PITCH_STEP         (WIDEBAND_PITCH_RANGE / GRID_SIZE_X)

#define SPLITBAND_PITCH_MID_HZ      (256.0f)
#define SPLITBAND_PITCH_STEP        (logf(1.5f))
#define SPLITBAND_TEMPO_MID_HZ      (2.0f)
#define SPLITBAND_TEMPO_STEP        (logf(2.0f))
#define SPLITBAND_SHARPNESS         (0.2f)

#define SYNCOPATORS_TEMPO_MIN_HZ    (0.5f)
#define SYNCOPATORS_ALICE_MIN_HZ    (1.0f)
#define SYNCOPATORS_TEMPO_MAX_HZ    (8.0f)
#define SYNCOPIPE_PITCH_SHIFT       (-2.0f)
#define SHEPARD4_MIN_FREQ_HZ        (200.0f)
#define SHEPARD7_MIN_FREQ_HZ        (60.0f)

#define BUCKET_ORGAN_NUM_OCTAVES    (4.0f)
#define BUCKET_ORGAN_MIN_PITCH      (-4.0f)

#define SYNTHESIS_ENERGY_TOL        (1e-8f)

namespace Synthesis
{

using namespace Gestures;

//====( elements of synthesis )===============================================

// pitch-frequency conversion
inline float pitch_to_freq (float pitch)
{
  const float pitch_shift = logf( DEFAULT_SYNTHESIS_MID_FREQ_HZ
                               / DEFAULT_SAMPLE_RATE );
  const float pitch_scale = DEFAULT_SYNTHESIS_PITCH_STEP;

  return expf(pitch_shift + pitch_scale * pitch);
}
inline float pitch_to_2_pi_freq (float pitch)
{
  const float pitch_shift = logf( DEFAULT_SYNTHESIS_MID_FREQ_HZ
                               / DEFAULT_SAMPLE_RATE
                               * 2 * M_PI);
  const float pitch_scale = DEFAULT_SYNTHESIS_PITCH_STEP;

  return expf(pitch_shift + pitch_scale * pitch);
}
inline float pitch_to_freq_mod_octave (float pitch)
{
  const float pitch_shift = logf( DEFAULT_SYNTHESIS_MID_FREQ_HZ
                               / DEFAULT_SAMPLE_RATE )
                         / logf(2);
  pitch += pitch_shift;
  pitch -= floorf(pitch);

  return powf(2.0f, pitch - 1);
}

inline float pitch_to_freq_shepard4 (float pitch)
{
  return powf(2.0f, 1 + pitch / 3.0f)
       * (2.0f * SHEPARD4_MIN_FREQ_HZ / DEFAULT_SAMPLE_RATE);
}

inline float pitch_to_freq_shepard7 (float pitch)
{
  return powf(2.0f, 1 + pitch / 3.0f)
       * SHEPARD7_MIN_FREQ_HZ / DEFAULT_SAMPLE_RATE;
}

inline float pitch_to_freq_wide (float pitch)
{
  static const float pitch_shift = logf(WIDEBAND_MID_HZ / DEFAULT_SAMPLE_RATE);
  static const float pitch_scale = WIDEBAND_PITCH_STEP;

  return 2 * M_PI * expf(pitch_shift + pitch_scale * pitch);
}

inline float pitch_to_freq_split (float pitch)
{
  const float pitch_shift = logf(SPLITBAND_PITCH_MID_HZ / DEFAULT_SAMPLE_RATE);
  const float pitch_scale = SPLITBAND_PITCH_STEP;

  return 2 * M_PI * expf(pitch_shift + pitch_scale * pitch);
}
inline float tempo_to_freq_split (float tempo)
{
  const float tempo_shift = logf( SPLITBAND_TEMPO_MID_HZ
                               / DEFAULT_AUDIO_FRAMERATE);
  const float tempo_scale = SPLITBAND_TEMPO_STEP;

  return 2 * M_PI * expf(tempo_shift + tempo_scale * tempo);
}

inline float tempo_to_freq_synco (float tempo_01)
{
  const float tempo_shift
    = logf(SYNCOPATORS_TEMPO_MIN_HZ / DEFAULT_AUDIO_FRAMERATE);
  const float tempo_scale
    = logf(SYNCOPATORS_TEMPO_MAX_HZ / SYNCOPATORS_TEMPO_MIN_HZ );

  return 2 * M_PI * expf(tempo_shift + tempo_scale * tempo_01);
}

inline float tempo_to_freq_synco_grid (float tempo_grid)
{
  const float tempo_scale = logf(2) / 2; // half octave per gridline
  const float tempo_shift
    = logf(SYNCOPATORS_ALICE_MIN_HZ / DEFAULT_AUDIO_FRAMERATE);

  return 2 * M_PI * expf((tempo_grid + 3.0f) * tempo_scale + tempo_shift);
}

//----( actuators )-----------------------------------------------------------

class BuzzingActuator
{
  int m_sawtooth;
public:
  BuzzingActuator () : m_sawtooth(0) {}
  inline float operator () (float pitch);
};

/** Shepard actuator.

  1 2 1 4 1 2 1 8 ... eventually returns 0

  This has a slightly blue energy spectrum.
*/
class ShepardActuator
{
  float m_phase;
public:
  ShepardActuator () : m_phase(0) {}
  inline float operator () (float pitch);
};

void sample_noise (StereoAudioFrame & sound, float power = 1);

//----( resonators )----------------------------------------------------------

/** Single-pole resonator for narrow bandpass filtering of signals.
*/
class Resonator
{
  complex m_state;

public:

  Resonator () : m_state(0,0) {}

  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return norm(m_state) > tol;
  }

  complex operator () (float pitch, float actuate)
  {
    float freq = pitch_to_freq(pitch);

    m_state *= SYNTHESIS_RESONATE_Q_FACTOR * exp_2_pi_i(freq);
    m_state += SYNTHESIS_RESONATE_LEAK * actuate;

    return m_state;
  }
};

/** Single-pole resonator for narrow bandpass filtering of noise.

  Decay is chosen to keep constant pitch accuracy across all frequencies.

  Perturbation amplitude is chosen so that unit white noise actuation
  yields unit variance resonance. An alternate choice would be

    perturb = low_freq * actuate

  so that unit actuation at the resonant frequency yields unit response.
*/
class NoiseResonator
{
  complex m_state;

public:

  NoiseResonator () : m_state(0,0) {}
  inline complex operator () (float pitch, float bandwidth, float actuate);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return norm(m_state) > tol;
  }
};

/** Octave-wide bandpass filter

  This real-valued filter approximates the lognormal energy envelope

    e(w) = exp(-(log(w/w0) / b)^2 / 2)
*/
class OctavePassFilter
{
  float x_coeff[3];
  float y_coeff[4];

  float x_hist[2];
  float y_hist[4];

public:

  OctavePassFilter ()
  {
    x_hist[0] = 0;
    x_hist[1] = 0;
    y_hist[0] = 0;
    y_hist[1] = 0;
    y_hist[2] = 0;
    y_hist[3] = 0;
  }

  void set_pitch (float pitch, float bandwidth = 0.25);
  float operator () (float x);

  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return tol
      < sqr(y_hist[0])
      + sqr(y_hist[1])
      + sqr(y_hist[2])
      + sqr(y_hist[3]);
  }
};

template<class T>
class Lowpass
{
  const float m_old_part;
  const float m_new_part;
  T m_state;

public:

  Lowpass (float timescale)
    : m_old_part(exp(-1 / timescale)),
      m_new_part(1 - m_old_part),
      m_state(0.0f)
  {}
  ~Lowpass () { LOG("lowpass state = " << m_state); }

  T operator() (T sound)
  {
    return m_state = m_old_part * m_state
                   + m_new_part * sound;
  }

  void sample (Vector<T> & sound_io)
  {
    const size_t I = sound_io.size;
    T * restrict sound = sound_io.data;
    for (size_t i = 0; i < I; ++i) {
      sound[i] = operator()(sound[i]);
    }

    if (not safe_isfinite(m_state)) {
      WARN("dropping nonfinite Lowpass state: " << m_state);
      m_state = 0.0f;
    }
  }
};

template<class T>
class Highpass
{
  const float m_new_part;
  const float m_old_part;
  T m_state;

public:

  Highpass (float timescale)
    : m_new_part(1 / timescale),
      m_old_part(1 - m_new_part),
      m_state(0.0f)
  {}

  T operator() (T sound)
  {
    m_state = m_old_part * m_state
            + m_new_part * sound;
    return sound - m_state;
  }

  void sample (Vector<T> & sound_io)
  {
    const size_t I = sound_io.size;
    T * restrict sound = sound_io.data;
    for (size_t i = 0; i < I; ++i) {
      sound[i] = operator()(sound[i]);
    }
  }
};

//----( sampling )------------------------------------------------------------

class VariableSpeedLoop
{
  StereoAudioFrame m_loop;

public:

  class State
  {
    friend class VariableSpeedLoop;

    float m_phase;
    float m_rate;
    float m_prev_scale;
    float m_curr_scale;

  public:

    State ()
      : m_phase(0),
        m_rate(1),
        m_prev_scale(0),
        m_curr_scale(0)
    {}

    void jump_to_phase (float phase)
    {
      m_phase = phase - floor(phase);
      m_prev_scale = 0;
      m_curr_scale = 0;
    }

    void set_rate_amp (float rate, float amplitude)
    {
      m_prev_scale = m_curr_scale;
      m_curr_scale = amplitude / (fabs(rate) + SYNTHESIS_LOOP_MIN_RATE);
      m_rate = rate;
    }
  };

  VariableSpeedLoop (StereoAudioFrame & loop)
    : m_loop(loop)
  {
    ASSERT(loop.alias, "expected loop to be alias");
  }

  ~VariableSpeedLoop () { free_complex(m_loop.data); }

  void sample (State & state, StereoAudioFrame & sound_accum);
  void sample (
      State & state,
      const StereoAudioFrame & sound_in,
      StereoAudioFrame & sound_out,
      float decay0,
      float decay1);
};

//====( voices )==============================================================

//----( glottis )-------------------------------------------------------------

/** Glottis

  Glottises are sustained sawtooths.
*/
class Glottis : public float3
{
public:

  typedef float3 Timbre;

  static float power   (const Timbre & t) { return t[0]; }
  static float sustain (const Timbre & t) { return t[1]; }
  static float pitch   (const Timbre & t) { return t[2]; }

  static float & power   (Timbre & t) { return t[0]; }
  static float & sustain (Timbre & t) { return t[1]; }
  static float & pitch   (Timbre & t) { return t[2]; }

  static Timbre layout (const Finger & finger)
  {
    float energy   = finger.get_energy();
    float sustain  = finger.get_z();
    float pitch    = finger.get_x();
    return Timbre(energy, sustain, pitch);
  }

private:

  BuzzingActuator m_actuator;
  float m_energy;

  void * operator new[] (size_t) { ERROR("cannot create array of glottises"); }
  void operator delete[] (void *) { ERROR("cannot delete array of glottises"); }

public:

  Glottis (const Timbre & t) : Timbre(t) {}
  Glottis () : Timbre(0,0,0) {}

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_energy > tol;
  }
};

//----( buzzers )-------------------------------------------------------------

/** Buzzers

  Buzzers are synthesized by a sawtooth actuator
  passed through a narrow bandpass filter.

  TODO add growl parameter for ~16Hz tremelo
*/
class Buzzer : public float4
{
public:

  typedef float4 Timbre;

  static float energy         (const Timbre & t) { return t[0]; }
  static float actuate_pitch  (const Timbre & t) { return t[1]; }
  static float resonate_pitch (const Timbre & t) { return t[2]; }

  static float & energy         (Timbre & t) { return t[0]; }
  static float & actuate_pitch  (Timbre & t) { return t[1]; }
  static float & resonate_pitch (Timbre & t) { return t[2]; }

  static Timbre layout (const Finger & finger)
  {
    float energy = finger.get_energy();
    float actuate  = finger.get_x();
    float resonate = finger.get_y();
    return Timbre(energy, actuate, resonate, 0);
  }

  static Timbre layout (const Chord & chord)
  {
    float energy = chord.get_energy();
    float actuate  = chord.get_x();
    float resonate = 3 * (2 * sqrtf(chord.get_length()) - 1);
    return Timbre(energy, actuate, resonate, 0);
  }

private:

  Resonator m_resonator;
  BuzzingActuator m_actuator;

  void * operator new[] (size_t) { ERROR("cannot create array of buzzers"); }
  void operator delete[] (void *) { ERROR("cannot delete array of buzzers"); }

public:

  Buzzer (const Timbre & t) : Timbre(t) {}
  Buzzer () : Timbre(0,0,0,0) {}

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_resonator.active(tol);
  }
};

//----( sine waves )----------------------------------------------------------

/** Simple sine-waves with envelope
*/
class Sine : public float4
{
public:

  typedef float4 Timbre;

  static float energy  (const Timbre & t) { return t[0]; }
  static float pitch   (const Timbre & t) { return t[1]; }
  static float sustain (const Timbre & t) { return t[2]; }

  static float & energy  (Timbre & t) { return t[0]; }
  static float & pitch   (Timbre & t) { return t[1]; }
  static float & sustain (Timbre & t) { return t[2]; }

  static Timbre layout (const Finger & finger)
  {
    float energy = finger.get_energy();
    float pitch = finger.get_x();
    float sustain = finger.get_z();
    return Timbre(energy, pitch, sustain, 0);
  }

  static Timbre layout (const Chord & chord)
  {
    float energy = chord.get_energy();
    float pitch = chord.get_x();
    float sustain = chord.get_z();
    return Timbre(energy, pitch, sustain, 0);
  }

private:

  complex m_phase;
  float m_energy;

  void * operator new[] (size_t) { ERROR("cannot create array of sines"); }
  void operator delete[] (void *) { ERROR("cannot delete array of sines"); }

public:

  Sine (const Timbre & t) : Timbre(t), m_phase(0,1), m_energy(0) {}
  Sine () : Timbre(0,0,0,0), m_phase(0,1), m_energy(0) {}

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_energy > tol;
  }
};

//----( exp-sine waves )------------------------------------------------------

/** Simple exp(sin)-waves with envelope, with Poisson power spectrum
*/
class ExpSine : public float4
{
public:

  typedef float4 Timbre;

  static float energy  (const Timbre & t) { return t[0]; }
  static float sustain (const Timbre & t) { return t[1]; }
  static float pitch   (const Timbre & t) { return t[2]; }
  static float rough   (const Timbre & t) { return t[3]; }

  static float & energy  (Timbre & t) { return t[0]; }
  static float & sustain (Timbre & t) { return t[1]; }
  static float & pitch   (Timbre & t) { return t[2]; }
  static float & rough   (Timbre & t) { return t[3]; }

  static Timbre layout (const Finger & finger)
  {
    float energy = finger.get_energy();
    float sustain = finger.get_z();
    float pitch = finger.get_x();
    float rough = finger.get_y();
    return Timbre(energy, sustain, pitch, rough);
  }

  static Timbre layout (const Chord & chord)
  {
    float energy = chord.get_energy();
    float sustain = chord.get_z();
    float pitch = chord.get_x();
    float rough = chord.get_y();
    return Timbre(energy, sustain, pitch, rough);
  }

private:

  complex m_phase;
  float m_energy;

  void * operator new[] (size_t) { ERROR("cannot create array of exp-sines"); }
  void operator delete[] (void *) { ERROR("cannot delete array of exp-sines"); }

public:

  ExpSine (const Timbre & t) : Timbre(t), m_phase(0,1), m_energy(0) {}
  ExpSine () : Timbre(0,0,0,0), m_phase(0,1), m_energy(0) {}

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_energy > tol;
  }
};

//----( bells )---------------------------------------------------------------

/** Bells

  Bells are like Buzzers, but with sustain.
*/
class Bell : public float4
{
public:

  typedef float4 Timbre;

  static float power          (const Timbre & t) { return t[0]; }
  static float sustain        (const Timbre & t) { return t[1]; }
  static float actuate_pitch  (const Timbre & t) { return t[2]; }
  static float resonate_pitch (const Timbre & t) { return t[3]; }

  static float & power          (Timbre & t) { return t[0]; }
  static float & sustain        (Timbre & t) { return t[1]; }
  static float & actuate_pitch  (Timbre & t) { return t[2]; }
  static float & resonate_pitch (Timbre & t) { return t[3]; }

  static Timbre layout (const Finger & finger)
  {
    float energy = finger.get_energy();
    float sustain  = finger.get_z();
    float actuate  = finger.get_x();
    float resonate = finger.get_y();
    return Timbre(energy, sustain, actuate, resonate);
  }

private:

  Resonator m_resonator;
  BuzzingActuator m_actuator;
  float m_energy;

  void * operator new[] (size_t) { ERROR("cannot create array of buzzers"); }
  void operator delete[] (void *) { ERROR("cannot delete array of buzzers"); }

public:

  Bell (const Timbre & t) : Timbre(t) {}
  Bell () : Timbre(0,0,0,0) {}

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_resonator.active(tol);
  }
};

//----( pipes )---------------------------------------------------------------

/** Pipes

  Pipes are synthesized by a white noise actuator
  passed through a variable-width bandpass filter.
*/
class Pipe : public float4
{
public:

  typedef float4 Timbre;

  static float energy    (const Timbre & t) { return t[0]; }
  static float pitch     (const Timbre & t) { return t[1]; }
  static float bandwidth (const Timbre & t) { return t[2]; }

  static float & energy    (Timbre & t) { return t[0]; }
  static float & pitch     (Timbre & t) { return t[1]; }
  static float & bandwidth (Timbre & t) { return t[2]; }

  static Timbre layout (const Finger & finger)
  {
    float energy = finger.get_energy();
    float pitch = finger.get_x();
    float bandwidth = sigmoid(0.5 * finger.get_y());
    return Timbre(energy, pitch, bandwidth, 0);
  }

private:

  NoiseResonator m_resonator;

  void * operator new[] (size_t) { ERROR("cannot create array of pipes"); }
  void operator delete[] (void *) { ERROR("cannot delete array of pipes"); }

public:

  Pipe (const Timbre & t) : Timbre(t) {}
  Pipe () : Timbre(0,0,0,0) {}

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_resonator.active(tol);
  }
};

//----( vibes )---------------------------------------------------------------

/** Vibes

  Vibees are synthesized by an freq & amplitude-modulated oscillator.

  energy   - the target energy this vibe is drifting towards
  pitch    - the base pitch at which this vibe is vibrating
  amp_mod  - the rate at which this vibe is drifting towards target energy
  freq_mod - controls sinusoidal freq modulation for timbre
*/
class Vibe : public float4
{
public:

  typedef float4 Timbre;

  static float energy   (const Timbre & t) { return t[0]; }
  static float pitch    (const Timbre & t) { return t[1]; }
  static float sustain  (const Timbre & t) { return t[2]; }

  static float & energy   (Timbre & t) { return t[0]; }
  static float & pitch    (Timbre & t) { return t[1]; }
  static float & sustain  (Timbre & t) { return t[2]; }

  static Timbre layout (const Finger & finger)
  {
    float energy = finger.get_energy();
    float pitch = finger.get_x();
    float sustain = finger.get_z();
    return Timbre(energy, pitch, sustain, 0);
  }

private:

  float m_energy;
  int m_phase;

  void * operator new[] (size_t) { ERROR("cannot create array of vibes"); }
  void operator delete[] (void *) { ERROR("cannot delete array of vibes"); }

public:

  Vibe (const Timbre & t) : Timbre(t), m_energy(0), m_phase(0) {}
  Vibe () : Timbre(0,0,0,0), m_energy(0), m_phase(0) {}

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_energy > tol;
  }
};

//----( gongs )---------------------------------------------------------------

/** Gongs

  Gongs are synthesized by an freq & amplitude-modulated oscillator.

  energy   - the target energy this gong is drifting towards
  pitch    - the base pitch at which this gong is vibrating
  amp_mod  - the rate at which this gong is drifting towards target energy
  freq_mod - controls sinusoidal freq modulation for timbre
*/
class Gong : public float4
{
public:

  typedef float4 Timbre;

  static float energy   (const Timbre & t) { return t[0]; }
  static float pitch    (const Timbre & t) { return t[1]; }
  static float amp_mod  (const Timbre & t) { return t[2]; }
  static float freq_mod (const Timbre & t) { return t[3]; }

  static float & energy   (Timbre & t) { return t[0]; }
  static float & pitch    (Timbre & t) { return t[1]; }
  static float & amp_mod  (Timbre & t) { return t[2]; }
  static float & freq_mod (Timbre & t) { return t[3]; }

  static Timbre layout (const Finger & finger)
  {
    float energy = finger.get_energy();
    float pitch = finger.get_x();
    float amp_mod = finger.get_z();
    float freq_mod = finger.get_y() / 3;
    return Timbre(energy, pitch, amp_mod, freq_mod);
  }

private:

  float m_energy;
  int m_phase;

  void * operator new[] (size_t) { ERROR("cannot create array of gongs"); }
  void operator delete[] (void *) { ERROR("cannot delete array of gongs"); }

public:

  Gong (const Timbre & t) : Timbre(t), m_energy(0), m_phase(0) {}
  Gong () : Timbre(0,0,0,0), m_energy(0), m_phase(0) {}

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_energy > tol;
  }
};

//----( formants )------------------------------------------------------------

/** Formants

  Formants are synthesized by a sawtooth actuator
  passed through multiple narrow bandpass filters.
*/
class Formant : public float4
{
public:

  typedef float4 Timbre;

  static float energy         (const Timbre & t) { return t[0]; }
  static float actuate_pitch  (const Timbre & t) { return t[1]; }
  static float resonate1      (const Timbre & t) { return t[2]; }
  static float resonate2      (const Timbre & t) { return t[3]; }

  static float & energy         (Timbre & t) { return t[0]; }
  static float & actuate_pitch  (Timbre & t) { return t[1]; }
  static float & resonate1      (Timbre & t) { return t[2]; }
  static float & resonate2      (Timbre & t) { return t[3]; }

  static Timbre layout (const Chord & chord)
  {
    float energy = chord.get_energy();
    float actuate  = chord.get_x();
    float resonate1 = chord.get_y();
    float resonate2 = 3 * (2 * chord.get_length() - 1);
    return Timbre(energy, actuate, resonate1, resonate2);
  }

private:

  Resonator m_resonator1;
  Resonator m_resonator2;
  BuzzingActuator m_actuator;

  void * operator new[] (size_t) { ERROR("cannot create array of formants"); }
  void operator delete[] (void *) { ERROR("cannot delete array of formants"); }

public:

  Formant (const Timbre & t) : Timbre(t) {}
  Formant () : Timbre(0,0,0,0) {}

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_resonator1.active(tol) or m_resonator2.active(tol);
  }
};

//----( shepard tones )-------------------------------------------------------

/** Shepard Tones

  Shepard Tones separate fine- and coarse- pitch by
  filtering fine-grained octave-offset clicks through a wide bandpass filter.

  energy - the signal energy
  fine   - the fine-scale periodic frequency, integer periodic
  coarse - the coarse-scale bandpass filter pitch
*/
class Shepard : public float4
{
public:

  typedef float4 Timbre;

  static float energy  (const Timbre & t) { return t[0]; }
  static float fine    (const Timbre & t) { return t[1]; }
  static float coarse  (const Timbre & t) { return t[2]; }
  static float formant (const Timbre & t) { return t[3]; }

  static float & energy  (Timbre & t) { return t[0]; }
  static float & fine    (Timbre & t) { return t[1]; }
  static float & coarse  (Timbre & t) { return t[2]; }
  static float & formant (Timbre & t) { return t[3]; }

  static Timbre layout (const Finger & finger)
  {
    float energy = finger.get_energy();
    float fine = finger.get_x();
    float coarse = finger.get_y();
    float formant = finger.get_y();

    return Timbre(energy, fine, coarse, formant);
  }

  static Timbre layout (const Chord & chord)
  {
    float energy = chord.get_energy();
    float fine = chord.get_angle(); // TODO soft clamp to 12-tone scale
    float coarse = chord.get_y();
    float formant = chord.get_length();

    return Timbre(energy, fine, coarse, formant);
  }

private:

  ShepardActuator m_actuator;
  OctavePassFilter m_filter;
  Resonator m_resonator;

  void * operator new[] (size_t) { ERROR("cannot create array of shepards"); }
  void operator delete[] (void *) { ERROR("cannot delete array of shepards"); }

public:

  Shepard (const Timbre & t) : Timbre(t) {}
  Shepard () : Timbre(0,0,0,0) {}

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_filter.active(tol);
  }
};

//----( shepard vibes )-------------------------------------------------------

/** Shepard Vibes

  Shepard Vibes separate fine- and coarse- pitch by
  Bernstein-blending multiple sinusoids, each an octave apart.

  energy - the signal energy
  fine   - the fine-scale periodic frequency, integer periodic
  coarse - the coarse-scale bandpass filter pitch
*/
class ShepardVibe : public float4
{
public:

  typedef float4 Timbre;

  static float energy  (const Timbre & t) { return t[0]; }
  static float fine    (const Timbre & t) { return t[1]; }
  static float coarse  (const Timbre & t) { return t[2]; }
  static float sustain (const Timbre & t) { return t[3]; }

  static float & energy  (Timbre & t) { return t[0]; }
  static float & fine    (Timbre & t) { return t[1]; }
  static float & coarse  (Timbre & t) { return t[2]; }
  static float & sustain (Timbre & t) { return t[3]; }

  static Timbre layout (const Finger & finger)
  {
    float energy = finger.get_energy();
    float fine = finger.get_x();
    if (safe_isfinite(SHEPARD_FINE_SPEED_UNIT)) {
      float fine_speed = finger.get_x_t() / SHEPARD_FINE_SPEED_UNIT;
      float hardness = Gestures::is_small(fine_speed);
      fine = soft_clamp_to_grid(fine * 12, hardness) / 12;
    }
    float coarse = finger.get_y();
    float sustain = finger.get_z();

    return Timbre(energy, fine, coarse, sustain);
  }

  static Timbre layout (const Chord & chord)
  {
    float energy = chord.get_energy();
    float fine = -chord.get_angle();
    if (safe_isfinite(SHEPARD_FINE_SPEED_UNIT)) {
      float fine_speed = -chord.get_angle_t() / SHEPARD_FINE_SPEED_UNIT;
      float hardness = Gestures::is_small(fine_speed);
      fine = soft_clamp_to_grid(fine * 12, hardness) / 12;
    }
    float coarse = chord.get_y();
    float sustain = chord.get_z();

    return Timbre(energy, fine, coarse, sustain);
  }

private:

  enum {
    num_octaves = 4,
    num_tones = 2 * num_octaves
  };

  complex m_phase[num_tones];
  float m_energy;

  void * operator new[] (size_t) { ERROR("cannot create ShepardVibes array"); }
  void operator delete[] (void *) { ERROR("cannot delete ShepardVibes array"); }

  void reset ()
  {
    m_energy = 0;
    for (size_t o = 0; o < num_tones; ++o) {
      m_phase[o] = exp_2_pi_i(random_01());
    }
  }

public:

  ShepardVibe (const Timbre & t) : Timbre(t) { reset(); }
  ShepardVibe () : Timbre(0,0,0,0) { reset(); }

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_energy > tol;
  }
};

//----( string )--------------------------------------------------------------

/** Strings

  Pitches with variously much energy in harmonics.
*/
class String : public float3
{
public:

  typedef float3 Timbre;

  static float energy  (const Timbre & t) { return t[0]; }
  static float pitch   (const Timbre & t) { return t[1]; }
  static float width   (const Timbre & t) { return t[2]; }

  static float & energy  (Timbre & t) { return t[0]; }
  static float & pitch   (Timbre & t) { return t[1]; }
  static float & width   (Timbre & t) { return t[2]; }

  static Timbre layout (const Finger & finger)
  {
    float energy = finger.get_z();
    float pitch = finger.get_x();
    float width = bound_to(0.0f, 1.0f, finger.get_y() / GRID_SIZE_Y + 0.5f);

    return Timbre(energy, pitch, width);
  }

private:

  complex m_phase;

  void * operator new[] (size_t) { ERROR("cannot create String array"); }
  void operator delete[] (void *) { ERROR("cannot delete String array"); }

public:

  String (const Timbre & t) : Timbre(t), m_phase(1,0) {}
  String () : Timbre(0,0,0), m_phase(1,0) {}

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return energy(* this) > tol;
  }
};

//----( plucked )-------------------------------------------------------------

/** Plucked Strings

  Pitches with variously much energy in harmonics.
*/
class Plucked : public float3
{
public:

  typedef float3 Timbre;

  static float energy  (const Timbre & t) { return t[0]; }
  static float sustain (const Timbre & t) { return t[1]; }
  static float pitch   (const Timbre & t) { return t[2]; }

  static float & energy  (Timbre & t) { return t[0]; }
  static float & sustain (Timbre & t) { return t[1]; }
  static float & pitch   (Timbre & t) { return t[2]; }

  static Timbre layout (const Finger & finger)
  {
    float energy = finger.get_energy();
    float sustain = finger.get_z();
    float pitch = finger.get_x();

    return Timbre(energy, sustain, pitch);
  }

private:

  complex m_phase;
  float m_energy;
  float m_attack;

  void * operator new[] (size_t) { ERROR("cannot create Plucked array"); }
  void operator delete[] (void *) { ERROR("cannot delete Plucked array"); }

public:

  Plucked (const Timbre & t)
    : Timbre(t),
      m_phase(1,0),
      m_energy(0),
      m_attack(0)
  {}
  Plucked ()
    : Timbre(0,0,0),
      m_phase(1,0),
      m_energy(0),
      m_attack(0)
  {}

  // sampling
  void sample (const Timbre & t_final, StereoAudioFrame & sound_accum);
  void fadein (StereoAudioFrame & sound_accum);
  void fadeout (StereoAudioFrame & sound_accum);
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_energy > tol;
  }
};

//----( wrapped grid )--------------------------------------------------------

template<class Voice, int offset = 0, int scale = 1, int shift = 0>
class Wrapped : public Voice
{
public:

  typedef typename Voice::Timbre Timbre;

  static Timbre layout (Finger finger)
  {
    float & energy = finger.energy();
    float & x = finger.x();
    float & y = finger.y();
    float level = round(y);

    x += offset * level + shift;
    x /= scale;
    y -= level;
    energy *= 1 - sqr(2 * y);
    y *= 6;

    return Voice::layout(finger);
  }

  Wrapped (const Timbre & t) : Voice(t) {}
  Wrapped () : Voice () {}
};

//----( vocoder )-------------------------------------------------------------

class Vocoder : public Synchronized::PhasorBank
{
public:

  struct Timbre : public Vector<float>
  {
    void operator= (const Vector<float> & other)
    {
      Vector<float>::operator=(other);
    }

    Timbre () : Vector<float>(SYNTHESIS_VOCODER_SIZE) {}
    Timbre (const Timbre & other)
      : Vector<float>(SYNTHESIS_VOCODER_SIZE)
    {
      operator=(other);
    }
  };

  Vocoder ()
    : Synchronized::PhasorBank(Synchronized::Bank(
        SYNTHESIS_VOCODER_SIZE,
        SYNTHESIS_VOCODER_MIN_FREQ_HZ / DEFAULT_SAMPLE_RATE,
        SYNTHESIS_VOCODER_MAX_FREQ_HZ / DEFAULT_SAMPLE_RATE,
        SYNTHESIS_VOCODER_ACUITY))
  {}

  void sample (Timbre & timbre, StereoAudioFrame & sound_accum)
  {
    timbre -= m_mass;
    sample_accum(timbre, sound_accum);
  }

  void retune_sample (Timbre & timbre, StereoAudioFrame & sound_accum)
  {
    retune(timbre, m_temp);
    m_temp -= m_mass;
    sample_accum(m_temp, sound_accum);
  }
};

class VocoPoint : public float2
{
  void * operator new[] (size_t) { ERROR("cannot create array of VocoPoints"); }
  void operator delete[] (void *)
  {
    ERROR("cannot delete array of VocoPoints");
  }

public:

  typedef float2 Timbre;

  static float mass  (const Timbre & t) { return t[0]; }
  static float pitch (const Timbre & t) { return t[1]; }

  static float & mass  (Timbre & t) { return t[0]; }
  static float & pitch (Timbre & t) { return t[1]; }

  static Timbre layout (const Finger & finger)
  {
    float mass = finger.get_z();
    float pitch = bound_to(0.f, 1.f, 0.5f + finger.get_x() / GRID_SIZE_X);

    return Timbre(mass, pitch);
  }

  VocoPoint (const Timbre & t) : Timbre(t) {}
  VocoPoint () : Timbre(0,0) {}

  // sampling
  void sample (Vocoder::Timbre & sound_accum);
  void sample (const Timbre & t_final, Vocoder::Timbre & sound_accum)
  {
    operator=(t_final);
    sample(sound_accum);
  }
  void fadein (Vocoder::Timbre & sound_accum) { sample(sound_accum); }
  void fadeout (Vocoder::Timbre & sound_accum) {}
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return mass(* this) > tol;
  }
};

class VocoBlob : public float3
{
  void * operator new[] (size_t) { ERROR("cannot create array of VocoBlobs"); }
  void operator delete[] (void *)
  {
    ERROR("cannot delete array of VocoBlobs");
  }

public:

  typedef float3 Timbre;

  static float mass  (const Timbre & t) { return t[0]; }
  static float pitch (const Timbre & t) { return t[1]; }
  static float width (const Timbre & t) { return t[2]; }

  static float & mass  (Timbre & t) { return t[0]; }
  static float & pitch (Timbre & t) { return t[1]; }
  static float & width (Timbre & t) { return t[2]; }

  static Timbre layout (const Finger & finger)
  {
    float mass = finger.get_z();
    float pitch = bound_to(0.f, 1.f, 0.5f + finger.get_x() / GRID_SIZE_X);
    float width = exp(8 * bound_to(-0.5f, 0.5f, finger.get_y() / GRID_SIZE_Y));

    return Timbre(mass, pitch, width);
  }

  VocoBlob (const Timbre & t) : Timbre(t) {}
  VocoBlob () : Timbre(0,0,0) {}

  // sampling
  void sample (Vocoder::Timbre & sound_accum);
  void sample (const Timbre & t_final, Vocoder::Timbre & sound_accum)
  {
    operator=(t_final);
    sample(sound_accum);
  }
  void fadein (Vocoder::Timbre & sound_accum) { sample(sound_accum); }
  void fadeout (Vocoder::Timbre & sound_accum) {}
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return mass(* this) > tol;
  }
};

//----( beater )--------------------------------------------------------------

class Beater : public Synchronized::LoopBank
{
protected:

  const float m_timestep;
  const float m_decay;
  const size_t m_blur_radius;

public:

  struct Timbre : public Vector<float>
  {
    void operator= (const Vector<float> & other)
    {
      Vector<float>::operator=(other);
    }

    Timbre () : Vector<float>(SYNTHESIS_BEATER_SIZE) {}
    Timbre (const Timbre & other)
      : Vector<float>(SYNTHESIS_BEATER_SIZE)
    {
      operator=(other);
    }
  };

  Beater (bool coalesce = true, float blur_factor = 1.0f);

  void advance () { Synchronized::LoopBank::advance(m_timestep, m_decay); }
  void sample (Timbre & amplitude);

private:
  Timbre m_temp;
};

//====( coupled voices )======================================================

namespace Coupled
{

extern const Synchronized::ScaledBeatFun g_pitch_param;
extern const Synchronized::ScaledBeatFun g_tempo_param;
extern const Synchronized::ScaledBeatFun g_synco_param;

//----( envelopes )----
// Envelopes interpolate mass while sampling

class SlowVoice
{
protected:

  float m_mass;

public:

  SlowVoice () : m_mass(0) {}

  void advance () { m_mass = 0; }
  void advance (float fadeout_factor) { m_mass *= fadeout_factor; }
  bool active (float tol = SYNTHESIS_ENERGY_TOL) { return m_mass > tol; }

  // abstract methods
  //void set_timbre (const Finger &);
  //Synchronized::Poll poll () const;
  //void sample (complex force, Sound & sound_accum);
};

class FastVoice
{
protected:

  float m_mass0;
  float m_mass1;
  float m_mass;

public:

  FastVoice () : m_mass0(0), m_mass1(0) {}

  void advance () { m_mass0 = m_mass1; m_mass1 = 0; }
  void advance (float fadeout_factor)
  {
    m_mass0 = m_mass1;
    m_mass1 *= fadeout_factor;
  }
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_mass0 + m_mass1 > tol;
  }
  void interpolate (float t) { m_mass = m_mass0 + t * (m_mass1 - m_mass0); }

  // abstract methods
  //void set_timbre (const Finger &);
  //Synchronized::Poll poll () const;
  //complex sample (complex force);
};

class ModulatedVoice
{
protected:

  float m_mass0;
  float m_mass1;
  float m_mass;
  float m_amp0;
  float m_amp1;
  float m_amp;

public:

  ModulatedVoice () : m_mass0(0), m_mass1(0), m_amp0(0), m_amp1(0) {}

  void advance ()
  {
    m_mass0 = m_mass1;
    m_amp0 = m_amp1;
    m_mass1 = 0;
    m_amp1 = 0;
  }
  void advance (float fadeout_factor)
  {
    m_mass0 = m_mass1;
    m_amp0 = m_amp1;
    m_mass1 *= fadeout_factor;
    m_amp1 *= fadeout_factor;
  }
  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_mass0 + m_mass1 > tol;
  }
  void interpolate (float t)
  {
    m_mass = m_mass0 + t * (m_mass1 - m_mass0);
    m_amp = m_amp0 + t * (m_amp1 - m_amp0);
  }

  // abstract methods
  //void set_timbre (const Finger &);
  //Synchronized::Poll poll_slow () const;
  //void sample_slow (complex force);
  //Synchronized::Poll poll_fast () const;
  //complex sample_fast (complex force);
};

//----( slow-coupled voices )----

class SyncoPipe : public SlowVoice
{
  Synchronized::Syncopator m_tempo;
  Pipe m_pipe;
  Pipe::Timbre m_timbre;
  float m_bandwidth;

public:

  typedef StereoAudioFrame Sound;

  SyncoPipe () : m_tempo(g_synco_param) {}

  void shift_pitch (float shift)
  {
    Pipe::pitch(m_timbre) += shift;
    m_tempo.freq *= exp(shift);
  }

  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_pipe.active(tol);
  }

  complex as_complex () const { return m_mass * m_tempo.phase; }

  void set_timbre (const Finger & polar_finger)
  {
    m_mass = polar_finger.get_z();
    m_tempo.freq = tempo_to_freq_synco(polar_finger.get_x());
    m_tempo.offset = exp_2_pi_i(polar_finger.get_y());
    Pipe::pitch(m_timbre) = 4 * polar_finger.get_x() + SYNCOPIPE_PITCH_SHIFT;
    m_bandwidth = (1 + polar_finger.get_x()) / 2;
  }

  Synchronized::Poll poll () const { return m_tempo.poll(); }

  void sample (complex force, Sound & sound_accum)
  {
    m_tempo.mass = m_mass;
    complex phase = m_tempo.sample(force);
    float time = wrap(arg(phase) / (2 * M_PI));

    Pipe::energy(m_timbre) = m_mass * 4.0f * sqr(sqr(sqr(1 - time)));
    Pipe::bandwidth(m_timbre) = m_bandwidth * sqr(1 - time);

    m_pipe.sample(m_timbre, sound_accum);
  }
};

class SyncoBlob : public SlowVoice
{
  Synchronized::Syncopator m_tempo;
  VocoBlob m_blob;
  VocoBlob::Timbre m_timbre;

public:

  typedef Vocoder::Timbre Sound;

  SyncoBlob () : m_tempo(g_synco_param) {}

  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_blob.active(tol);
  }

  complex as_complex () const { return m_mass * m_tempo.phase; }

  void set_timbre (const Finger & polar_finger)
  {
    m_mass = polar_finger.get_z();
    m_tempo.freq = tempo_to_freq_synco(polar_finger.get_x());
    m_tempo.offset = exp_2_pi_i(polar_finger.get_y());
    VocoBlob::pitch(m_timbre) = 0.2 + 0.2 * polar_finger.get_x();
  }

  Synchronized::Poll poll () const { return m_tempo.poll(); }

  void sample (complex force, Sound & sound_accum)
  {
    m_tempo.mass = m_mass;
    complex phase = m_tempo.sample(force);
    float time = wrap(arg(phase) / (2 * M_PI));

    VocoBlob::mass(m_timbre) = m_mass * exp(-4 * time);
    VocoBlob::width(m_timbre) = 20 * exp(-2 * time);

    m_blob.sample(m_timbre, sound_accum);
  }
};

class SyncoPoint : public SlowVoice
{
  Synchronized::Syncopator m_tempo;
  VocoPoint m_point;
  VocoPoint::Timbre m_timbre;

public:

  typedef Vocoder::Timbre Sound;

  SyncoPoint () : m_tempo(g_synco_param) {}

  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_point.active(tol);
  }

  complex as_complex () const { return m_mass * m_tempo.phase; }

  void set_timbre (const Finger & finger)
  {
    m_timbre = VocoPoint::layout(finger);
    m_mass = VocoPoint::mass(m_timbre);
    m_tempo.freq = tempo_to_freq_synco_grid(finger.get_y());
    m_tempo.offset = exp_2_pi_i(finger.get_y()); // one cycle per gridline
  }

  Synchronized::Poll poll () const { return m_tempo.poll(); }

  void sample (complex force, Sound & sound_accum)
  {
    complex phase = m_tempo.sample(force);
    float time = wrap(arg(phase) / (2 * M_PI));

    VocoPoint::mass(m_timbre) = m_mass * exp(-4 * time);

    m_point.sample(m_timbre, sound_accum);
  }
};

class SyncoWobble : public SlowVoice
{
  Synchronized::Syncopator m_tempo;

public:

  typedef complex Sound;

  SyncoWobble () : m_tempo(g_synco_param) {}

  void set_timbre (const Finger & polar_finger)
  {
    m_mass = polar_finger.get_z();
    m_tempo.freq = tempo_to_freq_synco(polar_finger.get_x());
    m_tempo.offset = exp_2_pi_i(polar_finger.get_y());
  }

  Synchronized::Poll poll () const { return m_tempo.poll(); }

  void sample (complex force, Sound & sound)
  {
    m_tempo.mass = m_mass;
    sound += m_mass * m_tempo.sample(force);
  }
};

class SyncoString : public SlowVoice
{
  float m_part1;
  float m_part2;
  float m_width1;
  float m_width2;
  String m_pitch1;
  String m_pitch2;
  String::Timbre m_timbre1;
  String::Timbre m_timbre2;

  Synchronized::Syncopator m_tempo;

public:

  typedef StereoAudioFrame Sound;

  SyncoString ()
    : m_part1(0.5f),
      m_part2(0.5f),
      m_width1(0),
      m_width2(0),
      m_tempo(g_synco_param)
  {}

  bool active (float tol = SYNTHESIS_ENERGY_TOL) const
  {
    return m_pitch1.active(tol) or m_pitch2.active(tol);
  }

  void set_timbre (const Finger & polar_finger);

  Synchronized::Poll poll () const { return m_tempo.poll(); }

  void sample (complex force, Sound & sound);
};

//----( fast-coupled voices )----

class Sine : public FastVoice
{
  float m_unused_empty_spacer_to_appease_gcc;
  Synchronized::Phasor m_pitch;

public:

  Sine () : m_pitch(g_pitch_param) {}

  void set_timbre (const Finger & grid_finger)
  {
    m_mass1 = grid_finger.get_z();
    m_pitch.freq = pitch_to_freq_wide(grid_finger.get_x());
  }

  Synchronized::Poll poll () const { return m_pitch.poll(); }

  complex sample (complex force)
  {
    m_pitch.mass = m_mass;
    complex phase = m_pitch.sample(force);
    return m_mass * phase;
  }
};

class Shepard4 : public ModulatedVoice
{
  float m_unused_empty_spacer_to_appease_gcc;
  Synchronized::Shepard4 m_pitch;

public:

  Shepard4 () : m_pitch(g_pitch_param) {}

  void set_timbre (const Finger & grid_finger)
  {
    m_mass1 = grid_finger.get_z();
    m_amp1 = (grid_finger.get_y() + 3.0f) * 0.5f;
    m_pitch.freq = pitch_to_freq_shepard4(grid_finger.get_x());
  }

  Synchronized::Poll poll () const { return m_pitch.poll(); }

  complex sample (complex force)
  {
    m_pitch.mass = m_mass;
    m_pitch.octave = m_amp;
    complex state = m_pitch.sample(force);
    return m_mass * state;
  }
};

class Shepard7 : public ModulatedVoice
{
  float m_unused_empty_spacer_to_appease_gcc;
  Synchronized::Shepard7 m_pitch;

public:

  Shepard7 () : m_pitch(g_pitch_param) {}

  void set_timbre (const Finger & grid_finger)
  {
    m_mass1 = grid_finger.get_z();
    m_amp1 = grid_finger.get_y() + 3.0f;
    m_pitch.freq = pitch_to_freq_shepard7(grid_finger.get_x());
  }

  Synchronized::Poll poll () const { return m_pitch.poll(); }

  complex sample (complex force)
  {
    m_pitch.mass = m_mass;
    m_pitch.octave = m_amp;
    complex state = m_pitch.sample(force);
    return m_mass * state;
  }
};

class SitarString : public ModulatedVoice
{
  Synchronized::Phasor m_pitch;

public:

  SitarString () : m_pitch(g_pitch_param) {}

  void set_timbre (const Finger & grid_finger)
  {
    float sustain = grid_finger.get_z();
    float energy = grid_finger.get_energy();
    float pitch = grid_finger.get_x();
    float sharp = bound_to(
        0.0f, 1.0f,
        grid_finger.get_y() / GRID_SIZE_Y + 0.5f);

    float timescale = ( DEFAULT_SYNTHESIS_RELEASE_SEC
                     + DEFAULT_SYNTHESIS_SUSTAIN_SEC * sustain )
                   * DEFAULT_AUDIO_FRAMERATE;
    float decay = expf(-1.0f / timescale);
    m_mass1 = max(decay * m_mass0, energy);
    m_amp1 = max(decay * m_amp0, sharp);
    m_pitch.freq = pitch_to_2_pi_freq(pitch);
  }

  Synchronized::Poll poll () const { return m_pitch.poll(); }

  complex sample (complex force)
  {
    m_pitch.mass = m_mass;
    complex phase = m_pitch.sample(force);
    return m_mass * phase / (1.2f - m_amp * phase);
  }
};

//----( bi-coupled voices )----

/** Amplitude-modulated oscillator
  see test/synchronize.py synth
*/
class BeatingSine : public ModulatedVoice
{
  Synchronized::Phasor m_pitch;
  Synchronized::Phasor m_tempo;

public:

  BeatingSine () : m_pitch(g_pitch_param), m_tempo(g_tempo_param) {}

  void set_timbre (const Finger & grid_finger)
  {
    m_mass1 = grid_finger.get_z();
    m_tempo.freq = tempo_to_freq_split(grid_finger.get_y());
    m_pitch.freq = pitch_to_freq_split(grid_finger.get_x());
  }

  Synchronized::Poll poll_slow () const { return m_tempo.poll(); }
  Synchronized::Poll poll_fast () const { return m_pitch.poll(); }

  void sample_slow (complex force)
  {
    m_tempo.mass = m_mass1;
    complex phase = m_tempo.sample(force);
    m_amp1 = m_mass1 * max(0.0f, phase.real());
  }

  complex sample_fast (complex force)
  {
    m_pitch.mass = m_mass;
    return m_amp * m_pitch.sample(force);
  }
};

class BeatingPlucked : public ModulatedVoice
{
  Synchronized::Phasor m_pitch;
  Synchronized::Phasor m_tempo;

public:

  BeatingPlucked () : m_pitch(g_pitch_param), m_tempo(g_tempo_param) {}

  void set_timbre (const Finger & grid_finger)
  {
    m_mass1 = bound_to(0.0f, 1.0f, grid_finger.get_z());
    m_tempo.freq = tempo_to_freq_split(grid_finger.get_y());
    m_pitch.freq = pitch_to_freq_split(grid_finger.get_x());
  }

  Synchronized::Poll poll_slow () const { return m_tempo.poll(); }
  Synchronized::Poll poll_fast () const { return m_pitch.poll(); }

  void sample_slow (complex force)
  {
    float mass = m_tempo.mass = m_mass1;
    complex phase = m_tempo.sample(force);
    float angle = wrap(arg(phase) / (2 * M_PI));
    m_amp1 = mass * exp(-2 * angle);
  }

  complex sample_fast (complex force)
  {
    m_pitch.mass = m_mass;
    complex phase = m_pitch.sample(force);
    float power = 0.85f * m_amp;
    return power * (1.0f - power) * phase / (1.0f - power * phase);
  }
};

class SyncoSine : public ModulatedVoice
{
  Synchronized::Phasor m_pitch;
  Synchronized::Syncopator m_tempo;

public:

  SyncoSine () : m_pitch(g_pitch_param), m_tempo(g_synco_param) {}

  void set_timbre (const Finger & grid_finger)
  {
    m_mass1 = grid_finger.get_z();
    m_tempo.freq = tempo_to_freq_split(grid_finger.get_y());
    m_tempo.offset = exp_2_pi_i(-grid_finger.get_y());
    m_pitch.freq = pitch_to_freq_split(grid_finger.get_x());
  }

  Synchronized::Poll poll_slow () const { return m_tempo.poll(); }
  Synchronized::Poll poll_fast () const { return m_pitch.poll(); }

  void sample_slow (complex force)
  {
    m_tempo.mass = m_mass1;
    complex phase = m_tempo.sample(force);
    m_amp1 = m_mass1 * max(0.0f, phase.real());
  }

  complex sample_fast (complex force)
  {
    m_pitch.mass = m_mass;
    return m_amp * m_pitch.sample(force);
  }
};

class SyncoPlucked : public ModulatedVoice
{
  Synchronized::Phasor m_pitch;
  Synchronized::Syncopator m_tempo;

public:

  SyncoPlucked () : m_pitch(g_pitch_param), m_tempo(g_synco_param) {}

  void set_timbre (const Finger & grid_finger)
  {
    m_mass1 = bound_to(0.0f, 1.0f, grid_finger.get_z());
    m_tempo.freq = tempo_to_freq_split(grid_finger.get_y());
    m_tempo.offset = exp_2_pi_i(-grid_finger.get_y());
    m_pitch.freq = pitch_to_freq_split(grid_finger.get_x());
  }

  Synchronized::Poll poll_slow () const { return m_tempo.poll(); }
  Synchronized::Poll poll_fast () const { return m_pitch.poll(); }

  void sample_slow (complex force)
  {
    float mass = m_tempo.mass = m_mass1;
    complex phase = m_tempo.sample(force);
    float angle = wrap(arg(phase) / (2 * M_PI));
    m_amp1 = mass * exp(-2 * angle);
  }

  complex sample_fast (complex force)
  {
    m_pitch.mass = m_mass;
    complex phase = m_pitch.sample(force);
    float power = 0.85f * m_amp;
    return power * (1.0f - power) * phase / (1.0f - power * phase);
  }
};

} // namespace Coupled

//====( effects )=============================================================

//----( loop resonator )------------------------------------------------------

/** A loop resonator.
  Inspired by the paper (R1), but using a closed loop rather than an open tube.

  (R1) "The Tube Resonance Model Speech Synthesizer" - Leonard Manzara
    http://www.gnu.org/software/gnuspeech/trm-write-up.pdf
*/

class LoopResonator
{
  const float m_decay;

  Vector<float> m_data;
  Vector<float> m_fwd;
  Vector<float> m_bwd;
  Vector<float> m_temp1;
  Vector<float> m_temp2;

public:

  LoopResonator (
      size_t size    = SYNTHESIS_LOOP_RESONATOR_SIZE,
      float timescale = SYNTHESIS_FORMANT_TIMESCALE);
  ~LoopResonator ();

  size_t size () const { return m_fwd.size; }

  void sample (
      const Vector<float> & impedance_in,
      StereoAudioFrame & sound_io);
};

//----( reverb )--------------------------------------------------------------

class EchoBox
{
  const float m_fade_old;
  const float m_fade_new;
  const float m_lowpass_old;
  const float m_lowpass_new;

  StereoAudioFrame m_memory;
  complex m_lowpass;
  size_t m_position;

public:

  EchoBox (
      size_t size,
      float decay    = DEFAULT_ECHOBOX_DECAY,
      float lowpass  = DEFAULT_ECHOBOX_LOWPASS);
  void transform (StereoAudioFrame & sound_io);
};

class Reverberator
{
  EchoBox m_echo0;
  EchoBox m_echo1;
  EchoBox m_echo2;
  EchoBox m_echo3;
  EchoBox m_echo4;
  EchoBox m_echo5;
  EchoBox m_echo6;
  EchoBox m_echo7;

  static size_t box_size (size_t base_size, size_t number);

public:

  Reverberator (size_t base_size = DEFAULT_REVERB_BASE_SIZE);

  void transform (StereoAudioFrame & sound);
};

template<int frame_size>
class FastLoop
{
  const size_t m_num_frames;
  Vector<complex> m_sound;
  size_t m_offset;

public:

  FastLoop (size_t num_frames)
    : m_num_frames(num_frames),
      m_sound(num_frames * frame_size),
      m_offset(0)
  {
    m_sound.zero();
  }

  void advance () { m_offset = (1 + m_offset) % m_num_frames; }

  StereoAudioFrame now  ()
  {
    return StereoAudioFrame(frame_size, m_sound.data + frame_size * m_offset);
  }
};

} // namespace Synthesis

#endif // KAZOO_SYNTHESIS_H

