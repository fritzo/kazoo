
#include "synthesis.h"
#include "splines.h"
#include "random.h"
#include <limits.h>
#include <algorithm>

#define LOG1(mess)
#define ASSERT2_LE(x,y)

#define SYNTHESIS_VIBE_SUSTAIN_TIMESCALE  (2.0f * DEFAULT_SAMPLE_RATE)
#define SYNTHESIS_VIBE_RELEASE_TIMESCALE  (0.1f * DEFAULT_SAMPLE_RATE)

#define SYNTHESIS_GONG_ATTACK_TIMESCALE (0.1f * DEFAULT_SAMPLE_RATE)
#define SYNTHESIS_GONG_SUSTAIN_TIMESCALE (6.0f * DEFAULT_SAMPLE_RATE)

#define SYNTHESIS_PLUCKED_SUSTAIN_TIMESCALE (2.0f * DEFAULT_AUDIO_FRAMERATE)
#define SYNTHESIS_PLUCKED_RELEASE_TIMESCALE (0.1f * DEFAULT_AUDIO_FRAMERATE)
#define SYNTHESIS_PLUCKED_ATTACK            (0.8f)

#define SYNTHESIS_STRING_SHARPNESS          (0.5f)
#define SYNTHESIS_STRING_MAX_WIDTH          (0.9f)

namespace Synthesis
{

//====( elements of synthesis )===============================================

/** Basic Synthesis.

  (N1) Fourier transform of one-sided exponential:
    f(t) = if t > 0 then exp(-k t) else 0
    g(w) = 1 / (k + w i)
  (N2) Fourier transform of one-sided twisted exponential:
    f(t) = if t > 0 then exp((w0 i - k) t) else 0
    g(w) = 1 / (k + (w - w0) i)
*/

//----( actuators )-----------------------------------------------------------

inline float BuzzingActuator::operator() (float pitch)
{
  float freq = pitch_to_freq(pitch);
  m_sawtooth += roundi(UINT_MAX * freq);
  return static_cast<float>(m_sawtooth) / INT_MAX;
}

// 1 2 1 4 1 2 1 8 ... eventually returns 0
inline int shepard_steps (int t)
{
  return ((t ^ (t + 1)) + 1) / 2;
}

// linearly interpolated
inline float shepard_steps (float t)
{
  int t0 = floor(t);
  int t1 = t0 + 1;
  return (t - t0) * shepard_steps(t1)
       + (t1 - t) * shepard_steps(t0);
}

inline float ShepardActuator::operator() (float pitch)
{
  float freq = pitch_to_freq_mod_octave(pitch);
  // TODO change these to ASSERT2
  ASSERT_LE(0.5, freq);
  ASSERT_LE(freq, 1);

  m_phase = fmodf(m_phase + freq, 1<<11);
  return shepard_steps(m_phase) / freq;
}

/** Low-frequency noise generator.

  Decay is chosen so that random frequency modulation happens a timescale
  much slower than the carrier timescale.

  Purturbation amplitude is chosen to match the variance of
  a given noise amplitude.
*/
inline void update_low_freq_noise (float & state, float freq, float amplitude)
{
  float low_freq = SYNTHESIS_NOISE_PITCH_PRECISION * freq;
  float decay = 1 - low_freq;                  // approximating exp(-low_freq)
  float actuate = amplitude * random_std();
  float perturb = sqrt(2 * low_freq) * actuate;

  state *= decay;
  state += perturb;
}

void sample_noise (StereoAudioFrame & sound, float power)
{
  for (size_t t = 0, T = sound.size; t < T; ++t) {
    sound[t] += power * random_normal_complex();
  }
}

//----( resonators )----------------------------------------------------------

inline complex NoiseResonator::operator () (
  float pitch,
  float bandwidth,
  float actuate)
{
  float freq = pitch_to_freq(pitch);
  float low_freq = SYNTHESIS_RESONATE_PITCH_PRECISION * bandwidth * freq;
  float decay = 1 - low_freq;                    // approximating exp(-low_freq)
  float perturb = sqrtf(2 * low_freq) * actuate;

  m_state *= decay * exp_2_pi_i(freq);
  m_state += perturb;
  return m_state;
}

/** Octave-wide bandpass filter

  The ideal filter would have lognormal energy envelope

    e(w) = exp(-(log(w/w0) / b)^2 / 2)

  with bandwidth b, typically 1/4.
  We approximate this by the filter with rational z-transform

           (z - 1)^2
    H(z) = ---------------------  x  const.
           (z - exp(-w0))^2
           (z - exp((-b + i) w0)
           (z - exp((-b - i) w0)

  normalized so that H(exp(i w0)) = 1.
  figure: python test/synthesis.py plot-z-trans

  We compute the filter coefficients by expanding

                   z^2 a2 + z^1 a1 + z^0 a0
    H(z) = ------------------------------------------
           z^4 a4 + z^3 a3 + z^2 a2 + z^1 a1 + z^0 a0

  where

    a2 = 1
    a1 = -2
    a0 = 1

    b3 = -2 exp(-w0) - 2 exp(-b w0) cos(w0)
    b2 = exp(-2 w0) + 4 exp(-(1+b) w0) cos(w0) + exp(-2 b w0)
    b1 = -2 exp(-(2 b + 1) w0) - 2 exp(-(2 + b) w0) cos(w0)
    b0 = exp(-2 (b+1) w0)

  The z-domain filter

    Y[z] = H(z) X[z]

  corresponds to the finite difference solution (modulo shift)

    Y[z] sum n:[0,4]. b_n z^(4-n) = X[z] sum n:[0,2] a_n z^(2-n)

  with difference equation (using b4 = 1)

    y(t) = a2 x(t)
         + a1 x(t-1)
         + a0 x(t-2)
         - b3 y(t-1)
         - b2 y(t-2)
         - b1 y(t-3)
         - b0 y(t-4)
*/
void OctavePassFilter::set_pitch (float pitch, float bandwidth)
{
  float w0 = pitch_to_freq(pitch);

  float e = exp(-w0);
  float e2 = sqr(e);
  float eb = exp(-bandwidth * w0);
  float eb2 = sqr(eb);
  float ebc = eb * cos(w0);
  y_coeff[0] = 2 * (e + ebc);               // = -b3
  y_coeff[1] = -(e2 + 4 * e * ebc + eb2);   // = -b2
  y_coeff[2] = 2 * (e * eb2 + e2 * ebc);    // = -b1
  y_coeff[3] = -(e2 * eb2);                 // = -b0
  //PRINT(y_coeff[0]);
  //PRINT(y_coeff[1]);
  //PRINT(y_coeff[2]);
  //PRINT(y_coeff[3]);

  complex z = exp(complex(0,w0));
  complex H0 = sqr(z - 1.0f)
             / ( sqr(z - expf(-w0))
               * (z - exp(complex(-bandwidth * w0, +w0)))
               * (z - exp(complex(-bandwidth * w0, -w0))) );
  float scale = 1 / abs(H0);
  x_coeff[0] = 1 * scale;
  x_coeff[1] = -2 * scale;
  x_coeff[2] = 1 * scale;
}

float OctavePassFilter::operator () (float x)
{
  float y = x_coeff[0] * x
         + x_coeff[1] * x_hist[0]
         + x_coeff[2] * x_hist[1]
         + y_coeff[0] * y_hist[0]
         + y_coeff[1] * y_hist[1]
         + y_coeff[2] * y_hist[2]
         + y_coeff[3] * y_hist[3];

  x_hist[1] = x_hist[0];
  x_hist[0] = x;
  y_hist[3] = y_hist[2];
  y_hist[2] = y_hist[1];
  y_hist[1] = y_hist[0];
  y_hist[0] = y;

  return y;
}

//----( sampling )------------------------------------------------------------

void VariableSpeedLoop::sample (State & state, StereoAudioFrame & sound_accum)
{
  const size_t I = m_loop.size;
  const size_t J = sound_accum.size;

  const complex * restrict loop = m_loop;
  complex * restrict sound = sound_accum;

  float phase = state.m_phase;
  float rate = state.m_rate;
  state.m_phase = wrap(phase + rate * J / I);

  const float amp0 = state.m_prev_scale;
  const float amp1 = state.m_curr_scale;
  state.m_prev_scale = state.m_curr_scale;

  if (rate > 1) {

    // push loop samples to sound

    ASSERTW_LE(rate, SYNTHESIS_LOOP_MAX_RATE);
    imin(rate, SYNTHESIS_LOOP_MAX_RATE);

    float i = phase * I;
    size_t i0 = static_cast<size_t>(floor(i));
    size_t i1 = static_cast<size_t>(ceil(i + rate * J));

    size_t K = i1 - i0;

    float j0 = (i0 - i) / rate;
    float dj = 1 / rate;

    for (size_t k = 0; k < K; ++k) {
      size_t i = (i0 + k) % I;
      float j = j0 + dj * k;

      float amp = (j * amp1 + (J - j) * amp0) / J;

      LinearInterpolate lin(j, J);

      complex source = amp * loop[i];
      sound[lin.i0] += lin.w0 * source;
      sound[lin.i1] += lin.w1 * source;
    }
  }

  else {

    // pull sound samples from loop

    ASSERTW_LE(SYNTHESIS_LOOP_MIN_RATE, rate);
    imax(rate, SYNTHESIS_LOOP_MIN_RATE);

    float i0 = phase * I;
    float di = rate;

    for (size_t j = 0; j < J; ++j) {
      float i = i0 + di * j;

      float amp = (j * amp1 + (J - j) * amp0) / J;

      CircularInterpolate circ(i, I);

      sound[j] += amp * ( circ.w0 * loop[circ.i0]
                        + circ.w1 * loop[circ.i1] );
    }
  }
}

inline void partial_decay_and_add (
    complex & restrict old,
    float part,
    float decay,
    complex add)
{
  old = (1 - part * (1 - decay)) * old + part * add;
}

void VariableSpeedLoop::sample (
    State & state,
    const StereoAudioFrame & sound_in,
    StereoAudioFrame & sound_out,
    float decay0,
    float decay1)
{
  const size_t I = m_loop.size;
  const size_t J = sound_out.size;

  const complex * restrict in = sound_in;
  complex * restrict out = sound_out;
  complex * restrict loop = m_loop;

  float phase = state.m_phase;
  float rate = state.m_rate;
  state.m_phase = wrap(phase + rate * J / I);

  const float amp0 = state.m_prev_scale;
  const float amp1 = state.m_curr_scale;
  state.m_prev_scale = state.m_curr_scale;

  sound_out = sound_in;

  if (rate > 1) {

    // push loop samples to sound

    ASSERTW_LE(rate, SYNTHESIS_LOOP_MAX_RATE);
    imin(rate, SYNTHESIS_LOOP_MAX_RATE);

    float i = phase * I;
    size_t i0 = static_cast<size_t>(floor(i));
    size_t i1 = static_cast<size_t>(ceil(i + rate * J));

    size_t K = i1 - i0;

    float j0 = (i0 - i) / rate;
    float dj = 1 / rate;

    for (size_t k = 0; k < K; ++k) {
      size_t i = (i0 + k) % I;
      float j = j0 + dj * k;

      float part1 = j / J;
      float part0 = 1.0f - part1;

      float amp = part0 * amp0 + part1 * amp1;
      float decay = part0 * decay0 + part1 * decay1;

      LinearInterpolate lin(j, J);

      complex old_sound = amp * loop[i];
      out[lin.i0] += lin.w0 * old_sound;
      out[lin.i1] += lin.w1 * old_sound;

      complex new_loop = ( lin.w0 * in[lin.i0]
                         + lin.w1 * in[lin.i1] ) / amp;
      loop[i] = decay * loop[i] + new_loop;
    }
  }

  else {

    // pull sound samples from loop

    ASSERTW_LE(SYNTHESIS_LOOP_MIN_RATE, rate);
    imax(rate, SYNTHESIS_LOOP_MIN_RATE);

    float i0 = phase * I;
    float di = rate;

    for (size_t j = 0; j < J; ++j) {
      float i = i0 + di * j;

      float part1 = static_cast<float>(j) / J;
      float part0 = 1.0f - part1;

      float amp = part0 * amp0 + part1 * amp1;
      float decay = part0 * decay0 + part1 * decay1;

      CircularInterpolate circ(i, I);

      complex old_sound = amp * ( circ.w0 * loop[circ.i0]
                                + circ.w1 * loop[circ.i1] );
      out[j] += old_sound;

      complex new_loop = in[j] / amp;
      partial_decay_and_add(loop[circ.i0], circ.w0, decay, new_loop);
      partial_decay_and_add(loop[circ.i1], circ.w1, decay, new_loop);
    }
  }
}

//====( voices )==============================================================

//----( glottis )---------------------------------------------------------------

/** Glottis sampling function.

  1. generate sustain/release envelope from energy
  2. generate a sawtooth tone at given pitch
  3. multiply sawtooth by envelope
*/
void Glottis::sample (const Timbre & t_final, StereoAudioFrame & sound_accum)
{
  float rate = 1.0f / sound_accum.size;
  Timbre & t = *this;
  Timbre dt = rate * (t_final - t);

  float max_power = max(power(t), power(t_final));
  float max_sustain = max(sustain(t), sustain(t_final));
  float timescale = ( DEFAULT_SYNTHESIS_RELEASE_SEC
                   + DEFAULT_SYNTHESIS_SUSTAIN_SEC * max_sustain )
                 * DEFAULT_SAMPLE_RATE;
  float decay = expf(-1 / timescale);
  float energy = m_energy;

  for (size_t i = 0, I = sound_accum.size; i < I; ++i) {

    t += dt;

    energy = max(decay * energy, max_power);
    sound_accum[i] += energy * m_actuator(pitch(t));
  }

  m_energy = energy;
}

void Glottis::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  power(* this) = 0;
  sample(t_final, sound_accum);
}

void Glottis::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  power(t_final) = 0;
  sample(t_final, sound_accum);
}

//----( buzzers )-------------------------------------------------------------

/** Buzzer sampling function.

  1. generate a base pitch modulated with low-frequency noise
  2. generate a sawtooth tone at the modulated pitch
  3. filter the sawtooth through a narrow bandpass filter
*/
void Buzzer::sample (const Timbre & t_final, StereoAudioFrame & sound_accum)
{
  float rate = 1.0f / sound_accum.size;
  Timbre & t = *this;
  Timbre dt = rate * (t_final - t);

  for (size_t i = 0; i < sound_accum.size; ++i) {

    t += dt;

    float actuate = energy(t) * m_actuator(actuate_pitch(t));
    sound_accum[i] += m_resonator(resonate_pitch(t), actuate);
  }
}

void Buzzer::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(* this) = 0;
  sample(t_final, sound_accum);
}

void Buzzer::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(t_final) = 0;
  sample(t_final, sound_accum);
}

//----( sines )---------------------------------------------------------------

void Sine::sample (const Timbre & t_final, StereoAudioFrame & sound_accum)
{
  Timbre & t = * this;

  float mean_energy = (energy(t) + energy(t_final)) / 2;

  complex rotate = exp_2_pi_i(pitch_to_freq(pitch(t_final)));

  float max_sustain = max(sustain(t), sustain(t_final));
  float timescale = ( DEFAULT_SYNTHESIS_RELEASE_SEC
                   + DEFAULT_SYNTHESIS_SUSTAIN_SEC * max_sustain )
                 * DEFAULT_SAMPLE_RATE;
  float old_energy = expf(-1 / timescale);
  float new_energy = 1 - old_energy;

  complex & restrict phase = m_phase;
  float & restrict energy = m_energy;

  for (size_t i = 0, I = sound_accum.size; i < I; ++i) {

    phase *= rotate;
    energy = old_energy * energy
           + new_energy * mean_energy;

    sound_accum[i] += energy * m_phase;
  }

  m_phase /= abs(m_phase);
  t = t_final;
}

void Sine::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(* this) = 0;
  sample(t_final, sound_accum);
}

void Sine::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(t_final) = 0;
  sustain(t_final) *= 0.5f;
  sample(t_final, sound_accum);
}

//----( exp-sines )-----------------------------------------------------------

void ExpSine::sample (const Timbre & t_final, StereoAudioFrame & sound_accum)
{
  Timbre & t = * this;

  float mean_energy = (energy(t) + energy(t_final)) / 2;

  float freq = pitch_to_freq(pitch(t_final));
  float equalize = 1 / sqrt(freq);
  complex rotate = exp_2_pi_i(freq);

  float max_sustain = max(sustain(t), sustain(t_final));
  float timescale = ( DEFAULT_SYNTHESIS_RELEASE_SEC
                   + DEFAULT_SYNTHESIS_SUSTAIN_SEC * max_sustain )
                 * DEFAULT_SAMPLE_RATE;
  float old_energy = expf(-1 / timescale);
  float new_energy = 1 - old_energy;

  float roughen = max(0.0f, rough(t_final) + GRID_SIZE_Y / 2.0f);

  complex & restrict phase = m_phase;
  float & restrict energy = m_energy;

  for (size_t i = 0, I = sound_accum.size; i < I; ++i) {

    phase *= rotate;
    energy = old_energy * energy
           + new_energy * mean_energy;

    sound_accum[i] += equalize * energy * exp((phase - 1.0f) * roughen);
  }

  m_phase /= abs(m_phase);
  t = t_final;
}

void ExpSine::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(* this) = 0;
  sample(t_final, sound_accum);
}

void ExpSine::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(t_final) = 0;
  sustain(t_final) *= 0.5f;
  sample(t_final, sound_accum);
}

//----( bells )---------------------------------------------------------------

/** Bell sampling function.

  1. generate a base pitch modulated with low-frequency noise
  2. generate a sawtooth tone at the modulated pitch
  3. filter the sawtooth through a narrow bandpass filter
*/
void Bell::sample (const Timbre & t_final, StereoAudioFrame & sound_accum)
{
  float rate = 1.0f / sound_accum.size;
  Timbre & t = *this;
  Timbre dt = rate * (t_final - t);

  float max_power = max(power(t), power(t_final));
  float max_sustain = max(sustain(t), sustain(t_final));
  float timescale = ( DEFAULT_SYNTHESIS_RELEASE_SEC
                   + DEFAULT_SYNTHESIS_SUSTAIN_SEC * max_sustain )
                 * DEFAULT_SAMPLE_RATE;
  float decay = expf(-1 / timescale);

  for (size_t i = 0; i < sound_accum.size; ++i) {

    t += dt;

    m_energy = decay * m_energy + (1 - decay) * max_power;
    float actuate = m_energy * m_actuator(actuate_pitch(t));
    sound_accum[i] += m_resonator(resonate_pitch(t), actuate);
  }
}

void Bell::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  power(* this) = 0;
  sample(t_final, sound_accum);
}

void Bell::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  power(t_final) = 0;
  sample(t_final, sound_accum);
}

//----( pipes )---------------------------------------------------------------

/** Pipe sampling function.

  1. generate a white noise
  2. filter the noise through a variable-width bandpass filter
*/
void Pipe::sample (const Timbre & t_final, StereoAudioFrame & sound_accum)
{
  float rate = 1.0f / sound_accum.size;
  Timbre & t = *this;
  Timbre dt = rate * (t_final - t);

  for (size_t i = 0; i < sound_accum.size; ++i) {

    t += dt;

    float actuate = energy(t) * random_std();
    sound_accum[i] += m_resonator(pitch(t), bandwidth(t), actuate);
  }
}

void Pipe::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(* this) = 0;
  sample(t_final, sound_accum);
}

void Pipe::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(t_final) = 0;
  sample(t_final, sound_accum);
}

//----( vibes )---------------------------------------------------------------

/** Vibe sampling function.
*/
void Vibe::sample (const Timbre & t_final, StereoAudioFrame & sound_accum)
{
  float rate = 1.0f / sound_accum.size;
  Timbre & t = *this;
  Timbre dt = rate * (t_final - t);

  for (size_t i = 0; i < sound_accum.size; ++i) {

    t += dt;

    float timescale = SYNTHESIS_VIBE_SUSTAIN_TIMESCALE * sustain(t)
                   + SYNTHESIS_VIBE_RELEASE_TIMESCALE;
    m_energy *= 1 - 1 / timescale;
    m_energy += rate * energy(t);

    float freq = pitch_to_freq(pitch(t));
    m_phase += roundi(UINT_MAX * freq);
    complex phase = exp_2_pi_i(m_phase * (1.0 / UINT_MAX));

    sound_accum[i] += m_energy * phase;
  }
}

void Vibe::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(* this) = 0;
  sample(t_final, sound_accum);
}

void Vibe::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(t_final) = 0;
  sample(t_final, sound_accum);
}

//----( gongs )---------------------------------------------------------------

/** Gong sampling function.
*/
void Gong::sample (const Timbre & t_final, StereoAudioFrame & sound_accum)
{
  float rate = 1.0f / sound_accum.size;
  Timbre & t = *this;
  Timbre dt = rate * (t_final - t);

  for (size_t i = 0; i < sound_accum.size; ++i) {

    t += dt;

    float am = 1 / ( SYNTHESIS_GONG_SUSTAIN_TIMESCALE
                  + SYNTHESIS_GONG_ATTACK_TIMESCALE * amp_mod(t) );
    m_energy = (1 - am) * m_energy + am * energy(t);

    float freq = pitch_to_freq(pitch(t));
    m_phase += roundi(UINT_MAX * freq);
    float phase = m_phase / INT_MAX;
    float mod = phase * (1 - 2 * sqr(phase));

    float fm = freq_mod(t);
    complex actuate = exp_2_pi_i(phase + fm * mod);

    sound_accum[i] += m_energy * actuate;
  }
}

void Gong::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(* this) = 0;
  sample(t_final, sound_accum);
}

void Gong::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(t_final) = 0;
  sample(t_final, sound_accum);
}

//----( formants )------------------------------------------------------------

/** Formant sampling function.

  1. generate a base pitch modulated with low-frequency noise
  2. generate a sawtooth tone at the modulated pitch
  3. filter the sawtooth through multiple parallel narrow bandpass filters
*/
void Formant::sample (const Timbre & t_final, StereoAudioFrame & sound_accum)
{
  float rate = 1.0f / sound_accum.size;
  Timbre & t = *this;
  Timbre dt = rate * (t_final - t);

  for (size_t i = 0; i < sound_accum.size; ++i) {

    t += dt;

    float actuate = energy(t) * m_actuator(actuate_pitch(t));
    sound_accum[i] += m_resonator1(resonate1(t), actuate)
                    + m_resonator2(resonate2(t), actuate);
;
  }
}

void Formant::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(* this) = 0;
  sample(t_final, sound_accum);
}

void Formant::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(t_final) = 0;
  sample(t_final, sound_accum);
}

//----( shepard tones )-------------------------------------------------------

/** Shepard sampling function.
*/
void Shepard::sample (const Timbre & t_final, StereoAudioFrame & sound_accum)
{
  float rate = 1.0f / sound_accum.size;
  Timbre & t = *this;
  Timbre dt = rate * (t_final - t);

  float mean_pitch = (coarse(t) + coarse(t_final)) / 2;
  float mean_formant = (formant(t) + formant(t_final)) / 2;
  m_filter.set_pitch(mean_pitch);

  for (size_t i = 0; i < sound_accum.size; ++i) {

    t += dt;

    float actuate = m_actuator(fine(t));
    //float filtered = m_filter(actuate); // DEBUG
    float filtered = actuate;
    sound_accum[i] += m_resonator(mean_formant, filtered);
  }
}

void Shepard::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(* this) = 0;
  sample(t_final, sound_accum);
}

void Shepard::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(t_final) = 0;
  sample(t_final, sound_accum);
}

//----( shepard vibes )-------------------------------------------------------

float blend (float t) { return sqr(1 - sqr(t)); }

void ShepardVibe::sample (
    const Timbre & t_final,
    StereoAudioFrame & sound_accum)
{
  Timbre & t = * this;

  float energy_value = energy(t_final);
  float coarse_value = coarse(t_final);
  float fine_value = fine(t_final);
  float sustain_value = max(sustain(t), sustain(t_final));

  float timescale = SYNTHESIS_VIBE_RELEASE_TIMESCALE
                 + SYNTHESIS_VIBE_SUSTAIN_TIMESCALE * sustain_value;
  float new_part = 1.0f / timescale;
  float old_part = 1.0f - new_part;

  float bandwidth = 0.5f * num_octaves;
  float base = coarse_value - bandwidth;
  complex rotate[num_tones];
  float weight[num_tones];
  const float major_fifth = log(1.5f) / log(2);
  for (size_t o = 0; o < num_tones; ++o) {

    float interval = o % num_octaves;
    if (o > num_octaves) interval += major_fifth;
    float pitch = wrap(fine_value + interval, num_octaves, base);
    float offset = (pitch - coarse_value) / bandwidth;

    //PRINT2(offset, weight[o]);
    //ASSERT_LE(-1, offset);
    //ASSERT_LE(offset, 1);

    weight[o] = blend(offset);
    if (o > num_octaves) weight[o] *= 0.5f;
    rotate[o] = exp_2_pi_i(pitch_to_freq(pitch));
  }

  for (size_t i = 0; i < sound_accum.size; ++i) {

    m_energy = new_part * energy_value
             + old_part * m_energy;

    for (size_t o = 0; o < num_tones; ++o) {
      m_phase[o] *= rotate[o];
      sound_accum[i] += m_energy * weight[o] * m_phase[o];
    }
  }

  for (size_t o = 0; o < num_tones; ++o) {
    m_phase[o] /= abs(m_phase[o]);
  }
}

void ShepardVibe::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(* this) = 0;
  sample(t_final, sound_accum);
}

void ShepardVibe::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(t_final) = 0;
  sample(t_final, sound_accum);
}

//----( string )--------------------------------------------------------------

void String::sample (const Timbre & t_final, StereoAudioFrame & sound_accum)
{
  Timbre & t = * this;

  float energy0 = SYNTHESIS_STRING_SHARPNESS * energy(t);
  float energy1 = SYNTHESIS_STRING_SHARPNESS * energy(t_final);
  float denergy = energy1 - energy0;

  float width0 = bound_to(0.0f, SYNTHESIS_STRING_MAX_WIDTH, width(t));
  float width1 = bound_to(0.0f, SYNTHESIS_STRING_MAX_WIDTH, width(t_final));
  float dwidth = width1 - width0;

  float mean_pitch = 0.5f * (pitch(t) + pitch(t_final));

  const complex trans = exp_2_pi_i(pitch_to_freq(mean_pitch));
  complex phase = m_phase;

  const size_t T = sound_accum.size;
  complex * restrict sound = sound_accum;

  for (size_t t = 0; t < T; ++t) {
    float dt = (t + 0.5f) / T;

    float energy = energy0 + denergy * dt;
    float width = width0 + dwidth * dt;

    phase *= trans;
    complex overtones = energy * phase / (1.0f - width * phase);
    sound[t] += overtones / (1.0f + abs(overtones)); // soft-clipped
  }

  m_phase = phase / abs(phase);
  t = t_final;
}

void String::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(* this) = 0;
  sample(t_final, sound_accum);
}

void String::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(t_final) = 0;
  sample(t_final, sound_accum);
}

//----( plucked )-------------------------------------------------------------

void Plucked::sample (const Timbre & t_final, StereoAudioFrame & sound_accum)
{
  Timbre & t = * this;

  float max_sustain = max(sustain(t), sustain(t_final));
  float timescale = SYNTHESIS_PLUCKED_RELEASE_TIMESCALE
                 + SYNTHESIS_PLUCKED_SUSTAIN_TIMESCALE * max_sustain;
  float old_part = exp(-timescale);

  float max_energy = max(energy(t), energy(t_final));
  float energy0 = m_energy;
  float attack0 = m_attack;

  float energy1 = old_part * energy0;
  float attack1 = old_part * attack0;
  float power = max(0.0f, max_energy - energy1);
  m_energy = energy1 = max(energy1, max_energy);
  m_attack = attack1 += (SYNTHESIS_PLUCKED_ATTACK - attack1)
                      * (power / max(1e-8f, energy1));

  float mean_pitch = 0.5f * (pitch(t) + pitch(t_final));
  const complex trans = exp_2_pi_i(pitch_to_freq(mean_pitch));
  complex phase = m_phase;

  const size_t T = sound_accum.size;
  complex * restrict sound = sound_accum;

  float denergy = (energy1 - energy0) / T;
  float dattack = (attack1 - attack0) / T;

  for (size_t t = 0; t < T; ++t) {
    float dt = t + 0.5f;

    float energy = energy0 + denergy * dt;
    float attack = attack0 + dattack * dt;

    phase *= trans;
    complex overtones = energy * phase / (1.0f - attack * phase);
    sound[t] += overtones / (1.0f + abs(overtones)); // soft-clipped
  }

  m_phase = phase / abs(phase);
  t = t_final;
}

void Plucked::fadein (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  m_energy = 0;
  m_attack = 0;
  sample(t_final, sound_accum);
}

void Plucked::fadeout (StereoAudioFrame & sound_accum)
{
  Timbre t_final = * this;
  energy(t_final) = 0;
  sustain(t_final) = 0;
  sample(t_final, sound_accum);
}

//----( vocoder )-------------------------------------------------------------

void VocoPoint::sample (Vocoder::Timbre & sound_accum)
{
  static const size_t P = SYNTHESIS_VOCODER_SIZE;

  float pitch_01 = pitch(* this);
  float pitch_P = pitch_01 * P - 0.5f;

  LinearInterpolate(pitch_P, P).iadd(sound_accum, mass(* this));
}

void VocoBlob::sample (Vocoder::Timbre & sound_accum)
{
  static const size_t P = SYNTHESIS_VOCODER_SIZE;
  static const float nat_to_P
    = P / log(SYNTHESIS_VOCODER_MAX_FREQ_HZ / SYNTHESIS_VOCODER_MIN_FREQ_HZ);

  float pitch_01 = pitch(* this);
  float pitch_P = pitch_01 * P - 0.5f;

  float width_nat = width(* this);
  float width_P = width_nat * nat_to_P;

  if (width_P < 0.5) {

    // two-point interpolation
    LinearInterpolate(pitch_P, P).iadd(sound_accum, mass(* this));

  } else {

    // kernal is lognormal in pitch, normal in freq

    float scale = mass(* this) / (sqrt(2 * M_PI) * width_P);

    size_t p0 = roundu(pitch_P);
    for (size_t p = p0;; --p) {
      float dfreq = expf((p - pitch_P) / nat_to_P) - 1;
      float m = scale * expf(-0.5f * sqr(dfreq / width_nat));
      sound_accum[p] += m;
      if ((m < 1e-8f) or (p == 0)) break;
    }
    for (size_t p = p0 + 1; p < P; ++p) {
      float dfreq = expf((p - pitch_P) / nat_to_P) - 1;
      float m = scale * expf(-0.5f * sqr(dfreq / width_nat));
      sound_accum[p] += m;
      if (m < 1e-8f) break;
    }
  }
}

//----( beater )--------------------------------------------------------------

Beater::Beater (bool coalesce, float blur_factor)
  : Synchronized::LoopBank(
      Synchronized::Bank(
          SYNTHESIS_BEATER_SIZE,
          SYNTHESIS_BEATER_MIN_FREQ, // top
          SYNTHESIS_BEATER_MAX_FREQ, // bottom
          SYNTHESIS_BEATER_ACUITY),
      coalesce,
      1.0f / DEFAULT_AUDIO_FRAMERATE),

    m_timestep(1.0f / DEFAULT_AUDIO_FRAMERATE),
    m_decay(expf(-m_timestep / SYNTHESIS_BEATER_TIMESCALE)),
    m_blur_radius(roundu(blur_factor * tone_size() / 2))
{
  PRINT3(m_blur_radius, size, num_tones());
  ASSERT_LT(m_blur_radius, size);
}

void Beater::sample (Timbre & amplitude)
{
  if (m_blur_radius) {

    // blurring allows nearby fingers to access coalesced amplitude
    Image::square_blur_1d(size, m_blur_radius, mass_now, amplitude);
    Image::linear_blur_1d(size, m_blur_radius, amplitude, m_temp);

  } else {
    amplitude = mass_now;
  }
}

//====( coupled voices )======================================================

namespace Coupled
{

const Synchronized::ScaledBeatFun g_pitch_param(COUPLED_PITCH_ACUITY);
const Synchronized::ScaledBeatFun g_tempo_param(COUPLED_TEMPO_ACUITY);
const Synchronized::ScaledBeatFun g_synco_param(COUPLED_SYNCO_ACUITY);

void SyncoString::set_timbre (const Finger & polar_finger)
{
  m_mass = polar_finger.get_z();
  float radius = polar_finger.get_x();
  float angle = polar_finger.get_y();

  m_tempo.freq = tempo_to_freq_synco((radius + 0.5f) / 2);
  m_tempo.offset = exp_2_pi_i(angle);

  // angle wraps around twice before returning to home
  float half_octave = radius * BUCKET_ORGAN_NUM_OCTAVES / 2;
  float half_angle1 = angle / 2;
  float half_angle2 = (1 + angle) / 2;

  float gap1 = wrap(half_angle1 - half_octave);
  float gap2 = wrap(half_angle2 - half_octave);

  float half_pitch1 = gap1 + half_octave;
  float half_pitch2 = gap2 + half_octave;

  float new_part1 = sqr(sinf(M_PI * (half_angle1 - half_octave)));
  float new_part2 = 1 - new_part1;
  float new_pitch1 = BUCKET_ORGAN_MIN_PITCH + half_pitch1 * 2;
  float new_pitch2 = BUCKET_ORGAN_MIN_PITCH + half_pitch2 * 2;

  float & pitch1 = String::pitch(m_timbre1);
  float & pitch2 = String::pitch(m_timbre2);

  // match old to new to minimize weighted frequency change
  float parity
    = sqr(pitch1 - new_pitch2) * (m_part1 + new_part2)
    + sqr(pitch2 - new_pitch1) * (m_part2 + new_part1)
    - sqr(pitch1 - new_pitch1) * (m_part1 + new_part1)
    - sqr(pitch2 - new_pitch2) * (m_part2 + new_part2);

  if (parity > 0) {

    m_part1 = new_part1;
    m_part2 = new_part2;
    m_width1 = 1 - gap1;
    m_width2 = 1 - gap2;
    pitch1 = new_pitch1;
    pitch2 = new_pitch2;

  } else {

    m_part1 = new_part2;
    m_part2 = new_part1;
    m_width1 = 1 - gap2;
    m_width2 = 1 - gap1;
    pitch1 = new_pitch2;
    pitch2 = new_pitch1;
  }
}

void SyncoString::sample (complex force, Sound & sound)
{
  float time = wrap(arg(m_tempo.sample(force)) / (2 * M_PI));

  float amplitude = m_mass * 8.0f * exp(-time);
  String::energy(m_timbre1) = m_part1 * amplitude;
  String::energy(m_timbre2) = m_part2 * amplitude;

  float width = exp(-time);
  String::width(m_timbre1) = m_width1 * width;
  String::width(m_timbre2) = m_width2 * width;

  m_pitch1.sample(m_timbre1, sound);
  m_pitch2.sample(m_timbre2, sound);
}

} // namespace Coupled

//====( effects )=============================================================

//----( loop resonator )------------------------------------------------------

LoopResonator::LoopResonator (size_t size, float timescale)
  : m_decay(exp(-1 / (size * timescale))),

    m_data(4 * size),
    m_fwd(size, m_data.begin()),
    m_bwd(size, m_fwd.end()),
    m_temp1(size, m_bwd.end()),
    m_temp2(size, m_temp1.end())
{
  ASSERT_DIVIDES(4, size);
  ASSERT_LE(2, timescale);

  m_data.zero();
}

LoopResonator::~LoopResonator ()
{
  PRINT3(min(m_fwd), rms(m_fwd), max(m_fwd));
  PRINT3(min(m_bwd), rms(m_bwd), max(m_bwd));
}

void LoopResonator::sample (
    const Vector<float> & impedance_in,
    StereoAudioFrame & sound_io)
{
  ASSERT_SIZE(impedance_in, size());
  ASSERT_DIVIDES(2, sound_io.size);

  const size_t I = size();
  const size_t T = sound_io.size;

  const float * restrict impedance = impedance_in;
  float * restrict fwd = m_fwd;
  float * restrict bwd = m_bwd;
  float * restrict new_fwd = m_temp1;
  float * restrict new_bwd = m_temp2;

  const float decay = m_decay;

  for (size_t t = 0; t < T; ++t) {

    complex z = sound_io[t];
    fwd[0] += z.real();
    bwd[0] += z.imag();

    for (size_t i = 0; i < I - 1; ++i) {
      size_t i0 = i;
      size_t i1 = i + 1;

      float imp0 = impedance[i0];
      float imp1 = impedance[i1];
      float mix = (imp1 - imp0) / (imp1 + imp0);

      new_fwd[i1] = (decay + mix) * fwd[i0] - mix * bwd[i1];
      new_bwd[i0] = (decay - mix) * bwd[i1] + mix * fwd[i0];
    }

    {
      size_t i0 = I - 1;
      size_t i1 = 0;

      float imp0 = impedance[i0];
      float imp1 = impedance[i1];
      float mix = (imp1 - imp0) / (imp1 + imp0);

      // the negative sign restricts to odd overtones
      new_fwd[i1] = -(decay + mix) * fwd[i0] - mix * bwd[i1];
      new_bwd[i0] = -(decay - mix) * bwd[i1] + mix * fwd[i0];
    }

    std::swap(fwd, new_fwd);
    std::swap(bwd, new_bwd);

    sound_io[t] = complex(fwd[0], bwd[0]);
  }
}

//----( reverb )--------------------------------------------------------------

EchoBox::EchoBox (
    size_t size,
    float decay,
    float lowpass)

  : m_fade_old(decay),
    m_fade_new(1 - decay),
    m_lowpass_old(lowpass),
    m_lowpass_new(1 - lowpass),

    m_memory(size),
    m_lowpass(0.0f),
    m_position(0)
{
  ASSERT_LT(0, decay);
  ASSERT_LT(decay, 1);
  ASSERT_LT(0, lowpass);
  ASSERT_LT(lowpass, 1);

  m_memory.zero();
}

void EchoBox::transform (StereoAudioFrame & sound_io)
{
  for (size_t i = 0; i < sound_io.size; ++i) {

    complex & memory = m_memory[m_position];
    complex & sound = sound_io[i];

    complex old_sound = sound;

    m_lowpass = m_lowpass_old * m_lowpass
              + m_lowpass_new * memory;

    sound = m_fade_old * old_sound
          + m_fade_new * m_lowpass;

    memory = m_fade_old * m_lowpass
           + m_fade_new * old_sound;

    m_position = (m_position + 1) % m_memory.size;
  }
}

size_t Reverberator::box_size(size_t base_size, size_t number)
{
  return roundu(base_size * powf(DEFAULT_REVERB_INTERVAL, number));
}

Reverberator::Reverberator (size_t base_size)
  : m_echo0(box_size(base_size, 0)),
    m_echo1(box_size(base_size, 1)),
    m_echo2(box_size(base_size, 2)),
    m_echo3(box_size(base_size, 3)),
    m_echo4(box_size(base_size, 4)),
    m_echo5(box_size(base_size, 5)),
    m_echo6(box_size(base_size, 6)),
    m_echo7(box_size(base_size, 7))
{}

void Reverberator::transform (StereoAudioFrame & sound)
{
  m_echo0.transform(sound);
  m_echo1.transform(sound);
  m_echo2.transform(sound);
  m_echo3.transform(sound);
  m_echo4.transform(sound);
  m_echo5.transform(sound);
  m_echo6.transform(sound);
  m_echo7.transform(sound);
}

} // namespace Synthesis

