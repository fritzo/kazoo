
#ifndef KAZOO_SYNCHRONY_H
#define KAZOO_SYNCHRONY_H

#include "common.h"
#include "filters.h"
#include "vectors.h"
#include "bounded.h"
#include <utility>

#define DEFAULT_SYNCHRONY_STRENGTH      (2.0f)

#define SYNCHRONY_MIN_MASS              (1e-2f)

namespace Synchronized
{

//----( oscillator tools )----------------------------------------------------

/** Standard Beat Function

  (see test/synchronize.py standard-beat)

  Starting with the beat function (letting a = pi/acuity)

    b(theta) = max(0, cos(theta) - cos(a))

  We enforce zero mean by subtracting by the mean value

                   1
    E[b(theta)] = ---- int theta:[-a,a]. cos(theta) - cos(a)
                  2 pi

                  sin(a) - a cos(a)
                = -----------------
                         pi

  Similarly, we scale variance to be 1.
*/
struct StdBeatFun
{
  float acuity;
  float beat_floor;
  float beat_shift;
  float beat_scale;

  StdBeatFun ()
    : acuity(NAN),
      beat_floor(NAN),
      beat_shift(NAN),
      beat_scale(NAN)
  {}
  StdBeatFun (float a)
    : acuity(a)
  {
    init(acuity, beat_floor, beat_shift, beat_scale);
  }

  static void init (float acuity, float & floor, float & shift, float & scale)
  {
    ASSERT_LE(1, acuity);

    float a = M_PI / acuity;
    float cos_a = cosf(a);
    float sin_a = sinf(a);

    // let f(theta) = max(0, cos(theta) - cos(a))
    float Ef = (sin_a - a * cos_a) / M_PI;
    float Ef2 = (a - 3 * sin_a * cos_a + 2 * a * sqr(cos_a)) / (2 * M_PI);
    float Vf = Ef2 - sqr(Ef);

    floor = cos_a;
    shift = -Ef;
    scale = 1 / sqrtf(Vf);
  }

  float value (float x) const
  {
    return beat_scale * (beat_shift + max(0.0f, x - beat_floor));
  }
  float deriv (float x, float y) const
  {
    return -beat_scale * (x > beat_floor ? y : 0.0f);
  }
  float max_deriv () const
  {
    float a = M_PI / max(2.0f, acuity);
    return beat_scale * sinf(a);
  }
};

struct ScaledBeatFun : public StdBeatFun
{
  float strength;

  ScaledBeatFun ()
    : StdBeatFun(),
      strength(NAN)
  {}
  ScaledBeatFun (float a, float s = DEFAULT_SYNCHRONY_STRENGTH)
    : StdBeatFun(a),
      strength(s)
  {
    ASSERT_LT(0, strength);

    beat_scale *= sqrtf(strength);
  }

  // WARNING ScaledBeatFun.value(-) is not zero-mean
  float value (float x) const { return beat_scale * max(0.0f, x - beat_floor); }
  float deriv (float x, float y) const
  {
    return -beat_scale * (x > 0 ? y : 0.0f);
  }
};

struct UnitBeatFun : public StdBeatFun
{
  UnitBeatFun ()
    : StdBeatFun()
  {}
  UnitBeatFun (float a, float strength = DEFAULT_SYNCHRONY_STRENGTH)
    : StdBeatFun(a)
  {
    ASSERT_LT(0, strength);

    beat_scale = sqrtf(strength) / (1 - beat_floor);
  }

  float value (float x) const
  {
    return beat_scale * max(0.0f, x - beat_floor);
  }
};

struct GeomBeatFun
{
  float acuity;
  float strength;
  float p;

  GeomBeatFun () : acuity(NAN), strength(NAN), p(NAN) {}
  GeomBeatFun (float a, float s = 3.0)
    : acuity(a), strength(s), p(1-1/a)
  {}

  complex value (complex z) const
  {
    return (1 - p) / (1.0f - p * z) * z;
  }
  complex deriv (complex z) const
  {
    complex dz(-z.imag(), z.real());
    return (1 - p + p * z) * sqr((1 - p) / (1.0f - p * z)) * dz;
  }
  float max_deriv () const { return acuity; }
};

//----( individual oscillators )----------------------------------------------

/** Synchronized oscillators with rational frequency locking.
  see test/synchronize.py

  (N1) To accurately scale coupling strength for 2-10 oscillators,
    we poll using a Bessel-corrected average mass.
    see notes/music.text (2011:04:14) (N1)
    see http://en.wikipedia.org/wiki/Bessel%27s_correction

  (N2) To lock pitch rather than frequency,
    we force oscillators at a rate proportional to their frequency.
    see test/synchronize.py staircases
    see test/synchronize.py stairs-lag
*/

class Poll
{
  complex m_force;
  float m_mass;
  float m_mass2;

public:

  Poll () : m_force(0,0), m_mass(0), m_mass2(0) {}
  Poll (complex f, float m) : m_force(f), m_mass(m), m_mass2(m * m) {}

  void operator+= (const Poll & other)
  {
    m_force += other.m_force;
    m_mass += other.m_mass;
    m_mass2 += other.m_mass2;
  }

  complex mean () const
  {
    float M = max(SYNCHRONY_MIN_MASS, m_mass);
    float M2 = max(sqr(SYNCHRONY_MIN_MASS), m_mass2);
    float Bessels_correction = max(0.0f, 1 - M2 / sqr(M));
    return m_force * Bessels_correction / M;
  }
};

struct Phasor
{
  typedef ScaledBeatFun BeatFun;

  complex phase;
  float mass;
  float freq;
  float beat;
  float bend;

  BeatFun beat_fun;

  Phasor (const BeatFun & f)
    : phase(1,0),
      mass(0),
      freq(0),
      beat(0),
      bend(0),
      beat_fun(f)
  {}

  Poll poll () const { return Poll(mass * beat * phase, mass); }

  complex sample (complex force);
};

struct Syncopator
{
  typedef ScaledBeatFun BeatFun;

  complex phase;
  complex offset;
  float mass;
  float freq;
  float beat;
  float bend;

  ScaledBeatFun beat_fun;

  Syncopator (const BeatFun & f)
    : phase(1,0),
      offset(1,0),
      mass(0),
      freq(0),
      beat(0),
      bend(0),
      beat_fun(f)
  {}

  Poll poll () const { return Poll(mass * beat * offset * phase, mass); }

  complex sample (complex force);
};

struct Shepard4
{
  typedef ScaledBeatFun BeatFun;

  complex phase;
  complex force;
  float mass;
  float freq;
  float octave;
  float bend;

  BeatFun beat_fun;

  Shepard4 (const BeatFun & f)
    : phase(1,0),
      force(0,0),
      mass(0),
      freq(0),
      octave(1.5f),
      bend(0),
      beat_fun(f)
  {}

  Poll poll () const { return Poll(mass * force, mass); }

  complex sample (complex force_in);
};

struct Shepard7
{
  typedef ScaledBeatFun BeatFun;

  complex phase;
  complex force;
  float mass;
  float freq;
  float octave;
  float bend;

  BeatFun beat_fun;

  Shepard7 (const BeatFun & f)
    : phase(1,0),
      force(0,0),
      mass(0),
      freq(0),
      octave(3.0f),
      bend(0),
      beat_fun(f)
  {}

  Poll poll () const { return Poll(mass * force, mass); }

  complex sample (complex force_in);
};

struct Geometric
{
  typedef GeomBeatFun BeatFun;

  complex phase;
  complex beat;
  complex dbeat;
  float mass;
  float freq;
  float bend;

  BeatFun beat_fun;

  Geometric (const BeatFun & f)
    : phase(1,0),
      beat(f.value(phase)),
      dbeat(f.deriv(phase)),
      mass(0),
      freq(0),
      bend(0),
      beat_fun(f)
  {}

  Poll poll () const { return Poll(mass * beat, mass); }

  complex sample (complex force);
};

struct Boltz
{
  typedef ScaledBeatFun BeatFun;

  complex phase;
  float mass;
  float freq;
  float beat;
  float dbeat;
  float bend;

  StdBeatFun beat_fun;

  Boltz (const StdBeatFun & f)
    : phase(1,0),
      mass(0),
      freq(0),
      beat(0),
      dbeat(0),
      bend(0),
      beat_fun(f)
  {}

  Poll poll () const { return Poll(complex(mass * beat, 0.0f), mass); }

  complex sample (complex force);
};

struct Phasor2
{
  typedef UnitBeatFun BeatFun;

  complex phase;
  complex beat;
  complex dbeat;
  float mass;
  float freq;
  float bend;

  BeatFun beat_fun;

  Phasor2 (const BeatFun & f)
    : phase(1,0),
      beat(0,0),
      dbeat(0,0),
      mass(0),
      freq(0),
      bend(0),
      beat_fun(f)
  {}

  Poll poll () const { return Poll(mass * beat, mass); }

  complex sample (complex mean_beat);
};

//----( arnold tongue testing )----

template<class Oscillator>
struct ArnoldTongues : public Rectangle
{
  Vector<float> bend;

  ArnoldTongues (Rectangle shape) : Rectangle(shape), bend(size()) {}

  void tongues (
      float acuity,
      float max_strength = 2,
      float pitch_octaves = 1,
      size_t num_periods = 1000);

  void keys (
      float min_acuity = 1,
      float max_acuity = 12,
      float pitch_octaves = 1,
      size_t num_periods = 1000);

  void islands (
      float acuity,
      float strength_scale = 1,
      float pitch_octaves = 1,
      size_t num_periods = 1000);

protected:

  static float mean_bend (
      const typename Oscillator::BeatFun & beat_fun,
      float freq1,
      float freq2,
      size_t num_steps);

  static float rms_bend (
      const typename Oscillator::BeatFun & beat_fun,
      float freq1,
      float freq2,
      float freq3,
      size_t num_steps);
};

//============================================================================

//----( abstract bank )-------------------------------------------------------

class Bank
{
public:

  size_t size;
  float freq0;
  float freq1;
  float acuity;
  float strength;
  bool debug;

  Bank () {}
  Bank (
      size_t n,
      float f0,
      float f1,
      float a = 0,
      float s = DEFAULT_SYNCHRONY_STRENGTH)

    : size(n),
      freq0(f0),
      freq1(f1),
      acuity(a),
      strength(s),
      debug(false)
  {}
  virtual ~Bank () {}

  //----( properties )----

  float max_freq () const { return max(freq0, freq1); }
  float min_freq () const { return min(freq0, freq1); }
  float num_octaves () const { return fabs(log(freq1 / freq0)) / log(2); }
  float num_tones () const { return num_octaves() * acuity; }
  float tone_size () const { return size / num_tones(); }

  double freq_at (size_t i) const
  {
    return freq0 * pow(freq1/freq0, (i + 0.5) / size);
  }
  size_t phase_acuity_at (size_t i) const
  {
    return roundu(acuity * max_freq() / freq_at(i));
  }
  size_t max_phase_acuity () const
  {
    return roundu(acuity * max_freq() / min_freq());
  }

protected:

  //----( initialization )----

  void init_transform (
      float * restrict frequency) const;

  void init_transform (
      float * restrict trans_real,
      float * restrict trans_imag) const;

  void init_decay (
      float * restrict decay,
      size_t order = 1) const;

  void init_decay_transform (
      float * restrict trans_real,
      float * restrict trans_imag,
      float * restrict rescale,
      size_t order = 1,
      float min_timescale = 0) const;

  void retune (
      const float * restrict freq_bend,
      const float * restrict mass_in,
      float * restrict mass_out,
      float rate = 1) const;

  void retune (
      const float * restrict freq_bend,
      const float * restrict mass_in,
      float * restrict mass_out,
      size_t vector_size,
      float rate = 1) const;

public:

  void retune (
      const Vector<float> & freq_bend,
      const Vector<float> & mass_in,
      Vector<float> & mass_out,
      float rate = 1) const;
};

//----( simple bank )---------------------------------------------------------

class SimpleBank : public Bank
{
protected:

  Vector<float> m_frequency;
  Vector<float> m_phase_real;
  Vector<float> m_phase_imag;

public:

  SimpleBank (Bank param);
  virtual ~SimpleBank ();

  void set_frequency (const Vector<float> & frequency)
  {
    m_frequency = frequency;
  }

  void sample_accum (
      const Vector<float> & amplitude0_in,
      const Vector<float> & damplitude_in,
      Vector<complex> & accum_out);
};

//----( fourier bank )--------------------------------------------------------

// TODO implement Goertzel algorithm for lower-cost transforms
// http://en.wikipedia.org/wiki/Goertzel_algorithm

class FourierBank : public Bank
{
  Vector<float> m_rescale;
  Vector<float> m_trans_real;
  Vector<float> m_trans_imag;
  Vector<float> m_phase_real;
  Vector<float> m_phase_imag;

public:

  FourierBank (Bank param, float min_timescale = 0);
  virtual ~FourierBank ();

  void sample (
      const Vector<float> & time_in,
      Vector<float> & freq_out);

  void sample (
      const Vector<complex> & time_in,
      Vector<float> & freq_out);

  void resonate (Vector<complex> & time_accum);
};

//----( fourier bank 2 )------------------------------------------------------

class FourierBank2 : public Bank
{
protected:

  Vector<float> m_rescale;
  Vector<float> m_trans_real;
  Vector<float> m_trans_imag;

private:

  Vector<float> m_pos_real;
  Vector<float> m_pos_imag;
  Vector<float> m_vel_real;
  Vector<float> m_vel_imag;

public:

  FourierBank2 (Bank param, float min_timescale = 0);
  virtual ~FourierBank2 ();

  void sample (
      const Vector<float> & time_in,
      Vector<float> & freq_out);

  void sample (
      const Vector<complex> & time_in,
      Vector<float> & freq_out);
};

//----( phasor bank )---------------------------------------------------------

class PhasorBank : public Bank
{
  const ScaledBeatFun m_beat_fun;

  Vector<float> m_data;

protected:

  Vector<float> m_mass;
  Vector<float> m_frequency;
  Vector<float> m_phase_real;
  Vector<float> m_phase_imag;
  Vector<float> m_beat;
  Vector<float> m_meso_bend;
  Vector<float> m_slow_bend;
  mutable Vector<float> m_temp;

public:

  PhasorBank (Bank param);
  virtual ~PhasorBank ();

  void sample_accum (
      const Vector<float> & amplitude0_in,
      const Vector<float> & damplitude_in,
      Vector<complex> & accum_out);

  void sample_accum (
      const Vector<float> & dmass_in,
      Vector<complex> & accum_out);

  void debias_slow_bend ();

  void retune (
      const Vector<float> & mass,
      Vector<float> & reass,
      float rate = 1) const
  {
    Bank::retune(m_slow_bend, mass, reass, rate);
  }
  void retune (float rate = 1)
  {
    m_temp = m_mass;
    retune(m_temp, m_mass, rate);
  }
  void retune_particles (float rate = 1);

  const Vector<float> & get_mass () const { return m_mass; }
  const Vector<float> & get_freq () const { return m_frequency; }
  const Vector<float> & get_phase_x () const { return m_phase_real; }
  const Vector<float> & get_phase_y () const { return m_phase_imag; }
  const Vector<float> & get_beat () const { return m_beat; }
  const Vector<float> & get_bend () const { return m_slow_bend; }
};

//----( syncopator bank )-----------------------------------------------------

class SyncopatorBank : public Bank
{
  const ScaledBeatFun m_beat_fun;

  Vector<float> m_data;

protected:

  Vector<float> m_mass;
  Vector<float> m_frequency;
  Vector<float> m_offset_real;
  Vector<float> m_offset_imag;
  Vector<float> m_phase_real;
  Vector<float> m_phase_imag;
  Vector<float> m_beat;
  mutable Vector<float> m_temp;

public:

  //----( state manipulation )----

  typedef float8 State;
  enum { state_dim = 8 };

  static void init (State & s) { s = State(0,1,1,0,1,0,0,0); }

  static void set_mass (State & s, float mass) { s[0] = mass; }
  static void set_freq (State & s, float freq) { s[1] = freq; }
  static void set_offset (State & s, float offset)
  {
    ASSERTW_LE(0, offset);
    ASSERTW_LE(offset, 1);

    complex c = exp_2_pi_i(offset);
    s[2] = c.real();
    s[3] = c.imag();
  }
  static void set_phase (State & s, float phase)
  {
    ASSERTW_LE(0, phase);
    ASSERTW_LE(phase, 1);

    complex c = exp_2_pi_i(phase);
    s[4] = c.real();
    s[5] = c.imag();
  }

  static float get_mass (const State & s) { return s[0]; }
  static float get_freq (const State & s) { return s[1]; }
  static float get_offset (const State & s)
  {
    return wrap(atan2f(s[3], s[2]) / (2 * M_PI));
  }
  static float get_phase (const State & s)
  {
    return wrap(atan2f(s[5], s[4]) / (2 * M_PI));
  }

  //----( bank interface )----

  SyncopatorBank (Bank param);
  virtual ~SyncopatorBank ();

  void set (const BoundedMap<Id, State> & states);
  void get (BoundedMap<Id, State> & states) const;

  void advance (float dt = 1.0f);
};

//----( geometric bank )------------------------------------------------------

class GeomBank : public Bank
{
  Vector<float> m_data;

protected:

  Vector<float> m_mass;
  Vector<float> m_frequency;
  Vector<float> m_phase_real;
  Vector<float> m_phase_imag;
  Vector<float> m_beat_real;
  Vector<float> m_beat_imag;
  Vector<float> m_dbeat_real;
  Vector<float> m_dbeat_imag;
  Vector<float> m_meso_energy;
  Vector<float> m_slow_energy;
  Vector<float> m_meso_bend;
  Vector<float> m_slow_bend;
  mutable Vector<float> m_temp;

  Filters::MaxGain m_force_gain;
  Filters::StdGain m_energy_gain;

  complex m_force_snapshot;

public:

  GeomBank (Bank param);
  ~GeomBank ();

  void sample_accum (
      const Vector<float> & dmass_in,
      Vector<complex> & accum_out);

  void retune (
      const Vector<float> & mass,
      Vector<float> & reass,
      float rate = 1) const
  {
    Bank::retune(m_slow_bend, mass, reass, rate);
  }
  void retune_zeromean (
      const Vector<float> & mass,
      Vector<float> & reass,
      float rate = 1) const;
  void retune (float rate = 1)
  {
    m_temp = m_mass;
    retune(m_temp, m_mass, rate);
  }

  const Vector<float> & get_mass () const { return m_mass; }
  const Vector<float> & get_freq () const { return m_frequency; }
  const Vector<float> & get_phase_x () const { return m_phase_real; }
  const Vector<float> & get_phase_y () const { return m_phase_imag; }
  const Vector<float> & get_beat () const { return m_beat_real; }
  const Vector<float> & get_energy () const { return m_slow_energy; }
  const Vector<float> & get_bend () const { return m_slow_bend; }

  complex get_force_snapshot () const { return m_force_snapshot; }
};

//----( geometric set )-------------------------------------------------------

/** Motivation for sparse beat induction.
  see ideas/music.text (2011:05-19-25) (Q2.A1)

  (1) we model beat as a sparse set of coupled resonating oscillators,
    heterogeneously distributed in frequency x duration x phase space.
    (1) frequency and duration are constrained by
      min_freq <= freq <= freq / duration <= max_freq
      min_duration <= 1
    (2) to maintain coverage in freq x duration space
      despite observation-induced contraction,
      each oscillator remembers and drifts towards a natrual frequency.
  (2) beat is learned fitting complex mass coefficients
    of an overcomplete oscillator basis
    to a complex-valued observed beat.
    (1) observed beat is complex-valued to allow highpass prefiltering such as
                pow(-,1/3)            i d/dt
        energy |----------> loudness |------> Im[beat]

      where the derivative term introdues a phase shift of pi/2.
*/

class GeomSet
{
protected:

  const size_t m_size;
  const float m_min_mass;
  const float m_min_omega;
  const float m_max_omega;
  const float m_min_duration;
  const float m_timescale;

private:

  Vector<float> m_data;

protected:

  Vector<float> m_mass;
  Vector<float> m_duration;
  Vector<float> m_omega0;
  Vector<float> m_omega;
  Vector<float> m_phase_real;
  Vector<float> m_phase_imag;
  Vector<float> m_beat_real;
  Vector<float> m_beat_imag;
  Vector<float> m_dbeat_real;
  Vector<float> m_dbeat_imag;

  float m_norm_beat;

  Filters::MaxGain m_force_gain;

public:

  bool debug;

  GeomSet (size_t size, float min_freq, float max_freq, float min_duration = 0);
  GeomSet (const GeomSet & other);
  void operator= (const GeomSet & other);
  ~GeomSet ();

  size_t size () const { return m_size; }
  float min_duration () const { return m_min_duration; }
  float min_omega () const { return m_min_omega; }
  float max_omega () const { return m_max_omega; }
  float min_freq () const { return m_min_omega / (2 * M_PI); }
  float max_freq () const { return m_max_omega / (2 * M_PI); }

  Vector<float> & get_mass () { return m_mass; }
  const Vector<float> & get_mass () const { return m_mass; }
  const Vector<float> & get_duration () const { return m_duration; }
  const Vector<float> & get_omega () const { return m_omega; }
  const Vector<float> & get_phase_x () const { return m_phase_real; }
  const Vector<float> & get_phase_y () const { return m_phase_imag; }

  float predict_value (const Vector<float> & mass_in) const
  {
    return dot(mass_in, m_beat_real);
  }
  float predict_value () const { return predict_value(m_mass); }

  void advance ();

  float learn (float observed_value, Vector<float> & mass_io) const;
  float learn (float observed_value) { return learn(observed_value, m_mass); }

  // this makes duration density uniform
  static inline float duration_cdf (float duration, float min_duration)
  {
    float L = logf(duration);
    float L1 = logf(min_duration);
    return L * (2 * L1 - L) / sqr(L1);
  }

private:

  void init_random (size_t i);

  void update_beat_function ();
};

//----( boltzmann bank )------------------------------------------------------

class BoltzBank : public Bank
{
  const ScaledBeatFun m_beat_fun;

  Vector<float> m_data;

protected:

  Vector<float> m_mass;
  Vector<float> m_frequency;
  Vector<float> m_phase_real;
  Vector<float> m_phase_imag;
  Vector<float> m_beat;
  Vector<float> m_dbeat;
  Vector<float> m_meso_energy;
  Vector<float> m_slow_energy;
  Vector<float> m_meso_bend;
  Vector<float> m_slow_bend;
  mutable Vector<float> m_temp;

  Filters::MaxGain m_force_gain;
  Filters::StdGain m_energy_gain;

  complex m_force_snapshot;

public:

  BoltzBank (Bank param);
  ~BoltzBank ();

  void sample_accum (
      const Vector<float> & dmass_in,
      Vector<complex> & accum_out);

  void retune (
      const Vector<float> & mass,
      Vector<float> & reass,
      float rate = 1) const
  {
    Bank::retune(m_slow_bend, mass, reass, rate);
  }
  void retune_zeromean (
      const Vector<float> & mass,
      Vector<float> & reass,
      float rate = 1) const;
  void retune (float rate = 1)
  {
    m_temp = m_mass;
    retune(m_temp, m_mass, rate);
  }

  void retune_particles (Vector<float> & pitch_table, float rate = 1);
  void retune_particles_zeromean (
      Vector<float> & pitch_table,
      const Vector<float> & mass,
      float rate = 1);

  const Vector<float> & get_mass () const { return m_mass; }
  const Vector<float> & get_freq () const { return m_frequency; }
  const Vector<float> & get_phase_x () const { return m_phase_real; }
  const Vector<float> & get_phase_y () const { return m_phase_imag; }
  const Vector<float> & get_beat () const { return m_beat; }
  const Vector<float> & get_energy () const { return m_slow_energy; }
  const Vector<float> & get_bend () const { return m_slow_bend; }

  complex get_force_snapshot () const { return m_force_snapshot; }
};

//----( loop bank )-----------------------------------------------------------

class LoopBank : public Bank
{
  const ScaledBeatFun m_beat_fun;

public:

  const size_t period;
  const float max_dt;

protected:

  const float m_drift_rate;
  const bool m_coalesce;

  Vector<float> m_freq_p;
  Vector<float> m_cos_t;
  Vector<float> m_sin_t;
  Vector<float> m_beat_t;

  Vector<float> m_mass_tp;
  Vector<float> m_temp_tp;

  Filters::DebugStats<float> m_mass_stats;
  Filters::DebugStats<float> m_force_x_stats;
  Filters::DebugStats<float> m_force_y_stats;
  Filters::DebugStats<float> m_bend_stats;

public:

  Vector<float> mass_now; // an alias

  LoopBank (
      Bank param,
      bool coalesce = true,
      float expected_dt = 1.0 / DEFAULT_VIDEO_FRAMERATE);
  virtual ~LoopBank ();

  bool coalesce () const { return m_coalesce; }

  const Vector<float> & get_freq () const { return m_freq_p; }
  float get_freq (size_t i) const { return m_freq_p[i]; }
  void set_freq (size_t i, float freq) { m_freq_p[i] = freq; }

  void advance (float dt, float decay = 1, float add = 0);
  void advance (
      Vector<float> & force_tp,
      float dt,
      float decay = 1,
      float add = 0);
};

//----( vector loop bank )----------------------------------------------------

class VectorLoopBank : public Bank
{
  const ScaledBeatFun m_beat_fun;

public:

  const size_t width;
  const size_t height;
  const size_t period;

protected:

  Vector<float> m_freq_y;
  Vector<float> m_phase_y;

  Vector<float> m_mass_ytx;
  Vector<float> m_temp_mass_tx;

  Filters::DebugStats<double> m_bend_stats;

public:

  VectorLoopBank (Bank param, size_t w);
  virtual ~VectorLoopBank ();

  void get_mass (Vector<float> & mass_yx) const;
  void add_mass (const Vector<float> & dmass_yx, float dt);
  void decay_add_mass (const Vector<float> & dmass_in, float dt);
  void scale_add_mass (
      const Vector<float> & scale_x,
      const Vector<float> & scale_y,
      const Vector<float> & dmass_in,
      float dt);
  void synchronize (float dt);
  void sample (Vector<float> & amplitude_x, float dt);
};

//====( transform wrappers )==================================================

//----( phasogram )-----------------------------------------------------------

class Phasogram : protected PhasorBank
{
  const size_t m_block_size;

  Vector<float> m_amplitude0;
  Vector<float> m_damplitude;

public:

  Phasogram (
      size_t block_size,
      Bank param);

  size_t size_in () const { return PhasorBank::size; }
  size_t size_out () const { return m_block_size; }

  void transform (
      const Vector<float> & mass_in,
      const Vector<float> & amplitude_in,
      Vector<complex> & sound_out,
      float timescale = 1.0f,
      bool do_retune = true);
};

//----( pitchgram )-----------------------------------------------------------

class Pitchgram : protected FourierBank
{
  const size_t m_size_in;

public:

  Pitchgram (
      size_t size_in,
      size_t size_out,
      float freq0,
      float freq1)

    : FourierBank(Bank(size_out, freq0, freq1)),

      m_size_in(size_in)
  {
    ASSERT_LT(0, size_in);
  }

  size_t size_in () const { return m_size_in; }
  size_t size_out () const { return FourierBank::size; }

  void transform (
      const Vector<float> & time_in,
      Vector<float> & freq_out)
  {
    ASSERT_SIZE(time_in, m_size_in);
    FourierBank::sample(time_in, freq_out);
  }

  void transform (
      const Vector<complex> & time_in,
      Vector<float> & freq_out)
  {
    ASSERT_SIZE(time_in, m_size_in);
    FourierBank::sample(time_in, freq_out);
  }
};

} // namespace Synchronized

#endif // KAZOO_SYNCHRONY_H

