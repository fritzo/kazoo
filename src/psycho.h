
/** Invertible Psychoacoustic Transformations.

  References on Psychoacoustics:
  (R1) http://en.wikipedia.org/wiki/Equal-loudness_contour
  (R2) "Bark and ERB bilinear transforms" -J.O.Smith 3, J.S.Abel
     http://citeseerx.ist.psu.edu/viewdoc/summary?doi = 10.1.1.9.613
  (R3) ISO 226:2003 standard equal-loudness curves, in phons
  (R4) ITU-R 468 noise weighting curve
    http://en.wikipedia.org/wiki/ITU-R_468_noise_weighting
  (R5) "the musical ear" a nontech article with refs to textbooks
    http://www.newmusicbox.org/article.nmbx?id = 4077
  (R6) "Auditory scales of frequency representation"
    -Hartmut Traunmüller
    http://www.ling.su.se/staff/hartmut/bark.htm

  Open Source Software for Psychoacoustics:
  (S1) Ogg Vorbis.
    in libvorbis:
      see lib/psy.[ch] for implementation
      see lib/psytune.c for example audio transform
  (S2) GPSYCHO: an LGPL'd psychoacoustic model (from LAME)
   http://lame.sourceforge.net/gpsycho.php

  Notes:
  (N1) The phon scale is a perceived loudness scale of pure tones,
    relative to the base phon: 1 phon = 1dBSPL @ 1kHz.
    The ISO 226:2003 standard describes phons at other frequencies.
  (N2) The brain has an automatic pitch detector (R5).
    (Q1) Should this be modeled directly in psychoacoustics,
      or learned by a neural net?
  (N3) H. Traunmüller in (R6) suggests pitch scale rather than bark scale,
    since a variety of psychoacoustic effects give us higher precision at
    low frequencies.
*/

#ifndef KAZOO_PSYCHO_H
#define KAZOO_PSYCHO_H

#include "common.h"
#include "vectors.h"
#include "splines.h"
#include "filters.h"
#include "synchrony.h"
#include "audio_types.h"
#include "cyclic_time.h"
#include "threads.h"
#include "config.h"

#define PSYCHO_MIN_PITCH_HZ             (30.0f)
#define PSYCHO_MAX_PITCH_HZ             (2e4f)
#define PSYCHO_PITCH_ACUITY             (7.0f)

#define PSYCHO_MIN_TEMPO_HZ             (0.25f)
#define PSYCHO_MAX_TEMPO_HZ             (8.0f)
#define PSYCHO_TEMPO_MIN_DURATION       (1 / 12.0f)

#define PSYCHO_HARMONY_SIZE             (1024)
#define PSYCHO_RHYTHM_SIZE              (2048)

#define PSYCHO_MIN_TIMESCALE_SEC        (0.001f)

#define PSYCHO_POLYRHYTHM_COUNT         (4)

namespace Psycho
{

//----( loudness transform )--------------------------------------------------

/** Loudness transform.

  Assumptions:
    energy is in [0,inf)
    loudness is in [0,1]
    fast time scale  <<  prediction time scale  <<  slow time scale
*/

class Loudness
{
  const size_t m_size;
  const float m_frame_rate;
  const float m_time_scale;
  const float m_ss_factor;

  const float m_tolerance;

  const float m_decay_factor_fast;
  const float m_decay_factor_slow;
  float m_max_loudness;

  Vector<float> m_current_fwd;
  Vector<float> m_history_fwd;
  Vector<float> m_current_bwd;
  Vector<float> m_history_bwd;

  Mutex m_max_loudness_mutex;

public:
  Loudness (
      size_t size,
      float frame_rate,
      float time_scale = 1.0,
      float ss_factor = 0.25);
  ~Loudness () { SAVE("Loudness.max", m_max_loudness); }

  // diagnostics
  size_t size () const { return m_size; }
  float frame_rate () const { return m_frame_rate; }
  float time_scale () const { return m_time_scale; }
  float ss_factor () const { return m_ss_factor; }

  // these can operate concurrently
  void transform_fwd (Vector<float> & energy_in, Vector<float> & loudness_out);
  void transform_bwd (Vector<float> & loudness_in, Vector<float> & energy_out);
};

class EnergyToLoudness
{
  Filters::RmsGain m_gain;

public:

  EnergyToLoudness (float sample_rate = DEFAULT_AUDIO_FRAMERATE)
    : m_gain(PSYCHO_MIN_TEMPO_HZ / 2.0f * sample_rate)
  {}

  float transform_fwd (float energy)
  {
    float loudness = powf(energy, 1/3.0f);
    return loudness * m_gain.update(sqr(loudness));
  }
  float transform_bwd (float loudness)
  {
    return powf(loudness / m_gain, 3.0f);
  }
};

//----( beat perception )-----------------------------------------------------

class EnergyToBeat
{
  ConfigParser m_config;

  const float m_min_tempo_hz;
  const float m_max_tempo_hz;

  Seconds m_time;

  float m_pos;
  float m_vel;
  float m_acc;
  float m_norm;

  float m_pos_mean;
  float m_pos_variance;
  float m_acc_variance;

  float m_lag;

  Mutex m_mutex;

public:

  EnergyToBeat (
      const char * config_filename = "config/default.psycho.conf",
      float min_tempo_hz = PSYCHO_MIN_TEMPO_HZ,
      float max_tempo_hz = PSYCHO_MAX_TEMPO_HZ);
  ~EnergyToBeat ();

  complex transform_fwd (Seconds time, float loudness);
  float transform_bwd (complex beat);
};

//----( logarithmic history )-------------------------------------------------

/** History with logrithmic resolution decay.

  Parameters:
    size     size of data to remember
    length   number of history pages to remember, going exponentially far back
    density  number of frames per history compression exponent,
             typically between 2 (compressing every frame)
             and length / 4 (for overall compression factor of exp(-4))
*/

class History
{
  const size_t m_size;
  const size_t m_length;
  const size_t m_density;
  const float m_tau;

  float log_time (float time) { return m_tau * logf(1 + time / m_tau); }

  //----( memory-managed frames )----

  struct Frame
  {
    Vector<float> data;
    float time;
    size_t rank;
    Frame * next;

    Frame (size_t size) : data(size), time(0), rank(0), next(NULL) {}
    ~Frame () { if (next) delete next; }

    void init (const Vector<float> & data);
    size_t num_terms () const { return 1 << rank; }
  };
  Frame * m_frames;
  Frame * m_free_frames;
  size_t m_num_cropped;

  void add_frame (const Vector<float> & present);
  void crop_to_frame (Frame * frame);
  void merge_frames (Frame * frame);

  //----( spline interpolation )----
  Vector<float> m_spline_weights;

public: //----( public interface )----

  History (
      size_t size,
      size_t length = 0,
      size_t density = 0);
  ~History ();

  // diagnostics
  size_t size () const { return m_size; }
  size_t length () const { return m_length; }
  size_t density () const { return m_density; }
  size_t size_in () const { return m_size; }
  size_t size_out () const { return m_size * m_length; }
  bool full () const { return m_num_cropped > 0; }

  History & add (const Vector<float> & present);
  History & get (Vector<float> & past); // arranged as past[time][channel]
  History & get_after (float delay, Vector<float> & past);
  History & at (float time, Vector<float> & moment);
};

//----( masking )-------------------------------------------------------------

class Masker : public Synchronized::FourierBank2
{
  const float m_radius;
  Vector<float> m_mask_dt;

  Vector<float> m_energy;
  Vector<float> m_time;

public:

  Masker (Synchronized::Bank param);
  ~Masker ();

  void sample (const Vector<float> & time_in, Vector<float> & masked_out);
  void sample (const Vector<complex> & time_in, Vector<float> & masked_out);

private:

  void sample (Vector<float> & masked_out);
};

//----( psychogram )----------------------------------------------------------

class Psychogram
{
  Synchronized::FourierBank2 m_bank;
  Loudness m_loudness;

  Vector<float> m_energy;

public:

  Psychogram (
      size_t size,
      float min_freq_hz = PSYCHO_MIN_PITCH_HZ,
      float max_freq_hz = PSYCHO_MAX_PITCH_HZ);
  ~Psychogram () {}

  void transform_fwd (
      const StereoAudioFrame & sound_in,
      Vector<float> & loudness_out);
  void transform_bwd (
      const Vector<float> & loudness_in,
      StereoAudioFrame & sound_out)
  {
    TODO("implement backward psychogram");
  }
};

//----( harmony )-------------------------------------------------------------

class Harmony : public Synchronized::Bank
{
public:

  typedef Synchronized::FourierBank2 Anal;
  typedef Synchronized::BoltzBank Synth;
  //typedef Synchronized::GeomBank Synth;

private:

  Anal m_anal_bank;
  Synth m_synth_bank;

  const float m_retune_rate;

  Vector<float> m_temp;

public:

  Harmony (
      size_t size = PSYCHO_HARMONY_SIZE,
      float acuity = PSYCHO_PITCH_ACUITY,
      float min_freq_hz = MIN_CHROMATIC_FREQ_HZ,
      float max_freq_hz = MAX_CHROMATIC_FREQ_HZ,
      float min_timescale_sec = PSYCHO_MIN_TIMESCALE_SEC);
  ~Harmony () {}

  const Synth & synth () const { return m_synth_bank; }

  void analyze (const MonoAudioFrame & sound_in, Vector<float> & mass_out);
  void analyze (const StereoAudioFrame & sound_in, Vector<float> & mass_out);

  void synthesize (Vector<float> & mass_io, StereoAudioFrame & sound_out);

  void synthesize_mix (
      const Vector<float> & weights,
      Vector<float> & masses_io,
      StereoAudioFrame & sound_out);

  void synthesize_control (
      const Vector<float> & mean,
      const Vector<float> & rise,
      const Vector<float> & flow,
      StereoAudioFrame & sound_out);
};

//----( rhythm )--------------------------------------------------------------

class Rhythm
{
  size_t m_num_observations;
  float m_total_residual;
public:

  typedef Synchronized::GeomSet Synth;

private:

  Synth m_set;

public:

  Rhythm (
      size_t size = PSYCHO_RHYTHM_SIZE,
      float min_freq_hz = PSYCHO_MIN_TEMPO_HZ,
      float max_freq_hz = PSYCHO_MAX_TEMPO_HZ,
      float duration = PSYCHO_TEMPO_MIN_DURATION);
  Rhythm (const Rhythm & other);
  ~Rhythm () { PRINT(residual()); }

  const Synth & synth () const { return m_set; }
  float residual () const { return m_total_residual / m_num_observations; }

  float sample ();
  float learn_and_sample (float observation);
  float predict (size_t num_steps);
};

//----( polyrhythm )----------------------------------------------------------

class Polyrhythm
{
public:

  typedef Synchronized::GeomSet Synth;

protected:

  Synth m_set;

  const size_t m_voice_count;

  Vector<float> m_masses;
  Vector<float> m_mass;
  Vector<float> m_values;

public:

  Polyrhythm (
      size_t voice_count = PSYCHO_POLYRHYTHM_COUNT,
      size_t tempo_size = PSYCHO_RHYTHM_SIZE,
      float min_tempo_hz = PSYCHO_MIN_TEMPO_HZ,
      float max_tempo_hz = PSYCHO_MAX_TEMPO_HZ,
      float duration = PSYCHO_TEMPO_MIN_DURATION);
  ~Polyrhythm () {}

  const Synth & synth () const { return m_set; }
  size_t voice_count () const { return m_voice_count; }

  void learn_one (size_t voice, float value);
  void learn_all (const Vector<float> & values);

  void sample (Vector<float> & values_out);

private:

  void advance (const Vector<float> & values_in);
  void project_mass ();
};

} // namespace Psycho

#endif // KAZOO_PSYCHO_H

