
#ifndef KAZOO_FLOCK_H
#define KAZOO_FLOCK_H

#include "common.h"
#include "psycho.h"
#include "synthesis.h"
#include "streaming.h"
#include "synchrony.h"
#include "regression.h"
#include "linalg.h"
#include "events.h"

#define FLOCK_PCA_TIMESCALE_SEC         (1.0f)
#define FLOCK_MOUSE_WHEEL_RATE          (0.2f)

namespace Streaming
{

//----( control interfaces )--------------------------------------------------

class KeyCounter
{
  size_t m_num_keys_down;

public:

  KeyCounter () : m_num_keys_down(0) {}

  void update (const SDL_KeyboardEvent & event);
  float get () const { return m_num_keys_down; }
};

struct Note
{
  float pitch;
  float power;

  Note () : pitch(0.5f), power(0) {}
  Note (float p, float e = 1.0f) : pitch(p), power(e) {}
};
typedef std::vector<Note> Notes;

void operator+= (Vector<float> & power, const Notes & notes);

//----( pitch flock )---------------------------------------------------------

class PitchFlock
{
private:

  Psycho::Harmony m_harmony;

  Vector<float> m_power;
  Vector<float> m_energy;

  Filters::RmsGain m_power_gain;

protected:

  Notes m_notes;
  Notes m_temp_notes;
  Mutex m_notes_mutex;

public:

  PitchFlock (
      size_t size = PSYCHO_HARMONY_SIZE,
      float acuity = PSYCHO_PITCH_ACUITY,
      float freq0  = MIN_CHROMATIC_FREQ_HZ,
      float freq1  = MAX_CHROMATIC_FREQ_HZ);
  ~PitchFlock ();

  const Psycho::Harmony::Synth & synth () const { return m_harmony.synth(); }
  const Vector<float> & get_power () const { return m_power; }
  const Vector<float> & get_energy () const { return m_energy; }

  void add_sound (const StereoAudioFrame & sound_in);
  void scale_energy (float scale) { m_energy *= scale; }

  void sample (StereoAudioFrame & sound_out);
};

//----( pitch flock viewer )--------------------------------------------------

class PitchFlockViewer
  : protected PitchFlock,
    public Rectangle,
    public Pushed<StereoAudioFrame>,
    public Pulled<StereoAudioFrame>,
    public Pulled<RgbImage>,
    public EventHandler
{
  const float m_plot_timescale;
  Filters::MaxGain m_power_gain;
  Filters::MaxGain m_energy_gain;
  Filters::RmsGain m_bend_gain;
  Filters::MaxGain m_force_gain;
  Filters::MaxGain m_image_gain;

  Regression::OnlinePca m_pca;
  Vector<float> m_select_x;
  Vector<float> m_select_y1;
  Vector<float> m_select_y2;

  const float m_image_timescale;
  BinarySemaphore m_image_mutex;
  RgbImage m_bend_image;
  Vector<float> m_beat_image;
  RgbImage m_pca_image;
  Vector<float> m_energy_image;
  Vector<float> m_temp;

  Seconds m_latest_image_time;

  enum PlotType
  {
    e_plot_bend,
    e_plot_beat,
    e_plot_pca,
    e_plot_pca_freq,
    e_plot_pca_4d,
    e_plot_none
  };
  PlotType m_plot_type;

  EventHandler::ButtonState m_mouse_state;
  EventHandler::WheelPosition m_wheel_position;
  LinAlg::Orientation3D m_angle3;
  LinAlg::Orientation4D m_angle4;

public:

  PitchFlockViewer (
      Rectangle shape,
      size_t size = PSYCHO_HARMONY_SIZE,
      float acuity = PSYCHO_PITCH_ACUITY,
      float freq0  = MIN_CHROMATIC_FREQ_HZ,
      float freq1  = MAX_CHROMATIC_FREQ_HZ);

  virtual ~PitchFlockViewer () {}

  virtual void push (Seconds time, const StereoAudioFrame & sound_in);
  virtual void pull (Seconds time, StereoAudioFrame & sound_out);
  virtual void pull (Seconds time, RgbImage & image);

private:

  void plot_bend ();
  void plot_beat ();
  void plot_pca ();
  void plot_pca_freq ();
  void plot_pca_4d ();

  void select_notes (float x, float y);

protected:

  virtual void keyboard (const SDL_KeyboardEvent & event);
  virtual void mouse_motion (const SDL_MouseMotionEvent & event);
  virtual void mouse_button (const SDL_MouseButtonEvent & event);
};

//----( tempo flock )---------------------------------------------------------

class TempoFlock
{
  Psycho::Rhythm m_rhythm;
protected:
  BinarySemaphore m_rhythm_mutex;
private:

  Psycho::EnergyToLoudness m_energy_to_loudness;

  float m_beat;
  Mutex m_beat_mutex;

public:

  TempoFlock (
      size_t size = PSYCHO_RHYTHM_SIZE,
      float min_freq_hz = PSYCHO_MIN_TEMPO_HZ,
      float max_freq_hz = PSYCHO_MAX_TEMPO_HZ);
  ~TempoFlock () {}

  const Psycho::Rhythm::Synth & synth () const { return m_rhythm.synth(); }

  void add_power (Seconds time, float power);
  void add_beat (float beat);

  float sample (bool learning);
};

//----( tempo flcok test )----------------------------------------------------

class TempoFlockTest : public Pushed<float>
{
  Psycho::EnergyToLoudness m_energy_to_loudness;
  Psycho::Rhythm m_estimator;
  Psycho::Rhythm m_predictor;

public:

  TempoFlockTest (
      size_t size = PSYCHO_RHYTHM_SIZE,
      float min_freq_hz = PSYCHO_MIN_TEMPO_HZ,
      float max_freq_hz = PSYCHO_MAX_TEMPO_HZ);

  virtual void push (Seconds time, const float & power_in);
};

//----( tempo flock viewer )--------------------------------------------------

class TempoFlockViewer
  : protected TempoFlock,
    public Rectangle,
    public Pushed<float>,
    public Pulled<StereoAudioFrame>,
    public Pulled<RgbImage>,
    public EventHandler
{
  const float m_image_timescale;
  BinarySemaphore m_image_mutex;
  RgbImage m_image;
  Vector<float> m_temp;

  Seconds m_latest_image_time;

  KeyCounter m_key_counter;
  EventHandler::ButtonState m_mouse_state;
  EventHandler::WheelPosition m_wheel_position;

public:

  TempoFlockViewer (
      Rectangle shape,
      size_t size = PSYCHO_RHYTHM_SIZE,
      float min_freq_hz = PSYCHO_MIN_TEMPO_HZ,
      float max_freq_hz = PSYCHO_MAX_TEMPO_HZ);

  virtual ~TempoFlockViewer () {}

  virtual void push (Seconds time, const float & power_in);
  virtual void pull (Seconds time, StereoAudioFrame & sound_out);
  virtual void pull (Seconds time, RgbImage & image);

private:

  void plot_beat ();

protected:

  virtual void keyboard (const SDL_KeyboardEvent & event);
  virtual void mouse_button (const SDL_MouseButtonEvent & event);
};

//----( flock )---------------------------------------------------------------

class Flock
{
  Psycho::Polyrhythm m_rhythm;
  Psycho::Harmony m_harmony;

  Psycho::EnergyToLoudness m_energy_to_loudness;

  Vector<float> m_power;
  Filters::MaxGain m_power_gain;

  Vector<float> m_beats;
  Vector<float> m_powers;

  Vector<float> m_pitch_masses;
  Vector<float> m_pitch_mass;

public:

  Flock (
      size_t voice_count = PSYCHO_POLYRHYTHM_COUNT,
      size_t pitch_size = PSYCHO_HARMONY_SIZE,
      size_t tempo_size = PSYCHO_RHYTHM_SIZE,
      float pitch_acuity = PSYCHO_PITCH_ACUITY,
      float min_pitch_hz = MIN_CHROMATIC_FREQ_HZ,
      float max_pitch_hz = MAX_CHROMATIC_FREQ_HZ,
      float min_tempo_hz = PSYCHO_MIN_TEMPO_HZ,
      float max_tempo_hz = PSYCHO_MAX_TEMPO_HZ);
  ~Flock () {}

  const Psycho::Polyrhythm::Synth & rhythm () const { return m_rhythm.synth(); }
  const Psycho::Harmony::Synth & harmony () const { return m_harmony.synth(); }

  size_t voice_count () const { return m_rhythm.voice_count(); }

  void set_rhythm (size_t voice, float beat);
  void set_harmony (size_t voice, const Vector<float> & selection);
  void add_sound (size_t voice, const StereoAudioFrame & sound_in);

  void sample (StereoAudioFrame & sound_out);
};

//----( flock viewer )--------------------------------------------------------

class FlockViewer
  : protected Flock,
    public Rectangle,
    public Pushed<StereoAudioFrame>,
    public Pulled<StereoAudioFrame>,
    public Pulled<RgbImage>,
    public EventHandler
{
  float m_beat;

  const float m_pca_timescale;
  Regression::OnlinePca m_pca;

  Vector<float> m_select_x;
  Vector<float> m_select_y;
  Vector<float> m_selection;

  const float m_image_timescale;
  Filters::MaxGain m_value_gain;
  Filters::MaxGain m_mass_gain;
  RgbImage m_image;
  BinarySemaphore m_image_mutex;
  Vector<float> m_temp;

  Seconds m_latest_image_time;

  KeyCounter m_key_counter;
  EventHandler::ButtonState m_mouse_state;
  EventHandler::WheelPosition m_wheel_position;
  LinAlg::Orientation3D m_angle3;
  float m_mouse_x, m_mouse_y;
  Mutex m_mouse_mutex;

public:

  FlockViewer (
      Rectangle shape,
      size_t voice_count = PSYCHO_POLYRHYTHM_COUNT,
      size_t pitch_size = PSYCHO_HARMONY_SIZE,
      size_t tempo_size = PSYCHO_RHYTHM_SIZE,
      float pitch_acuity = PSYCHO_PITCH_ACUITY,
      float min_pitch_hz = MIN_CHROMATIC_FREQ_HZ,
      float max_pitch_hz = MAX_CHROMATIC_FREQ_HZ,
      float min_tempo_hz = PSYCHO_MIN_TEMPO_HZ,
      float max_tempo_hz = PSYCHO_MAX_TEMPO_HZ);

  virtual ~FlockViewer () {}

  virtual void push (Seconds time, const StereoAudioFrame & sound_in);
  virtual void pull (Seconds time, StereoAudioFrame & sound_out);
  virtual void pull (Seconds time, RgbImage & image);

  size_t active_voice () const { return 0; } // TODO set voice value

private:

  void plot_pitch ();

  bool update_selection ();

protected:

  virtual void keyboard (const SDL_KeyboardEvent & event);
  virtual void mouse_motion (const SDL_MouseMotionEvent & event);
  virtual void mouse_button (const SDL_MouseButtonEvent & event);
};

} // namespace Streaming

#endif // KAZOO_FLOCK_H

