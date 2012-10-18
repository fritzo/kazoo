
#include "flock.h"
#include "splines.h"

#define TOL (1e-8f)

namespace Streaming
{

//----( control interfaces )--------------------------------------------------

void KeyCounter::update (const SDL_KeyboardEvent & event)
{
  switch (event.type) {
    case SDL_KEYDOWN: ++m_num_keys_down; break;
    case SDL_KEYUP:   --m_num_keys_down; break;
    default: return;
  }
}

void operator+= (Vector<float> & power, const Notes & notes)
{
  const size_t size = power.size;
  typedef Notes::const_iterator Auto;
  for (Auto i = notes.begin(); i != notes.end(); ++i) {
    LinearInterpolate(i->pitch * size, size).iadd(power, i->power);
  }
}

//----( pitch flock )---------------------------------------------------------

PitchFlock::PitchFlock (
    size_t size,
    float acuity,
    float freq0,
    float freq1)

  : m_harmony(size, acuity, freq0, freq1),

    m_power(size),
    m_energy(size),

    m_power_gain(DEFAULT_GAIN_TIMESCALE_SEC)
{
  m_power.zero();
  m_energy.zero();
}

PitchFlock::~PitchFlock ()
{
  PRINT3(min(m_power), mean(m_power), max(m_power));
  PRINT3(min(m_energy), mean(m_energy), max(m_energy));
  PRINT(m_power_gain);
}

void PitchFlock::add_sound (const StereoAudioFrame & sound_in)
{
  m_harmony.analyze(sound_in, m_power);
  m_power *= m_power_gain.update(norm(m_power));
}

void PitchFlock::sample (StereoAudioFrame & sound_out)
{
  const float sustain_timescale
    = DEFAULT_SYNTHESIS_SUSTAIN_SEC * DEFAULT_AUDIO_FRAMERATE;
  const float decay = expf(-1.0f / sustain_timescale);

  m_notes_mutex.lock();
  m_power += m_notes;
  m_notes_mutex.unlock();

  accumulate_step(decay, m_energy, m_power);
  m_power.zero();

  m_harmony.synthesize(m_energy, sound_out);
}

//----( pitch flock viewer )--------------------------------------------------

PitchFlockViewer::PitchFlockViewer (
    Rectangle shape,
    size_t size,
    float acuity,
    float freq0,
    float freq1)
  : PitchFlock(size, acuity, freq0, freq1),
    Rectangle(shape),

    m_plot_timescale(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_AUDIO_FRAMERATE),
    m_power_gain(m_plot_timescale),
    m_energy_gain(m_plot_timescale),
    m_bend_gain(m_plot_timescale),
    m_force_gain(m_plot_timescale),
    m_image_gain(1.0f / DEFAULT_SCREEN_FRAMERATE),

    m_pca(size, 6),
    m_select_x(size),
    m_select_y1(size),
    m_select_y2(size),

    m_image_timescale(0.2f),
    m_bend_image(shape.size()),
    m_beat_image(shape.size()),
    m_pca_image(shape.size()),
    m_energy_image(shape.size()),
    m_temp(shape.size()),

    m_latest_image_time(Seconds::now()),

    m_plot_type(e_plot_bend),

    m_mouse_state(),
    m_wheel_position(),
    m_angle3(),
    m_angle4()
{
  m_select_x.zero();
  m_select_y1.zero();
  m_select_y2.zero();

  m_bend_image.zero();
  m_beat_image.zero();
  m_pca_image.zero();
  m_energy_image.zero();
}

void PitchFlockViewer::push (Seconds time, const StereoAudioFrame & sound_in)
{
  PitchFlock::add_sound(sound_in);
}

void PitchFlockViewer::pull (Seconds time, StereoAudioFrame & sound_out)
{
  int wheel_position = m_wheel_position.pop();
  if (wheel_position) {
    float scale = expf(FLOCK_MOUSE_WHEEL_RATE * wheel_position);
    PitchFlock::scale_energy(scale);
  }

  PitchFlock::sample(sound_out);

  switch (m_plot_type) {
    case e_plot_bend: plot_bend(); break;
    case e_plot_beat: plot_beat(); break;
    case e_plot_pca: plot_pca(); break;
    case e_plot_pca_freq: plot_pca_freq(); break;
    case e_plot_pca_4d: plot_pca_4d(); break;
    case e_plot_none: break;
  }
}

void PitchFlockViewer::pull (Seconds time, RgbImage & image)
{
  ASSERT_SIZE(image.red, Rectangle::size());

  float dt = max(1e-8f, time - m_latest_image_time);
  m_latest_image_time = time;
  float decay = exp(-dt / m_image_timescale);

  switch (m_plot_type) {
    case e_plot_bend:
      m_image_mutex.lock();
      image = m_bend_image;
      m_bend_image *= decay;
      m_image_mutex.unlock();

      Image::linear_blur(width(), height(), 4,4, image.red, m_temp);
      Image::linear_blur(width(), height(), 4,1, image.green, m_temp);
      Image::linear_blur(width(), height(), 1,1, image.blue, m_temp);

      image.red *= 1 / max(TOL, max(image.red));
      image.green *= 1 / max(TOL, max(image.green));
      image.blue *= 1 / max(TOL, max(image.blue));

      break;

    case e_plot_beat:
      m_image_mutex.lock();
      image.red = m_beat_image;
      m_beat_image *= decay;
      m_image_mutex.unlock();

      Image::linear_blur(width(), height(), 1,4, image.red, m_temp);

      image.red *= 1 / max(TOL, max(image.red));
      image.green = image.red;
      image.blue = image.red;

      break;

    case e_plot_pca:
    case e_plot_pca_freq:
    case e_plot_pca_4d:

      m_image_mutex.lock();
      image.red = m_energy_image;
      m_energy_image *= powf(decay, 1.5);
      m_image_mutex.unlock();

      Image::linear_blur(width(), height(), 8, image.red, m_temp);
      image.red *= 0.2f / max(image.red);
      image.green = image.red;
      image.blue = image.red;

      m_image_mutex.lock();
      image += m_pca_image;
      m_pca_image *= decay;
      m_image_mutex.unlock();

      {
        const size_t R = 2;

        Image::linear_blur(width(), height(), R, image.red, m_temp);
        Image::linear_blur(width(), height(), R, image.green, m_temp);
        Image::linear_blur(width(), height(), R, image.blue, m_temp);

        const float scale = 1.4f * m_image_gain.update(max(image));
        float * restrict im = image;
        for (size_t i = 0, I = image.size; i < I; ++i) {
          im[i] = min(1.0f, scale * im[i]);
        }
      }

      break;

    case e_plot_none:
      break;
  }
}

void PitchFlockViewer::plot_bend ()
{
  const size_t I = synth().size;
  const size_t X = Rectangle::width();
  const size_t Y = Rectangle::height();

  const float * restrict power = get_power();
  const float * restrict energy = get_energy();
  const float * restrict bend = synth().get_bend();
  float * restrict red = m_bend_image.red;
  float * restrict green = m_bend_image.green;
  float * restrict blue = m_bend_image.blue;

  float energy_gain = m_energy_gain.update(max(get_energy()));
  float power_gain = m_power_gain.update(max(get_power()));
  float bend_variance = norm_squared(synth().get_bend()) / I;
  float bend_gain = m_bend_gain.update(bend_variance);
  float min_freq = min(synth().get_freq());
  float max_freq = max(synth().get_freq());
  float pitch_scale = 1 / log(max_freq / min_freq);

  if (m_image_mutex.try_lock()) {

    for (size_t i = 0; i < I; ++i) {

      float m = energy_gain * energy[i];
      float e = power_gain * power[i];

      float b = atanf(0.5f * bend_gain * bend[i]) / M_PI + 0.5f;
      float p = static_cast<float>(i) / I + pitch_scale * log(1 + bend[i]);

      BilinearInterpolate lin(b * X, X, p * Y, Y);

      lin.imax(red, e);
      lin.imax(green, m);
      lin.imax(blue, 1);
    }

    m_image_mutex.unlock();
  }
}

void PitchFlockViewer::plot_beat ()
{
  const size_t I = synth().size;
  const size_t X = Rectangle::width();
  const size_t Y = Rectangle::height();

  const float * restrict phase_x = synth().get_phase_x();
  const float * restrict phase_y = synth().get_phase_y();
  float * restrict image = m_beat_image;

  complex force = synth().get_force_snapshot();
  float beat = abs(force) * m_force_gain.update(abs(force));

  if (m_image_mutex.try_lock()) {

    for (size_t i = 0; i < I; ++i) {
      float x = wrap(atan2f(phase_y[i], phase_x[i]) / (2 * M_PI)) * X;
      float y = (i + 0.5f) / I * Y;

      BilinearInterpolate(x, X, y, Y).imax(image, beat);
    }

    m_image_mutex.unlock();
  }
}

void PitchFlockViewer::plot_pca ()
{
  float timescale = FLOCK_PCA_TIMESCALE_SEC * DEFAULT_AUDIO_FRAMERATE;
  float dt = 1 / timescale;

  m_pca.add_sample(synth().get_beat(), dt);

  const size_t I = synth().size;
  const size_t X = Rectangle::width();
  const size_t Y = Rectangle::height();

  float * restrict select_x = m_select_x;
  float * restrict select_y1 = m_select_y1;
  float * restrict select_y2 = m_select_y2;

  float pca_gain = sqrtf(I) / 8;
  const float * restrict energy = get_energy();
  const float * restrict pca_x = m_pca.component(0);
  const float * restrict pca_y = m_pca.component(1);
  const float * restrict pca_z = m_pca.component(2);
  const float * restrict pca_u = m_pca.component(3);
  const float * restrict pca_v = m_pca.component(4);

  const float ru = 1.0f;
  const float bu = cos(2 * M_PI / 3);
  const float bv = sin(2 * M_PI / 3);
  const float gu = cos(-2 * M_PI / 3);
  const float gv = sin(-2 * M_PI / 3);

  float * restrict energy_image = m_energy_image;
  float * restrict red = m_pca_image.red;
  float * restrict green = m_pca_image.green;
  float * restrict blue = m_pca_image.blue;

  const LinAlg::Orientation3D angle = m_angle3;
  const float skew = 1.0f / 20;

  if (m_image_mutex.try_lock()) {

    for (size_t i = 0; i < I; ++i) {
      float m = energy[i];
      float xyz[3] = {
        pca_x[i] * pca_gain,
        pca_y[i] * pca_gain,
        pca_z[i] * pca_gain
      };
      float u = pca_u[i] * pca_gain;
      float v = pca_v[i] * pca_gain;

      float uv_scale = 1 / (1 + sqrt(max(1e-20f, sqr(u) + sqr(v))));
      u *= uv_scale;
      v *= uv_scale;

      float r = sqr((1 + ru * u) / 2);
      float g = sqr((1 + gu * u + gv * v) / 2);
      float b = sqr((1 + bu * u + bv * v) / 2);

      float x = angle.coord_dot(0, xyz);
      float y = angle.coord_dot(1, xyz);
      float z = angle.coord_dot(2, xyz);

      float y1 = y - skew * z;
      float y2 = y + skew * z;

      float x_01 = select_x[i] = (1 + x) / 2;
      float y1_01 = select_y1[i] = (1 + y1) / 4;
      float y2_01 = select_y2[i] = (1 + y2 + 2) / 4;

      BilinearInterpolate lin(x_01 * X, X, y1_01 * Y, Y);
      lin.imax(energy_image, m);
      lin.imax(red, r);
      lin.imax(green, g);
      lin.imax(blue, b);

      lin.y(y2_01 * Y, Y);
      lin.imax(energy_image, m);
      lin.imax(red, r);
      lin.imax(green, g);
      lin.imax(blue, b);
    }

    m_image_mutex.unlock();
  }
}

void PitchFlockViewer::plot_pca_freq ()
{
  float timescale = FLOCK_PCA_TIMESCALE_SEC * DEFAULT_AUDIO_FRAMERATE;
  float dt = 1 / timescale;

  m_pca.add_sample(synth().get_beat(), dt);

  const size_t I = synth().size;
  const size_t X = Rectangle::width();
  const size_t Y = Rectangle::height();

  float * restrict select_x = m_select_x;
  float * restrict select_y1 = m_select_y1;
  float * restrict select_y2 = m_select_y2;

  float pca_gain = sqrtf(I) / 8;
  float dx_gain = sqrtf(I) / synth().num_tones();
  const float * restrict energy = get_energy();
  const float * restrict pca_y = m_pca.component(0);
  const float * restrict pca_z = m_pca.component(1);
  const float * restrict pca_u = m_pca.component(2);
  const float * restrict pca_v = m_pca.component(3);
  const float * restrict pca_dx = m_pca.component(4);

  const float ru = 1.0f;
  const float bu = cos(2 * M_PI / 3);
  const float bv = sin(2 * M_PI / 3);
  const float gu = cos(-2 * M_PI / 3);
  const float gv = sin(-2 * M_PI / 3);

  float * restrict energy_image = m_energy_image;
  float * restrict red = m_pca_image.red;
  float * restrict green = m_pca_image.green;
  float * restrict blue = m_pca_image.blue;

  const LinAlg::Orientation3D angle = m_angle3;
  const float skew = 1.0f / 20;

  if (m_image_mutex.try_lock()) {

    for (size_t i = 0; i < I; ++i) {
      float m = energy[i];
      float xyz[3] = {
        1 - (i + 0.5f) / I * 2 + pca_dx[i] * dx_gain,
        pca_y[i] * pca_gain,
        pca_z[i] * pca_gain
      };
      float u = pca_u[i] * pca_gain;
      float v = pca_v[i] * pca_gain;

      float uv_scale = 1 / (1 + sqrt(max(1e-20f, sqr(u) + sqr(v))));
      u *= uv_scale;
      v *= uv_scale;

      float r = sqr((1 + ru * u) / 2);
      float g = sqr((1 + gu * u + gv * v) / 2);
      float b = sqr((1 + bu * u + bv * v) / 2);

      float x = angle.coord_dot(0, xyz);
      float y = angle.coord_dot(1, xyz);
      float z = angle.coord_dot(2, xyz);

      float y1 = y - skew * z;
      float y2 = y + skew * z;

      float x_01 = select_x[i] = (1 + x) / 2;
      float y1_01 = select_y1[i] = (1 + y1) / 4;
      float y2_01 = select_y2[i] = (1 + y2 + 2) / 4;

      BilinearInterpolate lin(x_01 * X, X, y1_01 * Y, Y);
      lin.imax(energy_image, m);
      lin.imax(red, r);
      lin.imax(green, g);
      lin.imax(blue, b);

      lin.y(y2_01 * Y, Y);
      lin.imax(energy_image, m);
      lin.imax(red, r);
      lin.imax(green, g);
      lin.imax(blue, b);
    }

    m_image_mutex.unlock();
  }
}

void PitchFlockViewer::plot_pca_4d ()
{
  float timescale = FLOCK_PCA_TIMESCALE_SEC * DEFAULT_AUDIO_FRAMERATE;
  float dt = 1 / timescale;

  m_pca.add_sample(synth().get_beat(), dt);

  const size_t I = synth().size;
  const size_t X = Rectangle::width();
  const size_t Y = Rectangle::height();

  float * restrict select_x = m_select_x;
  float * restrict select_y = m_select_y1;

  float pca_gain = sqrtf(I) / 8;
  const float * restrict energy = get_energy();
  const float * restrict pca_w = m_pca.component(0);
  const float * restrict pca_x = m_pca.component(1);
  const float * restrict pca_y = m_pca.component(2);
  const float * restrict pca_z = m_pca.component(3);
  const float * restrict pca_u = m_pca.component(4);
  const float * restrict pca_v = m_pca.component(5);

  const float ru = 1.0f;
  const float bu = cos(2 * M_PI / 3);
  const float bv = sin(2 * M_PI / 3);
  const float gu = cos(-2 * M_PI / 3);
  const float gv = sin(-2 * M_PI / 3);

  float * restrict energy_image = m_energy_image;
  float * restrict red = m_pca_image.red;
  float * restrict green = m_pca_image.green;
  float * restrict blue = m_pca_image.blue;

  const LinAlg::Orientation4D angle = m_angle4;

  if (m_image_mutex.try_lock()) {

    for (size_t i = 0; i < I; ++i) {
      float m = energy[i];
      float wxyz[4] = {pca_w[i], pca_x[i], pca_y[i], pca_z[i]};
      float u = pca_u[i] * pca_gain;
      float v = pca_v[i] * pca_gain;

      float uv_scale = 1 / (1 + sqrt(max(1e-20f, sqr(u) + sqr(v))));
      u *= uv_scale;
      v *= uv_scale;

      float r = sqr((1 + ru * u) / 2);
      float g = sqr((1 + gu * u + gv * v) / 2);
      float b = sqr((1 + bu * u + bv * v) / 2);

      float x = pca_gain * angle.coord_dot(0, wxyz);
      float y = pca_gain * angle.coord_dot(1, wxyz);

      float x_01 = select_x[i] = (1 + x) / 2;
      float y_01 = select_y[i] = (1 + y) / 2;

      BilinearInterpolate lin(x_01 * X, X, y_01 * Y, Y);
      lin.imax(energy_image, m);
      lin.imax(red, r);
      lin.imax(green, g);
      lin.imax(blue, b);
    }

    m_image_mutex.unlock();
  }
}

void PitchFlockViewer::select_notes (float x, float y)
{
  const size_t I = synth().size;

  const float * restrict select_x = m_select_x;
  const float * restrict select_y1 = m_select_y1;
  const float * restrict select_y2 = m_select_y2;

  const float radius = 1 / sqrtf(I);
  const float r2 = sqr(radius);

  bool stereo;
  switch (m_plot_type) {
    case e_plot_pca:
    case e_plot_pca_freq:
      stereo = true;
      break;

    default:
      stereo = false;
      break;
  }

  float total_power = 0;
  Notes & restrict notes = m_temp_notes;
  for (size_t i = 0; i < I; ++i) {

    float xi = select_x[i];
    float dx2 = sqr(xi - x);
    if (dx2 < r2) {

      float pitch = (i + 0.5f) / I;

      {
        float yi = select_y1[i];
        float dy2 = sqr(yi - y) / 2;
        if (dx2 + dy2 < r2) {
          float power = r2 - dx2 - dy2;
          notes.push_back(Note(pitch, power));
          total_power += power;
        }
      }
      if (stereo) {
        float yi = select_y2[i];
        float dy2 = sqr(yi - y) / 2;
        if (dx2 + dy2 < r2) {
          float power = r2 - dx2 - dy2;
          notes.push_back(Note(pitch, power));
          total_power += power;
        }
      }
    }
  }

  if (total_power > 0) {
    float scale = 1.0f / total_power;
    typedef Notes::iterator Auto;
    for (Auto n = notes.begin(); n < notes.end(); ++n) {
      n->power *= scale;
    }
  }

  m_notes_mutex.lock();
  std::swap(m_notes, m_temp_notes);
  m_notes_mutex.unlock();
  m_temp_notes.clear();
}

void PitchFlockViewer::keyboard (const SDL_KeyboardEvent & event)
{
  switch (event.keysym.sym) {
    case SDLK_SPACE:
      if (event.type == SDL_KEYDOWN) {
        m_plot_type = static_cast<PlotType>(
            (m_plot_type + 1) % (e_plot_none + 1));
      }
      break;

    default:
      break;
  }
}

void PitchFlockViewer::mouse_motion (const SDL_MouseMotionEvent & event)
{
  switch (m_plot_type) {
    case e_plot_bend:
    case e_plot_beat:

      if (m_mouse_state.left_down()) {
        float pitch = bound_to(0.0f, 1.0f, (event.x + 0.5f) / height());
        m_notes_mutex.lock();
        m_notes.clear();
        m_notes.push_back(Note(pitch));
        m_notes_mutex.unlock();
      }

      break;

    case e_plot_pca:
    case e_plot_pca_freq:
    case e_plot_pca_4d:

      if (m_mouse_state.left_down()) {
        float x = (event.y + 0.5f) / width();
        float y = (event.x + 0.5f) / height();
        select_notes(x,y);
      }

      if (m_mouse_state.middle_down()) {
        float dx = M_PI * event.yrel / width();
        float dy = M_PI * event.xrel / height();
        m_angle4.drag2(dx, dy);
      }

      if (m_mouse_state.right_down()) {
        float dx = M_PI * event.yrel / width();
        float dy = M_PI * event.xrel / height();
        m_angle3.drag(dx, dy);
        m_angle4.drag1(dx, dy);
      }

      break;

    case e_plot_none:
      break;
  }
}

void PitchFlockViewer::mouse_button (const SDL_MouseButtonEvent & event)
{
  m_mouse_state.update(event);
  m_wheel_position.update(event);

  switch (m_plot_type) {
    case e_plot_bend:
    case e_plot_beat:

      if (m_mouse_state.left_down()) {
        float pitch = bound_to(0.0f, 1.0f, (event.x + 0.5f) / height());
        m_notes_mutex.lock();
        m_notes.clear();
        m_notes.push_back(Note(pitch));
        m_notes_mutex.unlock();
      }

      break;

    case e_plot_pca:
    case e_plot_pca_freq:
    case e_plot_pca_4d:

      if (m_mouse_state.left_down()) {
        float x = (event.y + 0.5f) / width();
        float y = (event.x + 0.5f) / height();
        select_notes(x,y);
      } else {
        m_notes_mutex.lock();
        m_notes.clear();
        m_notes_mutex.unlock();
      }

      break;

    case e_plot_none:
      break;
  }
}

//----( tempo flock )---------------------------------------------------------

TempoFlock::TempoFlock (size_t size, float min_freq_hz, float max_freq_hz)
  : m_rhythm(size, min_freq_hz, max_freq_hz),
    m_energy_to_loudness(),
    m_beat(0)
{}

void TempoFlock::add_power (Seconds time, float power)
{
  float beat = m_energy_to_loudness.transform_fwd(power);
  m_beat_mutex.lock();
  m_beat += beat;
  m_beat_mutex.unlock();
}

void TempoFlock::add_beat (float beat)
{
  m_beat_mutex.lock();
  m_beat += beat;
  m_beat_mutex.unlock();
}

float TempoFlock::sample (bool learning)
{
  m_beat_mutex.lock();
  float beat = m_beat;
  m_beat = 0.0f;
  m_beat_mutex.unlock();

  m_rhythm_mutex.lock();
  if (learning) {
    beat = m_rhythm.learn_and_sample(beat);
  } else {
    beat = m_rhythm.sample();
  }
  m_rhythm_mutex.unlock();

  return m_energy_to_loudness.transform_bwd(beat);
}

//----( tempo flcok test )----------------------------------------------------

TempoFlockTest::TempoFlockTest (
    size_t size,
    float min_freq_hz,
    float max_freq_hz)
  : m_energy_to_loudness(),
    m_estimator(size, min_freq_hz, max_freq_hz),
    m_predictor(m_estimator)
{
}

void TempoFlockTest::push (Seconds time, const float & power_in)
{
  float beat = m_energy_to_loudness.transform_fwd(power_in);
  m_estimator.learn_and_sample(beat);
  /* TODO
  m_predictor = m_estimator;
  float prediction = m_predictor.predict(m_num_steps);
  */
}

//----( tempo flock viewer )--------------------------------------------------

TempoFlockViewer::TempoFlockViewer (
    Rectangle shape,
    size_t size,
    float min_freq_hz,
    float max_freq_hz)

  : TempoFlock(size, min_freq_hz, max_freq_hz),
    Rectangle(shape),

    m_image_timescale(0.2f),
    m_image(shape.size()),
    m_temp(shape.size()),

    m_latest_image_time(Seconds::now()),

    m_key_counter(),
    m_mouse_state(),
    m_wheel_position()
{
  m_image.zero();
}

void TempoFlockViewer::push (Seconds time, const float & power_in)
{
  TempoFlock::add_power(time, power_in);
}

void TempoFlockViewer::pull (Seconds time, StereoAudioFrame & sound_out)
{
  bool learning = m_mouse_state.any_down();

  if (learning) {
    TempoFlock::add_beat(m_key_counter.get());
  }

  float power = TempoFlock::sample(learning);
  Synthesis::sample_noise(sound_out, power);

  plot_beat();
}

void TempoFlockViewer::pull (Seconds time, RgbImage & image)
{
  ASSERT_SIZE(image.red, Rectangle::size());

  float dt = max(1e-8f, time - m_latest_image_time);
  m_latest_image_time = time;
  float decay = exp(-dt / m_image_timescale);

  m_image_mutex.lock();
  image = m_image;
  m_image *= decay;
  m_image_mutex.unlock();

  Image::linear_blur(width(), height(), 1,3, image.red, m_temp);
  Image::linear_blur(width(), height(), 1,2, image.green, m_temp);
  Image::linear_blur(width(), height(), 1,1, image.blue, m_temp);

  image.red *= 1 / max(TOL, max(image.red));
  image.green *= 1 / max(TOL, max(image.green));
  image.blue *= 1 / max(TOL, max(image.blue));
}

void TempoFlockViewer::plot_beat ()
{
  const size_t I = synth().size();
  const size_t X = Rectangle::width();
  const size_t Y = Rectangle::height();

  const float * restrict mass = synth().get_mass();
  const float * restrict phase_x = synth().get_phase_x();
  const float * restrict phase_y = synth().get_phase_y();

  const complex r_part = exp_2_pi_i(0 / 3.0f);
  const complex g_part = exp_2_pi_i(1 / 3.0f);
  const complex b_part = exp_2_pi_i(2 / 3.0f);

  float * restrict red = m_image.red;
  float * restrict green = m_image.green;
  float * restrict blue = m_image.blue;

  //const float * restrict acuity = synth().get_acuity();
  //const float acuity1 = synth().max_acuity();
  const float * restrict duration = synth().get_duration();
  const float duration0 = synth().min_duration();

  if (m_image_mutex.try_lock()) {
    if (m_rhythm_mutex.try_lock()) {

      for (size_t i = 0; i < I; ++i) {
        complex phase(phase_x[i], phase_y[i]);

        float x = wrap(arg(phase) / (2 * M_PI) + 0.5f);
        //float y = synth().acuity_cdf(acuity[i], acuity1);
        float y = synth().duration_cdf(duration[i], duration0);

        float m = min(mass[i], 1.0f);
        float r = m;
        float g = sqr(m);
        float b = 1;

        BilinearInterpolate lin(x * X, X, y * Y, Y);

        lin.imax(red, r);
        lin.imax(green, g);
        lin.imax(blue, b);
      }
      m_rhythm_mutex.unlock();
    }
    m_image_mutex.unlock();
  }
}

void TempoFlockViewer::keyboard (const SDL_KeyboardEvent & event)
{
  m_key_counter.update(event);
}

void TempoFlockViewer::mouse_button (const SDL_MouseButtonEvent & event)
{
  m_mouse_state.update(event);
  m_wheel_position.update(event);
}

//----( flock )---------------------------------------------------------------

Flock::Flock (
    size_t voice_count,
    size_t pitch_size,
    size_t tempo_size,
    float pitch_acuity,
    float min_pitch_hz,
    float max_pitch_hz,
    float min_tempo_hz,
    float max_tempo_hz)

  : m_rhythm(voice_count, tempo_size, min_tempo_hz, max_tempo_hz),
    m_harmony(pitch_size, min_pitch_hz, max_pitch_hz, pitch_acuity),

    m_energy_to_loudness(),

    m_power(pitch_size),
    m_power_gain(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_AUDIO_FRAMERATE),

    m_beats(voice_count),
    m_powers(voice_count),

    m_pitch_masses(voice_count * pitch_size),
    m_pitch_mass(pitch_size)
{
  ASSERT_LT(0, voice_count);

  LOG("creating Flock with " << voice_count << " voices");

  m_power.zero();
}

void Flock::set_rhythm (size_t voice, float beat)
{
  m_rhythm.learn_one(voice, beat);
}

void Flock::set_harmony (size_t voice, const Vector<float> & selection)
{
  ASSERT_LT(voice, voice_count());

  float selection_decay = expf(-rhythm().min_freq());
  Vector<float> pitch_mass = m_pitch_masses.block(m_harmony.size, voice);
  accumulate_step(selection_decay, pitch_mass, selection);
}

void Flock::add_sound (size_t voice, const StereoAudioFrame & sound_in)
{
  m_harmony.analyze(sound_in, m_power);
  m_power *= m_power_gain.update(norm(m_power));

  set_harmony(voice, m_power);
}

void Flock::sample (StereoAudioFrame & sound_out)
{
  m_rhythm.sample(m_beats);

  for (size_t v = 0; v < voice_count(); ++v) {
    m_powers[v] = m_energy_to_loudness.transform_bwd(m_beats[v]);
  }

  m_harmony.synthesize_mix(m_powers, m_pitch_masses, sound_out);
}

//----( flock viewer )--------------------------------------------------------

FlockViewer::FlockViewer (
    Rectangle shape,
    size_t voice_count,
    size_t pitch_size,
    size_t tempo_size,
    float pitch_acuity,
    float min_pitch_hz,
    float max_pitch_hz,
    float min_tempo_hz,
    float max_tempo_hz)

  : Flock(
      voice_count,
      pitch_size,
      tempo_size,
      pitch_acuity,
      min_pitch_hz,
      max_pitch_hz,
      min_tempo_hz,
      max_tempo_hz),

    Rectangle(shape),

    m_beat(0),

    m_pca_timescale(FLOCK_PCA_TIMESCALE_SEC * DEFAULT_AUDIO_FRAMERATE),
    m_pca(pitch_size, 5),

    m_select_x(pitch_size),
    m_select_y(pitch_size),
    m_selection(pitch_size),

    m_image_timescale(0.2f),
    m_value_gain(m_image_timescale),
    m_mass_gain(m_image_timescale),
    m_image(shape.size()),
    m_temp(shape.size()),

    m_latest_image_time(Seconds::now()),

    m_key_counter(),
    m_mouse_state(),
    m_wheel_position(4),
    m_angle3(),
    m_mouse_x(0.5),
    m_mouse_y(0.5)
{
  m_select_x.zero();
  m_select_y.zero();
  m_selection.zero();

  m_image.zero();
}

void FlockViewer::push (Seconds time, const StereoAudioFrame & sound_in)
{
  Flock::add_sound(active_voice(), sound_in);
}

void FlockViewer::pull (Seconds time, StereoAudioFrame & sound_out)
{
  bool selected = update_selection();

  if (m_mouse_state.left_down() and selected) {

    float beat = m_beat + m_key_counter.get();

    size_t v = active_voice();
    Flock::set_rhythm(v, beat);
    Flock::set_harmony(v, m_selection);
  }

  Flock::sample(sound_out);

  m_pca.add_sample(harmony().get_beat(), 1 / m_pca_timescale);

  plot_pitch();
}

void FlockViewer::pull (Seconds time, RgbImage & image)
{
  ASSERT_SIZE(image.red, Rectangle::size());

  float dt = max(1e-8f, time - m_latest_image_time);
  m_latest_image_time = time;
  float decay = exp(-dt / m_image_timescale);

  m_image_mutex.lock();
  image = m_image;
  m_image *= decay;
  m_image_mutex.unlock();

  const size_t R = 1;
  Image::linear_blur(width(), height(), R, image.red, m_temp);
  Image::linear_blur(width(), height(), R, image.green, m_temp);
  Image::linear_blur(width(), height(), R, image.blue, m_temp);

  float * restrict im = image;
  float scale = 2.0f / max(image);
  for (size_t i = 0, I = image.size; i < I; ++i) {
    im[i] = min(1.0f, scale * im[i]);
  }
}

void FlockViewer::plot_pitch ()
{
  const size_t I = harmony().size;
  const size_t X = Rectangle::width();
  const size_t Y = Rectangle::height();

  const float * restrict selection = m_selection;

  float pca_gain = sqrtf(I) / 8;
  const float * restrict pca_x = m_pca.component(0);
  const float * restrict pca_y = m_pca.component(1);
  const float * restrict pca_z = m_pca.component(2);
  const float * restrict pca_u = m_pca.component(3);
  const float * restrict pca_v = m_pca.component(4);

  const float ru = 1.0f;
  const float bu = cos(2 * M_PI / 3);
  const float bv = sin(2 * M_PI / 3);
  const float gu = cos(-2 * M_PI / 3);
  const float gv = sin(-2 * M_PI / 3);

  float * restrict red = m_image.red;
  float * restrict green = m_image.green;
  float * restrict blue = m_image.blue;
  float * restrict select_x = m_select_x;
  float * restrict select_y = m_select_y;

  const LinAlg::Orientation3D angle = m_angle3;

  if (m_image_mutex.try_lock()) {

    for (size_t i = 0; i < I; ++i) {
      float xyz[3] = {
        pca_x[i] * pca_gain,
        pca_y[i] * pca_gain,
        pca_z[i] * pca_gain
      };
      float u = pca_u[i] * pca_gain;
      float v = pca_v[i] * pca_gain;

      float uv_scale = 1 / (1 + sqrt(max(1e-20f, sqr(u) + sqr(v))));
      u *= uv_scale;
      v *= uv_scale;

      float s = selection[i];
      float r = max(s, sqr((1 + ru * u) / 2));
      float g = max(s, sqr((1 + gu * u + gv * v) / 2));
      float b = max(s, sqr((1 + bu * u + bv * v) / 2));

      float x = angle.coord_dot(0, xyz);
      float y = angle.coord_dot(1, xyz);

      float x_01 = select_x[i] = (1 + x) / 2;
      float y_01 = select_y[i] = (1 + y) / 2;

      BilinearInterpolate lin(x_01 * X, X, y_01 * Y, Y);
      lin.imax(red, r);
      lin.imax(green, g);
      lin.imax(blue, b);
    }

    m_image_mutex.unlock();
  }
}

bool FlockViewer::update_selection ()
{
  const float x0 = m_mouse_x;
  const float y0 = m_mouse_y;
  const float radius = exp(1.0f + m_wheel_position.get_scaled())
                    / sqrtf(harmony().size);
  const float inv_R2 = 1.0f / sqr(radius);

  const float * restrict sx = m_select_x;
  const float * restrict sy = m_select_y;
  float * restrict s = m_selection;

  float total = 0.0f;
  for (size_t i = 0, I = m_selection.size; i < I; ++i) {
    float r2 = sqr(sx[i] - x0) + sqr(sy[i] - y0);
    total += s[i] = max(0.0f, 1.0f - r2 * inv_R2);
  }

  return total > 0.1f;
}

void FlockViewer::keyboard (const SDL_KeyboardEvent & event)
{
  m_key_counter.update(event);
}

void FlockViewer::mouse_motion (const SDL_MouseMotionEvent & event)
{
  m_mouse_mutex.lock();

  if (m_mouse_state.right_down()) {
    float dx = M_PI * event.yrel / width();
    float dy = M_PI * event.xrel / height();
    m_angle3.drag(dx, dy);
  }

  m_mouse_x = event.y * 1.0f / width();
  m_mouse_y = event.x * 1.0f / height();

  m_mouse_mutex.unlock();
}

void FlockViewer::mouse_button (const SDL_MouseButtonEvent & event)
{
  m_mouse_mutex.lock();

  m_mouse_state.update(event);
  m_wheel_position.update(event);

  m_mouse_x = event.y * 1.0f / width();
  m_mouse_y = event.x * 1.0f / height();

  m_mouse_mutex.unlock();
}

} // namespace Streaming

