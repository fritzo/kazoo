
#include "dirt.h"
#include "images.h"
#include "splines.h"

#define DIRT_POWER_TIMESCALE            (2.0f / DEFAULT_VIDEO_FRAMERATE)

#define DIRT_TEMPO_TIMESCALE            (30.0f)
#define DIRT_TEMPO_MIN_FREQ             (0.2f)
#define DIRT_TEMPO_MAX_FREQ             (3.2f)
#define DIRT_TEMPO_ACUITY               (3.0f)

#define DIRT_PITCH_TIMESCALE            (8.0f)
#define DIRT_PITCH_MIN_FREQ             (2e1f)
#define DIRT_PITCH_MAX_FREQ             (7e3f)
#define DIRT_PITCH_ACUITY               (7.0f)

#define TOL                             (1e-8f)

#define LOG1(message)

namespace Streaming
{

//----( control )-------------------------------------------------------------

DirtPitchControl::DirtPitchControl (Rectangle shape)
  : Rectangle(shape),

    m_image_x(width()),
    m_image_y(height()),

    impact_in("DirtPitchControl.impact_in"),
    amplitude_out("DirtPitchControl.amplitude_out", width()),
    shadow_monitor("DirtPitchControl.shadow_monitor", size())
{}

DirtPitchControl::~DirtPitchControl ()
{
  PRINT3(min(m_image_x), rms(m_image_x), max(m_image_x));
}

void DirtPitchControl::push (Seconds time, const MonoImage & image_xy)
{
  ASSERT_SIZE(image_xy, size());

  if (shadow_monitor) shadow_monitor.push(time, image_xy);

  const size_t I = width();
  const size_t J = height();

  Image::project_axes_sum(I, J, image_xy, m_image_x, m_image_y);

  if (impact_in) {
    float impact;
    impact_in.pull(time, impact);
    m_image_x *= impact;
  }

  amplitude_out.push(time, m_image_x);
}

DirtControl::DirtControl (Rectangle shape)
  : Rectangle(shape),

    m_old_image_xy(size()),
    m_image_y(height()),
    m_impact_y(height()),

    m_amplitude_y(height()),
    m_amplitude_x(width()),

    impact_in("DirtControl.impact_in"),
    tempo_io("DirtControl.tempo_io", height()),
    amplitude_out("DirtControl.amplitude_out", width())
{}

DirtControl::~DirtControl ()
{
  PRINT3(min(m_amplitude_x), rms(m_amplitude_x), max(m_amplitude_x));
  PRINT3(min(m_amplitude_y), rms(m_amplitude_y), max(m_amplitude_y));
}

void DirtControl::push (Seconds time, const MonoImage & image_xy)
{
  ASSERT_SIZE(image_xy, size());

  const size_t I = width();
  const size_t J = height();

  // detect impact & project
  const float * restrict image1_xy = image_xy;
  float * restrict image0_xy = m_old_image_xy;
  float * restrict image1_y = m_image_y;
  float * restrict impact_y = m_impact_y;

  for (size_t j = 0; j < J; ++j) {
    image1_y[j] = 0;
    impact_y[j] = 0;
  }

  for (size_t i = 0; i < I; ++i) {
    for (size_t j = 0; j < J; ++j) {
      const size_t ij = J * i + j;
      float im0 = image0_xy[ij];
      float im1 = image1_xy[ij];

      image0_xy[ij] = im1;
      image1_y[j] += im1;
      impact_y[j] += im1 * max(0.0f, im1 - im0); // proportional to dt
    }
  }

  if (impact_in) {
    float impact;
    impact_in.pull(time, impact);
    m_impact_y *= impact;
  }

  tempo_io.bounce(time, m_impact_y, m_amplitude_y);

  Image::transduce_yx(I,J, image1_xy, image1_y, m_amplitude_y, m_amplitude_x);

  amplitude_out.push(time, m_amplitude_x);
}

//----( tempo )---------------------------------------------------------------

DirtTempoSynth::DirtTempoSynth (
    size_t size,
    float expected_dt,
    bool coalesce,
    float blur_factor)

  : Synchronized::LoopBank(
      Synchronized::Bank(
          size,
          DIRT_TEMPO_MIN_FREQ, // top
          DIRT_TEMPO_MAX_FREQ, // bottom
          DIRT_TEMPO_ACUITY),
      coalesce,
      expected_dt),

    m_blur_radius(max(size_t(1), roundu(blur_factor * tone_size() / 2))),
    m_time(Seconds::now())
{
  PRINT3(m_blur_radius, size, num_tones());
  ASSERT_LT(m_blur_radius, size);
}

DirtTempoSynth::~DirtTempoSynth ()
{
  PRINT3(min(mass_now), rms(mass_now), max(mass_now));
}

void DirtTempoSynth::bounce (
    Seconds time,
    const Vector<float> & impact_in,
    Vector<float> & amplitude_out)
{
  ASSERT_SIZE(impact_in, size);
  ASSERT_SIZE(amplitude_out, size);

  float dt = max(1e-8f, time - m_time);
  m_time = time;

  float decay = exp(-dt / DIRT_TEMPO_TIMESCALE);
  ASSERT_LT(decay, 1);

  advance(dt, decay);

  const size_t I = size;
  const size_t R = m_blur_radius;

  if (R) {

    // WARNING HACK this uses impact_in as working memory
    float * restrict impact = const_cast<float *>(impact_in.data);
    float * restrict mass = mass_now;

    for (size_t i = 0; i < I; ++i) {
      float m = mass[i];
      mass[i] += impact[i];
      impact[i] = m;
    }

    // blurring allows nearby fingers to access coalesced amplitude
    Image::quadratic_blur_1d(I, R, impact, amplitude_out);

  } else {

    const float * restrict impact = impact_in;
    float * restrict amp = amplitude_out;
    float * restrict mass = mass_now;

    for (size_t i = 0; i < I; ++i) {
      amp[i] = mass[i] += impact[i];
    }
  }
}

//----( pitch )---------------------------------------------------------------

DirtPitchSynth::DirtPitchSynth (size_t size, float freq0, float freq1)
  : Synchronized::PhasorBank(Bank(
        size,
        (freq1 ? freq1 : DIRT_PITCH_MAX_FREQ) / DEFAULT_SAMPLE_RATE, // right
        (freq0 ? freq0 : DIRT_PITCH_MIN_FREQ) / DEFAULT_SAMPLE_RATE, // left
        DIRT_PITCH_ACUITY)),

    m_amplitude0(size),
    m_damplitude(size),

    amplitude_in(size),

    amplitude_monitor("DirtPitchSynth.amplitude_monitor", size),
    mass_monitor("DirtPitchSynth.mass_monitor", size),
    bend_monitor("DirtPitchSynth.bend_monitor", size)
{
  amplitude_in.unsafe_access().set(TOL);
  m_amplitude0.set(TOL);
  m_damplitude.zero();
}

DirtPitchSynth::~DirtPitchSynth ()
{
  PRINT2(rms(m_amplitude0), rms(m_damplitude));
}

void DirtPitchSynth::pull (Seconds time, StereoAudioFrame & sound_accum)
{
  amplitude_in.pull(time, m_damplitude);

  if (amplitude_monitor) amplitude_monitor.push(time, m_damplitude);

  float dt = sound_accum.size / (DIRT_PITCH_TIMESCALE * DEFAULT_SAMPLE_RATE);
  float old_part = exp(-dt);
  accumulate_step(old_part, m_mass, m_damplitude);
  retune();

  if (mass_monitor) mass_monitor.push(time, m_mass);

  m_damplitude -= m_amplitude0;
  sample_accum(m_amplitude0, m_damplitude, sound_accum);
  m_amplitude0 += m_damplitude;

  if (bend_monitor) bend_monitor.push(time, get_bend());
}

//----( keyboard )------------------------------------------------------------

void DirtKeys::pull (Seconds time, MonoImage & image_yx)
{
  ASSERT_SIZE(image_yx, size());

  const size_t I = m_image_x.size;
  const size_t J = m_image_y.size;

  bend_in.pull(time, m_image_x);

  for (size_t i = 0; i < I; ++i) {
    m_image_x[i] = sqr(m_image_x[i]);
  }

  Image::lift_axes_sum(J, I, m_image_y, m_image_x, image_yx);
}

void DirtColorKeys::pull (Seconds time, RgbImage & image_yx)
{
  ASSERT_SIZE(image_yx.red, size());

  const size_t I = height();
  const size_t J = width();

  // shadow is red
  MonoImage temp(image_yx.blue);
  shadow_in.pull(time, temp);
  Image::transpose(I, J, temp, image_yx.red);

  // all others are cyan
  amp0_in.pull(time, m_image_x);
  for (size_t j = 0; j < J/4; ++j) {
    copy_float(m_image_x, image_yx.blue + I * j, I);
    copy_float(m_image_x, image_yx.green + I * j, I);
  }

  amp1_in.pull(time, m_image_x);
  for (size_t j = J/4; j < 2*J/4; ++j) {
    copy_float(m_image_x, image_yx.blue + I * j, I);
    copy_float(m_image_x, image_yx.green + I * j, I);
  }

  mass_in.pull(time, m_image_x);
  for (size_t j = 2*J/4; j < 3*J/4; ++j) {
    copy_float(m_image_x, image_yx.blue + I * j, I);
    copy_float(m_image_x, image_yx.green + I * j, I);
  }

  bend_in.pull(time, m_image_x);
  for (size_t i = 0; i < I; ++i) {
    m_image_x[i] = fabs(m_image_x[i]);
  }
  m_image_x /= max(1e-8f, max(m_image_x));
  for (size_t j = 3*J/4; j < J; ++j) {
    copy_float(m_image_x, image_yx.green + I * j, I);
    copy_float(m_image_x, image_yx.blue + I * j, I);
  }
}

//----( systems )-------------------------------------------------------------

Dirt::Dirt (
    bool deaf,
    bool coalesce,
    float blur_factor,
    float freq0,
    float freq1)

  : m_table(deaf),
    m_points(m_table.shadow_out),
    m_impact(DIRT_POWER_TIMESCALE, TOL),
    m_control(m_points.out),
    m_tempo(
        m_points.out.height(),
        2.0f / DEFAULT_VIDEO_FRAMERATE,
        coalesce,
        blur_factor),
    m_pitch(
        m_points.out.width(),
        freq0,
        freq1)
{
  ASSERTW_LT(m_table.finger_diameter(), m_tempo.tone_size());
  ASSERTW_LT(m_table.finger_diameter(), m_pitch.tone_size());

  m_table.shadow_out - m_points;
  m_points.out - m_control;
  if (not deaf) {
    m_table.impact_out - m_impact;
    m_control.impact_in - m_impact;
  }
  m_control.tempo_io - m_tempo;
  m_control.amplitude_out - m_pitch.amplitude_in;
  m_table.sound_in - m_pitch;
}

ShowDirt::ShowDirt (
    Rectangle screen_shape,
    bool deaf,
    bool coalesce,
    float blur_factor,
    float freq0,
    float freq1)

  : Dirt(deaf, coalesce, blur_factor, freq0, freq1),
    m_keys(m_control),
    m_screen(m_keys, screen_shape)
{
  m_pitch.bend_monitor - m_keys.bend_in;
  m_screen.in - m_keys;
}

DirtPitchTest::DirtPitchTest (
    float freq0,
    float freq1,
    bool deaf,
    Rectangle screen_shape)
  : m_table(deaf),
    m_points(m_table.shadow_out),
    m_impact(TOL),
    m_control(m_points.out),
    m_pitch(m_points.out.width(), freq0, freq1),
    m_keys(m_control),
    m_screen(m_keys, screen_shape)
{
  ASSERTW_LT(m_table.finger_diameter(), m_pitch.tone_size());

  m_table.shadow_out - m_points;
  m_points.out - m_control;
  if (not deaf) {
    m_table.impact_out - m_impact;
    m_control.impact_in - m_impact;
  }
  m_control.amplitude_out - m_pitch.amplitude_in;
  m_table.sound_in - m_pitch;

  m_control.shadow_monitor - m_keys.shadow_in;
  m_pitch.amplitude_monitor - m_keys.amp0_in;
  m_pitch.mass_monitor - m_keys.mass_in;
  m_pitch.bend_monitor - m_keys.bend_in;
  m_screen.in - m_keys;
}

} // namespace Streaming

