
#include "streaming_devices.h"

namespace Streaming
{

//----( screen )--------------------------------------------------------------

extern bool g_screen_exists;

ScreenThread::ScreenThread (
    Rectangle shape,
    float framerate)

  : TimedThread(framerate),
    m_screen(shape.transposed())
{
  ASSERT(not g_screen_exists, "SDL only supports one screen");
  g_screen_exists = true;
}

ScreenThread::~ScreenThread ()
{
  ASSERT(g_screen_exists, "no screen found when deleting ScreenThread");
  g_screen_exists = false;
}

void ShowMono::step ()
{
  Seconds time = Seconds::now();

  in.pull(time, m_image);

  m_screen.draw(m_image);
  m_screen.update();

  PROGRESS_TICKER('O');
}

void ShowRgb::step ()
{
  Seconds time = Seconds::now();

  in.pull(time, m_image);

  m_screen.draw(m_image.red, m_image.green, m_image.blue);
  m_screen.update();

  PROGRESS_TICKER('O');
}

void ShowMono8::step ()
{
  Seconds time = Seconds::now();

  in.pull(time, m_image);

  m_screen.draw(m_image);
  m_screen.update();

  PROGRESS_TICKER('O');
}

void ShowRgb8::step ()
{
  Seconds time = Seconds::now();

  in.pull(time, m_image);

  m_screen.draw(m_image.red, m_image.green, m_image.blue);
  m_screen.update();

  PROGRESS_TICKER('O');
}

void GraphValue::step ()
{
  Seconds time = Seconds::now();

  float value;
  in.pull(time, value);

  if (value > 0) {
    if (value > m_UB) {
      m_UB = value;
      value = 1;
    } else {
      value /= m_UB;
    }
  }

  PRINT(value); // DEBUG

  for (size_t i = 0, I = m_bar.size; i < I; ++i) {
    float y = (i + 0.5f) / I;
    m_bar[i] = max(0.0f, min(1.0f, 64 * (value - y)));
  }

  m_screen.vertical_sweep(m_bar);

  PROGRESS_TICKER('O')
}

SweepVector::SweepVector (size_t size, size_t duration, bool transposed)
  : m_screen(transposed ? Rectangle(size, duration)
                        : Rectangle(duration, size)),
    m_transposed(transposed)
{
  g_screen_exists = true;
}

SweepVector::~SweepVector ()
{
  ASSERT(g_screen_exists, "no screen found when deleting SweepVector");
  g_screen_exists = false;
}

//----( audio )---------------------------------------------------------------

//----( threads )----

void MonoAudioThread::process (
    const float * restrict samples_in,
    float * restrict samples_out,
    size_t size)
{
  Seconds time = Seconds::now();
  ASSERT_EQ(size, ::AudioThread::size());

  MonoAudioFrame sound_in(size, const_cast<float *>(samples_in));
  MonoAudioFrame sound_out(size, samples_out);
  sound_out.zero();

  if (reading() and writing()) {
    if (io) {
      io.bounce(time, sound_in, sound_out);
    } else {
      out.push(time, sound_in);
      in.pull(time, sound_out);
    }
  } else {
    if (reading()) out.push(time, sound_in);
    if (writing()) in.pull(time, sound_out);
  }

  PROGRESS_TICKER('-');
}

void StereoAudioThread::process (
    const complex * restrict samples_in,
    complex * restrict samples_out,
    size_t size)
{
  Seconds time = Seconds::now();
  ASSERT_EQ(size, ::AudioThread::size());

  StereoAudioFrame sound_in(size, const_cast<complex *>(samples_in));
  StereoAudioFrame sound_out(size, samples_out);
  sound_out.zero();

  if (reading() and writing()) {
    if (io) {
      io.bounce(time, sound_in, sound_out);
    } else {
      out.push(time, sound_in);
      in.pull(time, sound_out);
    }
  } else {
    if (reading()) out.push(time, sound_in);
    if (writing()) in.pull(time, sound_out);
  }

  PROGRESS_TICKER('-');
}

//----( files )----

void MonoAudioFile::run ()
{
  if (out) {
    while (m_running and not done()) {
      read_frame(m_sound);
      out.push(Seconds::now(), m_sound);
    }
  }
}

size_t MonoAudioFile::run (float rate)
{
  Seconds time = Seconds::now();
  float timestep = 1 / rate;

  Timer timer;
  size_t num_frames = 0;

  while (not done()) {

    read_frame(m_sound);

    if (out) out.push(time, m_sound);

    time += timestep;
    ++num_frames;
  }

  float speed = num_frames / timer.elapsed() / DEFAULT_AUDIO_FRAMERATE;
  LOG("processed " << num_frames << " audio frames at "
      << speed << " x realtime");

  return num_frames;
}

void StereoAudioFile::run ()
{
  if (out) {
    while (m_running and not done()) {
      read_frame(m_sound);
      out.push(Seconds::now(), m_sound);
    }
  }
}

size_t StereoAudioFile::run (float rate)
{
  Seconds time = Seconds::now();
  float timestep = 1 / rate;

  Timer timer;
  size_t num_frames = 0;

  while (not done()) {

    read_frame(m_sound);

    if (out) out.push(time, m_sound);

    time += timestep;
    ++num_frames;
  }

  float speed = num_frames / timer.elapsed() / DEFAULT_AUDIO_FRAMERATE;
  LOG("processed " << num_frames << " audio frames at "
      << speed << " x realtime");

  return num_frames;
}

//----( mouse )---------------------------------------------------------------

MouseTest::MouseTest (Rectangle shape)
  : Rectangle(shape),

    m_image(size()),
    m_time(Seconds::now())
{
  m_image.zero();
}

void MouseTest::pull (Seconds time, MonoImage & image)
{
  float dt = max(1e-8f, time - m_time);
  m_time = time;

  m_image_mutex.lock();

  m_image *= expf(-dt);
  image = m_image;

  m_image_mutex.unlock();
}

void MouseTest::mouse_motion (const SDL_MouseMotionEvent & event)
{
  if (m_mouse_state.any_down()) {
    size_t Y = height();
    size_t x = event.y;
    size_t y = event.x;

    m_image_mutex.lock();
    imax(m_image[Y * x + y], 1.0f);
    m_image_mutex.unlock();
  }
}

void MouseTest::mouse_button (const SDL_MouseButtonEvent & event)
{
  m_mouse_state.update(event);

  if (m_mouse_state.any_down()) {
    size_t Y = height();
    size_t x = event.y;
    size_t y = event.x;

    m_image_mutex.lock();
    imax(m_image[Y * x + y], 1.0f);
    m_image_mutex.unlock();
  }
}

} // namespace Streaming

