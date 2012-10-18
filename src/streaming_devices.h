
#ifndef KAZOO_STREAMING_DEVICES_H
#define KAZOO_STREAMING_DEVICES_H

#include "common.h"
#include "streaming.h"
//#include "streaming_audio.h"
#include "streaming_video.h"
#include "animate.h"
#include "events.h"
#include "audio.h"

namespace Streaming
{

//----( screens )-------------------------------------------------------------

class ScreenThread : public TimedThread
{
protected:

  Screen m_screen;

public:

  ScreenThread (
      Rectangle shape,
      float framerate = DEFAULT_SCREEN_FRAMERATE);
  virtual ~ScreenThread ();

  size_t size () const { return m_screen.size(); }
  size_t width () const { return m_screen.height(); }
  size_t height () const { return m_screen.width(); }
};

class ShowMono : public ScreenThread
{
  MonoImage m_image;

public:

  RectangularPort<Pulled<MonoImage> > in;

  ShowMono (Rectangle shape)
    : ScreenThread(shape),
      m_image(shape.size()),
      in("ShowMono.in", shape)
  {}
  virtual ~ShowMono () {}

protected:

  virtual void step ();
};

class ShowRgb : public ScreenThread
{
  RgbImage m_image;

public:

  RectangularPort<Pulled<RgbImage> > in;

  ShowRgb (Rectangle shape)
    : ScreenThread(shape),
      m_image(shape.size()),
      in("ShowRgb.in", shape)
  {}
  virtual ~ShowRgb () {}

protected:

  virtual void step ();
};

class ShowMono8 : public ScreenThread
{
  Mono8Image m_image;

public:

  RectangularPort<Pulled<Mono8Image> > in;

  ShowMono8 (Rectangle shape)
    : ScreenThread(shape),
      m_image(shape.size()),
      in("ShowMono8.in", shape)
  {}
  virtual ~ShowMono8 () {}

protected:

  virtual void step ();
};

class ShowRgb8 : public ScreenThread
{
  Rgb8Image m_image;

public:

  RectangularPort<Pulled<Rgb8Image> > in;

  ShowRgb8 (Rectangle shape)
    : ScreenThread(shape),
      m_image(shape.size()),
      in("ShowRgb8.in", shape)
  {}
  virtual ~ShowRgb8 () {}

protected:

  virtual void step ();
};

class GraphValue : public ScreenThread
{
  float m_UB;

  Vector<float> m_bar;

public:

  Port<Pulled<float> > in;

  GraphValue (Rectangle shape)

    : ScreenThread(shape),

      m_UB(0),
      m_bar(shape.width()),

      in("GraphValue.in")
  {}
  virtual ~GraphValue () {}

protected:

  virtual void step ();
};

class SweepVector
  : public Pushed<Vector<float> >,
    public Pushed<Vector<uint8_t> >
{
  Screen m_screen;
  const bool m_transposed;

public:

  SweepVector (size_t size, size_t duration, bool transposed = false);
  virtual ~SweepVector ();

  virtual void push (Seconds time, const Vector<float> & vector)
  {
    if (m_transposed) m_screen.horizontal_sweep(vector);
    else              m_screen.vertical_sweep(vector);
  }
  virtual void push (Seconds time, const Vector<uint8_t> & vector)
  {
    if (m_transposed) m_screen.horizontal_sweep(vector);
    else              m_screen.vertical_sweep(vector);
  }
};

//----( wrappers )----

class ShowMonoZoom
{
  NormalizeTo01 m_normalize;
  ZoomMono m_zoom;
  ShowMono m_screen;

public:

  RectangularPort<Pulled<MonoImage> > & in;

  ShowMonoZoom (
      Rectangle shape_in,
      Rectangle screen_shape = Rectangle(0,0))

    : m_normalize(shape_in),
      m_zoom(shape_in, screen_shape.size() ? screen_shape : shape_in),
      m_screen(m_zoom.out),

      in(m_normalize.in)
  {
    m_zoom.in - m_normalize;
    m_screen.in - m_zoom;
  }
};

class ShowRgbZoom
{
  ZoomRgb m_zoom;
  ShowRgb m_screen;

public:

  RectangularPort<Pulled<RgbImage> > & in;

  ShowRgbZoom (
      Rectangle shape_in,
      Rectangle screen_shape = Rectangle(0,0))

    : m_zoom(shape_in, screen_shape.size() ? screen_shape : shape_in),
      m_screen(m_zoom.out),

      in(m_zoom.in)
  {
    m_screen.in - m_zoom;
  }
};

class ShowMono8Zoom
{
  ZoomMono8 m_zoom;
  ShowMono8 m_screen;

public:

  RectangularPort<Pulled<Mono8Image> > & in;

  ShowMono8Zoom (
      Rectangle shape_in,
      Rectangle screen_shape = Rectangle(0,0))

    : m_zoom(shape_in, screen_shape.size() ? screen_shape : shape_in),
      m_screen(m_zoom.out),

      in(m_zoom.in)
  {
    m_screen.in - m_zoom;
  }
};

class ShowRgb8Zoom
{
  ZoomRgb8 m_zoom;
  ShowRgb8 m_screen;

public:

  RectangularPort<Pulled<Rgb8Image> > & in;

  ShowRgb8Zoom (
      Rectangle shape_in,
      Rectangle screen_shape = Rectangle(0,0))

    : m_zoom(shape_in, screen_shape.size() ? screen_shape : shape_in),
      m_screen(m_zoom.out),

      in(m_zoom.in)
  {
    m_screen.in - m_zoom;
  }
};

//----( audio )---------------------------------------------------------------

//----( threads )----

class AudioThread
  : public Thread,
    protected ::AudioThread
{
public:

  AudioThread (bool listening, bool speaking, bool stereo)
    : ::AudioThread(
        DEFAULT_FRAMES_PER_BUFFER,
        DEFAULT_SAMPLE_RATE,
        listening,
        speaking,
        stereo)
  {}
  virtual ~AudioThread () {}

  bool deaf () const { return not reading(); }
  bool mute () const { return not writing(); }

  virtual void start () { ::AudioThread::start(); }
  virtual void stop () { ::AudioThread::stop(); }
  virtual void wait () {}

protected:

  virtual void step () { ERROR("AudioThread::step() should not be called"); }
};

class MonoAudioThread : public AudioThread
{
public:

  Port<Pushed<MonoAudioFrame> > out;
  Port<Pulled<MonoAudioFrame> > in;
  Port<Bounced<MonoAudioFrame> > io;

  MonoAudioThread (bool listening = true, bool speaking = true)
    : AudioThread(listening, speaking, false),

      out("MonoAudioThread.out"),
      in("MonoAudioThread.in"),
      io("MonoAudioThread.io")
  {}
  virtual ~MonoAudioThread () {}

protected:

  virtual void process (
      const float * restrict samples_in,
      float * restrict samples_out,
      size_t size);
};

class StereoAudioThread : public AudioThread
{
public:

  Port<Pushed<StereoAudioFrame> > out;
  Port<Pulled<StereoAudioFrame> > in;
  Port<Bounced<StereoAudioFrame> > io;

  StereoAudioThread (bool listening = true, bool speaking = true)
    : AudioThread(listening, speaking, true),

      out("StereoAudioThread.out"),
      in("StereoAudioThread.in"),
      io("StereoAudioThread.io")
  {}
  virtual ~StereoAudioThread () {}

protected:

  virtual void process (
      const complex * restrict samples_in,
      complex * restrict samples_out,
      size_t size);
};

// TODO factor into stereo and mono classes

//----( files )----

class MonoAudioFile
  : public AudioFile,
    public Pulled<MonoAudioFrame>,
    public Thread
{
  MonoAudioFrame m_sound;

public:

  Port<Pushed<MonoAudioFrame> > out;

  MonoAudioFile (string filename)
    : AudioFile(filename, false),
      out("MonoAudioFile.out")
  {}
  virtual ~MonoAudioFile () {}

  virtual void pull (Seconds time, MonoAudioFrame & sound)
  {
    if (done()) sound.zero(); else read_frame(sound);
  }

  virtual void step () { ERROR("MonoAudioFile::step should not be called"); }

  virtual void run ();
  size_t run (float rate);
};

class StereoAudioFile
  : public AudioFile,
    public Pulled<StereoAudioFrame>,
    public Thread
{
  StereoAudioFrame m_sound;

public:

  Port<Pushed<StereoAudioFrame> > out;

  StereoAudioFile (string filename)
    : AudioFile(filename, true),
      out("StereoAudioFile.out")
  {}
  virtual ~StereoAudioFile () {}

  virtual void pull (Seconds time, StereoAudioFrame & sound)
  {
    if (done()) sound.zero(); else read_frame(sound);
  }

  virtual void step () { ERROR("StereoAudioFile::step should not be called"); }

  virtual void run ();
  size_t run (float rate);
};

//----( mouse )---------------------------------------------------------------

class MouseTest
  : public Rectangle,
    public EventHandler,
    public Pulled<MonoImage>
{
  MonoImage m_image;
  Mutex m_image_mutex;

  Seconds m_time;

  EventHandler::ButtonState m_mouse_state;

public:

  MouseTest (Rectangle shape);
  virtual ~MouseTest () {}

  virtual void pull (Seconds time, MonoImage & image);

protected:

  virtual void mouse_motion (const SDL_MouseMotionEvent & event);
  virtual void mouse_button (const SDL_MouseButtonEvent & event);
};

} // namespace Streaming

#endif // KAZOO_STREAMING_DEVICES_H

