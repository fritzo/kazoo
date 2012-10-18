
#ifndef KAZOO_ANIMATE_H
#define KAZOO_ANIMATE_H

#include "common.h"
#include "vectors.h"
#include "cyclic_time.h"
#include "threads.h"
#include <SDL/SDL.h>

#define DEFAULT_ANIMATE_FRAMERATE       (60.0f)

//----( screen class )--------------------------------------------------------

class Screen : public Rectangle
{
  SDL_Surface * m_surface;

public:
  Screen (); // fullscreen
  Screen (
      Rectangle shape,
      const char * long_title = "kazoo - any key exits",
      const char * short_title = "kazoo");
  ~Screen ();

  static void set_title (
      const char * long_title,
      const char * short_title = NULL);

  // animation
  void draw (const Vector<uint8_t> & data, bool transpose = false);
  void draw (
      const Vector<uint8_t> & red,
      const Vector<uint8_t> & green,
      const Vector<uint8_t> & blue,
      bool transpose = false);
  void draw (const Vector<float> & data, bool transpose = false);
  void draw (
      const Vector<float> & red,
      const Vector<float> & green,
      const Vector<float> & blue,
      bool transpose = false);
  void vertical_sweep (const Vector<float> & data, bool update = true);
  void vertical_sweep (const Vector<uint8_t> & data, bool update = true);
  void horizontal_sweep (const Vector<float> & data, bool update = true);
  void horizontal_sweep (const Vector<uint8_t> & data, bool update = true);
  void draw_point (int x, int y, float color = 1);
  void draw_cross (int x, int y, int radius = 4, float color = 1);
  void draw_box (int x, int y, int radius = 4, float color = 1);
  void add_blob (int x0, int y0, float radius, float value, float * pixels);
  void update () { SDL_UpdateRect(m_surface, 0,0,0,0); }
};

//----( screen threads )------------------------------------------------------

class ScreenThread : private Thread
{
protected:

  Screen * m_screen;

private:

  const float m_delay;
  Seconds m_time;
  bool m_running;

public:

  ScreenThread (
      Screen * screen,
      float framerate = DEFAULT_ANIMATE_FRAMERATE)

    : m_screen(screen),
      m_delay(1.0f / framerate),
      m_running(false)
  {}
  ScreenThread (
      Rectangle shape,
      float framerate = DEFAULT_ANIMATE_FRAMERATE)

    : m_screen(new Screen(shape)),
      m_delay(1.0f / framerate),
      m_running(false)
  {}
  virtual ~ScreenThread () { delete m_screen; }

  size_t width () const { return m_screen->width(); }
  size_t height () const { return m_screen->height(); }
  size_t size () const { return m_screen->size(); }
  float delay () const { return m_delay; }

  bool running () const { return m_running; }

  void start ();
  void stop ();

protected:

  virtual void process (Seconds time) = 0;

private:

  virtual void run ();
};

//----( testing )-------------------------------------------------------------

void animate_test (size_t size = 512);

#endif // KAZOO_ANIMATE_H

