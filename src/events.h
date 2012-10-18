
#ifndef KAZOO_EVENTS_H
#define KAZOO_EVENTS_H

#include "common.h"
#include <SDL/SDL_events.h>
#include <SDL/SDL_mouse.h>

//----( sdl events )----------------------------------------------------------

bool key_pressed ();

SDLKey wait_for_keypress ();

void sdl_event_loop ();

//----( event handler )----

class EventHandler
{
  static EventHandler * s_unique_instance;
  static size_t s_key_count;
  static size_t s_motion_count;
  static size_t s_button_count;

public:

  class ButtonState
  {
    Uint8 m_state;

  public:

    ButtonState () : m_state(0) {}

    bool any_down () const { return m_state; }
    bool left_down () const { return m_state & SDL_BUTTON_LMASK; }
    bool middle_down () const { return m_state & SDL_BUTTON_MMASK; }
    bool right_down () const { return m_state & SDL_BUTTON_RMASK; }

    void update (const SDL_MouseButtonEvent & event);
  };

  class WheelPosition
  {
    const int m_radius;
    int m_position;

  public:

    WheelPosition (int radius = 9999)
      : m_radius(radius),
        m_position(0)
    {
      ASSERT_LT(0, radius);
    }

    int get () const { return m_position; }
    float get_scaled () const { return m_position * 1.0f / m_radius; }
    int pop () { int result = m_position; m_position = 0; return result; }

    void update (const SDL_MouseButtonEvent & event);
  };

  EventHandler ();
  virtual ~EventHandler ();

  static void handle (const SDL_KeyboardEvent & event);
  static void handle (const SDL_MouseMotionEvent & event);
  static void handle (const SDL_MouseButtonEvent & event);

protected:

  virtual void keyboard (const SDL_KeyboardEvent & event) {}
  virtual void mouse_motion (const SDL_MouseMotionEvent & event) {}
  virtual void mouse_button (const SDL_MouseButtonEvent & event) {}
};

#endif // KAZOO_EVENTS_H

