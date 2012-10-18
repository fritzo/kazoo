
#include "events.h"

//----( sdl events )----------------------------------------------------------

bool key_pressed ()
{
  SDL_Event event;

  while (SDL_PollEvent(& event)) {
    switch (event.type) {
      case SDL_QUIT:      return true;
      case SDL_KEYDOWN:   return true;
    }
  }

  return false;
}

SDLKey wait_for_keypress ()
{
  SDL_Event event;

  while (true) {
    SDL_WaitEvent(& event);
    switch (event.type) {
      case SDL_QUIT:    return SDLK_ESCAPE;
      case SDL_KEYDOWN: return event.key.keysym.sym;
    }
  }
}

void sdl_event_loop ()
{
  SDL_Event event;

  while (true) {

    SDL_WaitEvent(& event);

    switch (event.type) {

      case SDL_QUIT:
        return;

      case SDL_KEYDOWN:
        // for other SDLKey codes, see
        // http://www.libsdl.org/cgi/docwiki.cgi/SDLKey
        switch (event.key.keysym.sym) {
          case SDLK_ESCAPE: return;
          case SDLK_RETURN: return;
          default:
            EventHandler::handle(event.key);
        }
        break;

      case SDL_KEYUP:
        EventHandler::handle(event.key);
        break;

      case SDL_MOUSEMOTION:
        EventHandler::handle(event.motion);
        break;

      case SDL_MOUSEBUTTONDOWN:
      case SDL_MOUSEBUTTONUP:
        EventHandler::handle(event.button);
        break;


      default:
        break;
    }
  }
}

//----( event handler )----

EventHandler * EventHandler::s_unique_instance = NULL;

size_t EventHandler::s_key_count = 0;
size_t EventHandler::s_motion_count = 0;
size_t EventHandler::s_button_count = 0;

EventHandler::EventHandler ()
{
  ASSERT(s_unique_instance == NULL, "duplicate mouse handler");
  s_unique_instance = this;
}
EventHandler::~EventHandler ()
{
  ASSERT(s_unique_instance == this, "zombie mouse handler");
  s_unique_instance = NULL;

  PRINT3(s_key_count, s_motion_count, s_button_count);
}

void EventHandler::handle (const SDL_KeyboardEvent & event)
{
  if (s_unique_instance) {
    s_unique_instance->keyboard(event);
    ++s_key_count;
  }
}

void EventHandler::handle (const SDL_MouseMotionEvent & event)
{
  if (s_unique_instance) {
    s_unique_instance->mouse_motion(event);
    ++s_motion_count;
  }
}

void EventHandler::handle (const SDL_MouseButtonEvent & event)
{
  if (s_unique_instance) {
    s_unique_instance->mouse_button(event);
    ++s_button_count;
  }
}

void EventHandler::ButtonState::update (const SDL_MouseButtonEvent & event)
{
  switch (event.type) {

    case SDL_MOUSEBUTTONDOWN:
      cout << 'D' << flush;
      switch (event.button) {

        case SDL_BUTTON_LEFT:
        case SDL_BUTTON_MIDDLE:
        case SDL_BUTTON_RIGHT:
          m_state |= SDL_BUTTON(event.button);
          break;

        default:
          break;
      }
      break;

    case SDL_MOUSEBUTTONUP:
      cout << 'U' << flush;
      switch (event.button) {

        case SDL_BUTTON_LEFT:
        case SDL_BUTTON_MIDDLE:
        case SDL_BUTTON_RIGHT:
          m_state &= ~SDL_BUTTON(event.button);
          break;

        default:
          break;
      }
      break;

    default:
      break;
  }
}

void EventHandler::WheelPosition::update (const SDL_MouseButtonEvent & event)
{
  switch (event.button) {

    case SDL_BUTTON_WHEELUP:
      if (event.type == SDL_MOUSEBUTTONDOWN) {
        m_position = min(m_radius, m_position + 1);
      }
      break;

    case SDL_BUTTON_WHEELDOWN:
      if (event.type == SDL_MOUSEBUTTONDOWN) {
        m_position = max(-m_radius, m_position - 1);
      }
      break;

    default:
      break;
  }
}

