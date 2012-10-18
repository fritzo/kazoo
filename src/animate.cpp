
#include "animate.h"
#include "vectors.h"

#define BPP 4
#define DEPTH 32

//----( screen class )--------------------------------------------------------

Screen::Screen ()
  : Rectangle(0,0),
    m_surface(NULL)
{
  if (int info = SDL_Init(SDL_INIT_VIDEO) < 0) {
    ERROR("SDL_Init failed with error " << info);
  }

  Uint32 flags = SDL_HWSURFACE | SDL_HWACCEL | SDL_FULLSCREEN;
  SDL_Rect ** modes = SDL_ListModes(NULL, flags);
  m_surface = SDL_SetVideoMode(modes[0]->w, modes[0]->h, DEPTH, flags);
  if (m_surface == NULL) {
    SDL_Quit();
    ERROR("SDL_SetVideoMode failed");
  }

  m_width  = m_surface->w;
  m_height = m_surface->h;
}

Screen::Screen (
    Rectangle(shape),
    const char * long_title,
    const char * short_title)

  : Rectangle(shape),
    m_surface(NULL)
{
  if (int info = SDL_Init(SDL_INIT_VIDEO) < 0) {
    ERROR("SDL_Init failed with error " << info);
  }

  Uint32 flags = SDL_HWSURFACE | SDL_HWACCEL;
  m_surface = SDL_SetVideoMode(width(), height(), DEPTH, flags);
  if (m_surface == NULL) {
    SDL_Quit();
    ERROR("SDL_SetVideoMode failed");
  }

  SDL_WM_SetCaption(long_title, short_title);
}

Screen::~Screen()
{
  SDL_FreeSurface(m_surface);

  SDL_Quit();
}

void Screen::set_title (
    const char * long_title,
    const char * short_title)
{
  SDL_WM_SetCaption(long_title, short_title ? short_title : long_title);
}

//----( color maps )----------------------------------------------------------

/** Color maps are all bernstein polynomials mixing various colors. */

inline uint32_t map_mono (SDL_PixelFormat * format, uint8_t y)
{
  return SDL_MapRGB(format, y, y, y);
}

//wrapper for SDL's RGB converter
inline uint32_t map_rgb (SDL_PixelFormat * format, float r, float g, float b)
{
  return SDL_MapRGB(
      format,
      static_cast<Uint8>(255 * r),
      static_cast<Uint8>(255 * g),
      static_cast<Uint8>(255 * b));
}

// black - white
inline uint32_t color_map2 (float x, SDL_PixelFormat * format)
{
  return map_rgb(format, x,x,x);
}

// black - blue - red - white
inline uint32_t color_map4 (float x, SDL_PixelFormat * format)
{
  float y = 1-x;
  float w = x*x*x;
  float r = w + 3 * x*x * y;
  float g = w;
  float b = w + 3 * x * y*y;
  return map_rgb(format, r,g,b);
}

// black - blue - green - red - white
inline uint32_t color_map5 (float x, SDL_PixelFormat * format)
{
  float y = 1-x;
  float w = x*x*x*x;
  float r = w + 4 * x*x*x * y;
  float g = w + 6 * x*x * y*y;
  float b = w + 4 * x * y*y*y;
  return map_rgb(format, r,g,b);
}

//----( drawing functions )---------------------------------------------------

void Screen::draw (const Vector<uint8_t> & data, bool transpose)
{
  ASSERT_SIZE(data, size());

  if(SDL_MUSTLOCK(m_surface)) {
      if(SDL_LockSurface(m_surface) < 0) return;
  }

  SDL_PixelFormat * format = m_surface->format;
  uint32_t * restrict pixels = static_cast<uint32_t *>(m_surface->pixels);

  if (transpose) {
    for (size_t x = 0, X = m_width; x < X; ++x) {
      for (size_t y = 0, Y = m_height; y < Y; ++y) {
        size_t xy = x + y * X;
        size_t xy1 = Y * (x+1) - (y+1);
        uint8_t gray = data[xy1];
        pixels[xy] = SDL_MapRGB(format, gray,gray,gray);
      }
    }
  } else {
    for (size_t xy = 0, XY = size(); xy < XY; ++xy) {
      uint8_t gray = data[xy];
      pixels[xy] = SDL_MapRGB(format, gray,gray,gray);
    }
  }

  if(SDL_MUSTLOCK(m_surface)) SDL_UnlockSurface(m_surface);

  // instead of updating here, caller must manually .update()
  //SDL_UpdateRect(m_surface, 0, 0, m_surface->w, m_surface->h);
}

void Screen::draw (
    const Vector<uint8_t> & red,
    const Vector<uint8_t> & green,
    const Vector<uint8_t> & blue,
    bool transpose)
{
  ASSERT_SIZE(red, size());
  ASSERT_SIZE(green, size());
  ASSERT_SIZE(blue, size());

  if(SDL_MUSTLOCK(m_surface)) {
      if(SDL_LockSurface(m_surface) < 0) return;
  }

  SDL_PixelFormat * format = m_surface->format;
  uint32_t * restrict pixels = static_cast<uint32_t *>(m_surface->pixels);

  if (transpose) {
    for (size_t x = 0, X = m_width; x < X; ++x) {
      for (size_t y = 0, Y = m_height; y < Y; ++y) {
        size_t xy = x + y * X;
        size_t xy1 = Y * (x+1) - (y+1);
        pixels[xy] = SDL_MapRGB(format, red[xy1],green[xy1],blue[xy1]);
      }
    }
  } else {
    for (size_t xy = 0, XY = size(); xy < XY; ++xy) {
        pixels[xy] = SDL_MapRGB(format, red[xy],green[xy],blue[xy]);
    }
  }

  if(SDL_MUSTLOCK(m_surface)) SDL_UnlockSurface(m_surface);

  // instead of updating here, caller must manually .update()
  //SDL_UpdateRect(m_surface, 0, 0, m_surface->w, m_surface->h);

}

void Screen::draw (const Vector<float> & data, bool transpose)
{
  ASSERT_SIZE(data, size());

  if(SDL_MUSTLOCK(m_surface)) {
      if(SDL_LockSurface(m_surface) < 0) return;
  }

  SDL_PixelFormat * format = m_surface->format;
  uint32_t * restrict pixels = static_cast<uint32_t *>(m_surface->pixels);

  if (transpose) {
    for (size_t x = 0, X = m_width; x < X; ++x) {
      for (size_t y = 0, Y = m_height; y < Y; ++y) {
        size_t xy = x + y * X;
        size_t xy1 = Y * (x+1) - (y+1);
        pixels[xy] = color_map5(data[xy1], format);
      }
    }
  } else {
    for (size_t xy = 0, XY = size(); xy < XY; ++xy) {
      pixels[xy] = color_map5(data[xy], format);
    }
  }

  if(SDL_MUSTLOCK(m_surface)) SDL_UnlockSurface(m_surface);

  // instead of updating here, caller must manually .update()
  //SDL_UpdateRect(m_surface, 0, 0, m_surface->w, m_surface->h);
}

void Screen::draw (
    const Vector<float> & red,
    const Vector<float> & green,
    const Vector<float> & blue,
    bool transpose)
{
  ASSERT_SIZE(red, size());
  ASSERT_SIZE(green, size());
  ASSERT_SIZE(blue, size());

  if(SDL_MUSTLOCK(m_surface)) {
      if(SDL_LockSurface(m_surface) < 0) return;
  }

  SDL_PixelFormat * format = m_surface->format;
  uint32_t * restrict pixels = static_cast<uint32_t *>(m_surface->pixels);

  if (transpose) {
    for (size_t x = 0, X = m_width; x < X; ++x) {
      for (size_t y = 0, Y = m_height; y < Y; ++y) {
        size_t xy = x + y * X;
        size_t xy1 = Y * (x+1) - (y+1);
        pixels[xy] = map_rgb(format, red[xy1], green[xy1], blue[xy1]);
      }
    }
  } else {
    for (size_t xy = 0, XY = size(); xy < XY; ++xy) {
      pixels[xy] = map_rgb(format, red[xy], green[xy], blue[xy]);
    }
  }

  if(SDL_MUSTLOCK(m_surface)) SDL_UnlockSurface(m_surface);

  // instead of updating here, caller must manually .update()
  //SDL_UpdateRect(m_surface, 0, 0, m_surface->w, m_surface->h);
}

void Screen::vertical_sweep (const Vector<float> & data, bool update)
{
  ASSERT_SIZE(data, m_height);

  static size_t sweep_position = 0;
  if(SDL_MUSTLOCK(m_surface)) { if(SDL_LockSurface(m_surface) < 0) return; }

  SDL_PixelFormat * format = m_surface->format;
  uint32_t * restrict pixels = static_cast<uint32_t *>(m_surface->pixels);
  pixels += sweep_position;

  for (size_t y = 0; y < m_height; ++y) {
    pixels[(m_height - y - 1) * m_width] = color_map5(data[y], format);
  }

  if(SDL_MUSTLOCK(m_surface)) SDL_UnlockSurface(m_surface);

  if (update) SDL_UpdateRect(m_surface, sweep_position, 0, 1, m_surface->h);

  sweep_position = (1 + sweep_position) % m_width;
}

void Screen::vertical_sweep (const Vector<uint8_t> & data, bool update)
{
  ASSERT_SIZE(data, m_height);

  static size_t sweep_position = 0;
  if(SDL_MUSTLOCK(m_surface)) { if(SDL_LockSurface(m_surface) < 0) return; }

  SDL_PixelFormat * format = m_surface->format;
  uint32_t * restrict pixels = static_cast<uint32_t *>(m_surface->pixels);
  pixels += sweep_position;

  for (size_t y = 0; y < m_height; ++y) {
    pixels[(m_height - y - 1) * m_width] = map_mono(format, data[y]);
  }

  if(SDL_MUSTLOCK(m_surface)) SDL_UnlockSurface(m_surface);

  if (update) SDL_UpdateRect(m_surface, sweep_position, 0, 1, m_surface->h);

  sweep_position = (1 + sweep_position) % m_width;
}

void Screen::horizontal_sweep (const Vector<float> & data, bool update)
{
  ASSERT_SIZE(data, m_width);

  static size_t sweep_position = 0;
  if(SDL_MUSTLOCK(m_surface)) { if(SDL_LockSurface(m_surface) < 0) return; }

  SDL_PixelFormat * format = m_surface->format;
  uint32_t * restrict pixels = static_cast<uint32_t *>(m_surface->pixels);
  pixels += sweep_position * m_width;

  for (size_t x = 0; x < m_width; ++x) {
    pixels[x] = color_map5(data[x], format);
  }

  if(SDL_MUSTLOCK(m_surface)) SDL_UnlockSurface(m_surface);

  if (update) SDL_UpdateRect(m_surface, 0, sweep_position, m_surface->w, 1);

  sweep_position = (1 + sweep_position) % m_height;
}

void Screen::horizontal_sweep (const Vector<uint8_t> & data, bool update)
{
  ASSERT_SIZE(data, m_width);

  static size_t sweep_position = 0;
  if(SDL_MUSTLOCK(m_surface)) { if(SDL_LockSurface(m_surface) < 0) return; }

  SDL_PixelFormat * format = m_surface->format;
  uint32_t * restrict pixels = static_cast<uint32_t *>(m_surface->pixels);
  pixels += sweep_position * m_width;

  for (size_t x = 0; x < m_width; ++x) {
    pixels[x] = map_mono(format, data[x]);
  }

  if(SDL_MUSTLOCK(m_surface)) SDL_UnlockSurface(m_surface);

  if (update) SDL_UpdateRect(m_surface, 0, sweep_position, m_surface->w, 1);

  sweep_position = (1 + sweep_position) % m_height;
}

void Screen::draw_point (int x, int y, float value)
{
  if ((x < 0) or (y < 0) or (x >= (int)m_width)
                         or (y >= (int)m_height)) return;

  if(SDL_MUSTLOCK(m_surface)) { if(SDL_LockSurface(m_surface) < 0) return; }

  SDL_PixelFormat * format = m_surface->format;
  uint32_t color = color_map5(value, format);
  uint32_t * restrict pixels = static_cast<uint32_t *>(m_surface->pixels);

  pixels[x + (m_height - y - 1) * m_width] = color;

  if(SDL_MUSTLOCK(m_surface)) SDL_UnlockSurface(m_surface);

  //SDL_UpdateRect(m_surface, x, m_height - y - 1, 1, 1);
}

void Screen::draw_cross (int x0, int y0, int radius, float value)
{
  ASSERT_LE(0, radius);
  if ((x0 < 0) or (y0 < 0) or (x0 >= (int)m_width)
                           or (y0 >= (int)m_height)) return;

  if(SDL_MUSTLOCK(m_surface)) { if(SDL_LockSurface(m_surface) < 0) return; }

  size_t min_x = max(0, x0 - radius);
  size_t max_x = min((int) m_width - 1, x0 + radius);
  size_t min_y = max(0, y0 - radius);
  size_t max_y = min((int) m_height - 1, y0 + radius);

  SDL_PixelFormat * format = m_surface->format;
  uint32_t color = color_map5(value, format);
  uint32_t * restrict pixels = static_cast<uint32_t *>(m_surface->pixels);

  for (size_t y = min_y; y <= max_y; ++y) {
    pixels[x0 + (m_height - y - 1) * m_width] = color;
  }
  for (size_t x = min_x; x <= max_x; ++x) {
    pixels[x + (m_height - y0 - 1) * m_width] = color;
  }

  if(SDL_MUSTLOCK(m_surface)) SDL_UnlockSurface(m_surface);

  //SDL_UpdateRect(m_surface, min_x,         m_height - min_y - 1,
  //                          1+max_x-min_x, 1+max_y-min_y);
}

void Screen::draw_box (int x0, int y0, int radius, float value)
{
  ASSERT_LE(0, radius);
  if ((x0 < 0) or (y0 < 0) or (x0 >= (int)m_width)
                           or (y0 >= (int)m_height)) return;

  if(SDL_MUSTLOCK(m_surface)) { if(SDL_LockSurface(m_surface) < 0) return; }

  size_t min_x = max(0, x0 - radius);
  size_t max_x = min((int) m_width - 1, x0 + radius);
  size_t min_y = max(0, y0 - radius);
  size_t max_y = min((int) m_height - 1, y0 + radius);

  SDL_PixelFormat * format = m_surface->format;
  uint32_t color = color_map5(value, format);
  uint32_t * restrict pixels = static_cast<uint32_t *>(m_surface->pixels);

  for (size_t y = min_y; y <= max_y; ++y) {
    int x1 = max(0, x0 - radius);
    int x2 = min((int) m_width - 1, x0 + radius);

    pixels[x1 + (m_height - y - 1) * m_width] = color;
    pixels[x2 + (m_height - y - 1) * m_width] = color;
  }
  for (size_t x = min_x + 1; x < max_x; ++x) {
    int y1 = max(0, y0 - radius);
    int y2 = min((int) m_height - 1, y0 + radius);

    pixels[x + (m_height - y1 - 1) * m_width] = color;
    pixels[x + (m_height - y2 - 1) * m_width] = color;
  }

  if(SDL_MUSTLOCK(m_surface)) SDL_UnlockSurface(m_surface);

  //SDL_UpdateRect(m_surface, min_x,         m_height - min_y - 1,
  //                          1+max_x-min_x, 1+max_y-min_y);
}

void Screen::add_blob (
    int x0,
    int y0,
    float radius,
    float value,
    float * pixels)
{
  const int R = 3 * radius;

  int min_x = max(0, x0 - R);
  int max_x = min((int) m_width - 1, x0 + R);
  int min_y = max(0, y0 - R);
  int max_y = min((int) m_height - 1, y0 + R);

  if ((min_x == max_x) or (min_y == max_y)) return;

  for (int x = min_x; x <= max_x; ++x)
  for (int y = min_y; y <= max_y; ++y)
  {
    float z = value * expf(-0.5f * (sqr(x - x0) + sqr(y - y0)) / sqr(radius));
    pixels[x + (m_height - y - 1) * m_width] += z;
  }
}

//----( screen threads )------------------------------------------------------

void ScreenThread::start ()
{
  ASSERT(not m_running, "started ScreenThread twice");
  m_time = Seconds::now();
  m_running = true;
  Thread::start();
}

void ScreenThread::stop ()
{
  ASSERT(m_running, "stopped ScreenThread twice");
  m_running = false;
  Thread::wait();
}

void ScreenThread::run ()
{
  while (m_running) {
    m_time += m_delay;
    while (Seconds::now() < m_time) usleep(3000);
    process(m_time);
  }
}

