
#include "common.h"
#include "animate.h"
#include "events.h"

inline void chaotic_logistic_map (float & x) { x = 3.5699456 * x * (1-x); }

void animate_test (size_t size)
{
  Vector<float> buffer(size);

  for (size_t i = 0; i < buffer.size; ++i) {
    float t = (0.5f + i) / buffer.size;
    buffer[i] = t;
  }

  Screen screen(Rectangle(2 * size, size));

  SDL_Event event;
  int keypress = 0;
  while (not keypress) {

    screen.vertical_sweep(buffer);

    for (size_t i = 0; i < buffer.size; ++i) {
      // chaotic_logistic_map(buffer[i]);
      float t = (0.5 * i) / buffer.size / M_PI;
      buffer[i] = fmod(buffer[i] + t, 1);
    }

    while(SDL_PollEvent(&event)) {
      switch (event.type) {
        case SDL_QUIT:      keypress = 1;   break;
        case SDL_KEYDOWN:   keypress = 1;   break;
      }
    }
  }
}

int main ()
{
  animate_test ();

  return 0;
}

