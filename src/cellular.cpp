
#include "cellular.h"

void test_cellular (size_t X,
                    size_t Y,
                    size_t Z,
                    size_t num_cycles)
{
  Box<float> box(X,Y,Z);

  LOG("pseudorandomizing box");
  for (size_t i = 0; i < box.size; ++i) {
    box[i] = i % 61 + sqr(i) % 59;
  }

  for (size_t n = 0; n < num_cycles; ++n) {
    LOG("automaton iteration " << n);

    for (size_t x = 0; x < box.X; ++x) {
      size_t x0 = (x + box.X - 1) % box.X;
      size_t x1 = x;
      size_t x2 = (x + 1) % box.X;
      for (size_t y = 1; y + 1 < box.Y; ++y) {
        for (size_t z = 1; z + 1 < box.Z; ++z) {
          box(x,y,z) = ( box(x0,y,z)
                       + box(x1,y,z)
                       + box(x1,y+1,z)
                       + box(x1,y-1,z)
                       + box(x1,y,z+1)
                       + box(x1,y,z-1)
                       + box(x2,y,z)
                       ) / 7;
        }
      }
    }
  }

  LOG("done");
}

