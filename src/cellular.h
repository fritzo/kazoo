#ifndef KAZOO_CELLULAR_H
#define KAZOO_CELLULAR_H

#include "common.h"
#include "vectors.h"

template<class T>
class Box : public Vector<T>
{
public:

  const size_t X;
  const size_t Y;
  const size_t Z;

  Box (size_t x, size_t y, size_t z)
    : Vector<T>::Vector(x * y * z),
      X(x), Y(y), Z(z)
  {}

  T & operator() (size_t x, size_t y, size_t z)
  {
    return Vector<T>::data[z + Z * (y + Y * x)];
  }
};

void test_cellular (size_t X = 320,
                    size_t Y = 200,
                    size_t Z = 200,
                    size_t num_cycles = 10);

#endif // KAZOO_CELLULAR_H
