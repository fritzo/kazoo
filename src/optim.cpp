
#include "optim.h"
#include <algorithm>

#define LOG1(message)

float minimize_grid_search (const Function & f, float x0, float dx)
{
  float y0 = f(x0);
  //PRINT2(x0, f(x0))

  // search below x0
  float x = x0 - dx;
  float y = f(x);
  while (y < y0) {
    x0 = x;
    y0 = y;
    //PRINT2(x0, f(x0))
    x = x0 - dx;
    y = f(x);
  }

  // search above x0
  x = x0 + dx;
  y = f(x);
  while (y < y0) {
    x0 = x;
    y0 = y;
    //PRINT2(x0, f(x0))
    x = x0 + dx;
    y = f(x);
  }

  return x0;
}

float minimize_bisection_search (
    const Function & f,
    float x0,
    float x1,
    float dx)
{
  if (x0 > x1) std::swap(x0, x1);

  while (x1 > x0 + dx) {

    float x = 0.5f * (x0 + x1);
    //PRINT2(x, f(x))

    float dy = f(x + dx) - f(x - dx);
    (dy > 0 ? x1 : x0) = x;
    //PRINT(dy)
  }

  return 0.5f * (x0 + x1);
}

