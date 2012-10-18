
#include "filters.h"

using namespace Filters;

void test_NCV (size_t steps)
{
  float dt = 1.0f;
  float3 Ex(0);
  float3 Vxx(1,2,3);
  float3 Vyy(3,2,1);

  NCP<3> detection(Ex,Vxx);
  NCV<3> state(detection, Vyy);

  PRINT(detection);
  PRINT(state);
  for (size_t i = 0; i < steps; ++i) {
    state.advance(dt, Vyy);
    state.update(detection);
  }
  PRINT(state);
}

int main ()
{
  size_t num_steps = 1 << 22;

  test_NCV(num_steps);

  float average_time = get_elapsed_time() / (2 * num_steps);
  float average_rate = 1 / average_time;

  PRINT(average_time);
  PRINT(average_rate);

  return 0;
}

