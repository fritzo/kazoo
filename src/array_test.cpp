
#include "common.h"
#include "array.h"

void simple_test (size_t num_steps)
{
  float4 v(0,0,0,0);
  float4 dv(1,2,3,4);

  PRINT(dv);
  PRINT(v);
  for (size_t i = 0; i < num_steps; ++i) {
    v += dv; // TEST_LINE
  }
  PRINT(v);
}

int main ()
{
  size_t num_steps = 100000000;

  simple_test(num_steps);

  float average_time = get_elapsed_time() / (2 * num_steps);
  float average_rate = 1 / average_time;

  PRINT(average_time);
  PRINT(average_rate);

  LOG("---------------------------------");

  PRINT(sizeof(Array<float, 0>));
  PRINT(sizeof(Array<float, 1>));
  PRINT(sizeof(Array<float, 2>));
  PRINT(sizeof(Array<float, 3>));
  PRINT(sizeof(Array<float, 4>));
  PRINT(sizeof(Array<float, 5>));
  PRINT(sizeof(Array<float, 6>));
  PRINT(sizeof(Array<float, 7>));
  PRINT(sizeof(Array<float, 8>));
  PRINT(sizeof(Array<float, 9>));
  PRINT(sizeof(Array<float, 10>));

  return 0;
}

