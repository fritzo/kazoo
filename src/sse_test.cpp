
#include "common.h"
#include "args.h"
#include <stddef.h>

typedef float v4sf __attribute__ ((vector_size (16)));

union float4
{
  float array[4];
  v4sf vector;
};

void test_array (Args & args)
{
  size_t steps = args.pop(1000000);

  float4 w,x,y,z;

  for (size_t i = 0; i < 4; ++i) {
    w.array[i] = 1.0;
    x.array[i] = 1.0 / i;
    y.array[i] = 1.0 / i * i;
    z.array[i] = 0.0;
  }

  for (size_t n = 0; n < steps; ++n) {
    z.vector += x.vector + y.vector;
    x.vector *= w.vector + z.vector;
    y.vector *= w.vector + z.vector;
  }

  PRINT2(x.array[0], y.array[0]);
}

void test_conditional (Args & args)
{
  size_t size = args.pop(1024);
  size_t steps = args.pop(100000);

  float * restrict x = malloc_float(size);
  float * restrict y = malloc_float(size);
  float * restrict z = malloc_float(size);

  for (size_t i = 0; i < size; ++i) {
    x[i] = random_std();
    y[i] = random_std();
  }

  for (size_t t = 0; t < steps; ++t) {
    for (size_t i = 0; i < size; ++i) {
      //z[i] = y[i] > 0 ? x[i] : 0.0f;  // bad
      //z[i] = y[i] > 0 ? y[i] : 0;     // good
      //z[i] = y[i] > 0 ? y[i] : x[i];  // bad
      z[i] = y[i] > x[i] ? y[i] : 0;  // good
      //z[i] = y[i] > x[i] ? y[i] : x[i]; // good
    }
  }

  free_float(x);
  free_float(y);
  free_float(z);
}

const char * help_message =
"Usage: sse_test COMMAND [OPTIONS]"
"\nCommands:"
"\n  array [NUM_STEPS]"
"\n  if [SIZE] [NUM_STEPS]"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);

  args
    .case_("array", test_array)
    .case_("if", test_conditional)
    .default_error();
}

