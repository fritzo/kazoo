
#include "common.h"
#include "vectors.h"
#include <fftw3.h>
#include <iomanip>

size_t w = 320, h = 240;

void complex_2d (size_t steps)
{
  Vector<complex> orig(w * h);
  Vector<complex> input(w * h);
  Vector<complex> output(w * h);
  orig.zero();

  fftwf_plan plan = fftwf_plan_dft_2d(
      w,
      h,
      reinterpret_cast<fftwf_complex *>(input.data),
      reinterpret_cast<fftwf_complex *>(output.data),
      FFTW_FORWARD,
      FFTW_MEASURE);

  for (size_t i = 0; i < steps; ++i) {
    input = orig;
    fftwf_execute(plan);
  }
}

void real_2d (size_t steps)
{
  Vector<float> orig(w * h);
  Vector<float> input(w * h);
  Vector<complex> output(w * h);
  orig.zero();

  fftwf_plan plan = fftwf_plan_dft_r2c_2d(
      w,
      h,
      input.data,
      reinterpret_cast<fftwf_complex *>(output.data),
      FFTW_MEASURE);

  for (size_t i = 0; i < steps; ++i) {
    input = orig;
    fftwf_execute(plan);
  }
}

float print_rate (size_t steps = 0)
{
  static float last_time = 0;
  float new_time = get_elapsed_time();
  float time_diff = new_time - last_time;
  last_time = new_time;

  if (not steps) return 0;

  float average_time = time_diff / steps;
  float average_rate = 1 / average_time;

  LOG(std::setw(10) << average_rate << " Hz"
      << std::setw(10) << (average_time * 1000) << " ms");

  return average_time;
}

#define test_rate(fun) \
  cout << std::setw(24) << #fun ": "; \
  fun(steps); \
  print_rate(steps);

void test_all (size_t width, size_t height, size_t steps = 1000)
{
  PRINT2(width, height);
  w = width;
  h = height;

  test_rate(complex_2d);
  test_rate(real_2d);
}

int main ()
{
  test_all(20, 15);
  test_all(40, 30);
  test_all(80, 60);
  test_all(160, 120);
  test_all(320, 240);

  return 0;
}

