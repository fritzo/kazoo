
#include "common.h"
#include "vectors.h"
#include "splines.h"
#include <iomanip>

//----( implementation )------------------------------------------------------

enum { width = 64 };

namespace Mud
{

void advance (
    float dt,
    const float * restrict freq,
    float * restrict mass_real,
    float * restrict mass_imag,
    size_t size)
{
  // advance
  for (size_t j = 0; j < size; ++j) {
    const float c = cos(freq[j] * dt);
    const float s = sin(freq[j] * dt);

    for (size_t i = 0; i < width; ++i) {
      float x = mass_real[width * j + i];
      float y = mass_imag[width * j + i];

      mass_real[width * j + i] = c * x + s * y;
      mass_imag[width * j + i] = c * y - s * x;
    }
  }
}

// slower
void sample_v1 (
    float * restrict mass_real,
    float * restrict mass_out,
    size_t size)
{
  for (size_t i = 0; i < width; ++i) {
    float m = 0;
    for (size_t j = 0; j < size; ++j) {
      m += mass_real[width * j + i];
    }
    mass_out[i] = m;
  }
}

// faster
void sample_v2 (
    float * restrict mass_real,
    float * restrict mass_out,
    size_t size)
{
  zero_float(mass_out, width);
  for (size_t j = 0; j < size; ++j) {
    for (size_t i = 0; i < width; ++i) {
      mass_out[i] += mass_real[width * j + i];
    }
  }
}

} // namespace Mud

//----( unit tests )----------------------------------------------------------

void randomize (Vector<float> & a)
{
  for (size_t i = 0; i < a.size; ++i) {
    a[i] = random_std();
  }
}

enum { size = 1 << 10 };

void advance (size_t steps)
{
  Vector<float> freq(size);
  Vector<float> mass_real(size * width);
  Vector<float> mass_imag(size * width);

  randomize(freq);
  mass_real.set(1.0f);
  mass_imag.set(0.0f);

  for (size_t t = 0; t < steps; ++t) {
    Mud::advance(0.1f, freq, mass_real, mass_imag, size);
  }
}

void sample_v1 (size_t steps)
{
  Vector<float> mass_real(size * width);
  Vector<float> mass_out(width);

  randomize(mass_real);

  for (size_t t = 0; t < steps; ++t) {
    Mud::sample_v1(mass_real, mass_out, size);
  }
}

void sample_v2 (size_t steps)
{
  Vector<float> mass_real(size * width);
  Vector<float> mass_out(width);

  randomize(mass_real);

  for (size_t t = 0; t < steps; ++t) {
    Mud::sample_v2(mass_real, mass_out, size);
  }
}

void full (size_t steps, size_t large_width = 320)
{
  Vector<float> freq(size);
  Vector<float> mass_real(size * width);
  Vector<float> mass_imag(size * width);
  Vector<float> mass_out(width);
  Vector<float> mass_scaled(large_width);

  Spline spline(large_width, width);

  randomize(freq);
  mass_real.set(1.0f);
  mass_imag.set(0.0f);

  for (size_t t = 0; t < steps; ++t) {
    Mud::advance(0.1f, freq, mass_real, mass_imag, size);
    Mud::sample_v2(mass_real, mass_out, size);
    spline.transform_bwd(mass_out, mass_scaled);
  }
}

//----( timing )--------------------------------------------------------------

// HACK this was copied & adapted from speed_test.cpp

float print_time (size_t steps)
{
  static float last_time = 0;
  float new_time = get_elapsed_time();
  float time_diff = new_time - last_time;
  last_time = new_time;

  float average_time = time_diff / (2 * steps);
  float average_rate = 1 / average_time;

  LOG(average_rate << " Hz");

  return average_time;
}

#define test_rate(fun) \
  cout << std::setw(24) << #fun ": "; \
  fun(steps); \
  print_time(steps);

void test_all (size_t steps)
{
  LOG("--------------------------------");
  PRINT2(width, size);
  LOG("--------------------------------");
  test_rate(advance);
  test_rate(sample_v1);
  test_rate(sample_v2);
  test_rate(full);
}

int main (int argc, char ** argv)
{
  size_t steps =  argc >= 2 ? atoi(argv[1]) : 1000;

  test_all(steps);
}

