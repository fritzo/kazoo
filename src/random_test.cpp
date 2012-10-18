
#include "common.h"
#include <vector>

void test_random_int (size_t num_samples)
{
  LOG("\nTesting random_int()");

  int64_t min_val = 0xFFFFFFFFFFFFl, max_val = -0xFFFFFFFFFFFFl;

  for (size_t i = 0; i < num_samples; ++i) {
    int64_t r = random_int();
    imin(min_val, r);
    imax(max_val, r);
  }

  PRINT2(min_val, max_val);
}

void test_random_01 (size_t num_samples)
{
  LOG("\nTesting random_01()");

  float min_val = INFINITY, max_val = -INFINITY;
  double sum_val = 0, sum_float2 = 0;

  for (size_t i = 0; i < num_samples; ++i) {
    float r = random_01();
    imin(min_val, r);
    imax(max_val, r);
    sum_val += r;
    sum_float2 += r * r;
  }

  double mean_val = sum_val / num_samples;
  double var_val = sum_float2 / num_samples - sqr(mean_val);

  PRINT2(min_val, max_val);
  PRINT2(mean_val - 0.5, var_val * 12);
}

void test_random_std (size_t num_samples)
{
  LOG("\nTesting random_std()");

  float min_val = INFINITY, max_val = -INFINITY;
  double sum_val = 0, sum_float2 = 0;

  for (size_t i = 0; i < num_samples; ++i) {
    float r = random_std();
    imin(min_val, r);
    imax(max_val, r);
    sum_val += r;
    sum_float2 += r * r;
  }

  double mean_val = sum_val / num_samples;
  double var_val = sum_float2 / num_samples - sqr(mean_val);

  PRINT2(min_val, max_val);
  PRINT2(mean_val, var_val);
}

void test_random_choice (size_t num_samples, size_t size = 65536)
{
  LOG("\nTesting random_choice()");

  std::vector<size_t> bins(size, 0);

  for (size_t i = 0; i < num_samples; ++i) {
    ++bins[random_choice(size)];
  }

  size_t max_count = sqrt(num_samples);
  std::vector<size_t> hist(max_count, 0);
  for (size_t i = 0; i < size; ++i) {
    size_t count = bins[i];
    if (count < max_count) {
      ++hist[count];
    }
  }

  for (size_t i = 0; i < max_count; ++i) {
    size_t count = hist[i];
    if (count) {
      LOG(i << "\t@ " << count);
    }
  }
}

// we expect overflows below
#pragma GCC diagnostic ignored "-Woverflow"

int main ()
{
  PRINT(RAND_MAX);
  PRINT(RAND_MAX + 1); // overflows
  PRINT(RAND_MAX + 1.0f);
  PRINT(int64_t(RAND_MAX) * RAND_MAX);
  PRINT(int64_t(RAND_MAX) * RAND_MAX + RAND_MAX);

  size_t num_samples = 1 << 24;

  test_random_int(num_samples);
  test_random_01(num_samples);
  test_random_std(num_samples);
  //test_random_choice(num_samples);

  PRINT4(int(-1.5), int(-0.5), int(0.5), int(1.5));

  return 0;
}

