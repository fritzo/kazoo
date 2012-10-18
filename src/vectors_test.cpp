
#include "common.h"
#include "vectors.h"
#include "args.h"
#include <iomanip>

#ifndef USE_INTEL_MKL
int main ()
{
  ERROR("test not compiled with -DUSE_INTEL_MKL");
  return 1;
}
#else // USE_INTEL_MKL

#include <mkl.h>
#include <mkl_vml.h>

void randomize (Vector<float> & a)
{
  for (size_t i = 0; i < a.size; ++i) {
    a[i] = random_std();
  }
}

void log_glibc (const size_t size, const size_t steps)
{
  Vector<float> a(size);
  Vector<float> b(size);
  randomize(a);
  b.zero();

  const float * restrict a_(a.data);
  float * restrict b_(b.data);

  for (size_t n = 0; n < steps; ++n) {
    for (size_t i = 0; i < size; ++i) {
      b_[i] = logf(a_[i]);
    }
  }
}

void log_mkl (const size_t size, const size_t steps)
{
  Vector<float> a(size);
  Vector<float> b(size);
  randomize(a);
  b.zero();

  const float * restrict a_(a.data);
  float * restrict b_(b.data);

  for (size_t n = 0; n < steps; ++n) {
    vsLn(size, a_, b_);
  }
}

void exp_glibc (const size_t size, const size_t steps)
{
  Vector<float> a(size);
  Vector<float> b(size);
  randomize(a);
  b.zero();

  const float * restrict a_(a.data);
  float * restrict b_(b.data);

  for (size_t n = 0; n < steps; ++n) {
    for (size_t i = 0; i < size; ++i) {
      b_[i] = expf(a_[i]);
    }
  }
}

void exp_mkl (const size_t size, const size_t steps)
{
  Vector<float> a(size);
  Vector<float> b(size);
  randomize(a);
  b.zero();

  const float * restrict a_(a.data);
  float * restrict b_(b.data);

  for (size_t n = 0; n < steps; ++n) {
    vsExp(size, a_, b_);
  }
}

void lgamma_glibc (const size_t size, const size_t steps)
{
  Vector<float> a(size);
  Vector<float> b(size);
  randomize(a);
  b.zero();

  const float * restrict a_(a.data);
  float * restrict b_(b.data);

  for (size_t n = 0; n < steps; ++n) {
    for (size_t i = 0; i < size; ++i) {
      b_[i] = lgammaf(a_[i]);
    }
  }
}

void lgamma_mkl (const size_t size, const size_t steps)
{
  Vector<float> a(size);
  Vector<float> b(size);
  randomize(a);
  b.zero();

  const float * restrict a_(a.data);
  float * restrict b_(b.data);

  for (size_t n = 0; n < steps; ++n) {
    vsLGamma(size, a_, b_);
  }
}

//----------------------------------------------------------------------------

// computed using ginac/doc/example/lanczos.cpp

static const float coeffs_4_0 = 0.99993070858945212916f;
static const float coeffs_4_1 = 24.715529106293455768f;
static const float coeffs_4_2 = -19.207130007054814744f;
static const float coeffs_4_3 = 2.4510433865385976987f;

static const float coeffs_5_0 = 1.0000018972739439654f;
static const float coeffs_5_1 = 76.18008222264213687f;
static const float coeffs_5_2 = -86.50509203705485249f;
static const float coeffs_5_3 = 24.012898581922666602f;
static const float coeffs_5_4 = -1.229602849028567646f;

static const float coeffs_6_0 = 0.9999999561013040786f;
static const float coeffs_6_1 = 228.93440277636919233f;
static const float coeffs_6_2 = -342.81040354442876322f;
static const float coeffs_6_3 = 151.38423671180975885f;
static const float coeffs_6_4 = -20.0115271443821241f;
static const float coeffs_6_5 = 0.46162611352712712431f;

float lgammaf_lanczos_5(float x)
{
  const size_t order = 5;

  // XXX the reflection check is not sse optimizable
  if (x < 0.5f) {
    return logf(M_PI) - logf(sinf(M_PI * x)) - lgammaf_lanczos_5(1 - x);
  }

  float A = coeffs_5_0
          + coeffs_5_1 / (x + 0)
          + coeffs_5_2 / (x + 1)
          + coeffs_5_3 / (x + 2)
          + coeffs_5_4 / (x + 3);

  float temp = x + order - 0.5f;

  return logf(2 * M_PI) / 2
       + (x - 0.5f) * logf(temp)
       - temp
       + logf(A);
}

void lgamma_lanczos (const size_t size, const size_t steps)
{
  Vector<float> a(size);
  Vector<float> b(size);
  randomize(a);
  b.zero();

  const float * restrict a_(a.data);
  float * restrict b_(b.data);

  for (size_t n = 0; n < steps; ++n) {
    for (size_t i = 0; i < size; ++i) {
      b_[i] = lgammaf_lanczos_5(a_[i]);
    }
  }
}

//----------------------------------------------------------------------------

void print_rate (size_t size = 0, size_t steps = 0)
{
  static double last_time = 0;
  double new_time = get_elapsed_time();
  double time_diff = new_time - last_time;
  last_time = new_time;

  if (not steps) return; // initialization

  double rate = size * steps / time_diff * 1e-6f;

  LOG(std::setw(10) << rate << " MHz");
}

#define test_rate(fun) \
  cout << std::setw(24) << #fun ": "; \
  fun(size, steps); \
  print_rate(size, steps);

void test_all (size_t size, size_t steps)
{
  LOG("\n--------( " << size << " x " << steps << " )--------");

  print_rate();

  test_rate(log_glibc);
  test_rate(log_mkl);
  test_rate(exp_glibc);
  test_rate(exp_mkl);
  test_rate(lgamma_glibc);
  test_rate(lgamma_mkl);
  test_rate(lgamma_lanczos);
}

const char * help_message =
"Usage: vectors_test [SIZE_EXPONENT] [STEPS_EXPONENT]\n"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);

  size_t size = 1 << args.pop(10);
  size_t steps = 1 << args.pop(10);

  mkl_set_num_threads(1);
  vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

  test_all(size, steps);
}

#endif // USE_INTEL_MKL
