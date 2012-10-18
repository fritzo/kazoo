
#include "splines.h"

class TestWarpFwd : public FunctionAndDeriv
{
public:
  virtual float value (float t_large) const
  {
    float t_small = 0.5f * (1 - cos(M_PI * t_large));
    return t_small;
  }
  virtual float deriv (float t_large) const
  {
    float t_small = M_PI * 0.5f * sin(M_PI * t_large);
    return t_small;
  }
};

void test_warping (size_t large_size = 1000, size_t small_size = 300)
{
  LOG("building warping object : " << large_size << " --> " << small_size);

  TestWarpFwd warp_fwd;
  Spline w(large_size, small_size, warp_fwd);

  LOG("defining input sequence");
  Vector<float> large_in(large_size);
  Vector<float> small_out(small_size);
  Vector<float> large_out(large_size);

  for (size_t i = 0; i < large_size; ++i) {
    float t = (0.5f + i) / large_size;
    large_in[i] = sin(4 * 2 * M_PI * t) * t + 0.1 * random_std();
  }

  LOG("transforming forward and backward");
  w.transform_fwd(large_in, small_out);
  w.transform_bwd(small_out, large_out);

  LOG("RMS inversion error = "
   << rms_error(large_in, large_out)
   << " (should be ~ 0.1)");
}

int main ()
{
  test_warping();

  return 0;
}

