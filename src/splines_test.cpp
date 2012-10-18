
#include "splines.h"

using Image::Point;
using Image::Transform;

void test_Spline (size_t large = 1000, size_t small = 20)
{
  Spline spline(large, small);

  Vector<float> x(large), y(small);

  for (size_t i = 0; i < large; ++i) {
    x[i] = sqr(i * (large - i - 1)) * sqr(4.0f / sqr(large)) * random_01();
  }

  LOG("Testing spline : " << large << " -> " << small);

  spline.transform_fwd(x,y);
  float sum_x = sum(x);
  float sum_y = sum(y);
  float diff = sum_y - sum_x;
  float tot = sum_x + sum_y;
  LOG(" transformed mass " << sum_x << " -> " << sum_y);
  ASSERT_LT(fabs(diff / tot), 1e-4f);

  spline.transform_bwd(y,x);
  sum_x = sum(x);
  sum_y = sum(y);
  diff = sum_y - sum_x;
  tot = sum_x + sum_y;
  LOG(" transformed mass " << sum_x << " <- " << sum_y);
  ASSERT_LT(fabs(diff / tot), 1e-4f);
}

struct TestTransform : public Transform
{
  virtual void operator () (Point & xy) const
  {
    float r2 = sqr(xy.x) + sqr(xy.y);
    float scale = (1 + r2) / 2;

    xy.x *= scale;
    xy.y *= scale;
  }
};

void test_Spline2D (
    size_t width = DEFAULT_VIDEO_WIDTH,
    size_t height = DEFAULT_VIDEO_HEIGHT,
    size_t steps = DEFAULT_VIDEO_FRAMERATE * 10)
{
  LOG("building " << width << "x" << height << " Spline2D");
  const size_t size = width * height;
  TestTransform transform;
  Spline2D spline(width, height, width, height, transform);

  Vector<float> e_in(size), e_out(size);
  for (size_t i = 0; i < size; ++i) e_in[i] = random_01();

  LOG("transforming " << steps << " frames");
  float start_time = get_elapsed_time();
  for (size_t step = 0; step < steps; ++step) {
    spline.transform_fwd(e_in, e_out);
  }
  float end_time = get_elapsed_time();

  float time = end_time - start_time;
  float rate = steps / time;
  LOG("averaged " << rate << " frames/sec = "
      << (rate / DEFAULT_VIDEO_FRAMERATE) << " x frame rate");
}

int main ()
{
  test_Spline();
  test_Spline2D();

  return 0;
}

