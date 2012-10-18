
#include "common.h"
#include "images.h"
#include "vectors.h"
#include <iomanip>

size_t width;
size_t height;
size_t radius;
const float width_in_fingers = 48;
const float timescale = 1000;
const size_t peak_capacity = 24;

void randomize (Vector<float> & a)
{
  for (size_t i = 0; i < a.size; ++i) {
    a[i] = random_std();
  }
}

void subtract_background (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  randomize(a);
  b.zero();

  for (size_t n = 0; n < steps; ++n) {
    Image::subtract_background(width * height, timescale, a, b);
  }
}

void matrix_transpose (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::transpose_8(width, height, a, b);
  }
}

void scale_by_half (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height / 4);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::scale_by_half_8(width, height, a, b);
  }
}

void scale_by_two (const size_t steps)
{
  Vector<float> a(width * height / 4);
  Vector<float> b(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::scale_by_two_8(width / 2, height / 2, a, b);
  }
}

void gradient (const size_t steps)
{
  Vector<float> f(width * height);
  Vector<float> fx(width * height);
  Vector<float> fy(width * height);
  randomize(f);

  for (size_t n = 0; n < steps; ++n) {
    Image::gradient(width, height, f, fx, fy);
  }
}

void gradient_wrap_repeat (const size_t steps)
{
  Vector<float> f(width * height);
  Vector<float> fx(width * height);
  Vector<float> fy(width * height);
  randomize(f);

  for (size_t n = 0; n < steps; ++n) {
    Image::gradient_wrap_repeat(width, height, f, fx, fy);
  }
}

void scharr_gradient (const size_t steps)
{
  Vector<float> f(width * height);
  Vector<float> fx(width * height);
  Vector<float> fy(width * height);
  randomize(f);

  for (size_t n = 0; n < steps; ++n) {
    Image::scharr_gradient(width, height, f, fx, fy);
  }
}

void scharr_gradient_wrap_repeat (const size_t steps)
{
  Vector<float> f(width * height);
  Vector<float> fx(width * height);
  Vector<float> fy(width * height);
  randomize(f);

  for (size_t n = 0; n < steps; ++n) {
    Image::scharr_gradient_wrap_repeat(width, height, f, fx, fy);
  }
}

void square_blur_axis (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::square_blur_axis(width, height, radius, a, b);
  }
}

void linear_blur_axis (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::linear_blur_axis(width, height, radius, a, b);
  }
}

void quadratic_blur_axis (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::quadratic_blur_axis(width, height, radius, a, b);
  }
}

void exp_blur_axis (const size_t steps)
{
  Vector<float> a(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::exp_blur_axis_zero(width, height, radius, a);
  }
}

void enhance_points (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::enhance_points(width, height, radius, a, b);
  }
}

template<size_t R>
void enhance_points_ (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::enhance_points_<R>(width, height, a, b);
  }
}

template<size_t R>
void enhance_fingers_ (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  Vector<float> c(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::enhance_fingers_<R>(width, height, a, b, c);
  }
}

template<size_t R>
void orientations_ (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  Vector<float> c(width * height);
  Vector<float> d(width * height);
  Vector<float> e(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::orientations_<R>(width, height, a, b,c,d,e);
  }
}

void project_axes_max (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> ax(width);
  Vector<float> ay(height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::project_axes_max(width, height, a, ax, ay);
  }
}

void project_axes_sum (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> ax(width);
  Vector<float> ay(height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::project_axes_sum(width, height, a, ax, ay);
  }
}

void transduce_yx (const size_t steps)
{
  Vector<float> f_xy(width * height);
  Vector<float> f_x(width);
  Vector<float> f_y(height);
  Vector<float> g_x(width);
  Vector<float> g_y(height);
  randomize(f_xy);

  Image::project_axes_sum(width, height, f_xy, f_x, f_y);
  g_y = f_y;

  for (size_t n = 0; n < steps; ++n) {
    Image::transduce_yx(width, height, f_xy, f_y, g_y, g_x);
  }
}

void full_detect (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  Vector<float> c(width * height);
  Vector<float> d(width * height);
  randomize(c);
  d.zero();

  Image::Peaks peaks;

  for (size_t n = 0; n < steps; ++n) {
    a = c; // incurs little overhead
    Image::subtract_background(width * height, timescale, a, d);
    Image::quadratic_blur_axis(height, width, radius, a, b);
    Image::transpose_8(height, width, b, a);
    Image::quadratic_blur_axis(width, height, radius, a, b);
    Image::enhance_points(width, height, radius, b, a);
    Image::find_peaks(width, height, peak_capacity, 0, a, peaks);
  }
}

void full_fingers (const size_t steps)
{
  Vector<float> orig(width * height);
  Vector<float> bg(width * height);
  Vector<float> image(width * height);
  Vector<float> temp(width * height);
  Vector<float> tips(width * height);
  Vector<float> shafts(width * height);
  randomize(orig);
  bg.zero();

  Image::Peaks peaks;
  Image::Peaks moments;

  enum { radius = 2 };

  for (size_t n = 0; n < steps; ++n) {
    image = orig; // incurs little overhead

    Image::subtract_background(width * height, timescale, image, bg);
    Image::quadratic_blur_axis(height, width, radius, image, temp);
    Image::transpose_8(height, width, temp, image);
    Image::quadratic_blur_axis(width, height, radius, image, temp);
    Image::enhance_fingers(width, height, radius, temp, tips, shafts);
    Image::find_peaks(width, height, peak_capacity, 0, tips, peaks);
    Image::extract_moments_at(width, height, 4*radius, shafts, peaks, moments);
  }
}

void full_project (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  Vector<float> c(width * height);
  Vector<float> d(width * height);
  Vector<float> ax(width);
  Vector<float> ay(height);
  randomize(c);
  d.zero();

  for (size_t n = 0; n < steps; ++n) {
    a = c; // incurs little overhead
    Image::subtract_background(width * height, timescale, a, d);
    Image::square_blur_axis(height, width, radius, a, b);
    Image::transpose_8(height, width, b, a);
    Image::square_blur_axis(width, height, radius, a, b);
    Image::enhance_points(width, height, radius, b, a);
    Image::project_axes_max(width, height, a, ax, ay);
  }
}

void moments_along_y (const size_t steps)
{
  Vector<float> mass(width * height);
  Vector<float> sum_m(height);
  Vector<float> sum_mx(height);
  randomize(mass);

  for (size_t n = 0; n < steps; ++n) {
    Image::moments_along_y(width, height, mass, sum_m, sum_mx);
  }
}

void moments_along_x (const size_t steps)
{
  Vector<float> mass(width * height);
  Vector<float> y(height);
  Vector<float> sum_m(width);
  Vector<float> sum_my(width);
  randomize(mass);

  for (size_t j = 0; j < height; ++j) {
    y[j] = (j + 0.5f) / height;
  }

  for (size_t n = 0; n < steps; ++n) {
    Image::moments_along_x(width, height, mass, y, sum_m, sum_my);
  }
}

void local_moments (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  Vector<float> c(width * height);
  Vector<float> d(width * height);
  Vector<float> e(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    b = a; // incurs little overhead
    Image::local_moments_transpose(width, height, 4 * radius, b, c, d, e);
  }
}

void full_orient (const size_t steps)
{
  Vector<float> orig(width * height);
  Vector<float> image(width * height);
  Vector<float> tips(width * height);
  Vector<float> shafts(width * height);
  Vector<float> dx(width * height);
  Vector<float> dy(width * height);
  Vector<float> temp(width * height);
  randomize(orig);

  enum { R = 4 };

  for (size_t n = 0; n < steps; ++n) {
    image = orig; // incurs little overhead

    Image::quadratic_blur_axis(width, height, R, image, temp);
    Image::transpose_8(height, width, image, temp);
    image = temp; // incurs little overhead
    Image::quadratic_blur_axis(height, width, R, temp, image);

    Image::enhance_fingers_<R>(width, height, image, tips, shafts);
    Image::transpose_8(height, width, tips, temp);
    tips = temp; // incurs little overhead

    Image::local_moments_transpose(width, height, 4 * R, shafts, dx, dy, image);
  }
}

void orientations (const size_t steps)
{
  Vector<float> orig(width * height);
  Vector<float> image(width * height);

  Vector<float> small(width/2 * height/2);
  Vector<float> palms(width/4 * height/4);

  Vector<float> nn(width * height);
  Vector<float> ne(width * height);
  Vector<float> ee(width * height);
  Vector<float> se(width * height);

  randomize(orig);

  enum { R = 2 };

  for (size_t n = 0; n < steps; ++n) {
    image = orig; // incurs little overhead

    Image::quadratic_blur(width, height, R, image, nn);

    Image::scale_by_half_8(width, height, image, small);
    Image::scale_by_half_8(width/2, height/2, small, image);

    Image::enhance_points_<R>(width/4, height/4, image, palms);
    Image::orientations_<R>(width, height, image, nn,ne,ee,se);

#define SCALE_AND_BLUR(dir) \
    Image::scale_by_half_8(width, height, dir, small); \
    Image::scale_by_half_8(width/2, height/2, small, dir); \
    Image::quadratic_blur(width/4, height/4, 3 * R, dir, small);

    SCALE_AND_BLUR(nn);
    SCALE_AND_BLUR(ne);
    SCALE_AND_BLUR(ee);
    SCALE_AND_BLUR(se);

#undef SCALE_AND_BLUR
  }
}

void change_chi2 (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  Vector<float> c(width * height);
  randomize(a);
  b.set(1.0f);
  c.set(1.0f);

  for (size_t n = 0; n < steps; ++n) {
    Image::change_chi2(width * height, a, b, c, 1.0f / timescale);
  }
}

void detect_changes (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  Vector<float> c(width * height);
  Vector<float> d(width * height);
  randomize(a);
  d.zero();

  for (size_t n = 0; n < steps; ++n) {
    b = a; // incurs little overhead
    Image::detect_changes(width * height, b, c, d, 1 / timescale);
  }
}

void detect_change_moment_x (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  Vector<float> c(width * height);
  Vector<float> d(height);
  Vector<float> e(height);
  randomize(a);
  b.zero();
  c.zero();

  for (size_t n = 0; n < steps; ++n) {
    Image::detect_change_moment_x(width, height, a, b, c, d, e, 1/timescale);
  }
}

void detect_change_moment_y (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  Vector<float> c(width * height);
  Vector<float> d(width);
  Vector<float> e(width);
  randomize(a);
  b.zero();
  c.zero();

  for (size_t n = 0; n < steps; ++n) {
    Image::detect_change_moment_y(width, height, a, b, c, d, e, 1/timescale);
  }
}

void sand (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  Vector<float> c(width * height);
  Vector<float> d(width * height);

  Vector<float> f_x(width);
  Vector<float> f_y(height);
  Vector<float> g_x(width);
  Vector<float> g_y(height);

  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    b = a; // incurs little overhead
    Image::subtract_ceil(width * height, timescale, b, c);
    Image::square_blur_axis(height, width, 4, b, d);
    Image::transpose_8(height, width, d, b);
    Image::square_blur_axis(width, height, 4, b, d);
    Image::enhance_points_<4>(width, height, d, b);
    Image::project_axes_sum(width, height, b, f_x, f_y);
    g_y = f_y;
    Image::transduce_yx(width, height, b, f_y, g_y, g_x);
  }
}

void reassign_flow (const size_t steps)
{
  Vector<float> dx(width * height);
  Vector<float> dy(width * height);
  Vector<float> m0(width * height);
  Vector<float> m1(width * height);
  randomize(dx);
  randomize(dy);
  randomize(m0);

  for (size_t n = 0; n < steps; ++n) {
    Image::reassign_flow(width, height, dx, dy, m0, m1);
  }
}

void reassign (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::reassign_wrap_repeat(width, height, a, b);
  }
}

void reassign_wrap_x (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::reassign_wrap_x(width, height, a, b);
  }
}

void reassign_xy (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  Vector<float> c(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    b = a; // incurs little overhead
    Image::reassign_wrap_repeat_xy(width, height, b, c);
  }
}
void reassign_repeat_x (const size_t steps)
{
  Vector<float> a(width * height);
  Vector<float> b(width * height);
  randomize(a);

  for (size_t n = 0; n < steps; ++n) {
    Image::reassign_repeat_x(width, height, a, b);
  }
}

void local_optical_flow1 (const size_t steps)
{
  Vector<float> im0(width * height);
  Vector<float> im1(width * height);
  Vector<float> im0_highpass(width * height);
  Vector<float> im1_highpass(width * height);
  Vector<float> x(width * height);
  Vector<float> y(width * height);
  Vector<float> temp1(width * height);
  Vector<float> temp2(width * height);

  randomize(im0);
  randomize(im1);

  Image::highpass(width, height, radius, im0, im0_highpass, temp1);

  for (size_t n = 0; n < steps; ++n) {
    Image::highpass(width, height, radius, im1, im1_highpass, temp1);
    Image::local_optical_flow(
        width, height,
        im0_highpass, im1_highpass,
        x, y,
        temp1, temp2);
  }
}

void local_optical_flow2 (const size_t steps)
{
  Vector<float> im0(width * height);
  Vector<float> im1(width * height);
  Vector<float> im0_highpass(width * height);
  Vector<float> im1_highpass(width * height);
  Vector<float> x(width * height);
  Vector<float> y(width * height);
  Vector<float> xx(width * height);
  Vector<float> xy(width * height);
  Vector<float> yy(width * height);

  randomize(im0);
  randomize(im1);

  Image::highpass(width, height, radius, im0, im0_highpass, yy);

  for (size_t n = 0; n < steps; ++n) {
    Image::highpass(width, height, radius, im1, im1_highpass, yy);
    Image::local_optical_flow(
        width, height,
        im0_highpass, im1_highpass,
        x, y,
        xx, xy, yy);
  }
}

void local_optical_flow_pyramid (const size_t steps)
{
  Vector<float> im0(width * height);
  Vector<float> im1(width * height);
  Vector<float> im0_highpass(width * height);
  Vector<float> im1_highpass(width * height);
  Vector<float> im_sum(width * height);
  Vector<float> im_diff(width * height);
  Vector<float> x(width * height);
  Vector<float> y(width * height);
  Vector<float> temp1(width * height);
  Vector<float> temp2(width * height);
  Vector<float> temp3(width * height);

  randomize(im0);
  randomize(im1);

  Image::highpass(width, height, radius, im0, im0_highpass, temp1);

  for (size_t n = 0; n < steps; ++n) {
    Image::highpass(width, height, radius, im1, im1_highpass, temp1);
    Image::local_optical_flow_pyramid(
        width, height,
        im0_highpass, im1_highpass,
        im_sum, im_diff,
        x, y,
        temp1, temp2, temp3);
  }
}

void full_optical_flow (const size_t steps)
{
  const size_t krig_radius = 16;

  Vector<float> im0(width * height);
  Vector<float> im1(width * height);
  Vector<float> im0_highpass(width * height);
  Vector<float> im1_highpass(width * height);
  Vector<float> sx(width * height);
  Vector<float> sy(width * height);
  Vector<float> ixx(width * height);
  Vector<float> ixy(width * height);
  Vector<float> iyy(width * height);
  Vector<float> fx(width * height);
  Vector<float> fy(width * height);
  float dt = 1.0f / DEFAULT_VIDEO_FRAMERATE;

  randomize(im0);
  randomize(im1);

  Image::highpass(width, height, radius, im0, im0_highpass, iyy);

  for (size_t n = 0; n < steps; ++n) {
    Image::highpass(width, height, radius, im1, im1_highpass, iyy);
    Image::local_optical_flow(width, height, im0, im1, sx, sy, ixx, ixy, iyy);
    Image::krig_optical_flow(
        width, height, krig_radius, dt,
        sx, sy,
        ixx, ixy, iyy,
        fx, fy);
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

void test_all (size_t w, size_t h, size_t steps)
{
  width = w;
  height = h;
  radius = roundu(width / width_in_fingers / 2);

  LOG("\n--------( "
      << width << " x " << height << " +- " << radius
      << " )--------");

  print_rate();

  test_rate(subtract_background);
  test_rate(matrix_transpose);
  test_rate(scale_by_half);
  test_rate(scale_by_two);
  test_rate(gradient);
  test_rate(gradient_wrap_repeat);
  test_rate(scharr_gradient);
  test_rate(scharr_gradient_wrap_repeat);
  test_rate(square_blur_axis);
  test_rate(linear_blur_axis);
  test_rate(quadratic_blur_axis);
  test_rate(exp_blur_axis);
  test_rate(enhance_points);
  test_rate(enhance_points_<2>);
  test_rate(enhance_points_<4>);
  test_rate(enhance_fingers_<2>);
  test_rate(enhance_fingers_<4>);
  test_rate(orientations_<2>);
  test_rate(orientations_<4>);
  test_rate(moments_along_y);
  test_rate(moments_along_x);
  test_rate(local_moments);
  test_rate(change_chi2);
  test_rate(detect_changes);
  test_rate(detect_change_moment_x);
  test_rate(detect_change_moment_y);
  test_rate(project_axes_max);
  test_rate(project_axes_sum);
  test_rate(transduce_yx);
  test_rate(full_detect);
  test_rate(full_fingers);
  test_rate(full_project);
  test_rate(full_orient);
  test_rate(orientations);
  test_rate(sand);
  test_rate(reassign_flow);
  test_rate(reassign);
  test_rate(reassign_wrap_x);
  test_rate(reassign_repeat_x);
  test_rate(reassign_xy);
  test_rate(local_optical_flow1);
  test_rate(local_optical_flow2);
  test_rate(local_optical_flow_pyramid);
  test_rate(full_optical_flow);
}

int main (int argc, char ** argv)
{
  size_t steps =  argc >= 2 ? atoi(argv[1]) : 1000;

  test_all(320,240, steps);
  test_all(640,480, steps / 4);
}

