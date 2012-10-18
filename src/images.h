#ifndef KAZOO_IMAGE_H
#define KAZOO_IMAGE_H

#include "common.h"
#include "image_types.h"

/** Fast image processing tools.

  TODO Work out transpose convention & fix transpose bugs in images_test
*/

namespace Image
{

//----( edge policies )-------------------------------------------------------

enum EdgePolicy { REPEAT_AT_EDGES, WRAP_AT_EDGES };

struct Wrap
{
  static size_t prev (size_t i, size_t I) { return (i + I - 1) % I; }
  static size_t next (size_t i, size_t I) { return (i + 1) % I; }
};

struct Repeat
{
  static size_t prev (size_t i, size_t  ) { return (i == 0) ? 0 : i - 1; }
  static size_t next (size_t i, size_t I)
  {
    return (i == (I - 1)) ? I - 1 : i + 1;
  }
};

//----( persistence )---------------------------------------------------------

void write_image (
    const char * filename,
    size_t I,
    size_t J,
    const float * image);

void read_image (
    const char * filename,
    size_t & I,
    size_t & J,
    float * & image);

bool save_png (
    const char * filename,
    size_t I,
    size_t J,
    uint8_t * image);

bool save_png (
    const char * filename,
    size_t I,
    size_t J,
    const float * image);

//----( algorithms )----------------------------------------------------------

void project_axes_max (
    const size_t X,
    const size_t Y,
    const float * restrict image,
    float * restrict image_x,
    float * restrict image_y);

void project_axes_sum (
    const size_t X,
    const size_t Y,
    const float * restrict image,
    float * restrict image_x,
    float * restrict image_y);

void lift_axes_sum (
    const size_t X,
    const size_t Y,
    const float * restrict image_x,
    const float * restrict image_y,
    float * restrict image);

void subtract_background (
    size_t size,
    float timescale,
    float * restrict fg,
    float * restrict bg);

void subtract_ceil (
    size_t size,
    float timescale,
    float * restrict fg,
    float * restrict bg);

void transpose_2 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

void transpose_4 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

void transpose_8 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

// optimized for multiples of 8,4,2
void transpose (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

void scale_by_half_2 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

void scale_by_half_4 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

void scale_by_half_8 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

// optimized for multiples of 8,4,2
void scale_by_half (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

void scale_by_two_2 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

void scale_by_two_4 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

void scale_by_two_8 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

// optimized for multiples of 8,4,2
void scale_by_two (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

void transduce_yx (
    const size_t width,
    const size_t height,
    const float * restrict f_xy,
    const float * restrict f_y,
    const float * restrict g_y,
    float * restrict g_x,
    const float tol = 1e-8f);

void integrate_axis (
    const size_t width,
    const size_t height,
    float * restrict source,
    float * restrict destin);

//----( blurring )----

void square_blur_axis (
    const size_t width,
    const size_t height,
    const size_t radius,
    const float * restrict source,
    float * restrict destin);

void square_blur_axis_wrap (
    const size_t width,
    const size_t height,
    const size_t radius,
    const float * restrict source,
    float * restrict destin);

inline void square_blur (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict image,
    float * restrict temp)
{
  square_blur_axis(width, height, radius, image, temp);
  transpose(width, height, temp, image);
  square_blur_axis(height, width, radius, image, temp);
  transpose(height, width, temp, image);
}

inline void linear_blur_axis (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict image,
    float * restrict temp)
{
  square_blur_axis(width, height, radius, image, temp);
  square_blur_axis(width, height, radius, temp, image);
}

inline void linear_blur_axis_wrap (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict image,
    float * restrict temp)
{
  square_blur_axis_wrap(width, height, radius, image, temp);
  square_blur_axis_wrap(width, height, radius, temp, image);
}

inline void linear_blur (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict image,
    float * restrict temp)
{
  linear_blur_axis(width, height, radius, image, temp);
  transpose(width, height, image, temp);
  linear_blur_axis(height, width, radius, temp, image);
  transpose(height, width, temp, image);
}

inline void linear_blur (
    const size_t width,
    const size_t height,
    const size_t radius_x,
    const size_t radius_y,
    float * restrict image,
    float * restrict temp)
{
  linear_blur_axis(width, height, radius_x, image, temp);
  transpose(width, height, image, temp);
  linear_blur_axis(height, width, radius_y, temp, image);
  transpose(height, width, temp, image);
}

inline void quadratic_blur_axis (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict source,
    float * restrict destin)
{
  square_blur_axis(width, height, radius, source, destin);
  square_blur_axis(width, height, radius, destin, source);
  square_blur_axis(width, height, radius, source, destin);
}

inline void quadratic_blur_axis_wrap (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict source,
    float * restrict destin)
{
  square_blur_axis_wrap(width, height, radius, source, destin);
  square_blur_axis_wrap(width, height, radius, destin, source);
  square_blur_axis_wrap(width, height, radius, source, destin);
}

inline void quadratic_blur (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict image,
    float * restrict temp)
{
  quadratic_blur_axis(width, height, radius, image, temp);
  transpose(width, height, temp, image);
  quadratic_blur_axis(height, width, radius, image, temp);
  transpose(height, width, temp, image);
}

void exp_blur_axis_zero (
    const size_t width,
    const size_t height,
    const float radius,
    float * restrict image);

inline void exp_blur_zero (
    const size_t width,
    const size_t height,
    const float radius,
    float * restrict image,
    float * restrict temp)
{
  exp_blur_axis_zero(width, height, radius, image);
  transpose(width, height, image, temp);
  exp_blur_axis_zero(height, width, radius, temp);
  transpose(height, width, temp, image);
}

void exp_blur_1d_zero (
    const size_t size,
    float radius,
    const float * restrict source,
    float * restrict destin);

void exp_blur_1d_zero (
    const size_t size,
    float radius_lo,
    float radius_hi,
    const float * restrict source,
    float * restrict destin);

inline void square_blur_1d (
    const size_t size,
    const size_t radius,
    const float * restrict source,
    float * restrict destin)
{
  square_blur_axis(size, 1, radius, source, destin);
}

inline void linear_blur_1d (
    const size_t size,
    const size_t radius,
    float * restrict image,
    float * restrict temp)
{
  linear_blur_axis(size, 1, radius, image, temp);
}

inline void quadratic_blur_1d (
    const size_t size,
    const size_t radius,
    float * restrict source,
    float * restrict destin)
{
  quadratic_blur_axis(size, 1, radius, source, destin);
}

void highpass (
    const size_t width,
    const size_t height,
    const size_t R,
    const float * restrict im,
    float * restrict im_highpass,
    float * restrict temp);

void highpass_1d (
    const size_t size,
    const size_t R,
    const float * restrict im,
    float * restrict im_highpass,
    float * restrict temp);

//----( feature enhancement )----

void enhance_points (
    const size_t width,
    const size_t height,
    const size_t radius,
    const float * restrict source,
    float * restrict destin);

// defined for R = 2,4,8
template<size_t R>
void enhance_points_ (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

// defined for R = 2,4,8
template<size_t R>
void enhance_fingers_ (
    const size_t width,
    const size_t height,
    const float * restrict image,
    float * restrict tips,
    float * restrict shafts);

void enhance_fingers (
    const size_t width,
    const size_t height,
    const size_t radius, // defined for 2,4,8
    const float * restrict image,
    float * restrict tips,
    float * restrict shafts);

// defined for R = 2,4,8
template<size_t R>
void orientations_ (
    const size_t width,
    const size_t height,
    const float * restrict image,
    float * restrict nn,
    float * restrict ne,
    float * restrict ee,
    float * restrict se);

void orientations (
    const size_t width,
    const size_t height,
    const size_t radius, // defined for 2,4,8
    const float * restrict image,
    float * restrict nn,
    float * restrict ne,
    float * restrict ee,
    float * restrict se);

void enhance_crosses (
    const size_t width,
    const size_t height,
    const size_t radius,
    const float * restrict source,
    float * restrict destin);

void enhance_lines (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict image,
    float * restrict temp1,
    float * restrict temp2,
    bool dark = true);

float gradient_x (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx);

float gradient_x_repeat (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx);

float gradient (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx,
    float * restrict fy);

float gradient_wrap_repeat (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx,
    float * restrict fy);

float scharr_gradient (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx,
    float * restrict fy);

float scharr_gradient_wrap_repeat (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx,
    float * restrict fy);

void sharpen (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict image,
    float * restrict sharp,
    float * restrict temp,
    bool dark = true);

void hdr_real (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict image,
    float * restrict blur1,
    float * restrict blur2,
    float * restrict temp);

void hdr_01 (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict image,
    float * restrict temp1,
    float * restrict temp2,
    float * restrict temp3);

void hdr_real_color (
    const size_t width,
    const size_t height,
    float * restrict r,
    float * restrict g,
    float * restrict b);

void find_peaks (
    const size_t width,
    const size_t height,
    const size_t peak_capacity,
    const float min_value,
    const float * restrict frame,
    Peaks & peaks);

void measure_extent (
    const size_t width,
    const size_t height,
    const size_t x0,
    const size_t y0,
    const float * restrict integral,
    size_t & radius_guess);

void extract_blob (
    const size_t width,
    const size_t height,
    const size_t x0,
    const size_t y0,
    const size_t radius,
    const float * restrict frame,
    Blob & blob);

void extract_moments (
    const size_t width,
    const size_t height,
    const float * restrict frame,
    float & mass,
    float & Ex,
    float & Ey);

void local_moments_transpose (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict mass,
    float * restrict mx,
    float * restrict my,
    float * restrict temp);

void moments_along_y (
    const size_t width,
    const size_t height,
    const float * restrict mass,
    float * restrict sum_m,
    float * restrict sum_mx);

void moments_along_x (
    const size_t width,
    const size_t height,
    const float * restrict mass,
    const float * restrict y,
    float * restrict sum_m,
    float * restrict sum_my);

// input bounds: 0.5 <= x <= width - 0.5, 0.5 <= y <= height - 0.5
Peak extract_moments_at (
    const size_t width,
    const size_t height,
    const float radius,
    float x0,
    float y0,
    const float * restrict mass,
    const float tol = 1e-8);

void extract_moments_at (
    const size_t width,
    const size_t height,
    const float radius,
    const float * restrict mass,
    const Peaks & positions,
    Peaks & moments);

void vh_convex_floodfill (
    const size_t width,
    const size_t height,
    const float threshold,
    const float * restrict frame,
    uint8_t * mask);

//----( change )----

void change_chi2 (
    const size_t size,
    float * restrict image,
    float * restrict mean,
    float * restrict variance,
    const float dt,
    float tol = 1e-16f);

void detect_changes (
    const size_t size,
    float * restrict image,
    float * restrict mean,
    float * restrict variance,
    const float dt);

void detect_change_moment_x (
    const size_t width,
    const size_t height,
    float * restrict image_xy,
    float * restrict mean_xy,
    float * restrict variance_xy,
    float * restrict mass_x,
    float * restrict moment_x,
    const float dt);

void detect_change_moment_y (
    const size_t width,
    const size_t height,
    float * restrict image_xy,
    float * restrict mean_xy,
    float * restrict variance_xy,
    float * restrict mass_y,
    float * restrict moment_y,
    const float dt);

void update_momentum (
    const size_t size,
    const float * restrict mass_new,
    const float * restrict moment_new,
    float * restrict mass_old,
    float * restrict moment_old,
    float * restrict momentum,
    const float dt);

//----( reassignment )----

void reassign_flow (
    const size_t I,
    const size_t J,
    const float * restrict flow_x,
    const float * restrict flow_y,
    const float * restrict mass_in,
    float * restrict mass_out);

void reassign_flow_wrap_repeat (
    const size_t I,
    const size_t J,
    const float * restrict flow_x,
    const float * restrict flow_y,
    const float * restrict mass_in,
    float * restrict mass_out);

void reassign_wrap_repeat (
    const size_t width,
    const size_t height,
    const float * restrict mass_in,
    float * restrict mass_out,
    float scale = 1,
    float shift = 0);

void reassign_wrap_x (
    const size_t width,
    const size_t height,
    const float * restrict mass_in,
    float * restrict mass_out,
    float scale = 1,
    float shift = 0);

void reassign_repeat_x (
    const size_t width,
    const size_t height,
    const float * restrict mass_in,
    float * restrict mass_out,
    float scale = 1,
    float shift = 0);

//----( optical flow )----

// surprise : 1 / (pix frame)
// info : 1 / pix^2
// cov : pix^2
// dt : sec / frame

void local_optical_flow_1d (
    const size_t I,
    const float * restrict im0_highpass,
    const float * restrict im1_highpass,
    float * restrict surprise_x,
    float * restrict info_xx,
    float * restrict temp);

void krig_optical_flow_1d (
    const size_t I,
    const float R,
    const float dt,
    float * restrict surprise_x,
    float * restrict info_xx,
    float * restrict flow_x,
    const float prior_info);

void local_optical_flow (
    const size_t width,
    const size_t height,
    const float * restrict im0_highpass,
    const float * restrict im1_highpass,
    float * restrict surprise_x,
    float * restrict surprise_y,
    float * restrict im_sum,
    float * restrict im_diff);

void local_optical_flow (
    const size_t width,
    const size_t height,
    const float * restrict im0_highpass,
    const float * restrict im1_highpass,
    float * restrict surprise_x,
    float * restrict surprise_y,
    float * restrict info_xx,
    float * restrict info_xy,
    float * restrict info_yy);

void local_optical_flow_pyramid (
    const size_t width,
    const size_t height,
    const float * restrict im0,
    const float * restrict im1,
    float * restrict im_sum,
    float * restrict im_diff,
    float * restrict surprise_x,
    float * restrict surprise_y,
    float * restrict temp1,
    float * restrict temp2,
    float * restrict temp3);

void solve_optical_flow (
    const size_t width,
    const size_t height,
    const float dt,
    const float * restrict surprise_x,
    const float * restrict surprise_y,
    const float * restrict info_xx,
    const float * restrict info_xy,
    const float * restrict info_yy,
    float * restrict flow_x,
    float * restrict flow_y,
    const float prior_info = 0.1);

void krig_optical_flow (
    const size_t width,
    const size_t height,
    const float krig_radius,
    const float dt,
    float * restrict surprise_x,
    float * restrict surprise_y,
    float * restrict info_xx,
    float * restrict info_xy,
    float * restrict info_yy,
    float * restrict flow_x,
    float * restrict flow_y,
    const float prior_info = 0.1);

void advance_optical_flow (
    const size_t I,
    const size_t J,
    const float dt,
    float * restrict dx,
    float * restrict dy,
    const float * restrict old_sx,
    const float * restrict old_sy,
    const float * restrict old_ixx,
    const float * restrict old_ixy,
    const float * restrict old_iyy,
    float * restrict new_sx,
    float * restrict new_sy,
    float * restrict new_ixx,
    float * restrict new_ixy,
    float * restrict new_iyy);

void fuse_optical_flow (
    const size_t I,
    const float dt,
    const float process_noise,
    const float prior_info,
    const float * restrict old_sx,
    const float * restrict old_sy,
    const float * restrict old_ixx,
    const float * restrict old_ixy,
    const float * restrict old_iyy,
    float * restrict new_sx,
    float * restrict new_sy,
    float * restrict new_ixx,
    float * restrict new_ixy,
    float * restrict new_iyy,
    float * restrict flow_x,
    float * restrict flow_y);

//----( high-level wrappers )-------------------------------------------------

inline void integrate (
    const size_t width,
    const size_t height,
    float * restrict image,
    float * restrict temp)
{
  integrate_axis(width, height, image, temp);
  transpose_8(width, height, temp, image);
  integrate_axis(height, width, image, temp);
  transpose_8(height, width, temp, image);
}

inline void scale_smoothly_by_four (
    size_t width,
    size_t height,
    const float * restrict source_11,
    float * restrict destin_44,
    float * restrict temp_44)
{
  scale_by_two(width, height, source_11, temp_44);
  width *= 2;
  height *= 2;

  square_blur_axis(width, height, 1, temp_44, destin_44);
  transpose(width, height, destin_44, temp_44);
  square_blur_axis(height, width, 1, temp_44, destin_44);

  scale_by_two(height, width, destin_44, temp_44);
  width *= 2;
  height *= 2;

  square_blur_axis(height, width, 1, temp_44, destin_44);
  transpose(height, width, destin_44, temp_44);
  square_blur_axis(width, height, 1, temp_44, destin_44);
}

inline void square_blur_scaled (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict image,
    float * restrict temp)
{
  if (not radius) return;

  square_blur_axis(width, height, radius, image, temp);
  transpose_8(width, height, temp, image);
  square_blur_axis(height, width, radius, image, temp);
  transpose_8(height, width, temp, image);

  float scale = powf(2 * radius + 1, -2);
  for (size_t xy = 0; xy < width * height; ++xy) {
    image[xy] *= scale;
  }
}

inline void quadratic_blur_scaled (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict image,
    float * restrict temp)
{
  if (not radius) return;

  quadratic_blur(width, height, radius, image, temp);

  float scale = powf(2 * radius + 1, -6);
  for (size_t xy = 0; xy < width * height; ++xy) {
    image[xy] *= scale;
  }
}

inline void dilate (
    const size_t width,
    const size_t height,
    const size_t radius,
    float * restrict image,
    float * restrict temp,
    float sharpness = 1.0f, // +20 for dilate, -20 for erode
    float tol = 1e-6f)
{
  for (size_t xy = 0; xy < width * height; ++xy) {
    image[xy] = max(0.0f, expf(sharpness * image[xy]) - tol);
  }

  quadratic_blur_scaled(width, height, radius, image, temp);

  for (size_t xy = 0; xy < width * height; ++xy) {
    image[xy] = logf(image[xy] + tol) / sharpness;
  }
}

inline void extract_blob (
    const size_t X,
    const size_t Y,
    const size_t x0,
    const size_t y0,
    const float * restrict frame,
    Blob & blob)
{
  size_t radius = min(min(x0, X - x0),
                      min(y0, Y - y0));

  measure_extent(X,Y,x0,y0,frame, radius);

  radius = roundu(radius * M_SQRT2);

  extract_blob(X,Y,x0,y0,radius,frame,blob);
}

inline void reassign_wrap_repeat_xy (
    const size_t I,
    const size_t J,
    float * restrict mass,
    float * restrict temp,
    float scale = 1.0f)
{
  reassign_wrap_x(I, J, mass, temp, scale);
  transpose_8(I, J, temp, mass);
  reassign_repeat_x(J, I, mass, temp);
  transpose_8(J, I, temp, mass);
}

} // namespace Image

#endif // KAZOO_IMAGE_H
