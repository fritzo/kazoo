
#include "images.h"
#include "vectors.h"
#include "sym33.h"
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <png.h>

//#define ASSERT2_EQ(x,y) ASSERT_EQ(x,y)
//#define ASSERT2_LE(x,y) ASSERT_LE(x,y)
//#define ASSERT2_LT(x,y) ASSERT_LT(x,y)

#define ASSERT2_EQ(x,y)
#define ASSERT2_LE(x,y)
#define ASSERT2_LT(x,y)

#define LOG1(message)

#define TOL (1e-20f)

namespace Image
{

//----( persistence )---------------------------------------------------------

void write_image (
    const char * filename,
    size_t I,
    size_t J,
    const float * image)
{
  LOG("writing " << I << " x " << J << " image to file " << filename);
  std::ofstream file(filename);
  file << I << ' ' <<  J << '\n';
  for (size_t ij = 0, IJ = I * J; ij < IJ; ++ij) {
    file << image[ij] << (ij % 8 ? ' ' : '\n');
  }
}

void read_image (
    const char * filename,
    size_t & I,
    size_t & J,
    float * & image)
{
  LOG("reading image from file " << filename);
  std::ifstream file(filename);
  ASSERT(file, "file not found: " << filename);

  file >> I >> J;
  LOG(" allocating " << I << " x " << J << " image");
  ASSERT_LE(I, 1920); // sanity check
  ASSERT_LE(J, 1920); // sanity check

  image = malloc_float(I * J);
  for (size_t ij = 0, IJ = I * J; ij < IJ; ++ij) {
    file >> image[ij];
  }
}

bool save_png (
    const char * filename,
    size_t I,
    size_t J,
    uint8_t * image)
{
  static_assert(sizeof(uint8_t) == sizeof(png_byte),
      "png_byte is note identical to uint8_t");

  // png images are row-major
  const size_t width = J;
  const size_t height = I;

  FILE *file = fopen(filename, "wb");
  if (not file) {
    WARN("failed to open png file for writing");
    return false;
  }
  LOG("saving image to " << filename);

  LOG1("writing header");
  png_structp writer = png_create_write_struct(
      PNG_LIBPNG_VER_STRING,
      NULL,
      NULL,
      NULL);
  png_infop info = png_create_info_struct(writer);
  png_init_io(writer, file);
  const int bit_depth = 8;
  png_set_IHDR(
      writer,
      info,
      width,
      height,
      bit_depth,
      PNG_COLOR_TYPE_GRAY,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT);
  png_write_info(writer, info);

  LOG1("writing png rows");

  png_byte ** rows = new png_byte * [height];
  for (size_t y = 0; y < height; ++y) {
    rows[y] = image + width * y;
  }

  png_write_image(writer, rows);

  delete[] rows;

  LOG1("finishing file");

  png_write_end(writer, NULL);
  fclose(file);

  return true;
}

bool save_png (
    const char * filename,
    size_t I,
    size_t J,
    const float * image)
{
  LOG1("converting float [0,1] -> 8-bit gray");

  const size_t IJ = I * J;

  const float * restrict image_real = image;
  png_byte * restrict image_int = new png_byte[IJ];

  for (size_t ij = 0; ij < IJ; ++ij) {
    image_int[ij] = static_cast<png_byte>(image_real[ij] * 255.0f + 0.5f);
  }

  bool result = save_png(filename, I, J, image_int);

  delete[] image_int;

  return result;
}

//----( algorithms )----------------------------------------------------------

namespace { inline float nonneg (float x) { return max(0.0f, x); } }

// block size of 8 seems to be optimal
const size_t BLOCK_SIZE = 8;

//----( projecting to axes )--------------------------------------------------

void project_axes_max (
    const size_t X,
    const size_t Y,
    const float * restrict image,
    float * restrict image_x,
    float * restrict image_y)
{
  for (size_t y = 0; y < Y; ++y) image_y[y] = 0;

  for (size_t x = 0; x < X; ++x) {

    float line = 0;

    for (size_t y = 0; y < Y; ++y) {
      float image_xy = image[Y * x + y];

      imax(line, image_xy);
      imax(image_y[y], image_xy);
    }

    image_x[x] = line;
  }
}

void project_axes_sum (
    const size_t X,
    const size_t Y,
    const float * restrict image,
    float * restrict image_x,
    float * restrict image_y)
{
  for (size_t y = 0; y < Y; ++y) image_y[y] = 0;

  for (size_t x = 0; x < X; ++x) {

    float line = 0;

    for (size_t y = 0; y < Y; ++y) {
      float image_xy = image[Y * x + y];

      line += image_xy;
      image_y[y] += image_xy;
    }

    image_x[x] = line;
  }
}

void lift_axes_sum (
    const size_t X,
    const size_t Y,
    const float * restrict image_x,
    const float * restrict image_y,
    float * restrict image)
{
  float sum_x = 0;
  for (size_t x = 0; x < X; ++x) {
    sum_x += image_x[x];
  }

  float sum_y = 0;
  for (size_t y = 0; y < Y; ++y) {
    sum_y += image_y[y];
  }

  float scale = pow(max(1e-20f, sum_x * sum_y), -0.5);

  for (size_t x = 0; x < X; ++x) {
    float scale_x = scale * image_x[x];

    for (size_t y = 0; y < Y; ++y) {
      image[Y * x + y] = scale_x * image_y[y];
    }
  }
}

//----( subtract background )-------------------------------------------------

void subtract_background (
    size_t size,
    float timescale,
    float * restrict fg,
    float * restrict bg)
{
  ASSERT_LT(0, timescale);

  const float new_part = 1 / (1 + timescale);
  const float old_part = 1 - new_part;

  for (size_t i = 0; i < size; ++i) {
    fg[i] -= bg[i] = old_part * bg[i] + new_part * fg[i];
  }
}

//----( subtract ceil )-------------------------------------------------------

void subtract_ceil (
    size_t size,
    float timescale,
    float * restrict fg,
    float * restrict bg)
{
  ASSERT_LT(0, timescale);

  const float scale_bg = expf(-1 / timescale);

  for (size_t i = 0; i < size; ++i) {
    fg[i] -= bg[i] = max(fg[i], scale_bg * bg[i]);
  }
}

//----( transpose )-----------------------------------------------------------

template<unsigned block_size>
inline void transpose_block_ (
    const size_t I,
    const size_t J,
    const float * restrict x,
    float * restrict y)
{
  for (size_t i = 0; i < block_size; ++i)
  for (size_t j = 0; j < block_size; ++j)
  {
    y[i + j*I] = x[i*J + j];
  }
}

template<unsigned block_size>
void transpose_ (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  ASSERT_DIVIDES(block_size, width);
  ASSERT_DIVIDES(block_size, height);

  const size_t I = width;
  const size_t J = height;

  const float * const restrict x = source;
        float * const restrict y = destin;

  for (size_t i = 0; i < I; i += block_size)
  for (size_t j = 0; j < J; j += block_size)
  {
    transpose_block_<block_size>(I, J, x + i*J + j, y + i + j*I);
  }
}

void transpose_2 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  transpose_<2>(width, height, source, destin);
}

void transpose_4 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  transpose_<4>(width, height, source, destin);
}

void transpose_8 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  transpose_<8>(width, height, source, destin);
}

void transpose (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  if (not ((width % 8) || (height % 8))) {
    transpose_8(width, height, source, destin);
    return;
  }

  if (not ((width % 4) || (height % 4))) {
    transpose_4(width, height, source, destin);
    return;
  }

  if (not ((width % 2) || (height % 2))) {
    transpose_2(width, height, source, destin);
    return;
  }

  const size_t I = width;
  const size_t J = height;

  const float * const restrict x = source;
        float * const restrict y = destin;

  for (size_t i = 0; i < I; ++i)
  for (size_t j = 0; j < J; ++j)
  {
    y[i + j*I] = x[i*J + j];
  }
}

//----( scaling down )--------------------------------------------------------

template<size_t block_size>
inline void scale_by_half_block_ (
    const size_t I,
    const size_t J,
    const float * restrict x,
    float * restrict y)
{
  for (size_t i = 0; i < block_size / 2; ++i)
  for (size_t j = 0; j < block_size / 2; ++j)
  {
    y[J * i + j] = x[2 * J * (2 * i + 0) + 2 * j + 0]
                 + x[2 * J * (2 * i + 0) + 2 * j + 1]
                 + x[2 * J * (2 * i + 1) + 2 * j + 0]
                 + x[2 * J * (2 * i + 1) + 2 * j + 1];
  }
}

template<size_t block_size>
void scale_by_half_ (
    const size_t width,
    const size_t height,
    const float * restrict x,
    float * restrict y)
{
  ASSERT_DIVIDES(block_size, width);
  ASSERT_DIVIDES(block_size, height);

  const size_t I = width / 2;
  const size_t J = height / 2;

  for (size_t i = 0; i < I; i += block_size / 2)
  for (size_t j = 0; j < J; j += block_size / 2)
  {
    scale_by_half_block_<block_size>(
        I,
        J,
        x + 2 * J * 2 * i + 2 * j,
        y + J * i + j);
  }
}

void scale_by_half_2 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  scale_by_half_<2>(width, height, source, destin);
}

void scale_by_half_4 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  scale_by_half_<4>(width, height, source, destin);
}

void scale_by_half_8 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  scale_by_half_<8>(width, height, source, destin);
}

void scale_by_half (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  if (not ((width % 8) || (height % 8))) {
    scale_by_half_8(width, height, source, destin);
    return;
  }

  if (not ((width % 4) || (height % 4))) {
    scale_by_half_4(width, height, source, destin);
    return;
  }

  ASSERT_DIVIDES(2, width);
  ASSERT_DIVIDES(2, height);

  scale_by_half_2(width, height, source, destin);
}

//----( scaling up )----------------------------------------------------------

template<size_t block_size>
inline void scale_by_two_block_ (
    const size_t I,
    const size_t J,
    const float * restrict x,
    float * restrict y)
{
  for (size_t i = 0; i < block_size / 2; ++i)
  for (size_t j = 0; j < block_size / 2; ++j)
  {
    float x_ij = x[J * i + j];

    y[2 * J * (2 * i + 0) + 2 * j + 0] = x_ij;
    y[2 * J * (2 * i + 0) + 2 * j + 1] = x_ij;
    y[2 * J * (2 * i + 1) + 2 * j + 0] = x_ij;
    y[2 * J * (2 * i + 1) + 2 * j + 1] = x_ij;
  }
}

template<size_t block_size>
void scale_by_two_ (
    const size_t I,
    const size_t J,
    const float * restrict x,
    float * restrict y)
{
  ASSERT_DIVIDES(block_size / 2, I);
  ASSERT_DIVIDES(block_size / 2, J);

  for (size_t i = 0; i < I; i += block_size / 2)
  for (size_t j = 0; j < J; j += block_size / 2)
  {
    scale_by_two_block_<block_size>(
        I,
        J,
        x + J * i + j,
        y + 2 * J * 2 * i + 2 * j);
  }
}

void scale_by_two_2 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  scale_by_two_<2>(width, height, source, destin);
}

void scale_by_two_4 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  scale_by_two_<4>(width, height, source, destin);
}

void scale_by_two_8 (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  scale_by_two_<8>(width, height, source, destin);
}

void scale_by_two (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin)
{
  if (not ((width % 4) || (height % 4))) {
    scale_by_two_8(width, height, source, destin);
    return;
  }

  if (not ((width % 2) || (height % 2))) {
    scale_by_two_4(width, height, source, destin);
    return;
  }

  scale_by_two_2(width, height, source, destin);
}

//----( transduction )--------------------------------------------------------

/** Transduction from projections.

 Given f_xy(-,-), marginals f_x(-), f_y(-), and a signal g_y(-), define

                   g_y(y) f_xy(x,y)
   g_x(x) = int y. ----------------
                        f_y(y)
*/
void transduce_yx (
    const size_t width,
    const size_t height,
    const float * restrict f_xy,
    const float * restrict f_y,
    const float * restrict g_y,
    float * restrict g_x,
    const float tol)
{
  for (size_t i = 0; i < width; ++i) {
    float sum = 0;
    for (size_t j = 0; j < height; ++j) {
      sum += g_y[j] * f_xy[height * i + j] / (f_y[j] + tol);
    }
    g_x[i] = sum;
  }
}

//----( integration )---------------------------------------------------------

void integrate_axis (
    const size_t width,
    const size_t height,
    float * restrict source,
    float * restrict destin)
{
  const size_t I = width;
  const size_t J = height;

  for (size_t j = 0; j < J; ++j) {
    destin[j] = source[j];
  }

  for (size_t i = 0; i+1 < I; ++i) {
    float * restrict destin_i0 = destin + J * (i + 0);
    float * restrict destin_i1 = destin + J * (i + 1);
    float * restrict source_i1 = source + J * (i + 1);

    for (size_t j = 0; j < J; ++j) {
      destin_i1[j] = destin_i0[j] + source_i1[j];
    }
  }
}

//----( blur )----------------------------------------------------------------

inline void shift_window (
    float * restrict center,
    const float * restrict prev,
    const float * restrict left,
    const float * restrict right,
    const size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    center[i] = right[i] - left[i] + prev[i];
  }
}

void square_blur_axis (
    const size_t width,
    const size_t height,
    const size_t radius,
    const float * restrict source,
    float * restrict destin)
{
  ASSERT_LT(0, radius);
  ASSERT_LT(radius, width / 2);

  const size_t I = width;
  const size_t J = height;
  const size_t R = radius;

  const float * restrict f = source;
  float * restrict g = destin;

  size_t i_left = 0, i = 0, i_right = 0;

  for (size_t j = 0; j < J; ++j) {
    g[J*i+j] = R * f[J*i_right+j];
  }
  while (i_right <= R) {
    for (size_t j = 0; j < J; ++j) {
      g[J*i+j] += f[J*i_right+j];
    }

    ++i_right;
  }
  ++i;

  ASSERT2_EQ(i_left, 0);
  ASSERT2_EQ(i, 1);
  ASSERT2_EQ(i_right, 1+R);

  while (i <= R) {
    shift_window(g + J * i,
                 g + J * (i - 1),
                 f + J * i_left,
                 f + J * i_right,
                 J);

    ++i; ++i_right;
  }

  ASSERT2_EQ(i_left, 0);
  ASSERT2_EQ(i, R+1);
  ASSERT2_EQ(i_right, R+1+R);

  while (i_right < I) {
    shift_window(g + J * i,
                 g + J * (i - 1),
                 f + J * i_left,
                 f + J * i_right,
                 J);

    ++i_left; ++i; ++i_right;
  }
  --i_right;

  ASSERT2_EQ(i_left, I - R - R - 1);
  ASSERT2_EQ(i, I - R);
  ASSERT2_EQ(i_right, I - 1);

  while (i < I) {
    shift_window(g + J * i,
                 g + J * (i - 1),
                 f + J * i_left,
                 f + J * i_right,
                 J);

    ++i_left; ++i;
  }

  ASSERT2_EQ(i_left, I - R - 1);
  ASSERT2_EQ(i, I);
  ASSERT2_EQ(i_right, I - 1);
}

void square_blur_axis_wrap (
    const size_t width,
    const size_t height,
    const size_t radius,
    const float * restrict source,
    float * restrict destin)
{
  ASSERT_LT(0, radius);
  ASSERT_LT(radius, width / 2);

  const size_t I = width;
  const size_t J = height;
  const size_t R = radius;

  const float * restrict f = source;
  float * restrict g = destin;

  for (size_t j = 0; j < J; ++j) {
    g[j] = f[j];
  }

  for (size_t r = 1; r <= R; ++r) {
    const float * restrict f0 = f + J * r;
    const float * restrict f1 = f + J * (I - r);

    for (size_t j = 0; j < J; ++j) {
      g[j] += f0[j] + f1[j];
    }
  }

  for (size_t i = 1; i < I; ++i) {
    const float * restrict f_left = f + J * ((I + i - R - 1) % I);
    const float * restrict f_right = f + J * ((I + i + R) % I);
    const float * restrict g_prev = g + J * (i - 1);
    float * restrict g_curr = g + J * i;

    for (size_t j = 0; j < J; ++j) {
      g_curr[j] = f_right[j] - f_left[j] + g_prev[j];
    }
  }
}

void exp_blur_axis_zero (
    const size_t I,
    const size_t J,
    const float radius,
    float * restrict image)
{
  ASSERT_LT(0, radius);
  const float p = 1 / (1 + radius);

  // backward pass "f -> g", boundary condition = zero
  { size_t i = I;
    float * restrict curr = image + J * (i-1);

    for (size_t j = 0; j < J; ++j) {
      curr[j] *= p;
    }
  }
  for (size_t i = I - 1; i > 0; --i) {
    const float * restrict prev = image + J * i;
    float * restrict curr = image + J * (i-1);

    for (size_t j = 0; j < J; ++j) {
      float prev_j = prev[j];
      curr[j] = prev_j + p * (curr[j] - prev_j);
    }
  }

  // forward pass "g -> h", boundary condition = zero

  // The previous backward pass yields negative g[-] values of
  //   g[-n] = g[0] q^n
  // where f[0] is the repeated boundary value.
  // The forward pass over (-inf,0) then yields (letting q = 1 - p)
  //   h[0] = g[0] p (sum n>=0. q^(2 n))
  //        = g[0] p / (1 - q^2)
  // Note that for q in (0,1)
  //   q < 1  ==>  q^2 < q
  //          ==>  1 - q^2 > 1 - q
  //          ==>  1 - q^2 > p
  //          ==>  1 > p / (1 - q^2)
  // as required.

  { size_t i = 0;
    float * restrict curr = image + J * i;

    for (size_t j = 0; j < J; ++j) {
      curr[j] *= p / (1 - sqr(p));
    }
  }
  for (size_t i = 0; i < I - 1; ++i) {
    const float * restrict prev = image + J * i;
    float * restrict curr = image + J * (i+1);

    for (size_t j = 0; j < J; ++j) {
      float prev_j = prev[j];
      curr[j] = prev_j + p * (curr[j] - prev_j);
    }
  }
}

void exp_blur_1d_zero (
    const size_t I,
    float radius,
    const float * restrict source,
    float * restrict destin)
{
  ASSERT_LT(0, radius);
  const float p = 1 / (1 + radius);

  // backward pass "f -> g", boundary condition = zero
  destin[I-1] = p * source[I-1];
  for (size_t i = I - 1; i > 0; --i) {
    destin[i-1] = destin[i] + p * (source[i-1] - destin[i]);
  }

  // forward pass "g -> h", boundary condition = zero

  // The previous backward pass yields negative g[-] values of
  //   g[-n] = g[0] q^n
  // where f[0] is the repeated boundary value.
  // The forward pass over (-inf,0) then yields (letting q = 1 - p)
  //   h[0] = g[0] p (sum n>=0. q^(2 n))
  //        = g[0] p / (1 - q^2)
  // Note that for q in (0,1)
  //   q < 1  ==>  q^2 < q
  //          ==>  1 - q^2 > 1 - q
  //          ==>  1 - q^2 > p
  //          ==>  1 > p / (1 - q^2)
  // as required.

  destin[0] *= p / (1 - sqr(1 - p));
  for (size_t i = 0; i < I - 1; ++i) {
    destin[i+1] = destin[i] + p * (destin[i+1] - destin[i]);
  }
}

void exp_blur_1d_zero (
    const size_t I,
    float radius_bwd,
    float radius_fwd,
    const float * restrict source,
    float * restrict destin)
{
  ASSERT_LT(0, radius_fwd);
  ASSERT_LT(0, radius_bwd);

  const float p_fwd = 1 / (1 + radius_fwd);
  const float p_bwd = 1 / (1 + radius_bwd);

  // backward pass "f -> g", boundary condition = zero
  destin[I-1] = p_bwd * source[I-1];
  for (size_t i = I - 1; i > 0; --i) {
    destin[i-1] = destin[i] + p_bwd * (source[i-1] - destin[i]);
  }

  // forward pass "g -> h", boundary condition = zero
  destin[0] *= p_fwd / (1 - (1 - p_fwd) * (1 - p_bwd));
  for (size_t i = 0; i < I - 1; ++i) {
    destin[i+1] = destin[i] + p_fwd * (destin[i+1] - destin[i]);
  }
}

void highpass (
    const size_t I,
    const size_t J,
    const size_t R,
    const float * restrict im,
    float * restrict im_highpass,
    float * restrict temp)
{
  square_blur_axis(I, J, R, im, temp);
  square_blur_axis(I, J, R, temp, im_highpass);
  square_blur_axis(I, J, R, im_highpass, temp);
  transpose(I, J, temp, im_highpass);
  quadratic_blur_axis(J, I, R, im_highpass, temp);
  transpose(J, I, temp, im_highpass);

  float scale = -pow(R+1+R, -6);
  for (size_t ij = 0; ij < I*J; ++ij) {
    im_highpass[ij] = im[ij] + scale * im_highpass[ij];
  }
}

void highpass_1d (
    const size_t I,
    const size_t R,
    const float * restrict im,
    float * restrict im_highpass,
    float * restrict temp)
{
  square_blur_1d(I, R, im, temp);
  square_blur_1d(I, R, temp, im_highpass);
  square_blur_1d(I, R, im_highpass, temp);

  float scale = -pow(R+1+R, -3);
  for (size_t i = 0; i < I; ++i) {
    im_highpass[i] = im[i] + scale * temp[i];
  }
}

//----( peak filters )-------------------------------------------------------

#define FILTER_KERNEL \
  float f12 = f1[j2]; \
  float ev1 = nonneg(f1[j0] + f1[j4] - 2 * f12); \
  float ev2 = nonneg(f0[j1] + f2[j3] - 2 * f12); \
  float ev3 = nonneg(f0[j3] + f2[j1] - 2 * f12); \
  g1[j2] = ev1 * ev2 * ev3;

void enhance_points (
    const size_t I,
    const size_t J,
    const size_t R,
    const float * restrict f,
    float * restrict g)
{
  ASSERT_LT(0, R);
  ASSERT_LT(2 * R, I);
  ASSERT_LT(2 * R, J);

  LOG1("approximate difference determinant");
  for (size_t i = 0; i < I; ++i) {
    size_t i0 = i < R ? 0 : i - R;
    size_t i1 = i;
    size_t i2 = i + R < I ? i + R : I - 1;

    const float * restrict f0 = f + J * i0;
    const float * restrict f1 = f + J * i1;
    const float * restrict f2 = f + J * i2;
    float * restrict g1 = g + J * i1;

    for (size_t j = 0; j < R - R / 2; ++j) {
      size_t j0 = 0;
      size_t j1 = 0;
      size_t j2 = j;
      size_t j3 = j + R / 2;
      size_t j4 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = R - R / 2; j < R; ++j) {
      size_t j0 = 0;
      size_t j1 = j - R + R / 2;
      size_t j2 = j;
      size_t j3 = j + R / 2;
      size_t j4 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = R; j < J - R; ++j) {
      size_t j0 = j - R;
      size_t j1 = j - R + R / 2;
      size_t j2 = j;
      size_t j3 = j + R / 2;
      size_t j4 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = J - R; j < J - R / 2; ++j) {
      size_t j0 = j - R;
      size_t j1 = j - R + R / 2;
      size_t j2 = j;
      size_t j3 = j + R / 2;
      size_t j4 = J - 1;

      FILTER_KERNEL
    }

    for (size_t j = J - R / 2; j < J; ++j) {
      size_t j0 = j - R;
      size_t j1 = j - R + R / 2;
      size_t j2 = j;
      size_t j3 = J - 1;
      size_t j4 = J - 1;

      FILTER_KERNEL
    }
  }
}

#undef FILTER_KERNEL

#define FILTER_KERNEL \
  float two_f11 = 2 * f1[j1]; \
  float ev1 = max(0.0f, (f1[j0] + f1[j2] - two_f11)); \
  float ev2 = max(0.0f, (f0[j0] + f2[j2] - two_f11)); \
  float ev3 = max(0.0f, (f0[j1] + f2[j1] - two_f11)); \
  float ev4 = max(0.0f, (f0[j2] + f2[j0] - two_f11)); \
  g1[j1] = ev1 * ev2 * ev3 * ev4;

template<size_t R>
void enhance_points_ (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict g)
{
  ASSERT_LT(2 * R, I);
  ASSERT_LT(2 * R, J);

  LOG1("approximate difference determinant");
  for (size_t i = 0; i < I; ++i) {
    size_t i0 = i < R ? 0 : i - R;
    size_t i1 = i;
    size_t i2 = i + R < I ? i + R : I - 1;

    const float * restrict f0 = f + J * i0;
    const float * restrict f1 = f + J * i1;
    const float * restrict f2 = f + J * i2;
    float * restrict g1 = g + J * i1;

    for (size_t j = 0; j < R; ++j) {
      size_t j0 = 0;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = R; j < J - R; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = J - R; j < J; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = J - 1;

      FILTER_KERNEL
    }
  }
}

// explicit template declarations

void enhance_points_2 (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict g)
{
  enhance_points_<2>(I,J,f,g);
}

void enhance_points_4 (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict g)
{
  enhance_points_<4>(I,J,f,g);
}

void enhance_points_8 (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict g)
{
  enhance_points_<8>(I,J,f,g);
}

#undef FILTER_KERNEL

//----( fingers )-------------------------------------------------------------

// TODO optimize by splitting into two passes and adding one temp array
// (1) compute ev1,ev2,ev4, and sture in t,s,temp (not vectorizable)
// (2) compute ev3,t,s (vectorizable)
// Only the endpoints need be computed in pass 1,
//   with max(0, _ - two_f11) delayed until pass 2.

#define FILTER_KERNEL \
  float two_f11 = 2 * f1[j1]; \
  float ev1 = max(0.0f, (f1[j0] + f1[j2] - two_f11)); \
  float ev2 = max(0.0f, (f0[j0] + f2[j2] - two_f11)) * diag_scale; \
  float ev3 = max(0.0f, (f0[j1] + f2[j1] - two_f11)); \
  float ev4 = max(0.0f, (f0[j2] + f2[j0] - two_f11)) * diag_scale; \
  t1[j1] = ev1 * ev2 * ev3 * ev4; \
  float s = max(0.0f, max(max(ev1, ev2), max(ev3, ev4)) \
                   - min(min(ev1, ev2), min(ev3, ev4)) ); \
  s1[j1] = sqr(s);

template<size_t R>
void enhance_fingers_ (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict t,
    float * restrict s)
{
  ASSERT_LT(2 * R, I);
  ASSERT_LT(2 * R, J);

  const float diag_scale = 1 / M_SQRT2;

  LOG1("approximate difference trace");
  for (size_t i = 0; i < I; ++i) {
    size_t i0 = i < R ? 0 : i - R;
    size_t i1 = i;
    size_t i2 = i + R < I ? i + R : I - 1;

    const float * restrict f0 = f + J * i0;
    const float * restrict f1 = f + J * i1;
    const float * restrict f2 = f + J * i2;
    float * restrict t1 = t + J * i1;
    float * restrict s1 = s + J * i1;

    for (size_t j = 0; j < R; ++j) {
      size_t j0 = 0;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = R; j < J - R; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = J - R; j < J; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = J - 1;

      FILTER_KERNEL
    }
  }
}

#undef FILTER_KERNEL

void enhance_fingers (
    const size_t I,
    const size_t J,
    const size_t R,
    const float * restrict f,
    float * restrict t,
    float * restrict s)
{
  switch (R) {
    case 2: enhance_fingers_<2>(I,J, f,t,s); return;
    case 4: enhance_fingers_<4>(I,J, f,t,s); return;
    case 8: enhance_fingers_<8>(I,J, f,t,s); return;
    default: ERROR("invalid radius " << R << ", expected 2,4,8");
  }
}

//----( orientations )--------------------------------------------------------

// TODO optimize by splitting into two passes, as with enhance_fingers

#define FILTER_KERNEL \
  float two_f11 = 2 * f1[j1]; \
  nn1[j1] = max(0.0f, (f1[j0] + f1[j2] - two_f11)); \
  ne1[j1] = max(0.0f, (f0[j0] + f2[j2] - two_f11)) * diag_scale; \
  ee1[j1] = max(0.0f, (f0[j1] + f2[j1] - two_f11)); \
  se1[j1] = max(0.0f, (f0[j2] + f2[j0] - two_f11)) * diag_scale;

template<size_t R>
void orientations_ (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict nn,
    float * restrict ne,
    float * restrict ee,
    float * restrict se)
{
  ASSERT_LT(2 * R, I);
  ASSERT_LT(2 * R, J);

  const float diag_scale = 1 / M_SQRT2;

  LOG1("approximate difference trace");
  for (size_t i = 0; i < I; ++i) {
    size_t i0 = i < R ? 0 : i - R;
    size_t i1 = i;
    size_t i2 = i + R < I ? i + R : I - 1;

    const float * restrict f0 = f + J * i0;
    const float * restrict f1 = f + J * i1;
    const float * restrict f2 = f + J * i2;
    float * restrict nn1 = nn + J * i1;
    float * restrict ne1 = ne + J * i1;
    float * restrict ee1 = ee + J * i1;
    float * restrict se1 = se + J * i1;

    for (size_t j = 0; j < R; ++j) {
      size_t j0 = 0;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = R; j < J - R; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = J - R; j < J; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = J - 1;

      FILTER_KERNEL
    }
  }
}

#undef FILTER_KERNEL

void orientations (
    const size_t I,
    const size_t J,
    const size_t R,
    const float * restrict f,
    float * restrict nn,
    float * restrict ne,
    float * restrict ee,
    float * restrict se)
{
  switch (R) {
    case 2: orientations_<2>(I,J, f,nn,ne,ee,se); return;
    case 4: orientations_<4>(I,J, f,nn,ne,ee,se); return;
    case 8: orientations_<8>(I,J, f,nn,ne,ee,se); return;
    default: ERROR("invalid radius: " << R);
  }
}

//----( cross filters )-------------------------------------------------------

#define FILTER_KERNEL \
  float ev1 = nonneg(f2[j0] - f2[j1]); \
  float ev2 = nonneg(f0[j0] - f0[j1]); \
  float ev3 = nonneg(f2[j0] - f1[j0]); \
  float ev4 = nonneg(f2[j2] - f1[j2]); \
  float ev5 = nonneg(f2[j2] - f2[j1]); \
  float ev6 = nonneg(f0[j2] - f0[j1]); \
  float ev7 = nonneg(f0[j0] - f1[j0]); \
  float ev8 = nonneg(f0[j2] - f1[j2]); \
  g1[j1] = powf(ev1 * ev2 * ev3 * ev4 * ev5 * ev6 * ev7 * ev8, 0.25f);

void enhance_crosses (
    const size_t I,
    const size_t J,
    const size_t R,
    const float * restrict f,
    float * restrict g)
{
  ASSERT_LT(0, R);
  ASSERT_LT(2 * R, I);
  ASSERT_LT(2 * R, J);

  LOG1("approximate difference determinant");
  for (size_t i = 0; i < I; ++i) {
    size_t i0 = i < R ? 0 : i - R;
    size_t i1 = i;
    size_t i2 = i + R < I ? i + R : I - 1;

    const float * restrict f0 = f + J * i0;
    const float * restrict f1 = f + J * i1;
    const float * restrict f2 = f + J * i2;
    float * restrict g1 = g + J * i1;

    for (size_t j = 0; j < R; ++j) {
      size_t j0 = 0;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = R; j < J - R; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = J - R; j < J; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = J - 1;

      FILTER_KERNEL
    }
  }
}

#undef FILTER_KERNEL

//----( lines )---------------------------------------------------------------

void enhance_lines (
    const size_t I,
    const size_t J,
    const size_t R,
    float * restrict image,
    float * restrict temp1,
    float * restrict temp2,
    bool dark)
{
  sharpen(I,J,1, image, temp1, temp2, dark);

  // enhance autocorrelation
  zero_float(image, I*J);
  for (std::ptrdiff_t di = -R, DI = R; di <= DI; ++di) {
    size_t I0 = abs(di);
    size_t I1 = I - abs(di);

    for (std::ptrdiff_t dj = 0, DJ = R; dj <= DJ; ++dj) {
      size_t J0 = dj;
      size_t J1 = J - dj;

      //        R = 2
      //    ^
      //    | O O O O O
      //    | O O O O O   O is sampled
      // dj + . . . O O
      //    | . . . . .   . is not sampled
      //    | . . . . .
      //    +-----+---->
      //         di
      if (di <= 0 and dj == 0) continue;

      for (size_t i = I0; i < I1; ++i) {
        size_t i0 = J * (i - di);
        size_t i1 = J * i;
        size_t i2 = J * (i + di);

        for (size_t j = J0; j < J1; ++j) {
          size_t x = i0 + j - dj;
          size_t y = i1 + j;
          size_t z = i2 + j + dj;

          image[x] += temp1[y] * temp1[z];
          image[y] += temp1[z] * temp1[x];
          image[z] += temp1[x] * temp1[y];
        }
      }
    }
  }

  for (size_t ij = 0; ij < I*J; ++ij) {
    image[ij] = sqrtf(image[ij]);
  }
}

//----( derivatives )---------------------------------------------------------

void gradient_1d (
    const size_t I,
    const float * restrict f,
    float * restrict df)
{
  df[0] = f[1] - f[0];

  for (size_t i = 1; i < I-1; ++i) {
    df[i] = (f[i+1] - f[i-1]) / 2;
  }

  df[I-1] = f[I-1] - f[I-2];
}

float gradient_x (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx)
{
  { size_t i = 0;
    const float * restrict f1 = f + J * i;
    const float * restrict f2 = f + J * (i + 1);

    for (size_t j = 0; j < J; ++j) {
      fx[J * i + j] = f2[j] - f1[j];
    }
  }

  for (size_t i = 1; i < I - 1; ++i) {
    const float * restrict f0 = f + J * (i - 1);
    const float * restrict f2 = f + J * (i + 1);

    for (size_t j = 0; j < J; ++j) {
      fx[J * i + j] = (f2[j] - f0[j]) / 2;
    }
  }

  { size_t i = I - 1;
    const float * restrict f0 = f + J * (i - 1);
    const float * restrict f1 = f + J * i;

    for (size_t j = 0; j < J; ++j) {
      fx[J * i + j] = f1[j] - f0[j];
    }
  }

  return 0.5f;
}

float gradient_x_wrap (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx)
{
  for (size_t i = 0; i < I; ++i) {
    const float * restrict f0 = f + J * Wrap::prev(i,I);
    const float * restrict f2 = f + J * Wrap::next(i,I);
    float * restrict fx1 = fx + J * i;

    for (size_t j = 0; j < J; ++j) {
      fx1[j] = (f2[j] - f0[j]) / 2;
    }
  }

  return 0.5f;
}

float gradient (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx,
    float * restrict fy)
{
  transpose(I, J, f, fy);
  gradient_x(J, I, fy, fx);
  transpose(J, I, fx, fy);
  gradient_x(I, J, f, fx);

  return 0.5f;
}

float gradient_wrap_repeat (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx,
    float * restrict fy)
{
  transpose(I, J, f, fy);
  gradient_x(J, I, fy, fx);
  transpose(J, I, fx, fy);
  gradient_x_wrap(I, J, f, fx);

  return 0.5f;
}

#define FILTER_SCALE (1 / (1 + 33.3f + 1))
#define FILTER_KERNEL \
  fx1[j1] =         (f2[j0] - f0[j0]) \
          + 3.33f * (f2[j1] - f0[j1]) \
          +         (f2[j2] - f0[j2]); \
  fy1[j1] =         (f0[j2] - f0[j0]) \
          + 3.33f * (f1[j2] - f1[j0]) \
          +         (f2[j2] - f2[j0]);

template<size_t R>
inline float scharr_gradient_ (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx,
    float * restrict fy)
{
  ASSERT_LT(2 * R, I);
  ASSERT_LT(2 * R, J);

  for (size_t i = 0; i < I; ++i) {
    size_t i0 = i < R ? 0 : i - R;
    size_t i1 = i;
    size_t i2 = i + R < I ? i + R : I - 1;

    const float * restrict f0 = f + J * i0;
    const float * restrict f1 = f + J * i1;
    const float * restrict f2 = f + J * i2;
    float * restrict fx1 = fx + J * i1;
    float * restrict fy1 = fy + J * i1;

    for (size_t j = 0; j < R; ++j) {
      size_t j0 = 0;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = R; j < J - R; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = J - R; j < J; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = J - 1;

      FILTER_KERNEL
    }
  }

  return FILTER_SCALE;
}

float scharr_gradient (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx,
    float * restrict fy)
{
  return scharr_gradient_<1>(I, J, f, fx, fy);
}

float scharr_gradient_wrap_repeat (
    const size_t I,
    const size_t J,
    const float * restrict f,
    float * restrict fx,
    float * restrict fy)
{
  for (size_t i = 0; i < I; ++i) {

    const float * restrict f0 = f + J * Wrap::prev(i,I);
    const float * restrict f1 = f + J * i;
    const float * restrict f2 = f + J * Wrap::next(i,I);

    float * restrict fx1 = fx + J * i;
    float * restrict fy1 = fy + J * i;

    { size_t j = 0;

      size_t j0 = j;
      size_t j1 = j;
      size_t j2 = j + 1;

      FILTER_KERNEL
    }

    for (size_t j = 1; j < J - 1; ++j) {

      size_t j0 = j - 1;
      size_t j1 = j;
      size_t j2 = j + 1;

      FILTER_KERNEL
    }

    { size_t j = J - 1;

      size_t j0 = j - 1;
      size_t j1 = j;
      size_t j2 = j;

      FILTER_KERNEL
    }
  }

  return FILTER_SCALE;
}

#undef FILTER_KERNEL
#undef FILTER_SCALE

// Schurr used the weighting [3 10 3]
#define FILTER_KERNEL \
  fxx1[j1] =         (fx2[j0] - fx0[j0]) \
           + 3.33f * (fx2[j1] - fx0[j1]) \
           +         (fx2[j2] - fx0[j2]); \
  fxy1[j1] =         (fx0[j2] - fx0[j0]) \
           + 3.33f * (fx1[j2] - fx1[j0]) \
           +         (fx2[j2] - fx2[j0]); \
  fyy1[j1] =         (fy0[j2] - fy0[j0]) \
           + 3.33f * (fy1[j2] - fy1[j0]) \
           +         (fy2[j2] - fy2[j0]);

template<size_t R>
void scharr_hessian_ (
    const size_t I,
    const size_t J,
    const float * restrict fx,
    const float * restrict fy,
    float * restrict fxx,
    float * restrict fxy,
    float * restrict fyy)
{
  ASSERT_LT(2 * R, I);
  ASSERT_LT(2 * R, J);

  for (size_t i = 0; i < I; ++i) {
    size_t i0 = i < R ? 0 : i - R;
    size_t i1 = i;
    size_t i2 = i + R < I ? i + R : I - 1;

    const float * restrict fx0 = fx + J * i0;
    const float * restrict fx1 = fx + J * i1;
    const float * restrict fx2 = fx + J * i2;
    const float * restrict fy0 = fy + J * i0;
    const float * restrict fy1 = fy + J * i1;
    const float * restrict fy2 = fy + J * i2;
    float * restrict fxx1 = fxx + J * i1;
    float * restrict fxy1 = fxy + J * i1;
    float * restrict fyy1 = fyy + J * i1;

    for (size_t j = 0; j < R; ++j) {
      size_t j0 = 0;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = R; j < J - R; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = J - R; j < J; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = J - 1;

      FILTER_KERNEL
    }
  }
}

#undef FILTER_KERNEL

#define FILTER_KERNEL \
  float fxx = 3  * (fx2[j0] - fx0[j0]) \
           + 10 * (fx2[j1] - fx0[j1]) \
           + 3  * (fx2[j2] - fx0[j2]); \
  float fyy = 3  * (fy0[j2] - fy0[j0]) \
           + 10 * (fy1[j2] - fy1[j0]) \
           + 3  * (fy2[j2] - fy2[j0]); \
  trace1[j1] = max(fxx, 0.0f) + max(fyy, 0.0f);

template<size_t R>
void scharr_trace_hessian_ (
    const size_t I,
    const size_t J,
    const float * restrict fx,
    const float * restrict fy,
    float * restrict trace)
{
  ASSERT_LT(2 * R, I);
  ASSERT_LT(2 * R, J);

  for (size_t i = 0; i < I; ++i) {
    size_t i0 = i < R ? 0 : i - R;
    size_t i1 = i;
    size_t i2 = i + R < I ? i + R : I - 1;

    const float * restrict fx0 = fx + J * i0;
    //const float * restrict fx1 = fx + J * i1;
    const float * restrict fx2 = fx + J * i2;
    const float * restrict fy0 = fy + J * i0;
    const float * restrict fy1 = fy + J * i1;
    const float * restrict fy2 = fy + J * i2;
    float * restrict trace1 = trace + J * i1;

    for (size_t j = 0; j < R; ++j) {
      size_t j0 = 0;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = R; j < J - R; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = j + R;

      FILTER_KERNEL
    }

    for (size_t j = J - R; j < J; ++j) {
      size_t j0 = j - R;
      size_t j1 = j;
      size_t j2 = J - 1;

      FILTER_KERNEL
    }
  }
}

#undef FILTER_KERNEL

//----( maxima )--------------------------------------------------------------

// coordinate-wise approximation of optimal blob center
inline float subpixel_offset (float left, float center, float right)
{
  float grad = (right - left) / 2;
  float hessian = left + right - 2 * center;
  float optimum = -grad / hessian;

  ASSERT2_LT(-1, optimum);
  ASSERT2_LT(optimum, 1);

  return optimum;
}

void find_peaks (
    const size_t I,
    const size_t J,
    const size_t peak_capacity,
    const float min_value,
    const float * restrict frame,
    Peaks & peaks)
{
  const int di = J;
  const int dj = 1;

  peaks.clear();

  for (size_t i = 1; i < I - 1; ++i)
  for (size_t j = 1; j < J - 1; ++j)
  {
    const float * f = frame + J * i + j;

    float z = f[0];
    bool maximal = (z > min_value)
               and (z > f[-di -dj])
               and (z > f[    -dj])
               and (z > f[+di -dj])
               and (z > f[-di    ])
               and (z > f[+di    ])
               and (z > f[-di +dj])
               and (z > f[    +dj])
               and (z > f[+di +dj]);

    if (maximal) {
      float x = i + 0.5f + subpixel_offset(f[-di], z, f[di]);
      float y = j + 0.5f + subpixel_offset(f[-dj], z, f[dj]);

      peaks.push_back(Peak(x, y, z));
    }
  }

  if (peaks.size() > peak_capacity) {
    std::nth_element(peaks.begin(),
                     peaks.begin() + peak_capacity,
                     peaks.end());
    peaks.resize(peak_capacity);
  }
}

//----( blobs )---------------------------------------------------------------

inline float mean_over_square (
    const size_t X,
    const size_t Y,
    const size_t x,
    const size_t y,
    const size_t r,
    const float * restrict integral)
{
  ASSERT_LT(0, r);

  std::ptrdiff_t x0 = r <= x ? x - r : 0;
  std::ptrdiff_t x1 = x + r < X ? x + r : X - 1;
  std::ptrdiff_t y0 = r <= y ? y - r : 0;
  std::ptrdiff_t y1 = y + r < Y ? y + r : Y - 1;

  return ( integral[x1 * Y + y1]
         - integral[x0 * Y + y1]
         + integral[x0 * Y + y0]
         - integral[x1 * Y + y0] )
       / ( (x1 - x0) * (y1 - y0) );
}

void measure_extent (
    const size_t width,
    const size_t height,
    const size_t x0,
    const size_t y0,
    const float * restrict integral,
    size_t & radius_guess)
{
  ASSERT_LT(2 * radius_guess, width);
  ASSERT_LT(2 * radius_guess, height);

  const size_t X = width;
  const size_t Y = height;
  const size_t max_radius = 2 * radius_guess;

  float best_response = -INFINITY;
  for (size_t r = 1; r <= max_radius; ++r) {

    float response = mean_over_square(X,Y,x0,y0,  r  ,integral)
                  - mean_over_square(X,Y,x0,y0, 2*r ,integral);

    if (response > best_response) {
      best_response = response;
      radius_guess = r;
    }
  }
}

void extract_blob (
    const size_t width,
    const size_t height,
    const size_t x0,
    const size_t y0,
    const size_t radius,
    const float * restrict image,
    Blob & blob)
{
  ASSERT_LE(radius, x0);
  ASSERT_LE(x0 + radius + 1, width);
  ASSERT_LE(radius, y0);
  ASSERT_LE(y0 + radius + 1, height);

  float m00 = 0;
  float m10 = 0;
  float m01 = 0;
  float m20 = 0;
  float m11 = 0;
  float m02 = 0;

  std::ptrdiff_t R = radius;
  float R2 = sqr(R + 0.5f);
  for (std::ptrdiff_t dx = -R; dx <= R; ++dx) { size_t x = x0 + dx;
  for (std::ptrdiff_t dy = -R; dy <= R; ++dy) { size_t y = y0 + dy;
    if (sqr(dx) + sqr(dy) > R2) continue;

    float p = image[x * height + y];

    m00 += p;
    m10 += p * dx;
    m01 += p * dy;
    m20 += p * (dx * dx);
    m11 += p * (dx * dy);
    m02 += p * (dy * dy);
  }}

  blob.x = m10 / m00;
  blob.y = m01 / m00;

  blob.xx = m20 / m00 - sqr(blob.x);
  blob.xy = m11 / m00 - blob.x * blob.y;
  blob.yy = m02 / m00 - sqr(blob.y);

  blob.x += x0;
  blob.y += y0;
}

//----( moments )-------------------------------------------------------------

void extract_moments (
    const size_t I,
    const size_t J,
    const float * restrict frame,
    float & mass_out,
    float & Ex_out,
    float & Ey_out)
{
  float mass = 0;
  float sum_mx = 0;
  float sum_my = 0;

  for (size_t i = 0; i < I; ++i) {
    float x = (i + 0.5f) / I;

    for (size_t j = 0; j < J; ++j) {
      float y = (j + 0.5f) / J;

      float m = frame[J * i + j];

      mass += m;
      sum_mx += m * x;
      sum_my += m * y;
    }
  }

  mass_out = mass;
  Ex_out = (mass > 0 ? sum_mx / mass : 0.0f);
  Ey_out = (mass > 0 ? sum_my / mass : 0.0f);
}

void local_moments_axis (
    const size_t I,
    const size_t J,
    const size_t R,
    float * restrict mass,
    float * restrict mx,
    float * restrict temp)
{
  const float x0 = 0.5f * I;

  for (size_t i = 0; i < I; ++i) {
    float x = x0 - i;

    for (size_t j = 0; j < J; ++j) {
      size_t ij = J * i + j;

      mx[ij] = mass[ij] * x;
    }
  }

  linear_blur_axis(I, J, R, mx, temp);
  linear_blur_axis(I, J, R, mass, temp);

  for (size_t i = 0; i < I; ++i) {
    float x = x0 - i;

    for (size_t j = 0; j < J; ++j) {
      size_t ij = J * i + j;

      mx[ij] -= mass[ij] * x;
    }
  }
}

void local_moments_transpose (
    const size_t I,
    const size_t J,
    const size_t R,
    float * restrict mass,
    float * restrict mx,
    float * restrict my,
    float * restrict temp)
{
  local_moments_axis(I, J, R, mass, temp, mx);
  transpose_8(I, J, temp, mx);
  linear_blur_axis(J, I, R, mx, temp);

  transpose_8(I, J, mass, temp);
  mass = temp;
  local_moments_axis(J, I, R, mass, my, temp);
}

void moments_along_y (
    const size_t I,
    const size_t J,
    const float * restrict mass,
    float * restrict sum_m,
    float * restrict sum_mx)
{
  for (size_t j = 0; j < J; ++j) {
    sum_m[j] = 0;
    sum_mx[j] = 0;
  }

  for (size_t i = 0; i < I; ++i) {
    float x = (i + 0.5f) / I;

    for (size_t j = 0; j < J; ++j) {
      float m = mass[J * i + j];

      sum_m[j] += m;
      sum_mx[j] += m * x;
    }
  }
}

void moments_along_x (
    const size_t I,
    const size_t J,
    const float * restrict mass,
    const float * restrict y,
    float * restrict sum_m,
    float * restrict sum_my)
{
  for (size_t i = 0; i < I; ++i) {
    sum_m[i] = 0;
    sum_my[i] = 0;
  }

  for (size_t i = 0; i < I; ++i) {
    float sum_m_i = 0;
    float sum_my_i = 0;

    for (size_t j = 0; j < J; ++j) {
      size_t ij = J * i + j;

      float m = mass[ij];
      sum_m_i += m;
      sum_my_i += m * y[j];
    }

    sum_m[i] = sum_m_i;
    sum_my[i] += sum_my_i;
  }
}

Peak extract_moments_at (
    const size_t I,
    const size_t J,
    const float radius,
    float x0,
    float y0,
    const float * restrict mass,
    const float tol)
{
  ASSERT2_LE(0, x0);
  ASSERT2_LE(x0, I);
  ASSERT2_LE(0, y0);
  ASSERT2_LE(y0, I);

  x0 -= 0.5f;
  y0 -= 0.5f;

  size_t i0 = x0 > radius ? roundu(ceil(x0 - radius)) : 0;
  size_t j0 = y0 > radius ? roundu(ceil(y0 - radius)) : 0;
  size_t i1 = x0 + radius < I - 1 ? roundu(floor(x0 + radius)) : I - 1;
  size_t j1 = y0 + radius < J - 1 ? roundu(floor(y0 + radius)) : J - 1;

  float sum_m = 0;
  float sum_mx = 0;
  float sum_my = 0;

  for (size_t i = i0; i < i1; ++i) {
    float dx = (i - x0) / radius;
    float envelope_x = 1 - sqr(dx);

    for (size_t j = j0; j < j1; ++j) {
      float dy = (j - y0) / radius;
      float envelope_y = 1 - sqr(dy);

      float m = envelope_x * envelope_y * mass[J * i + j];

      sum_m += m;
      sum_mx += m * dx;
      sum_my += m * dy;
    }
  }

  return Peak(sum_mx / (sum_m + tol), sum_my / (sum_m + tol), sum_m);
}

void extract_moments_at (
    const size_t width,
    const size_t height,
    const float radius,
    const float * restrict mass,
    const Peaks & positions,
    Peaks & moments)
{
  size_t N = positions.size();

  moments.resize(N);

  for (size_t n = 0; n < N; ++n) {
    const Peak & position = positions[n];
    moments[n] = extract_moments_at(
        width,
        height,
        radius,
        position.x,
        position.y,
        mass);
  }
}

//----( sharpen filter )------------------------------------------------------

void sharpen (
    const size_t I,
    const size_t J,
    const size_t R,
    float * restrict image,
    float * restrict sharp,
    float * restrict temp,
    bool dark)
{
  copy_float(image, sharp, I*J);

  // lowpass filter
  quadratic_blur_axis(I,J,R, image, temp);
  transpose_8(I,J, temp, image);
  quadratic_blur_axis(J,I,R, image, temp);
  transpose_8(J,I, temp, image);

  // highpass filter
  float scale = powf(2 * R + 1, -6);
  if (dark) {
    for (size_t ij = 0; ij < I*J; ++ij) {
      sharp[ij] = max(0.0f, scale * image[ij] - sharp[ij]);
    }
  } else {
    for (size_t ij = 0; ij < I*J; ++ij) {
      sharp[ij] = max(0.0f, sharp[ij] - scale * image[ij]);
    }
  }
}

//----( high dynamic range )--------------------------------------------------

void hdr_real (
    const size_t I,
    const size_t J,
    const size_t R,
    float * restrict image,
    float * restrict blur1,
    float * restrict blur2,
    float * restrict temp)
{
  size_t IJ = I * J;

  // accumulate x
  copy_float(image, blur1, IJ);
  quadratic_blur_scaled(I,J,R, blur1, temp);

  // accumulate x^2
  for (size_t ij = 0; ij < IJ; ++ij) {
    blur2[ij] = sqr(image[ij]);
  }
  quadratic_blur_scaled(I,J,R, blur2, temp);

  float tol = 1e-3;
  for (size_t ij = 0; ij < IJ; ++ij) {
    float mean = blur1[ij];
    float var = blur2[ij] - sqr(mean) + tol;

    image[ij] = (image[ij] - mean) / sqrtf(var);
  }
}

void hdr_01 (
    const size_t I,
    const size_t J,
    const size_t R,
    float * restrict image,
    float * restrict temp1,
    float * restrict temp2,
    float * restrict temp3)
{
  size_t IJ = I * J;

  Vector<float> vect(IJ, image);
  affine_to_01(vect);

  float scale = M_PI * 0.99f;
  float shift = -0.5f;
  for (size_t ij = 0; ij < IJ; ++ij) {
    image[ij] = tanf(scale * (shift + image[ij]));
  }

  hdr_real(I,J,R, image, temp1, temp2, temp3);

  scale = 1.0 / M_PI;
  shift = 0.5f;
  for (size_t ij = 0; ij < IJ; ++ij) {
    image[ij] = scale * atanf(image[ij]) + shift;
  }
}

void rgb_to_xyz (
    float * restrict r,
    float * restrict g,
    float * restrict b,
    size_t K)
{
  for (size_t k = 0; k < K; ++k) {

    float x = 0.412453f * r[k] + 0.357580f * g[k] + 0.180423f * b[k];
    float y = 0.212671f * r[k] + 0.715160f * g[k] + 0.072169f * b[k];
    float z = 0.019334f * r[k] + 0.119193f * g[k] + 0.950227f * b[k];

    r[k] = x;
    g[k] = y;
    b[k] = z;
  }
}

void xyz_to_rgb (
    float * restrict x,
    float * restrict y,
    float * restrict z,
    size_t K)
{
  for (size_t k = 0; k < K; ++k) {

    float r =  3.240479f * x[k] + -1.537150f * y[k] + -0.498535f * z[k];
    float g = -0.969256f * x[k] +  1.875992f * y[k] +  0.041556f * z[k];
    float b =  0.055648f * x[k] + -0.204043f * y[k] +  1.057311f * z[k];

    x[k] = r;
    y[k] = g;
    z[k] = b;
  }
}

void hdr_real_color (
    const size_t I,
    const size_t J,
    float * restrict r_data,
    float * restrict g_data,
    float * restrict b_data)
{
  const size_t K = I*J;
  float luminance_scale = 4.0f;

  Vector<float> r(K, r_data);
  Vector<float> g(K, g_data);
  Vector<float> b(K, b_data);
  float * restrict rgb[3] = {r,g,b};
  rgb_to_xyz(r,g,b, K);
  g /= luminance_scale;

  Vector<float> dr(K);  dr.zero();
  Vector<float> dg(K);  dg.zero();
  Vector<float> db(K);  db.zero();
  float * restrict drgb[3] = {dr,dg,db};

  Vector<float> Er(K), Eg(K), Eb(K);
  float * restrict E1[3] = {Er, Eg, Eb};

  Vector<float> Err(K), Erg(K), Erb(K), Egg(K), Egb(K), Ebb(K);
  float * restrict E2[6] = {Err, Erg, Erb, Egg, Egb, Ebb};

  Vector<float> temp(K);

  const float radius = 0.5f * min(I,J) - 1;
  std::vector<size_t> radii;
  for (int level = 0;; ++level) {
    size_t R = floor(radius / (1 << level));
    if (R) radii.push_back(R);
    else break;
  }
  std::reverse(radii.begin(), radii.end());

  for (size_t i = 0; i < radii.size(); ++i) {
    size_t R = radii[i];
    LOG("whiten at scale " << R);

    LOG(" compute local means");
    for (size_t c = 0; c < 3; ++c) {

      Vector<float> x(K, rgb[c]);
      Vector<float> Ex(K, E1[c]);

      quadratic_blur_scaled(I,J,R/4, x,temp);
      Ex = x;
      quadratic_blur_scaled(I,J,R, Ex,temp);
    }

    LOG(" compute local covariances");
    const int fst[6] = {0,0,0,1,1,2};
    const int snd[6] = {0,1,2,1,2,2};
    for (size_t cc = 0; cc < 6; ++cc) {
      size_t c1 = fst[cc];
      size_t c2 = snd[cc];

      Vector<float> x(K, rgb[c1]);
      Vector<float> y(K, rgb[c2]);
      Vector<float> Ex(K, E1[c1]);
      Vector<float> Ey(K, E1[c2]);
      Vector<float> Exy(K, E2[cc]);

      multiply(x,y, Exy);
      quadratic_blur_scaled(I,J,R, Exy,temp);

      for (size_t k = 0; k < K; ++k) {
        Exy[k] -= Ex[k] * Ey[k];
      }
    }

    LOG(" locally whiten image");
    for (size_t k = 0; k < K; ++k) {

      Sym33::Sym33 cov(Err[k], Erg[k], Erb[k], Egg[k], Egb[k], Ebb[k]);
      Sym33::Sym33 isqrt_cov;
      Sym33::isqrt(cov, isqrt_cov);

      Sym33::float3 diff(r[k] - Er[k], g[k] - Eg[k], b[k] - Eb[k]);
      Sym33::float3 std;
      Sym33::multiply(isqrt_cov, diff, std);

      dr[k] += std[0];
      dg[k] += std[1];
      db[k] += std[2];
    }
  }

  for (size_t c = 0; c < 3; ++c) {
    Vector<float> x(K, rgb[c]);
    Vector<float> dx(K, drgb[c]);

    multiply(1.0f / radii.size(), dx, x);
  }

  g *= luminance_scale;
  xyz_to_rgb(r,g,b, K);
}

//----( floodfill )-----------------------------------------------------------

void vh_convex_floodfill (
    const size_t I,
    const size_t J,
    const float thresh,
    const float * restrict f,
    uint8_t * m)
{
  const uint8_t ALL = 0xFF;
  const uint8_t NONE = 0;
  const uint8_t COLORED = 1 << 0;
  const uint8_t FLOODED = 1 << 1;
  const uint8_t FROM_LEFT = 1 << 2;
  const uint8_t FROM_RIGHT = 1 << 3;
  const uint8_t FROM_ABOVE = 1 << 4;
  const uint8_t FROM_BELOW = 1 << 5;

  LOG("find center intensity");
  size_t i0 = I / 2;
  size_t j0 = J / 2;
  size_t ij0 = J * i0 + j0;
  float f0 = ( f[ij0 - J + 1] + f[ij0 + 1] + f[ij0 + J + 1]
            + f[ij0 - J    ] + f[ij0    ] + f[ij0 + J    ]
            + f[ij0 - J - 1] + f[ij0 - 1] + f[ij0 + J - 1] ) / 9;

  LOG("find all pixels within bounds");
  for (size_t ij = 0; ij < I*J; ++ij) {
    float df = f[ij] - f0;
    m[ij] = (fabs(df) < thresh) ? COLORED
                                : NONE;
  }
  ASSERT_EQ(m[J*i0+j0], 1);

  LOG("flood monotonically out from center");
  m[J*i0+j0] = 3;

  for (size_t r = 0; r < I / 2 - 1; ++r) {
    for (size_t j = 1; j < J - 1; ++j) {
      size_t ij = J * (i0 + r) + j;
      if (COLORED & m[ij])
        if (FLOODED & (m[ij-J-1] | m[ij-J] | m[ij-J+1]))
          m[ij] |= FLOODED;
    }
  }

  for (size_t r = 0; r < I / 2 - 1; ++r) {
    for (size_t j = 1; j < J - 1; ++j) {
      size_t ij = J * (i0 - r) + j;
      if (COLORED & m[ij])
        if (FLOODED & (m[ij+J-1] | m[ij+J] | m[ij+J+1]))
          m[ij] |= FLOODED;
    }
  }

  for (size_t r = 0; r < J / 2 - 1; ++r) {
    for (size_t i = 1; i < I - 1; ++i) {
      size_t ij = J * i + j0 + r;
      if (COLORED & m[ij])
        if (FLOODED & (m[ij-J-1] | m[ij-1] | m[ij+J-1]))
          m[ij] |= FLOODED;
    }
  }

  for (size_t r = 0; r < J / 2 - 1; ++r) {
    for (size_t i = 1; i < I - 1; ++i) {
      size_t ij = J * i + j0 - r;
      if (COLORED & m[ij])
        if (FLOODED & (m[ij-J+1] | m[ij+1] | m[ij+J+1]))
          m[ij] |= FLOODED;
    }
  }

  LOG("convexify in to center");

  for (size_t i = 1; i < I; ++i) {
    for (size_t j = 0; j < J; ++j) {
      size_t ij = J * i + j;
      if ((FLOODED & m[ij]) or (FROM_LEFT & m[ij-J])) m[ij] |= FROM_LEFT;
    }
  }

  for (size_t i = I - 1; i; --i) {
    for (size_t j = 0; j < J; ++j) {
      size_t ij = J * i + j;
      if ((FLOODED & m[ij-J]) or (FROM_RIGHT & m[ij])) m[ij-J] |= FROM_RIGHT;
    }
  }

  for (size_t j = 1; j < J; ++j) {
    for (size_t i = 0; i < I; ++i) {
      size_t ij = J * i + j;
      if ((FLOODED & m[ij]) or (FROM_BELOW & m[ij-1])) m[ij] |= FROM_BELOW;
    }
  }

  for (size_t j = J - 1; j; --j) {
    for (size_t i = 0; i < I; ++i) {
      size_t ij = J * i + j;
      if ((FLOODED & m[ij-1]) or (FROM_ABOVE & m[ij])) m[ij-1] |= FROM_ABOVE;
    }
  }

  LOG("convert to all-or-nothing mask");

  for (size_t ij = 0; ij < I*J; ++ij) {
    bool convex = ( (FROM_LEFT & m[ij]) and (FROM_RIGHT & m[ij]) )
               or ( (FROM_ABOVE & m[ij]) and (FROM_BELOW & m[ij]) );
    m[ij] = convex ? ALL : NONE;
  }
}

//----( change detection )----------------------------------------------------

void change_chi2 (
    const size_t I,
    float * restrict f,
    float * restrict Ef,
    float * restrict Vf,
    const float dt,
    float tol)
{
  const float new_part = 1 - exp(-dt);

  for (size_t i = 0; i < I; ++i) {

    float df = f[i] - Ef[i];
    Ef[i] += new_part * df;

    float df2 = df * df - Vf[i];
    Vf[i] += new_part * df2;

    f[i] = df2 / (Vf[i] + tol);
  }
}

void detect_changes (
    const size_t I,
    float * restrict f,
    float * restrict Ef,
    float * restrict Vf,
    const float dt)
{
  const float new_part = 1 - exp(-dt);

  for (size_t i = 0; i < I; ++i) {

    float df = f[i] - Ef[i];
    Ef[i] += new_part * df;

    float change = df * df - Vf[i];
    Vf[i] += new_part * change;

    f[i] = max(0.0f, change);
  }
}

void detect_change_moment_x (
    const size_t X,
    const size_t Y,
    float * restrict f,
    float * restrict Ef,
    float * restrict Vf,
    float * restrict mass,
    float * restrict moment,
    const float dt)
{
  const float new_part = 1 - exp(-dt);

  for (size_t y = 0; y < Y; ++y) {
    mass[y] = 0;
    moment[y] = 0;
  }

  for (size_t x = 0; x < X; ++x) {
    for (size_t y = 0; y < Y; ++y) {

      size_t i = Y * x + y;

      float df = f[i] - Ef[i];
      Ef[i] += new_part * df;

      float change = df * df - Vf[i];
      Vf[i] += new_part * change;

      float c = max(0.0f, change);
      f[i] = c;
      mass[y] += c;
      moment[y] += c * x;
    }
  }
}

void detect_change_moment_y (
    const size_t X,
    const size_t Y,
    float * restrict f,
    float * restrict Ef,
    float * restrict Vf,
    float * restrict mass,
    float * restrict moment,
    const float dt)
{
  const float new_part = 1 - exp(-dt);

  for (size_t x = 0; x < X; ++x) {

    float sum_c = 0;
    float sum_cy = 0;

    for (size_t y = 0; y < Y; ++y) {

      size_t i = Y * x + y;

      float df = f[i] - Ef[i];
      Ef[i] += new_part * df;

      float change = df * df - Vf[i];
      Vf[i] += new_part * change;

      float c = max(0.0f, change);
      f[i] = c;
      sum_c += c;
      sum_cy += c * y;
    }

    mass[x] = sum_c;
    moment[x] = sum_cy;
  }
}

void update_momentum (
    const size_t size,
    const float * restrict mass_new,
    const float * restrict moment_new,
    float * restrict mass_old,
    float * restrict moment_old,
    float * restrict momentum,
    const float dt)
{
  const float timescale = 1 / dt;

  for (size_t i = 0; i < size; ++i) {
    float m_old = mass_old[i];
    float m_new = mass_new[i];
    float m = min(m_old, m_new);

    float pos_old = moment_old[i] / m_old;
    float pos_new = moment_new[i] / m_new;
    float velocity = timescale * (pos_new - pos_old);
    momentum[i] = m * velocity;

    mass_old[i] = mass_new[i];
  }
}

//----( reassignment )--------------------------------------------------------

// this version simply copies mass at the edges
void reassign_flow (
    const size_t I,
    const size_t J,
    const float * restrict flow_x,
    const float * restrict flow_y,
    const float * restrict mass_in,
    float * restrict mass_out)
{
  for (size_t ij = J; ij < (I - 1) * J; ++ij) {
    mass_out[ij] = 0;
  }

  { size_t i = 0;
    for (size_t j = 0; j < J; ++j) {
      mass_out[J * i + j] = mass_in[J * i + j];
    }
  }

  for (size_t i = 1; i < I - 1; ++i) {

    const float * restrict flow_x1 = flow_x + J * i;
    const float * restrict flow_y1 = flow_y + J * i;
    const float * restrict mass1 = mass_in + J * i;
    float * restrict reas0 = mass_out + J * (i - 1);
    float * restrict reas1 = mass_out + J * (i + 0);
    float * restrict reas2 = mass_out + J * (i + 1);

    { size_t j = 0;
      mass_out[J * i + j] = mass_in[J * i + j];
    }

    for (size_t j = 1; j < J - 1; ++j) {

      // j:J indexes the interval 01
      size_t j0 = j - 1;
      size_t j1 = j;
      size_t j2 = j + 1;

      // compute weights
      float dx = flow_x1[j1]; ASSERT2_LE(1.0f, max(dx, -dx));
      float w0_ = max(0.0f, -dx);
      float w2_ = max(0.0f, dx);
      float w1_ = 1 - (w0_ + w2_);

      float dy = flow_y1[j1]; ASSERT2_LE(1.0f, max(dy, -dy));
      float w_0 = max(0.0f, -dy);
      float w_2 = max(0.0f, dy);
      float w_1 = 1 - (w_0 + w_2);

      // shift mass
      float mass = mass1[j1];
      w_0 *= mass;
      w_1 *= mass;
      w_2 *= mass;

      reas0[j0] += w0_ * w_0;
      reas0[j1] += w0_ * w_1;
      reas0[j2] += w0_ * w_2;
      reas1[j0] += w1_ * w_0;
      reas1[j1] += w1_ * w_1;
      reas1[j2] += w1_ * w_2;
      reas2[j0] += w2_ * w_0;
      reas2[j1] += w2_ * w_1;
      reas2[j2] += w2_ * w_2;
    }

    { size_t j = J - 1;
      mass_out[J * i + j] = mass_in[J * i + j];
    }
  }

  { size_t i = I - 1;
    for (size_t j = 0; j < J; ++j) {
      mass_out[J * i + j] = mass_in[J * i + j];
    }
  }
}

// this version accuratly moves mass at edges
void reassign_flow_wrap_repeat (
    const size_t I,
    const size_t J,
    const float * restrict flow_x,
    const float * restrict flow_y,
    const float * restrict mass_in,
    float * restrict mass_out)
{
  for (size_t ij = 0; ij < I*J; ++ij) {
    mass_out[ij] = 0;
  }

  for (size_t i = 0; i < I; ++i) {

    const float * restrict flow_x1 = flow_x + J * i;
    const float * restrict flow_y1 = flow_y + J * i;
    const float * restrict mass1 = mass_in + J * i;

    float * restrict reas0 = mass_out + J * Wrap::prev(i,I);
    float * restrict reas1 = mass_out + J * i;
    float * restrict reas2 = mass_out + J * Wrap::next(i,I);

    { size_t j = 0;

      // j:J indexes the interval 01
      size_t j1 = j;
      size_t j2 = j + 1;

      // compute weights
      float dx = flow_x1[j1]; ASSERT2_LE(1.0f, max(dx, -dx));
      float w0_ = max(0.0f, -dx);
      float w2_ = max(0.0f, dx);
      float w1_ = 1 - (w0_ + w2_);

      float dy = flow_y1[j1]; ASSERT2_LE(1.0f, max(dy, -dy));
      float w_2 = max(0.0f, dy);
      float w_1 = 1 - w_2;

      // shift mass
      float mass = mass1[j1];
      w_1 *= mass;
      w_2 *= mass;

      reas0[j1] += w0_ * w_1;
      reas0[j2] += w0_ * w_2;
      reas1[j1] += w1_ * w_1;
      reas1[j2] += w1_ * w_2;
      reas2[j1] += w2_ * w_1;
      reas2[j2] += w2_ * w_2;
    }

    for (size_t j = 1; j < J - 1; ++j) {

      // j:J indexes the interval 01
      size_t j0 = j - 1;
      size_t j1 = j;
      size_t j2 = j + 1;

      // compute weights
      float dx = flow_x1[j1]; ASSERT2_LE(1.0f, max(dx, -dx));
      float w0_ = max(0.0f, -dx);
      float w2_ = max(0.0f, dx);
      float w1_ = 1 - (w0_ + w2_);

      float dy = flow_y1[j1]; ASSERT2_LE(1.0f, max(dy, -dy));
      float w_0 = max(0.0f, -dy);
      float w_2 = max(0.0f, dy);
      float w_1 = 1 - (w_0 + w_2);

      // shift mass
      float mass = mass1[j1];
      w_0 *= mass;
      w_1 *= mass;
      w_2 *= mass;

      reas0[j0] += w0_ * w_0;
      reas0[j1] += w0_ * w_1;
      reas0[j2] += w0_ * w_2;
      reas1[j0] += w1_ * w_0;
      reas1[j1] += w1_ * w_1;
      reas1[j2] += w1_ * w_2;
      reas2[j0] += w2_ * w_0;
      reas2[j1] += w2_ * w_1;
      reas2[j2] += w2_ * w_2;
    }

    { size_t j = J - 1;

      // j:J indexes the interval 01
      size_t j0 = j - 1;
      size_t j1 = j;

      // compute weights
      float dx = flow_x1[j1]; ASSERT2_LE(1.0f, max(dx, -dx));
      float w0_ = max(0.0f, -dx);
      float w2_ = max(0.0f, dx);
      float w1_ = 1 - (w0_ + w2_);

      float dy = flow_y1[j1]; ASSERT2_LE(1.0f, max(dy, -dy));
      float w_0 = max(0.0f, -dy);
      float w_1 = 1 - w_0;

      // shift mass
      float mass = mass1[j1];
      w_0 *= mass;
      w_1 *= mass;

      reas0[j0] += w0_ * w_0;
      reas0[j1] += w0_ * w_1;
      reas1[j0] += w1_ * w_0;
      reas1[j1] += w1_ * w_1;
      reas2[j0] += w2_ * w_0;
      reas2[j1] += w2_ * w_1;
    }
  }
}

void reassign_wrap_repeat (
    const size_t I,
    const size_t J,
    const float * restrict mass_in,
    float * restrict mass_out,
    float scale,
    float shift)
{
  scale /= 9;
  imax(shift, TOL);

  for (size_t ij = 0; ij < I*J; ++ij) {
    mass_out[ij] = shift;
  }

  for (size_t i = 0; i < I; ++i) {

    // i:I indexes the circle S1
    size_t i0 = Wrap::prev(i,I);
    size_t i1 = i;
    size_t i2 = Wrap::next(i,I);

    const float * restrict mass0 = mass_in + J * i0;
    const float * restrict mass1 = mass_in + J * i1;
    const float * restrict mass2 = mass_in + J * i2;

    float * restrict reas0 = mass_out + J * i0;
    float * restrict reas1 = mass_out + J * i1;
    float * restrict reas2 = mass_out + J * i2;

    { size_t j = 0;

      // j:J indexes the interval 01
      size_t j1 = j;
      size_t j2 = j + 1;

      // compute local moments
      float m_boundary = mass0[j1] + mass1[j1] + mass2[j1];
      float m_interior = mass0[j2] + mass1[j2] + mass2[j2];

      float m = m_boundary + m_interior; ASSERT2_LE(0, m);

      float mx = ( mass2[j1] + mass2[j2] )
              - ( mass0[j1] + mass0[j2] );
      float my = m_interior;

      float dx = mx / (m + TOL); ASSERT2_LE(-1, dx); ASSERT2_LE(dx, 1);
      float dy = my / (m + TOL); ASSERT2_LE(0, dy); ASSERT2_LE(dy, 1);

      // compute weights
      float w0_ = bound_to(0.0f, 1.0f, -dx);
      float w2_ = bound_to(0.0f, 1.0f, dx);
      float w1_ = 1 - (w0_ + w2_);

      float w_2 = dy;
      float w_1 = 1 - w_2;

      // shift mass
      float mass = scale * (2 * m_boundary + m_interior);
      w_1 *= mass;
      w_2 *= mass;

      reas0[j1] += w0_ * w_1;
      reas0[j2] += w0_ * w_2;
      reas1[j1] += w1_ * w_1;
      reas1[j2] += w1_ * w_2;
      reas2[j1] += w2_ * w_1;
      reas2[j2] += w2_ * w_2;
    }

    for (size_t j = 1; j < J - 1; ++j) {

      // j:J indexes the interval 01
      size_t j0 = j - 1;
      size_t j1 = j;
      size_t j2 = j + 1;

      // compute local moments
      float m_left   = mass0[j0] + mass0[j1] + mass0[j2];
      float m_center = mass1[j0] + mass1[j1] + mass1[j2];
      float m_right  = mass2[j0] + mass2[j1] + mass2[j2];

      float m = m_left + m_center + m_right; ASSERT2_LE(0, m);

      float mx = m_right - m_left;
      float my = ( mass0[j2] + mass1[j2] + mass2[j2] )
              - ( mass0[j0] + mass1[j0] + mass2[j0] );

      float dx = mx / (m + TOL); ASSERT2_LE(-1, dx); ASSERT2_LE(dx, 1);
      float dy = my / (m + TOL); ASSERT2_LE(-1, dy); ASSERT2_LE(dy, 1);

      // compute weights
      float w0_ = bound_to(0.0f, 1.0f, -dx);
      float w2_ = bound_to(0.0f, 1.0f, dx);
      float w1_ = 1 - (w0_ + w2_);

      float w_0 = bound_to(0.0f, 1.0f, -dy);
      float w_2 = bound_to(0.0f, 1.0f, dy);
      float w_1 = 1 - (w_0 + w_2);

      // shift mass
      float mass = scale * m;
      w_0 *= mass;
      w_1 *= mass;
      w_2 *= mass;

      reas0[j0] += w0_ * w_0;
      reas0[j1] += w0_ * w_1;
      reas0[j2] += w0_ * w_2;
      reas1[j0] += w1_ * w_0;
      reas1[j1] += w1_ * w_1;
      reas1[j2] += w1_ * w_2;
      reas2[j0] += w2_ * w_0;
      reas2[j1] += w2_ * w_1;
      reas2[j2] += w2_ * w_2;
    }

    { size_t j = J - 1;

      // j:J indexes the interval 01
      size_t j0 = j - 1;
      size_t j1 = j;

      // compute local moments
      float m_interior = mass0[j0] + mass1[j0] + mass2[j0];
      float m_boundary = mass0[j1] + mass1[j1] + mass2[j1];

      float m = m_interior + m_boundary; ASSERT2_LE(0, m);

      float mx = ( mass2[j0] + mass2[j1] )
              - ( mass0[j0] + mass0[j1] );
      float my = -m_interior;

      float dx = mx / (m + TOL); ASSERT2_LE(-1, dx); ASSERT2_LE(dx, 1);
      float dy = my / (m + TOL); ASSERT2_LE(-1, dy); ASSERT2_LE(dy, 0);

      // compute weights
      float w0_ = bound_to(0.0f, 1.0f, -dx);
      float w2_ = bound_to(0.0f, 1.0f, dx);
      float w1_ = 1 - (w0_ + w2_);

      float w_0 = -dy;
      float w_1 = 1 - w_0;

      // shift mass
      float mass = scale * (m_interior + 2 * m_boundary);
      w_0 *= mass;
      w_1 *= mass;

      reas0[j0] += w0_ * w_0;
      reas0[j1] += w0_ * w_1;
      reas1[j0] += w1_ * w_0;
      reas1[j1] += w1_ * w_1;
      reas2[j0] += w2_ * w_0;
      reas2[j1] += w2_ * w_1;
    }
  }
}

void reassign_wrap_x (
    const size_t I,
    const size_t J,
    const float * restrict mass_in,
    float * restrict mass_out,
    float scale,
    float shift)
{
  scale /= 3;
  imax(shift, TOL);

  for (size_t ij = 0; ij < I*J; ++ij) {
    mass_out[ij] = shift;
  }

  for (size_t i = 0; i < I; ++i) {

    // i:I indexes the circle S1
    size_t i0 = (i+I-1) % I;
    size_t i1 = i;
    size_t i2 = (i+1) % I;

    const float * restrict mass0 = mass_in + J * i0;
    const float * restrict mass1 = mass_in + J * i1;
    const float * restrict mass2 = mass_in + J * i2;

    float * restrict reas0 = mass_out + J * i0;
    float * restrict reas1 = mass_out + J * i1;
    float * restrict reas2 = mass_out + J * i2;

    for (size_t j = 0; j < J; ++j) {

      float m0 = mass0[j];
      float m1 = mass1[j];
      float m2 = mass2[j];

      // compute local moments
      float m = m0 + m1 + m2; ASSERT2_LE(0, m);
      float mx = m2 - m0;
      float dx = mx / (m + TOL); ASSERT2_LE(-1, dx); ASSERT2_LE(dx, 1);

      // compute weights
      float w0 = bound_to(0.0f, 1.0f, -dx);
      float w2 = bound_to(0.0f, 1.0f, dx);
      float w1 = 1 - (w0 + w2);

      // shift mass
      float mass = scale * m;

      reas0[j] += mass * w0;
      reas1[j] += mass * w1;
      reas2[j] += mass * w2;
    }
  }
}

void reassign_repeat_x (
    const size_t I,
    const size_t J,
    const float * restrict mass_in,
    float * restrict mass_out,
    float scale,
    float shift)
{
  scale /= 3;
  imax(shift, TOL);

  for (size_t ij = 0; ij < I*J; ++ij) {
    mass_out[ij] = shift;
  }

  { size_t i = 0;

    // i:I indexes the circle S1
    size_t i1 = i;
    size_t i2 = (i+1) % I;

    const float * restrict mass1 = mass_in + J * i1;
    const float * restrict mass2 = mass_in + J * i2;

    float * restrict reas1 = mass_out + J * i1;
    float * restrict reas2 = mass_out + J * i2;

    for (size_t j = 0; j < J; ++j) {

      float m1 = mass1[j];
      float m2 = mass2[j];

      // compute local moments
      float m = m1 + m2; ASSERT2_LE(0, m);
      float mx = m2;
      float dx = mx / (m + TOL); ASSERT2_LE(-1, dx); ASSERT2_LE(dx, 1);

      // compute weights
      float w2 = bound_to(0.0f, 1.0f, dx);
      float w1 = 1 - w2;

      // shift mass
      float mass = scale * (2 * m1 + m2);

      reas1[j] += mass * w1;
      reas2[j] += mass * w2;
    }
  }

  for (size_t i = 1; i < I - 1; ++i) {

    // i:I indexes the circle S1
    size_t i0 = (i+I-1) % I;
    size_t i1 = i;
    size_t i2 = (i+1) % I;

    const float * restrict mass0 = mass_in + J * i0;
    const float * restrict mass1 = mass_in + J * i1;
    const float * restrict mass2 = mass_in + J * i2;

    float * restrict reas0 = mass_out + J * i0;
    float * restrict reas1 = mass_out + J * i1;
    float * restrict reas2 = mass_out + J * i2;

    for (size_t j = 0; j < J; ++j) {

      float m0 = mass0[j];
      float m1 = mass1[j];
      float m2 = mass2[j];

      // compute local moments
      float m = m0 + m1 + m2; ASSERT2_LE(0, m);
      float mx = m2 - m0;
      float dx = mx / (m + TOL); ASSERT2_LE(-1, dx); ASSERT2_LE(dx, 1);

      // compute weights
      float w0 = bound_to(0.0f, 1.0f, -dx);
      float w2 = bound_to(0.0f, 1.0f, dx);
      float w1 = 1 - (w0 + w2);

      // shift mass
      float mass = scale * m;

      reas0[j] += mass * w0;
      reas1[j] += mass * w1;
      reas2[j] += mass * w2;
    }
  }

  { size_t i = I - 1;

    // i:I indexes the circle S1
    size_t i0 = (i+I-1) % I;
    size_t i1 = i;

    const float * restrict mass0 = mass_in + J * i0;
    const float * restrict mass1 = mass_in + J * i1;

    float * restrict reas0 = mass_out + J * i0;
    float * restrict reas1 = mass_out + J * i1;

    for (size_t j = 0; j < J; ++j) {

      float m0 = mass0[j];
      float m1 = mass1[j];

      // compute local moments
      float m = m0 + m1; ASSERT2_LE(0, m);
      float mx = -m0;
      float dx = mx / (m + TOL); ASSERT2_LE(-1, dx); ASSERT2_LE(dx, 1);

      // compute weights
      float w0 = bound_to(0.0f, 1.0f, -dx);
      float w1 = 1 - w0;

      // shift mass
      float mass = scale * (m0 + 2 * m1);

      reas0[j] += mass * w0;
      reas1[j] += mass * w1;
    }
  }
}

//----( optical flow )--------------------------------------------------------

void local_optical_flow_1d (
    const size_t I,
    const float * restrict im0_highpass,
    const float * restrict im1_highpass,
    float * restrict surprise_x,
    float * restrict info_xx,
    float * restrict temp)
{
  LOG1("compute time-deriv and time-average");
  for (size_t i = 0; i < I; ++i) {
    float i0 = im0_highpass[i];
    float i1 = im1_highpass[i];

    surprise_x[i] = (i0 + i1) / 2;
    info_xx[i] = i0 - i1;
  }

  LOG1("compute space-derivs");
  gradient_1d(I, surprise_x, temp);

  LOG1("compute surprise and info");
  for (size_t i = 0; i < I; ++i) {
    float dx = temp[i];
    float neg_dt = info_xx[i];

    surprise_x[i] = neg_dt * dx;
    info_xx[i] = dx * dx;
  }
}

void krig_optical_flow_1d (
    const size_t I,
    const float R,
    const float dt,
    float * restrict surprise_x,
    float * restrict info_xx,
    float * restrict flow_x,
    const float prior_info)
{
  if (R > 0) {
    exp_blur_1d_zero(I, R, surprise_x, flow_x);
    exp_blur_1d_zero(I, R, info_xx, flow_x);
  }

  float fps = 1 / dt;

  for (size_t i = 0; i < I; ++i) {

    // covariance = inverse(fisher info)
    float covxx = 1.0f / (info_xx[i] + prior_info);

    // mean flow = surprise * covariance
    float sx = surprise_x[i] * fps;

    flow_x[i] = covxx * sx;
  }
}

void local_optical_flow (
    const size_t I,
    const size_t J,
    const float * restrict im0_highpass,
    const float * restrict im1_highpass,
    float * restrict surprise_x,
    float * restrict surprise_y,
    float * restrict im_sum,
    float * restrict im_diff)
{
  LOG1("compute time-deriv and time-average");
  for (size_t ij = 0; ij < I*J; ++ij) {
    float i0 = im0_highpass[ij];
    float i1 = im1_highpass[ij];

    im_sum[ij] = (i0 + i1) / 2;
    im_diff[ij] = i0 - i1;
  }

  LOG1("compute space-derivs");
  gradient(I, J, im_sum, surprise_x, surprise_y);

  LOG1("compute surprise");
  for (size_t ij = 0; ij < I*J; ++ij) {
    float neg_dt = im_diff[ij];

    surprise_x[ij] *= neg_dt;
    surprise_y[ij] *= neg_dt;
  }
}

void local_optical_flow (
    const size_t I,
    const size_t J,
    const float * restrict im0_highpass,
    const float * restrict im1_highpass,
    float * restrict surprise_x,
    float * restrict surprise_y,
    float * restrict info_xx,
    float * restrict info_xy,
    float * restrict info_yy)
{
  LOG1("compute time-deriv and time-average");
  for (size_t ij = 0; ij < I*J; ++ij) {
    float i0 = im0_highpass[ij];
    float i1 = im1_highpass[ij];

    info_xx[ij] = (i0 + i1) / 2;
    info_xy[ij] = i0 - i1;
  }

  LOG1("compute space-derivs");
  gradient(I, J, info_xx, surprise_x, surprise_y);

  LOG1("compute surprise and info");
  for (size_t ij = 0; ij < I*J; ++ij) {
    float dx = surprise_x[ij];
    float dy = surprise_y[ij];
    float neg_dt = info_xy[ij];

    surprise_x[ij] = neg_dt * dx;
    surprise_y[ij] = neg_dt * dy;

    info_xx[ij] = dx * dx;
    info_xy[ij] = dx * dy;
    info_yy[ij] = dy * dy;
  }
}

void local_optical_flow_pyramid (
    const size_t I,
    const size_t J,
    const float * restrict im_sum,
    const float * restrict im_diff,
    float * restrict surprise_x,
    float * restrict surprise_y,
    float * restrict temp1,
    float * restrict temp2,
    float * restrict temp3)
{
  LOG1("compute space-derivs");
  gradient(I, J, im_sum, surprise_x, surprise_y);

  LOG1("compute surprise and info");
  for (size_t ij = 0; ij < I*J; ++ij) {
    float neg_dt = im_diff[ij];

    surprise_x[ij] *= neg_dt;
    surprise_y[ij] *= neg_dt;
  }

  LOG1("decide whether at smallest pyramid");
  enum { smallest_level = 8 };
  bool at_smallest_level = (I % 2) || (I / 2 < smallest_level)
                        || (J % 2) || (J / 2 < smallest_level);
  if (at_smallest_level) return;

  ASSERT_DIVIDES(4, I * J);
  const size_t IJ4 = I * J / 4;

  float * restrict small_sum = temp1 + IJ4 * 0;
  float * restrict small_diff = temp1 + IJ4 * 1;
  float * restrict small_x = temp1 + IJ4 * 2;
  float * restrict small_y = temp1 + IJ4 * 3;
  float * restrict small1 = temp2 + IJ4 * 0;
  float * restrict small2 = temp2 + IJ4 * 1;
  float * restrict small3 = temp2 + IJ4 * 2;

  scale_by_half(I, J, im_sum, small_sum);
  scale_by_half(I, J, im_diff, small_diff);
  local_optical_flow_pyramid(
      I / 2,
      J / 2,
      small_sum,
      small_diff,
      small_x,
      small_y,
      small1,
      small2,
      small3);

  LOG1("account for misc factors");
  float small_scale = 1.0f / ( sqr(1+1+1)  // scale_by_half above
                            * 4           // square_blur below
                            );
  for (size_t ij = 0; ij < IJ4; ++ij) {
    small_x[ij] *= small_scale;
    small_y[ij] *= small_scale;
  }

  scale_by_two(I/2, J/2, small_x, temp2);
  square_blur(I,J,1, temp2, temp3);
  for (size_t ij = 0; ij < I*J; ++ij) {
    surprise_x[ij] += temp2[ij];
  }

  scale_by_two(I/2, J/2, small_y, temp2);
  square_blur(I,J,1, temp2, temp3);
  for (size_t ij = 0; ij < I*J; ++ij) {
    surprise_y[ij] += temp2[ij];
  }
}

void local_optical_flow_pyramid (
    const size_t I,
    const size_t J,
    const float * restrict im0,
    const float * restrict im1,
    float * restrict im_sum,
    float * restrict im_diff,
    float * restrict surprise_x,
    float * restrict surprise_y,
    float * restrict temp1,
    float * restrict temp2,
    float * restrict temp3)
{
  LOG1("compute time-deriv and time-average");
  for (size_t ij = 0; ij < I*J; ++ij) {
    float i0 = im0[ij];
    float i1 = im1[ij];

    im_sum[ij] = (i0 + i1) / 2;
    im_diff[ij] = i0 - i1;
  }

  LOG1("recurse down pryamid");
  local_optical_flow_pyramid(
    I, J,
    im_sum, im_diff,
    surprise_x, surprise_y,
    temp1, temp2, temp3);
}

void solve_optical_flow (
    const size_t I,
    const size_t J,
    const float dt,
    const float * restrict surprise_x,
    const float * restrict surprise_y,
    const float * restrict info_xx,
    const float * restrict info_xy,
    const float * restrict info_yy,
    float * restrict flow_x,
    float * restrict flow_y,
    const float prior_info)
{
  float fps = 1 / dt;

  for (size_t ij = 0; ij < I*J; ++ij) {

    // covariance = inverse(fisher info)
    float ixx = info_xx[ij] + prior_info;
    float ixy = info_xy[ij];
    float iyy = info_yy[ij] + prior_info;

    float det = ixx * iyy - ixy * ixy;

    float covxx = iyy / det;
    float covxy = -ixy / det;
    float covyy = ixx / det;

    // mean flow = surprise * covariance
    float sx = surprise_x[ij] * fps;
    float sy = surprise_y[ij] * fps;

    flow_x[ij] = covxx * sx + covxy * sy;
    flow_y[ij] = covxy * sx + covyy * sy;
  }
}

void krig_optical_flow (
    const size_t I,
    const size_t J,
    const float R,
    const float dt,
    float * restrict surprise_x,
    float * restrict surprise_y,
    float * restrict info_xx,
    float * restrict info_xy,
    float * restrict info_yy,
    float * restrict flow_x,
    float * restrict flow_y,
    const float prior_info)
{
  // 3 transposes could be avoided below, at elegance cost

  if (R > 0) {
    exp_blur_zero(I, J, R, surprise_x, flow_x);
    exp_blur_zero(I, J, R, surprise_y, flow_x);
    exp_blur_zero(I, J, R, info_xx, flow_x);
    exp_blur_zero(I, J, R, info_xy, flow_x);
    exp_blur_zero(I, J, R, info_yy, flow_x);
  }

  float fps = 1 / dt;

  for (size_t ij = 0; ij < I*J; ++ij) {

    // covariance = inverse(fisher info)
    float ixx = info_xx[ij] + prior_info;
    float ixy = info_xy[ij];
    float iyy = info_yy[ij] + prior_info;

    float det = ixx * iyy - ixy * ixy;

    float covxx = iyy / det;
    float covxy = -ixy / det;
    float covyy = ixx / det;

    // mean flow = surprise * covariance
    float sx = surprise_x[ij] * fps;
    float sy = surprise_y[ij] * fps;

    flow_x[ij] = covxx * sx + covxy * sy;
    flow_y[ij] = covxy * sx + covyy * sy;
  }
}

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
    float * restrict new_iyy)
{
  // scale & bound flow for reassign_flow
  for (size_t ij = 0; ij < I*J; ++ij) {
    dx[ij] = bound_to(-1.0f, 1.0f, dx[ij] * dt);
    dy[ij] = bound_to(-1.0f, 1.0f, dy[ij] * dt);
  }

  reassign_flow(I, J, dx, dy, old_sx, new_sx);
  reassign_flow(I, J, dx, dy, old_sy, new_sy);
  reassign_flow(I, J, dx, dy, old_ixx, new_ixx);
  reassign_flow(I, J, dx, dy, old_ixy, new_ixy);
  reassign_flow(I, J, dx, dy, old_iyy, new_iyy);
}

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
    float * restrict flow_y)
{
  // E0,V0 <- S0/I0,1/I0
  // V0 <- V0 + noise
  // S0,I0 <- E0/V0,1/V0
  // S0,I0 <- S0+S1,I0+I1
  // E <- S0/I0

  float fps = 1 / dt;
  float noise = process_noise * dt;

  for (size_t i = 0; i < I; ++i) {

    float Sx = old_sx[i];
    float Sy = old_sy[i];
    float Ixx = old_ixx[i];
    float Ixy = old_ixy[i];
    float Iyy = old_iyy[i];

    float det = Ixx * Iyy - Ixy * Ixy;

    float Vxx =  Iyy / det;
    float Vxy = -Ixy / det;
    float Vyy =  Ixx / det;

    float Ex = Vxx * Sx + Vxy * Sy;
    float Ey = Vxy * Sx + Vyy * Sy;

    Vxx += noise;
    Vyy += noise;

    det = Vxx * Vyy - Vxy * Vxy;

    Ixx =  Vyy / det;
    Ixy = -Vxy / det;
    Iyy =  Vxx / det;

    Sx = Ixx * Ex + Ixy * Ey;
    Sy = Ixy * Ex + Iyy * Ey;

    new_sx[i] = Sx += new_sx[i];
    new_sy[i] = Sy += new_sy[i];
    new_ixx[i] = Ixx += new_ixx[i];
    new_ixy[i] = Ixy += new_ixy[i];
    new_iyy[i] = Iyy += new_iyy[i];

    Ixx += prior_info;
    Iyy += prior_info;

    det = Ixx * Iyy - Ixy * Ixy;

    Vxx =  Iyy / det;
    Vxy = -Ixy / det;
    Vyy =  Ixx / det;

    flow_x[i] = fps * (Vxx * Sx + Vxy * Sy);
    flow_y[i] = fps * (Vxy * Sx + Vyy * Sy);
  }
}

} // namespace Image

