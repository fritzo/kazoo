
#include "vectors.h"
#include <fstream>

#define LOG1(mess)

//----( vector classes )------------------------------------------------------

template<>
void save_to_python (const Vector<float> & x, string filename)
{
  LOG("saving Vector<float> to " << filename);

  std::ofstream file(filename);

  file << "[";
  for (size_t i = 0; i < x.size; ++i) {
    file << "\n  " << x[i] << ",";
  }
  file << "\n]";
}

template<>
void save_to_python (const Vector<uint8_t> & x, string filename)
{
  LOG("saving Vector<uint8_t> to " << filename);

  std::ofstream file(filename);

  file << "[";
  for (size_t i = 0; i < x.size; ++i) {
    file << "\n  " << int(x[i]) << ",";
  }
  file << "\n]";
}

//----( mapped operations )---------------------------------------------------

void uchar_to_01 (const Vector<uint8_t> & u, Vector<float> & v)
{
  const size_t size = u.size;
  ASSERT_SIZE(v, size);

  const uint8_t * restrict u_ = u.data;
  float * restrict v_ = v.data;

  const float over_256 = 1.0f / 256.0f;

  for (size_t i = 0; i < size; ++i) {
    float u = u_[i];
    v_[i] = (u + 0.5f) * over_256;
  }
}

void real_to_uchar (const Vector<float> & x, Vector<uint8_t> & u)
{
  const size_t size = x.size;
  ASSERT_SIZE(u, size);

  const float * restrict x_ = x.data;
  uint8_t * restrict u_ = u.data;

  for (size_t i = 0; i < size; ++i) {
    float x = x_[i];
    float w = x / sqrtf(1 + sqr(x));
    float v = (w + 1) / 2;
    float u = 256 * v;
    u_[i] = static_cast<uint8_t>(u);
    //u_[i] = bound_to(0, 255, int(u));
  }
}

void uchar_to_real (const Vector<uint8_t> & u, Vector<float> & x)
{
  const size_t size = u.size;
  ASSERT_SIZE(x, size);

  const uint8_t * restrict u_ = u.data;
  float * restrict x_ = x.data;

  const float over_256 = 1.0f / 256.0f;

  for (size_t i = 0; i < size; ++i) {
    float u = u_[i];
    float v = (u + 0.5f) * over_256;
    float w = 2 * v - 1;
    x_[i] = w / sqrt(1 - sqr(w));
  }
}

void minimum (float x, const Vector<float> & y, Vector<float> & z)
{
  const size_t size = y.size;
  ASSERT_SIZE(z, size);

  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = min(x, y_[i]);
}

void minimum (
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = min(x_[i], y_[i]);
}

void maximum (float x, const Vector<float> & y, Vector<float> & z)
{
  const size_t size = y.size;
  ASSERT_SIZE(z, size);

  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = max(x, y_[i]);
}

void maximum (
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = max(x_[i], y_[i]);
}

void add (float x, const Vector<float> & y, Vector<float> & z)
{
  const size_t size = y.size;
  ASSERT_SIZE(z, size);

  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = x + y_[i];
}

void add (const Vector<float> & x, const Vector<float> & y, Vector<float> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = x_[i] + y_[i];
}

void add (
    const Vector<complex> & x,
    const Vector<complex> & y,
    Vector<complex> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const complex * restrict x_(x.data);
  const complex * restrict y_(y.data);
  complex * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = x_[i] + y_[i];
}

void subtract (
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = x_[i] - y_[i];
}

void subtract (
    const Vector<complex> & x,
    const Vector<complex> & y,
    Vector<complex> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const complex * restrict x_(x.data);
  const complex * restrict y_(y.data);
  complex * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = x_[i] - y_[i];
}

float dot (const Vector<float> & x, const Vector<float> & y)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);

  float result = 0;
  for (size_t i = 0; i < size; ++i) {
    result += x_[i] * y_[i];
  }
  return result;
}

double dot (const Vector<double> & x, const Vector<float> & y)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);

  const double * restrict x_(x.data);
  const float * restrict y_(y.data);

  double result = 0;
  for (size_t i = 0; i < size; ++i) {
    result += x_[i] * y_[i];
  }
  return result;
}

complex dot (const Vector<complex> & x, const Vector<complex> & y)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);

  const complex * restrict x_(x.data);
  const complex * restrict y_(y.data);

  complex result = 0;
  for (size_t i = 0; i < size; ++i) {
    result += conj(x_[i]) * y_[i];
  }
  return result;
}

void multiply (float x, const Vector<float> & y, Vector<float> & z)
{
  const size_t size = y.size;
  ASSERT_SIZE(z, size);

  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = x * y_[i];
}

void multiply (float x, const Vector<complex> & y, Vector<complex> & z)
{
  const size_t size = y.size;
  ASSERT_SIZE(z, size);

  const complex * restrict y_(y.data);
  complex * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = x * y_[i];
}

void multiply (
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = x_[i] * y_[i];
}

void multiply (
    const Vector<float> & x,
    const Vector<complex> & y,
    Vector<complex> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict x_(x.data);
  const complex * restrict y_(y.data);
  complex * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = x_[i] * y_[i];
}

void multiply_conj (
    const Vector<complex> & x,
    const Vector<complex> & y,
    Vector<complex> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const complex * restrict x_(x.data);
  const complex * restrict y_(y.data);
  complex * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = conj(x_[i]) * y_[i];
}

void multiply_add (float x, const Vector<float> & y, Vector<float> & z)
{
  const size_t size = y.size;
  ASSERT_SIZE(z, size);

  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] += x * y_[i];
}

void multiply_add (
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] += x_[i] * y_[i];
}

void multiply_add (
    const Vector<float> & x,
    const Vector<complex> & y,
    Vector<complex> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict x_(x.data);
  const complex * restrict y_(y.data);
  complex * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] += x_[i] * y_[i];
}

void divide (float x, const Vector<float> & y, Vector<float> & z)
{
  const size_t size = y.size;
  ASSERT_SIZE(z, size);

  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = x / y_[i];
}

void divide (
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = x_[i] / y_[i];
}

void affine_combine (
    float a,
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = a * x_[i] + (1-a) * y_[i];
}
void affine_combine (
    float a,
    const Vector<complex> & x,
    const Vector<complex> & y,
    Vector<complex> & z)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const complex * restrict x_(x.data);
  const complex * restrict y_(y.data);
  complex * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = a * x_[i] + (1-a) * y_[i];
}
void affine_combine (
    const Vector<float> & a,
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z)
{
  const size_t size = a.size;
  ASSERT_SIZE(x, size);
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict a_(a.data);
  const float * restrict x_(x.data);
  const float * restrict y_(y.data);
  float * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = a_[i] * x_[i] + (1-a_[i]) * y_[i];
}
void affine_combine (
    const Vector<float> & a,
    const Vector<complex> & x,
    const Vector<complex> & y,
    Vector<complex> & z)
{
  const size_t size = a.size;
  ASSERT_SIZE(x, size);
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict a_(a.data);
  const complex * restrict x_(x.data);
  const complex * restrict y_(y.data);
  complex * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = a_[i] * x_[i] + (1-a_[i]) * y_[i];
}

void linear_combine (
    const Vector<float> & a,
    const Vector<complex> & x,
    const Vector<float> & b,
    const Vector<complex> & y,
    Vector<complex> & z)
{
  const size_t size = a.size;
  ASSERT_SIZE(b, size);
  ASSERT_SIZE(x, size);
  ASSERT_SIZE(y, size);
  ASSERT_SIZE(z, size);

  const float * restrict a_(a.data);
  const float * restrict b_(b.data);
  const complex * restrict x_(x.data);
  const complex * restrict y_(y.data);
  complex * restrict z_(z.data);

  for (size_t i = 0; i < size; ++i) z_[i] = a_[i] * x_[i] + b_[i] * y_[i];
}

void accumulate_step (
    float factor,
    Vector<float> & x_old,
    const Vector<float> & x_new)
{
  const size_t size = x_old.size;
  ASSERT_SIZE(x_new, size);

  const float * restrict x_new_(x_new.data);
  float * restrict x_old_(x_old.data);

  float a_old = factor;
  float a_new = 1 - factor;

  if (a_new < a_old) {
    for (size_t i = 0; i < size; ++i) {
      x_old_[i] += a_new * (x_new_[i] - x_old_[i]);
    }
  } else {
    for (size_t i = 0; i < size; ++i) {
      x_old_[i] = x_new_[i] + a_old * (x_old_[i] - x_new_[i]);
    }
  }
}

void accumulate_step (
    const Vector<float> & factor,
    Vector<complex> & x_old,
    const Vector<complex> & x_new)
{
  const size_t size = factor.size;
  ASSERT_SIZE(x_old, size);
  ASSERT_SIZE(x_new, size);

  const float * restrict factor_(factor.data);
  const complex * restrict x_new_(x_new.data);
  complex * restrict x_old_(x_old.data);

  for (size_t i = 0; i < size; ++i) {
    float a_old = factor_[i];
    float a_new = 1 - factor_[i];

    x_old_[i] = x_old_[i] * a_old + x_new_[i] * a_new;
  }
}

void exp (
    const Vector<float> & x,
    Vector<float> & y)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);

  const float * restrict x_(x.data);
  float * restrict y_(y.data);

  for (size_t i = 0; i < size; ++i) y_[i] = expf(x_[i]);
}

void exp_inplace (
    Vector<float> & x)
{
  const size_t size = x.size;

  float * restrict x_(x.data);

  for (size_t i = 0; i < size; ++i) x_[i] = expf(x_[i]);
}

void log (
    const Vector<float> & x,
    Vector<float> & y)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);

  const float * restrict x_(x.data);
  float * restrict y_(y.data);

  for (size_t i = 0; i < size; ++i) y_[i] = logf(x_[i]);
}

void log_inplace (
    Vector<float> & x)
{
  const size_t size = x.size;

  float * restrict x_(x.data);

  for (size_t i = 0; i < size; ++i) x_[i] = logf(x_[i]);
}

void lgamma (
    const Vector<float> & x,
    Vector<float> & y)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);

  const float * restrict x_(x.data);
  float * restrict y_(y.data);

  for (size_t i = 0; i < size; ++i) y_[i] = lgammaf(x_[i]);
}

void lgamma_inplace (
    Vector<float> & x)
{
  const size_t size = x.size;

  float * restrict x_(x.data);

  for (size_t i = 0; i < size; ++i) x_[i] = lgammaf(x_[i]);
}

//----( retuctions )----------------------------------------------------------

float sum (const Vector<float> & x)
{
  const size_t size = x.size;

  const float * restrict  x_ = x.data;

  float total = 0;
  for (size_t i = 0; i < size; ++i) total += x_[i];
  return total;
}

double sum (const Vector<double> & x)
{
  const size_t size = x.size;

  const double * restrict  x_ = x.data;

  double total = 0;
  for (size_t i = 0; i < size; ++i) total += x_[i];
  return total;
}

complex sum (const Vector<complex> & x)
{
  const size_t size = x.size;

  const complex * restrict  x_ = x.data;

  complex total = 0;
  for (size_t i = 0; i < size; ++i) total += x_[i];
  return total;
}

uint64_t sum (const Vector<uint8_t> & x)
{
  const size_t size = x.size;

  const uint8_t * restrict  x_ = x.data;

  uint64_t total = 0;
  for (size_t i = 0; i < size; ++i) total += x_[i];
  return total;
}

float mean_wrt (const Vector<float> & x, const Vector<float> & like)
{
  const size_t size = x.size;
  ASSERT_SIZE(like, size);

  const float * restrict like_(like.data);
  const float * restrict x_(x.data);

  float sum_like = 0;
  float sum_like_x = 0;

  for (size_t i = 0; i < size; ++i) {
    float like_i = like_[i];
    float x_i = x_[i];

    sum_like += like_i;
    sum_like_x += like_i * x_i;
  }

  return sum_like_x / sum_like;
}

float dist_squared (const Vector<float> & x, const Vector<float> & y)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);

  float total = 0;
  for (size_t i = 0; i < size; ++i) total += sqr(x_[i] - y_[i]);
  return total;
}

float rms_error (const Vector<float> & x, const Vector<float> & y)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);

  float total = 0;
  for (size_t i = 0; i < size; ++i) total += sqr(x_[i] - y_[i]);

  return sqrt(total / size);
}

float norm_squared (const Vector<float> & x)
{
  const size_t size = x.size;

  const float * restrict x_(x.data);

  float total = 0;
  for (size_t i = 0; i < size; ++i) total += sqr(x_[i]);
  return total;
}

float max (const Vector<float> & x)
{
  const size_t size = x.size;

  const float * restrict x_(x.data);

  float total = x_[0];
  for (size_t i = 0; i < size; ++i) imax(total, x_[i]);
  return total;
}

float min (const Vector<float> & x)
{
  const size_t size = x.size;

  const float * restrict x_(x.data);

  float total = x_[0];
  for (size_t i = 0; i < size; ++i) imin(total, x_[i]);
  return total;
}

double max (const Vector<double> & x)
{
  const size_t size = x.size;

  const double * restrict x_(x.data);

  double total = x_[0];
  for (size_t i = 0; i < size; ++i) imax(total, x_[i]);
  return total;
}

double min (const Vector<double> & x)
{
  const size_t size = x.size;

  const double * restrict x_(x.data);

  double total = x_[0];
  for (size_t i = 0; i < size; ++i) imin(total, x_[i]);
  return total;
}

int max (const Vector<uint8_t> & x)
{
  const size_t size = x.size;

  const uint8_t * restrict x_(x.data);

  uint8_t total = x_[0];
  for (size_t i = 0; i < size; ++i) imax(total, x_[i]);
  return total;
}

int min (const Vector<uint8_t> & x)
{
  const size_t size = x.size;

  const uint8_t * restrict x_(x.data);

  uint8_t total = x_[0];
  for (size_t i = 0; i < size; ++i) imin(total, x_[i]);
  return total;
}

size_t argmin (const Vector<float> & x)
{
  const size_t size = x.size;

  const float * restrict x_(x.data);

  float x_best = x_[0];
  size_t i_best = 0;
  for (size_t i = 0; i < size; ++i) {
    if (x_[i] < x_best) {
      x_best = x_[i];
      i_best = i;
    }
  }

  return i_best;
}

size_t argmax (const Vector<float> & x)
{
  const size_t size = x.size;

  const float * restrict x_(x.data);

  float x_best = x_[0];
  size_t i_best = 0;
  for (size_t i = 0; i < size; ++i) {
    if (x_[i] > x_best) {
      x_best = x_[i];
      i_best = i;
    }
  }

  return i_best;
}

float max_norm_squared (const Vector<float> & x)
{
  const size_t size = x.size;

  const float * restrict x_(x.data);

  float result = 0;
  for (size_t i = 0; i < size; ++i) imax(result, sqr(x_[i]));
  return result;
}

float max_norm_squared (const Vector<complex> & x)
{
  const size_t size = x.size;

  const complex * restrict x_(x.data);

  float result = 0;
  for (size_t i = 0; i < size; ++i) imax(result, norm(x_[i]));
  return result;
}

int max_dist_squared (const Vector<uint8_t> & x, const Vector<uint8_t> & y)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);

  const uint8_t * restrict x_(x.data);
  const uint8_t * restrict y_(y.data);

  int result = 0;
  for (size_t i = 0; i < size; ++i) imax(result, sqr(int(x_[i]) - int(y_[i])));
  return result;
}

float max_dist_squared (const Vector<float> & x, const Vector<float> & y)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);

  float result = 0;
  for (size_t i = 0; i < size; ++i) imax(result, sqr(x_[i] - y_[i]));
  return result;
}

double max_dist_squared (const Vector<double> & x, const Vector<double> & y)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);

  const double * restrict x_(x.data);
  const double * restrict y_(y.data);

  double result = 0;
  for (size_t i = 0; i < size; ++i) imax(result, sqr(x_[i] - y_[i]));
  return result;
}

float density (const Vector<float> & x)
{
  const size_t size = x.size;

  const float * restrict x_(x.data);

  float sum_x1 = 0;
  float sum_x2 = 0;

  for (size_t i = 0; i < size; ++i) {
    float xi = x_[i];

    sum_x1 += max(-xi,xi);
    sum_x2 += xi * xi;
  }

  return sqr(sum_x1) / sum_x2 / size;
}

float entropy (const Vector<float> & x)
{
  const size_t size = x.size;

  const float * restrict x_(x.data);

  float result = 0;
  for (size_t i = 0; i < size; ++i) {

    float xi = x_[i];
    if (xi > 0) {

      result += xi * logf(xi);
    }
  }

  return -result;
}

float relentropy (
    const Vector<float> & x,
    const Vector<float> & y,
    bool non_normalized)
{
  const size_t size = x.size;
  ASSERT_SIZE(y, size);

  const float * restrict x_(x.data);
  const float * restrict y_(y.data);

  float result = 0;
  for (size_t i = 0; i < size; ++i) {

    float xi = x_[i];
    if (xi > 0) {

      float yi = y_[i];
      if (yi > 0) {

        result += xi * logf(xi / yi);
      } else {
        return INFINITY;
      }
    }
  }

  if (non_normalized) result += sum(x) - sum(y);

  return result;
}

void hard_clip (Vector<float> & x)
{
  const size_t size = x.size;

  const float * restrict x_(x.data);

  for (size_t i = 0; i < size; ++i) bound_to(-1.0f, 1.0f, x_[i]);
}

void soft_clip (Vector<float> & x)
{
  const size_t size = x.size;

  float * restrict x_(x.data);

  for (size_t i = 0; i < size; ++i) x_[i] /= 1 + max(x_[i], -x_[i]);
}

void affine_to_01 (Vector<float> & x)
{
  const size_t size = x.size;

  float * restrict x_(x.data);

  float LB = INFINITY;
  float UB = -INFINITY;
  for (size_t i = 0; i < size; ++i) {
    LB = min(LB, x_[i]);
    UB = max(UB, x_[i]);
  }

  float shift = -LB;
  float scale = 1.0f / (UB - LB);
  for (size_t i = 0; i < size; ++i) {
    x_[i] = scale * (x_[i] + shift);
  }
}

