#ifndef KAZOO_VECTORS_H
#define KAZOO_VECTORS_H

#include "common.h"

//----( vector classes )------------------------------------------------------

template<class T> struct Vector;
template<> struct Vector<float>;
template<> struct Vector<complex>;

template<class T> struct CudaVector;

template<class T>
struct Vector
{
  typedef T value_type;

  T * const data;
  const size_t size;
  const bool alias;

  // aliasing
  explicit Vector (size_t s, T * d = NULL)
    : data(d ? d : (T*) malloc_aligned(s * sizeof(T))), size(s), alias(d) {}
  Vector (const Vector<T> & other)
    : data(other.data), size(other.size), alias(true) {}
  ~Vector () { if (not alias) free_aligned(data); }

  // copying
  void operator= (const Vector<T> & other)
  {
    ASSERT_SIZE(other, size);
    for (size_t i = 0; i < size; ++i) { data[i] = other.data[i]; }
  }
  inline void operator= (const CudaVector<T> & other);

  // constant filling
  void zero () { zero_bytes(data, size * sizeof(T)); }
  void set (T value)
  {
    for (size_t i = 0; i < size; ++i) data[i] = value;
  }
  void operator= (T value) DEPRECATED { set(value); }

  // access
  operator const T * () const { return data; }
  operator       T * ()       { return data; }
  Vector<T> block (size_t stride, size_t number = 0)
  {
    ASSERT_DIVIDES(stride, size);
    ASSERT_LT(number, size / stride);

    return Vector<T>(stride, data + stride * number);
  }
  const Vector<T> block (size_t stride, size_t number = 0) const
  {
    ASSERT_DIVIDES(stride, size);
    ASSERT_LT(number, size / stride);

    return Vector<T>(stride, const_cast<T*>(data) + stride * number);
  }

  // stl-style bounds
  typedef T * iterator;
  iterator begin () { return data; }
  iterator end () { return data + size; }
  typedef T * const_iterator;
  const_iterator begin () const { return data; }
  const_iterator end () const { return data + size; }

  void print () const
  {
    for (size_t i = 0; i < size; ++i) { cout << data[i] << "\n"; }
    cout << endl;
  }
};

template<>
struct Vector<float>
{
  typedef float value_type;

  float * const data;
  const size_t size;
  const bool alias;

  // aliasing
  explicit Vector (size_t s, float * d = NULL)
    : data(d ? d : malloc_float(s)), size(s), alias(d) {}
  Vector (const Vector<float> & other)
    : data(other.data), size(other.size), alias(true) {}
  inline explicit Vector (Vector<complex> & other);
  ~Vector<float> () { if (not alias) free_float(data); }

  // copying
  void operator= (const Vector<float> & other)
  {
    ASSERT_SIZE(other, size);
    copy_float(other.data, data, size);
  }
  inline void operator= (const CudaVector<float> & other);

  // constant filling
  void zero () { zero_float(data, size); }
  void set (float value)
  {
    for (size_t i = 0; i < size; ++i) data[i] = value;
  }
  void operator= (float value) DEPRECATED { set(value); }

  // access
  operator const float * () const { return data; }
  operator       float * ()       { return data; }
  Vector<float> block (size_t stride, size_t number = 0)
  {
    ASSERT_DIVIDES(stride, size);
    ASSERT_LT(number, size / stride);

    return Vector<float>(stride, data + stride * number);
  }
  const Vector<float> block (size_t stride, size_t number = 0) const
  {
    ASSERT_DIVIDES(stride, size);
    ASSERT_LT(number, size / stride);

    return Vector<float>(stride, const_cast<float*>(data) + stride * number);
  }

  // stl-style bounds
  typedef float * iterator;
  iterator begin () { return data; }
  iterator end () { return data + size; }
  typedef const float * const_iterator;
  const_iterator begin () const { return data; }
  const_iterator end () const { return data + size; }

  void print () const { print_float(data, size); }
};

template<>
struct Vector<complex>
{
  typedef complex value_type;

  complex * const data;
  const size_t size;
  const bool alias;

  // aliasing
  explicit Vector (size_t s, complex * d = NULL)
    : data(d ? d : malloc_complex(s)), size(s), alias(d) {}
  Vector (const Vector<complex> & other)
    : data(other.data), size(other.size), alias(true) {}
  inline explicit Vector (Vector<float> & other);
  ~Vector<complex> () { if (not alias) free_complex(data); }

  // copying
  void operator= (const Vector<complex> & other)
  {
    ASSERT_SIZE(other, size);
    copy_complex(other.data, data, size);
  }
  inline void operator= (const CudaVector<complex> & other);

  // constant filling
  void zero () { zero_complex(data, size); }
  void set (complex value)
  {
    for (size_t i = 0; i < size; ++i) data[i] = value;
  }
  void operator= (complex value) DEPRECATED { set(value); }

  // access
  operator const complex * () const { return data; }
  operator       complex * ()       { return data; }
  Vector<complex> block (size_t stride, size_t number = 0)
  {
    ASSERT_DIVIDES(stride, size);
    ASSERT_LT(number, size / stride);

    return Vector<complex>(stride, data + stride * number);
  }
  const Vector<complex> block (size_t stride, size_t number = 0) const
  {
    ASSERT_DIVIDES(stride, size);
    ASSERT_LT(number, size / stride);

    return Vector<complex>(
        stride,
        const_cast<complex*>(data) + stride * number);
  }
  Vector<complex> prefix (size_t subsize)
  {
    ASSERT_LE(subsize, size);
    return Vector<complex>(subsize, data);
  }

  // stl-style bounds
  typedef complex * iterator;
  iterator begin () { return data; }
  iterator end () { return data + size; }
  typedef const complex * const_iterator;
  const_iterator begin () const { return data; }
  const_iterator end () const { return data + size; }

  void print () const { print_complex(data, size); }
};
namespace std
{

// deep swapping
template<class T> void swap (Vector<T> & x, Vector<T> & y)
{
  ASSERT_EQ(x.size, y.size);

  T * restrict x_data = x.data;
  T * restrict y_data = y.data;

  for (size_t i = 0, I = x.size; i < I; ++i) {
    T x = x_data[i];
    T y = y_data[i];

    x_data[i] = y;
    y_data[i] = x;
  }
}

} // namespace std

// aliased conversion : real <--> complex
Vector<float>::Vector (Vector<complex> & other)
  : data(reinterpret_cast<float *>(other.data)),
    size(other.size * 2),
    alias(true)
{}

Vector<complex>::Vector (Vector<float> & other)
  : data(reinterpret_cast<complex *>(other.data)),
    size(other.size / 2),
    alias(true)
{}

template<class T> ostream & operator<< (ostream & o, const Vector<T> & v)
{
  for (size_t i = 0; i < v.size; ++i) {
    o << ' ' << v[i];
  }
  return o;
}

template<class T>
void save_to_python (const Vector<T> & x, string filename);

//----( casting )-------------------------------------------------------------

template<class Src, class Dst>
void vector_cast (const Vector<Src> & src, Vector<Dst> & dst)
{
  ASSERT_EQ(src.size, dst.size);

  const Src * restrict src_data = src.data;
  Dst * restrict dst_data = dst.data;

  for (size_t i = 0, I = src.size; i < I; ++i) dst_data[i] = src_data[i];
}

// we map between the real line and [0,255] via the chain
//
//   u:[0,255] <---> v:(0,1) <---> w:(-1,1) <---> x:RR
//
//     u + 0.5                             v
// v = -------;   w = 2 v - 1;   x = -------------
//       256                         sqrt(1 - v^2)
//
//           x              w + 1
// v = -------------;   v = -----;  u = floor(v / 256)
//     sqrt(1 + x^2)          2
//
void uchar_to_01 (const Vector<uint8_t> & u, Vector<float> & v);
void real_to_uchar (const Vector<float> & x, Vector<uint8_t> & u);
void uchar_to_real (const Vector<uint8_t> & u, Vector<float> & x);

//----( in-place operators )--------------------------------------------------

//----( vector, scalar )----

template<class T>
void operator+= (Vector<T> & x, T y)
{
  size_t size = x.size;
  T * restrict x_ = x.data;
  for (size_t i = 0; i < size; ++i) x_[i] += y;
}

template<class T>
void operator-= (Vector<T> & x, T y)
{
  size_t size = x.size;
  T * restrict x_ = x.data;
  for (size_t i = 0; i < size; ++i) x_[i] -= y;
}

template<class T>
void operator*= (Vector<T> & x, T y)
{
  size_t size = x.size;
  T * restrict x_ = x.data;
  for (size_t i = 0; i < size; ++i) x_[i] *= y;
}

template<class T>
void operator/= (Vector<T> & x, T y)
{
  size_t size = x.size;
  T * restrict x_ = x.data;
  for (size_t i = 0; i < size; ++i) x_[i] /= y;
}

template<class T>
void idiv_store_rhs (T x, Vector<T> & Y)
{
  size_t size = Y.size;

  T * restrict y = Y.data;

  for (size_t i = 0; i < size; ++i) y[i] = x / y[i];
}

inline void operator*= (Vector<complex> & x, float y)
{
  Vector<float> real_x(x);
  real_x *= y;
}

inline void operator/= (Vector<complex> & x, float y)
{
  Vector<float> real_x(x);
  real_x /= y;
}

template<class T>
void imax (Vector<T> & x, T y)
{
  size_t size = x.size;
  T * restrict x_ = x.data;
  for (size_t i = 0; i < size; ++i) imax(x_[i], y);
}

template<class T>
void imin (Vector<T> & x, T y)
{
  size_t size = x.size;
  T * restrict x_ = x.data;
  for (size_t i = 0; i < size; ++i) imin(x_[i], y);
}

template<class T>
void ipow (Vector<T> & x, T p)
{
  size_t size = x.size;
  T * restrict x_ = x.data;
  for (size_t i = 0; i < size; ++i) x_[i] = pow(max(0.0f, x_[i]), p);
}

//----( vector, vector )----

template<class S, class T>
void operator+= (Vector<S> & X, const Vector<T> & Y)
{
  size_t size = X.size;
  ASSERT_SIZE(Y, size);

  S * restrict x = X.data;
  const T * restrict y = Y.data;

  for (size_t i = 0; i < size; ++i) x[i] += y[i];
}

template<class S, class T>
void operator-= (Vector<S> & X, const Vector<T> & Y)
{
  size_t size = X.size;
  ASSERT_SIZE(Y, size);

  S * restrict x = X.data;
  const T * restrict y = Y.data;

  for (size_t i = 0; i < size; ++i) x[i] -= y[i];
}

template<class S, class T>
void isub_store_rhs (const Vector<S> & X, Vector<T> & Y)
{
  size_t size = X.size;
  ASSERT_SIZE(Y, size);

  const S * restrict x = X.data;
  T * restrict y = Y.data;

  for (size_t i = 0; i < size; ++i) y[i] = x[i] - y[i];
}

template<class S, class T>
void operator*= (Vector<S> & X, const Vector<T> & Y)
{
  size_t size = X.size;
  ASSERT_SIZE(Y, size);

  S * restrict x = X.data;
  const T * restrict y = Y.data;

  for (size_t i = 0; i < size; ++i) x[i] *= y[i];
}

template<class S, class T>
void operator/= (Vector<S> & X, const Vector<T> & Y)
{
  size_t size = X.size;
  ASSERT_SIZE(Y, size);

  S * restrict x = X.data;
  const T * restrict y = Y.data;

  for (size_t i = 0; i < size; ++i) x[i] /= y[i];
}

template<class S, class T>
void idiv_store_rhs (const Vector<S> & X, Vector<T> & Y)
{
  size_t size = X.size;
  ASSERT_SIZE(Y, size);

  const S * restrict x = X.data;
  T * restrict y = Y.data;

  for (size_t i = 0; i < size; ++i) y[i] = x[i] / y[i];
}

template<class T>
void imax (Vector<T> & X, const Vector<T> & Y)
{
  size_t size = X.size;
  ASSERT_SIZE(Y, size);

  T * restrict x = X.data;
  const T * restrict y = Y.data;

  for (size_t i = 0; i < size; ++i) imax(x[i], y[i]);
}

template<class T>
void imin (Vector<T> & X, const Vector<T> & Y)
{
  size_t size = X.size;
  ASSERT_SIZE(Y, size);

  T * restrict x = X.data;
  const T * restrict y = Y.data;

  for (size_t i = 0; i < size; ++i) imin(x[i], y[i]);
}

//----( mapped operations )---------------------------------------------------

// z = min(x,y)
void minimum (float x, const Vector<float> & y, Vector<float> & z);
void minimum (
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z);

// z = max(x,y)
void maximum (float x, const Vector<float> & y, Vector<float> & z);
void maximum (
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z);

// z = x + y
void add (float x, const Vector<float> & y, Vector<float> & z);
void add (const Vector<float> & x, const Vector<float> & y, Vector<float> & z);
void add (
    const Vector<complex> & x,
    const Vector<complex> & y,
    Vector<complex> & z);

// z = x - y
void subtract (
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z);
void subtract (
    const Vector<complex> & x,
    const Vector<complex> & y,
    Vector<complex> & z);

// <x|y>
float dot (const Vector<float> & x, const Vector<float> & y);
double dot (const Vector<double> & x, const Vector<float> & y);
complex dot (const Vector<complex> & x, const Vector<complex> & y);
inline double dot (const Vector<float> & x, const Vector<double> & y)
{
  return dot(y,x);
}

// z = x y
void multiply (float x, const Vector<float> & y, Vector<float> & z);
void multiply (float x, const Vector<complex> & y, Vector<complex> & z);
void multiply (
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z);
void multiply (
    const Vector<float> & x,
    const Vector<complex> & y,
    Vector<complex> & z);

// z = x' y
void multiply_conj (
    const Vector<complex> & x,
    const Vector<complex> & y,
    Vector<complex> & z);

// z += x y
void multiply_add (float x, const Vector<float> & y, Vector<float> & z);
void multiply_add (
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z);
void multiply_add (
    const Vector<float> & x,
    const Vector<complex> & y,
    Vector<complex> & z);

// z = x / y
void divide (float x, const Vector<float> & y, Vector<float> & z);
void divide (
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z);

// z = a x + (1-a) y
void affine_combine (
    float a,
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z);
void affine_combine (
    float a,
    const Vector<complex> & x,
    const Vector<complex> & y,
    Vector<complex> & z);
void affine_combine (
    const Vector<float> & a,
    const Vector<float> & x,
    const Vector<float> & y,
    Vector<float> & z);
void affine_combine (
    const Vector<float> & a,
    const Vector<complex> & x,
    const Vector<complex> & y,
    Vector<complex> & z);

// z = a x + b y
void linear_combine (
    const Vector<float> & a,
    const Vector<complex> & x,
    const Vector<float> & b,
    const Vector<complex> & y,
    Vector<complex> & z);

// x_old = factor * x_old + (1-factor) * x_new
void accumulate_step (
    float factor,
    Vector<float> & x_old,
    const Vector<float> & x_new);
void accumulate_step (
    const Vector<float> & factor,
    Vector<complex> & x_old,
    const Vector<complex> & x_new);

// y = exp(x)
void exp (
    const Vector<float> & x,
    Vector<float> & y);
void exp_inplace (
    Vector<float> & x);

// y = log(x)
void log (
    const Vector<float> & x,
    Vector<float> & y);
void log_inplace (
    Vector<float> & x);

// y = log(gamma(x))
void lgamma (
    const Vector<float> & x,
    Vector<float> & y);
void lgamma_inplace (
    Vector<float> & x);

//----( reductions )----------------------------------------------------------

// sum i. x[i]
float sum (const Vector<float> & x);
double sum (const Vector<double> & x);
complex sum (const Vector<complex> & x);
uint64_t sum (const Vector<uint8_t> & x);

inline float mean (const Vector<float> & x) { return sum(x) / x.size; }
inline double mean (const Vector<double> & x) { return sum(x) / x.size; }
inline complex mean (const Vector<complex> & x)
{
  return sum(x) * (1.0f / x.size);
}
inline float mean (const Vector<uint8_t> & x)
{
  return sum(x) * (1.0f / x.size);
}

// (sum i. like[i] x[i]) / (sum i. like[i])
float mean_wrt (const Vector<float> & x, const Vector<float> & like);

// d = ||x-y||^2
float dist_squared (const Vector<float> & x, const Vector<float> & y);

// rms = sqrt( ||x-y||^2 / dim(x) )
float rms_error (const Vector<float> & x, const Vector<float> & y);

// ||x||^2 = sum i. |x[i]|^2
float norm_squared (const Vector<float> & x);
inline float norm_squared (const Vector<complex> & z)
{
  const Vector<float> x = Vector<float>(const_cast<Vector<complex> &>(z));
  return norm_squared(x);
}

inline float norm (const Vector<float> & x)
{
  float ns = norm_squared(x);
  if (ns > 0) return sqrt(ns); else return 0; // avoids sqrt(0) = nan
}
inline float norm (const Vector<complex> & x)
{
  float ns = norm_squared(x);
  if (ns > 0) return sqrt(ns); else return 0; // avoids sqrt(0) = nan
}

inline float rms (const Vector<float> & x)
{
  float ms = norm_squared(x) / x.size;
  if (ms > 0) return sqrt(ms); else return 0; // avoids sqrt(0) = nan
}
inline float rms (const Vector<complex> & x)
{
  float ms = norm_squared(x) / x.size;
  if (ms > 0) return sqrt(ms); else return 0; // avoids sqrt(0) = nan
}

// max i. x
float max (const Vector<float> & x);
float min (const Vector<float> & x);
double max (const Vector<double> & x);
double min (const Vector<double> & x);
int max (const Vector<uint8_t> & x);
int min (const Vector<uint8_t> & x);
size_t argmin (const Vector<float> & x);
size_t argmax (const Vector<float> & x);

// max i. |x|^2
float max_norm_squared (const Vector<float> & x);
float max_norm_squared (const Vector<complex> & x);
inline float max_abs (const Vector<float> & x)
{
  return sqrtf(max_norm_squared(x));
}

float max_dist_squared (const Vector<float> & x, const Vector<float> & y);
double max_dist_squared (const Vector<double> & x, const Vector<double> & y);
int max_dist_squared (const Vector<uint8_t> & x, const Vector<uint8_t> & y);

inline float max_dist (const Vector<float> & x, const Vector<float> & y)
{
  return sqrtf(max_dist_squared(x,y));
}
inline float max_dist (const Vector<uint8_t> & x, const Vector<uint8_t> & y)
{
  return sqrtf(max_dist_squared(x,y));
}

float density (const Vector<float> & x);
float entropy (const Vector<float> & x);
inline float perplexity (const Vector<float> & x) { return expf(entropy(x)); }
float relentropy (
    const Vector<float> & x,
    const Vector<float> & y,
    bool non_normalized = false);

// bound |x| to [0,1)
void hard_clip (Vector<float> & x);
inline void hard_clip (Vector<complex> & z)
{
  Vector<float> x = Vector<float>(z);
  hard_clip(x);
}
void soft_clip (Vector<float> & x);
inline void soft_clip (Vector<complex> & z)
{
  Vector<float> x = Vector<float>(z);
  soft_clip(x);
}
void affine_to_01 (Vector<float> & x);

//----( abstract functions )--------------------------------------------------

struct VectorFunction
{
  virtual ~VectorFunction () {}
  virtual size_t size_in () const = 0;
  virtual size_t size_out () const = 0;
  virtual void operator() (
      const Vector<float> & input,
      Vector<float> & output) = 0;
};

//----( function sampling wrappers )------------------------------------------

inline void sample_uniform (Vector<float> & out)
{
  for (size_t i = 0; i < out.size; ++i) {
    out[i] = (0.5f + i) / out.size;
  }
}

inline void sample_function (const Function & fun, Vector<float> & out)
{
  for (size_t i = 0; i < out.size; ++i) {
    out[i] = fun((0.5f + i) / out.size);
  }
}

#endif // KAZOO_VECTORS_H
