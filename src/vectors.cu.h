#ifndef KAZOO_VECTORS_CU_H
#define KAZOO_VECTORS_CU_H

#include "vectors.h"
#include "gpu.cu.h"

//----( vector classes )------------------------------------------------------

template<class T>
struct CudaVector
{
  T * const data;
  const size_t size;
  const bool alias;

  // aliasing
  explicit CudaVector (size_t s, T * d = NULL)
    : data(d ? d : Gpu::cuda_new<T>(s)), size(s), alias(d) {}
  CudaVector (const CudaVector<T> & other)
    : data(other.data), size(other.size), alias(true) {}
  ~CudaVector () { if (not alias) Gpu::cuda_delete(data); }

  // copying
  void operator= (const CudaVector<T> & other)
  {
    ASSERT_SIZE(other, size);
    Gpu::copy_dev_to_dev(data, other.data, size);
  }
  void operator= (const Vector<T> & other)
  {
    ASSERT_SIZE(other, size);
    Gpu::copy_host_to_dev(data, other.data, size);
  }

  // constant filling
  void zero () { Gpu::set_zero(data, size); }
  void set (T value) { Gpu::set_const(data, size, value); }
  void operator= (T value) DEPRECATED { set(value); }

  // access
  operator const T * () const { return data; }
  operator       T * ()       { return data; }
  T get (size_t i) const
  {
    T value;
    Gpu::copy_dev_to_host(& value, data + i, 1);
    return value;
  }
  void set (size_t i, T value) { Gpu::copy_host_to_dev(data + i, & value, 1); }
  CudaVector<T> block (size_t stride, size_t number = 0)
  {
    ASSERT_DIVIDES(stride, size);
    ASSERT_LT(number, size / stride);

    return CudaVector<T>(stride, data + stride * number);
  }
  const CudaVector<T> block (size_t stride, size_t number = 0) const
  {
    ASSERT_DIVIDES(stride, size);
    ASSERT_LT(number, size / stride);

    return CudaVector<T>(stride, const_cast<T*>(data) + stride * number);
  }
};

template<class T>
inline void Vector<T>::operator= (const CudaVector<T> & other)
{
  ASSERT_SIZE(other, size);
  Gpu::copy_dev_to_host(data, other.data, size);
}

inline void Vector<float>::operator= (const CudaVector<float> & other)
{
  ASSERT_SIZE(other, size);
  Gpu::copy_dev_to_host(data, other.data, size);
}

inline void Vector<complex>::operator= (const CudaVector<complex> & other)
{
  ASSERT_SIZE(other, size);
  Gpu::copy_dev_to_host(data, other.data, size);
}

//----( in-place operators )--------------------------------------------------

void operator+= (CudaVector<float> & x, float y);
void operator*= (CudaVector<float> & x, float y);

void operator+= (CudaVector<float> & x, const CudaVector<float> & y);
void operator-= (CudaVector<float> & x, const CudaVector<float> & y);
void operator*= (CudaVector<float> & x, const CudaVector<float> & y);
void operator/= (CudaVector<float> & x, const CudaVector<float> & y);

//----( reductions )----------------------------------------------------------

float min (const CudaVector<float> & x);
float max (const CudaVector<float> & x);
float sum (const CudaVector<float> & x);

inline float mean (const CudaVector<float> & x) { return sum(x) / x.size; }

float norm_squared (const CudaVector<float> & x);

inline float rms (const CudaVector<float> & x)
{
  return sqrtf(norm_squared(x) / x.size);
}

void parallel_min (const CudaVector<float> & x, CudaVector<float> & result);
void parallel_max (const CudaVector<float> & x, CudaVector<float> & result);
void parallel_sum (const CudaVector<float> & x, CudaVector<float> & result);

#endif // KAZOO_VECTORS_CU_H
