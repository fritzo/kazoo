
#include "vectors.cu.h"

//----( operations )----------------------------------------------------------

namespace CudaVector_private
{

template<class T>
struct IaddScalarOp
{
  int size;
  T * restrict x;
  T y;

  __device__ void load_map_store (int i) const { x[i] += y; }
};

template<class T>
struct ImulScalarOp
{
  int size;
  T * restrict x;
  T y;

  __device__ void load_map_store (int i) const { x[i] *= y; }
};

template<class T>
struct IaddOp
{
  int size;
  T * restrict x;
  const T * restrict y;

  __device__ void load_map_store (int i) const { x[i] += y[i]; }
};

template<class T>
struct IsubOp
{
  int size;
  T * restrict x;
  const T * restrict y;

  __device__ void load_map_store (int i) const { x[i] -= y[i]; }
};

template<class T>
struct ImulOp
{
  int size;
  T * restrict x;
  const T * restrict y;

  __device__ void load_map_store (int i) const { x[i] *= y[i]; }
};

template<class T>
struct IdivOp
{
  int size;
  T * restrict x;
  const T * restrict y;

  __device__ void load_map_store (int i) const { x[i] /= y[i]; }
};

template<class T>
struct MinOp
{
  int size;
  const T * x;
  T * result;

  typedef T Accum;

  __device__ Accum load (int i, int block = 0) { return x[i]; }
  __device__ Accum reduce () const { return INFINITY; }
  __device__ Accum reduce (Accum a1, Accum a2) const { return min(a1,a2); }
  __device__ void store (int block, Accum accum) { result[block] = accum; }
};

template<class T>
struct MaxOp
{
  int size;
  const T * x;
  T * result;

  typedef T Accum;

  __device__ Accum load (int i, int block = 0) { return x[i]; }
  __device__ Accum reduce () const { return -INFINITY; }
  __device__ Accum reduce (Accum a1, Accum a2) const { return max(a1,a2); }
  __device__ void store (int block, Accum accum) { result[block] = accum; }
};

template<class T>
struct SumOp
{
  int size;
  const T * x;
  T * result;

  typedef T Accum;

  __device__ Accum load (int i, int block = 0) { return x[i]; }
  __device__ Accum reduce () const { return 0; }
  __device__ Accum reduce (Accum a1, Accum a2) const { return a1 + a2; }
  __device__ void store (int block, Accum accum) { result[block] = accum; }
};

template<class T>
struct NormSquaredOp
{
  int size;
  const T * x;
  T * result;

  typedef T Accum;

  __device__ Accum load (int i, int block = 0) { T t = x[i]; return t * t; }
  __device__ Accum reduce () const { return 0; }
  __device__ Accum reduce (Accum a1, Accum a2) const { return a1 + a2; }
  __device__ void store (int block, Accum accum) { result[block] = accum; }
};

} // namespace CudaVector_private
using namespace CudaVector_private;

//----( in-place operators )--------------------------------------------------

void operator+= (CudaVector<float> & x, float y)
{
  IaddScalarOp<float> op = { x.size, x.data, y };
  Gpu::map(op);
}

void operator*= (CudaVector<float> & x, float y)
{
  ImulScalarOp<float> op = { x.size, x.data, y };
  Gpu::map(op);
}

void operator+= (CudaVector<float> & x, const CudaVector<float> & y)
{
  ASSERT_EQ(x.size, y.size);

  IaddOp<float> op = { x.size, x.data, y.data };
  Gpu::map(op);
}

void operator-= (CudaVector<float> & x, const CudaVector<float> & y)
{
  ASSERT_EQ(x.size, y.size);

  IsubOp<float> op = { x.size, x.data, y.data };
  Gpu::map(op);
}

void operator*= (CudaVector<float> & x, const CudaVector<float> & y)
{
  ASSERT_EQ(x.size, y.size);

  ImulOp<float> op = { x.size, x.data, y.data };
  Gpu::map(op);
}

void operator/= (CudaVector<float> & x, const CudaVector<float> & y)
{
  ASSERT_EQ(x.size, y.size);

  IdivOp<float> op = { x.size, x.data, y.data };
  Gpu::map(op);
}

//----( parallelized math )---------------------------------------------------

float min (const CudaVector<float> & x)
{
  static CudaVector<float> result(1);

  MinOp<float> op = { x.size, x.data, result.data };
  Gpu::reduce(op);

  return result.get(0);
}

float max (const CudaVector<float> & x)
{
  static CudaVector<float> result(1);

  MaxOp<float> op = { x.size, x.data, result.data };
  Gpu::reduce(op);

  return result.get(0);
}

float sum (const CudaVector<float> & x)
{
  static CudaVector<float> result(1);

  SumOp<float> op = { x.size, x.data, result.data };
  Gpu::reduce(op);

  return result.get(0);
}

float norm_squared (const CudaVector<float> & x)
{
  static CudaVector<float> result(1);

  NormSquaredOp<float> op = { x.size, x.data, result.data };
  Gpu::reduce(op);

  return result.get(0);
}

void parallel_min (const CudaVector<float> & x, CudaVector<float> & result)
{
  ASSERT_DIVIDES(result.size, x.size);
  const int blocks = result.size;
  const int size = x.size / blocks;

  MinOp<float> op = { size, x.data, result.data };
  Gpu::reduce(op, blocks);
}

void parallel_max (const CudaVector<float> & x, CudaVector<float> & result)
{
  ASSERT_DIVIDES(result.size, x.size);
  const int blocks = result.size;
  const int size = x.size / blocks;

  MaxOp<float> op = { size, x.data, result.data };
  Gpu::reduce(op, blocks);
}

void parallel_sum (const CudaVector<float> & x, CudaVector<float> & result)
{
  ASSERT_DIVIDES(result.size, x.size);
  const int blocks = result.size;
  const int size = x.size / blocks;

  SumOp<float> op = { size, x.data, result.data };
  Gpu::reduce(op, blocks);
}

