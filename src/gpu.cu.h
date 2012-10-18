#ifndef KAZOO_GPU_CU_H
#define KAZOO_GPU_CU_H

#include "gpu.h"
#include "cloud_stats.h"
#include <curand_kernel.h>

template<class T> class CudaVector;

namespace Gpu
{

//----( static assertions )---------------------------------------------------

template<bool cond>
struct cuda_static_assert;

template<>
struct cuda_static_assert<true>
{
  __device__ inline cuda_static_assert () {}
  __device__ inline cuda_static_assert (string cond) {}
};

#define ASSERT_CUDA(info) ASSERT((info)==cudaSuccess, cudaGetErrorString(info))

// WARNING some of this is specific to the GeForce GTX 580
static const int MAX_THREAD_DIM = 1024;
static const int MAX_GRID_DIM = 65535;
static const int THREADS_PER_WARP = 32; // this should be equal to warpSize

//----( memory )--------------------------------------------------------------

template<class T>
inline T * cuda_new (size_t size)
{
  T * data;
  ASSERT_CUDA(cudaMalloc((void**) & data, size * sizeof(T)));
  return data;
}

template<class T>
inline void cuda_delete (const T * data)
{
  ASSERT_CUDA(cudaFree((void*) data));
}

template<class T>
inline T * cuda_host_new (
    size_t size,
    int flags = cudaHostAllocDefault)
{
  T * data;
  ASSERT_CUDA(cudaMallocHost((void**) & data, size * sizeof(T), flags));
  return data;
}

template<class T>
inline void cuda_host_delete (const T * data)
{
  ASSERT_CUDA(cudaFreeHost((void*) data));
}

template<class T>
inline void copy_host_to_dev (T * dst, const T * src, size_t size)
{
  ASSERT_CUDA(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
inline void copy_dev_to_host (T * dst, const T * src, size_t size)
{
  ASSERT_CUDA(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template<class T>
inline void copy_dev_to_dev (T * dst, const T * src, size_t size)
{
  ASSERT_CUDA(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice));
}

//----( simple vector operations )--------------------------------------------

template<class T>
inline void set_zero (T * dev_data, size_t size)
{
  cudaMemset(dev_data, 0, size * sizeof(T));
}

template<class T>
inline void set_zero (T * dev_data, size_t size1, size_t size2)
{
  cudaMemset(dev_data, 0, size1 * size2 * sizeof(T));
}

template<class T>
__global__ void set_const_kernel (T * dev_data, int size, T value)
{
  int pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos < size) dev_data[pos] = value;
}

template<class T>
void set_const (T * dev_data, size_t size, T value)
{
  int block_dim = 64;
  int grid_dim = (size + block_dim - 1) / block_dim;

  set_const_kernel<<<grid_dim, block_dim>>>(dev_data, size, value);
}

template<class T>
__global__ void set_const_kernel (T * dev_data, int size1, int size2, T value)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if ((x < size1) and (y < size2)) dev_data[size1 * y + x] = value;
}

template<class T>
void set_const (T * dev_data, size_t size1, size_t size2, T value)
{
  ASSERT_DIVIDES(16, size1); // required for aligned strides

  dim3 block_dim(16, 16);
  dim3 grid_dim((size2 + 15) / 16, (size1 + 15) / 16);

  set_const_kernel<<<grid_dim, block_dim>>>(dev_data, size2, size1, value);
}

//----( random generators )---------------------------------------------------

void curand_init (CudaVector<curandState> & state);

void curand_unif (
    float LB,
    float UB,
    CudaVector<curandState> & state,
    CudaVector<float> & out);

void curand_normal (
    float mean,
    float sigma,
    CudaVector<curandState> & state,
    CudaVector<float> & out);

//----( map )-----------------------------------------------------------------

template<class Operation>
__global__ void map_kernel (const Operation op)
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int X = op.size;

  if (x < X) {
    op.load_map_store(x);
  }
}

template<class Operation>
inline void map (const Operation op)
{
  int blocks = (op.size + 31) / 32;
  int threads = 32;

  // target between 64 and 127 blocks
  while (blocks > 128 and threads < 1024) {
    blocks = (blocks + 1) / 2;
    threads *= 2;
  }

  map_kernel
      <Operation>
      <<<blocks, threads>>>
      (op);
}

//----( parallel map )--------------------------------------------------------

template<class Operation>
__global__ void parallel_map_kernel (const Operation op)
{
  const int b = blockIdx.x;
  const int x0 = b * op.size + threadIdx.x;
  const int X = (b + 1) * op.size;
  const int dx = blockDim.x;

  for (int x = x0; x < X; x += dx) {
    op.load_map_store(x,b);
  }
}

template<class Operation>
inline void map (const Operation op, int blocks)
{
  if (blocks == 1) {

    map(op);

  } else {

    int warps = min(32, (op.size + 31) / 32);
    int threads = 32 * warps;

    parallel_map_kernel
        <Operation>
        <<<blocks, threads>>>
        (op);
  }
}

//----( reduction )-----------------------------------------------------------

// this is optimized to do small reductions in parallel
// and uses only one block per reduction

template<class Operation>
__global__ void reduce_kernel_1024 (Operation op)
{
  const int b = blockIdx.x;
  const int t = threadIdx.x;

  const int dx = 1024;
  const int X = (b + 1) * op.size;
  int x = b * op.size + t;

  typename Operation::Accum local_accum
    = x < X ? op.load(x,b) : op.reduce();

  for (x += dx; x < X; x += dx) {
    local_accum = op.reduce(local_accum, op.load(x,b));
  }

  __shared__ typename Operation::Accum shared_accum[1024];
  shared_accum[t] = local_accum;

  int T = 1024;

#pragma unroll
  for (int i = 0; i < 10; ++i) {

    __syncthreads();

    T /= 2;
    if (t < T) {
      shared_accum[t] = op.reduce(shared_accum[t], shared_accum[t + T]);
    }
  }

  if (t == 0) {
    op.store(b, shared_accum[0]);
  }
}

template<class Operation>
inline void reduce (const Operation op, int blocks = 1)
{
  reduce_kernel_1024
      <Operation>
      <<<blocks, 1024>>>
      (op);
}

} // namespace Gpu

#endif // KAZOO_GPU_CU_H
