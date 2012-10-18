
#include "gpu.cu.h"
#include "vectors.cu.h"

#define LOG1(message)

namespace Gpu
{

//----( general info )--------------------------------------------------------

static int cuda_device_count_noncached ()
{
  int device_count;
  cudaError_t info = cudaGetDeviceCount(& device_count);

  return info == cudaSuccess ? device_count : 0;
}

int cuda_device_count ()
{
  static const int device_count = cuda_device_count_noncached();
  return device_count;
}

namespace { bool g_using_cuda = cuda_device_count(); }

bool using_cuda () { return g_using_cuda; }

void set_using_cuda (bool whether)
{
  g_using_cuda = (cuda_device_count > 0) and whether;
}

void print_gpu_info ()
{
  int device_count;
  ASSERT_CUDA( cudaGetDeviceCount(& device_count) );

  for (int i = 0; i < device_count; ++i) {

    cudaDeviceProp  prop;
    ASSERT_CUDA( cudaGetDeviceProperties(& prop, i) );

    float bus_width_gb = prop.memoryBusWidth * pow(2, -33);
    float mem_clock_rate_hz = prop.memoryClockRate * 1000.0f;
    float bandwidth_gb_per_sec = bus_width_gb * mem_clock_rate_hz;

    LOG("Cuda device " << i << ", " << prop.name << ":");
    LOG("  compute capability = " << prop.major << "." << prop.minor);
    LOG("  clock rate = " << (prop.clockRate / pow(10,6)) << "GHz");
    LOG("  memory = " << (prop.totalGlobalMem * pow(2,-30)) << "GB global"
        " + " << (prop.totalConstMem * pow(2,-10)) << "KB constant");
    LOG("  bandwidth = " << bandwidth_gb_per_sec << " GB/sec");
    LOG("  features:"
        << (prop.deviceOverlap ? " copy overlap," : "")
        << (prop.concurrentKernels ? " concurrent kernels," : "")
        << (prop.kernelExecTimeoutEnabled ? " kernel timeout," : ""));

    //LOG("    max pitch = " << prop.memPitch);
    //LOG("    texture alignment = " << prop.textureAlignment);

    LOG("  has " << prop.multiProcessorCount << " multiprocessors");
    LOG("  mp limits = "
        << prop.regsPerBlock << " registers, "
        << prop.maxThreadsPerMultiProcessor << " threads");
    LOG("  block limits = "
        << prop.sharedMemPerBlock << "B shared mem, "
        << prop.maxThreadsPerBlock << " threads");
    LOG("  max grid dims"
        << " = " << prop.maxGridSize[0]
        << " x " << prop.maxGridSize[1]
        << " x " << prop.maxGridSize[2]);
    LOG("  max thread dims"
        << " = " << prop.maxThreadsDim[0]
        << " x " << prop.maxThreadsDim[1]
        << " x " << prop.maxThreadsDim[2]);
    LOG("  threads per warp = " << prop.warpSize);
  }
}

//----( memory )--------------------------------------------------------------

void * cuda_malloc (size_t size) { return cuda_new<char>(size); }

void cuda_free (void * data) { cuda_delete<char>((char *) data); }

void cuda_memcpy_h2d (void * dst, const void * src, size_t size)
{
  copy_host_to_dev<char>((char *) dst, (const char *) src, size);
}

void cuda_memcpy_d2h (void * dst, const void * src, size_t size)
{
  copy_dev_to_host<char>((char *) dst, (const char *) src, size);
}

void cuda_bzero (void * dst, size_t size)
{
  set_zero<char>((char *) dst, size);
}

//----( cuda timer )----------------------------------------------------------

class CudaTimer
{
  cudaEvent_t m_start;
  cudaEvent_t m_stop;

public:

  CudaTimer ()
  {
    ASSERT_CUDA(cudaEventCreate(& m_start));
    ASSERT_CUDA(cudaEventCreate(& m_stop));
  }
  ~CudaTimer ()
  {
    ASSERT_CUDA(cudaEventDestroy(m_start));
    ASSERT_CUDA(cudaEventDestroy(m_stop));
  }

  void tick ()
  {
    ASSERT_CUDA(cudaEventRecord(m_start, 0));
  }
  float tock ()
  {
    ASSERT_CUDA(cudaEventRecord(m_stop, 0));
    ASSERT_CUDA(cudaEventSynchronize(m_stop));

    float time_ms;
    ASSERT_CUDA(cudaEventElapsedTime(& time_ms, m_start, m_stop));

    return time_ms * 1e-3f;
  }
};

//----( random generators )---------------------------------------------------

__global__ void curand_init_kernel (int size, curandState * state)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < size) {
    int seed = 0;
    int seq_number = x;
    int offset = 0;
    curand_init(seed, seq_number, offset, state + x);
  }
}

__global__ void curand_unif_kernel (
    int size,
    float LB,
    float UB,
    curandState * restrict state,
    float * restrict out)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < size) {
    out[x] = LB + (UB - LB) * curand_uniform(state + x);
  }
}

__global__ void curand_normal_kernel (
    int size,
    float mean,
    float sigma,
    curandState * restrict state,
    float * restrict out)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x < size) {
    out[x] = mean + sigma * curand_normal(state + x);
  }
}

void curand_init (CudaVector<curandState> & state)
{
  // curand initialization requires up to 16k of stack space
  // (see curand manual, pp. 18)
  // however, this only works in devices of compute capability >= 2.0

  size_t old_size;
  ASSERT_CUDA(cudaDeviceGetLimit(& old_size, cudaLimitStackSize));

  size_t new_size = 1 << 14;
  if (new_size > old_size) {
    LOG1("temporarily increasing cuda stack size "
        << old_size << " -> " << new_size);
    ASSERT_CUDA(cudaDeviceSetLimit(cudaLimitStackSize, new_size));
  }

  curand_init_kernel
      <<<(state.size + 63) / 64, 64>>>
      (state.size, state);

  if (new_size > old_size) {
    ASSERT_CUDA(cudaDeviceSetLimit(cudaLimitStackSize, old_size));
  }
}

void curand_unif (
    float LB,
    float UB,
    CudaVector<curandState> & state,
    CudaVector<float> & out)
{
  ASSERT_EQ(state.size, out.size);

  curand_unif_kernel
      <<<(state.size + 63) / 64, 64>>>
      (state.size, LB, UB, state, out);
}

void curand_normal (
    float mean,
    float sigma,
    CudaVector<curandState> & state,
    CudaVector<float> & out)
{
  ASSERT_EQ(state.size, out.size);

  curand_normal_kernel
      <<<(state.size + 63) / 64, 64>>>
      (state.size, mean, sigma, state, out);
}

} // namespace Gpu

