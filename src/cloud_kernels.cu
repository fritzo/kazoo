
#include "cloud_kernels.cu.h"

using Cloud::QuantizeStats;
using Cloud::ConstructStats;

namespace Gpu
{

__device__ inline uint8_t float_to_uchar (const float & restrict x)
{
  return max(0.0f, min(255.0f, rintf(x)));
}

__device__ inline float4 uchar4_to_float4 (const uchar4 & restrict vect)
{
  return make_float4(vect.x, vect.y, vect.z, vect.w);
}

__device__ inline uchar4 float4_to_uchar4 (const float4 & restrict vect)
{
  return make_uchar4(float_to_uchar(vect.x),
                     float_to_uchar(vect.y),
                     float_to_uchar(vect.z),
                     float_to_uchar(vect.w));
}

//----( distance measurements )-----------------------------------------------

//----( one probe vs all points )----

template<class Accum>
__device__ inline Accum dist_squared (Accum p, Accum q)
{
  Accum diff = p - q;
  return diff * diff;
}

template<class Accum>
__device__ inline Accum dist_squared (uchar4 p, uchar4 q)
{
  return + dist_squared<Accum>(p.x, q.x)
         + dist_squared<Accum>(p.y, q.y)
         + dist_squared<Accum>(p.z, q.z)
         + dist_squared<Accum>(p.w, q.w);
}

//----( simple version )----

// each of num_points blocks computes squared_distance of one probe to one point
// each of dim/4 threads compues 4 terms in the squared distance
template<int dim>
__global__ void measure_one_kernel_64 (
    const uchar4 * restrict probe,
    const uchar4 * restrict points,
    float * restrict squared_distances)
{
  cuda_static_assert<dim % 4 == 0>();

  const int t = threadIdx.x;
  const uchar4 * restrict point = points + dim/4 * blockIdx.x;
  float & restrict squared_distance = squared_distances[blockIdx.x];

  // sum in register

  typedef int Accum;
  Accum register_sum = 0;

  #pragma unroll 4
  for (int x = t; x < dim/4; x += 64) {
    register_sum += dist_squared<Accum>(probe[x], point[x]);
  }

  // reduce

  __shared__ volatile Accum shared_sums[64];
  shared_sums[t] = register_sum;

  __syncthreads();

  if (t >= 32) return;

  shared_sums[t] += shared_sums[t + 32];
  shared_sums[t] += shared_sums[t + 16];
  shared_sums[t] += shared_sums[t + 8];
  shared_sums[t] += shared_sums[t + 4];
  shared_sums[t] += shared_sums[t + 2];
  shared_sums[t] += shared_sums[t + 1];

  if (t == 0) {
    squared_distance = shared_sums[0];
  }
}

void measure_one_64 (
    const CudaVector<uint8_t> & probe,
    const CudaVector<uint8_t> & points,
    CudaVector<float> & squared_distances)
{
  int dim = probe.size;
  int num_points = squared_distances.size;
  ASSERT_SIZE(points, dim * num_points);

#define kernel(DIM) \
  if (DIM == dim) { \
    measure_one_kernel_64<DIM><<<num_points, 64>>>( \
        (uchar4 *) probe.data, \
        (uchar4 *) points.data, \
        squared_distances.data); \
  } else

  /*
  kernel( 128)
  kernel( 256)
  */
  kernel( 384)
  kernel( 512)
  kernel( 768)
  kernel(1024)
  kernel(1152)
  kernel(2048)

#undef kernel

  TODO("implement measure_one_kernel_64<" << dim << ">");
}

//----( fully unrolled version )----

// XXX this has erroneous results at size = 2048
// each of num_points blocks computes squared_distance of one probe to one point
// each of dim/4 threads compues 4 terms in the squared distance
template<int num_rows, int num_cols>
__global__ void __launch_bounds__(num_cols) measure_one_unrolled_kernel (
    const uchar4 * restrict probe,
    const uchar4 * restrict points,
    float * restrict squared_distances)
{
  cuda_static_assert<(num_cols == 32) or (num_cols == 64)>();

  typedef int Accum;

  const int col = threadIdx.x;

  probe += col;
  points += blockIdx.x * num_rows * num_cols + col;
  squared_distances += blockIdx.x;

  // sum rows (1 iteration per row)

  Accum row_sum = dist_squared<Accum>(probe[0], points[0]);

  #pragma unroll
  for (int row = 1; row < num_rows; ++row) {
    uchar4 q = probe[row * num_cols];
    uchar4 p = points[row * num_cols];

    row_sum += dist_squared<Accum>(q,p);
  }

  // reduce

  __shared__ volatile Accum row_sums[num_cols];
  row_sums[col] = row_sum;

  if (col >= num_cols / 2) return;

  if (num_cols > 32) {
    __syncthreads();
    row_sums[col] += row_sums[col + 32];
  }

  __syncthreads(); // DEBUG
  row_sums[col] += row_sums[col + 16];
  __syncthreads(); // DEBUG
  row_sums[col] += row_sums[col + 8];
  __syncthreads(); // DEBUG
  row_sums[col] += row_sums[col + 4];
  __syncthreads(); // DEBUG
  row_sums[col] += row_sums[col + 2];
  __syncthreads(); // DEBUG
  row_sums[col] += row_sums[col + 1];
  __syncthreads(); // DEBUG

  if (col == 0) {
    squared_distances[0] = row_sums[0];
  }
}

void measure_one_unrolled (
    const CudaVector<uint8_t> & probe,
    const CudaVector<uint8_t> & points,
    CudaVector<float> & squared_distances)
{
  int dim = probe.size;
  int size = squared_distances.size;
  ASSERT_SIZE(points, dim * size);

  ASSERT_DIVIDES(4 * THREADS_PER_WARP, dim);

#define kernel(DIM, ROWS, COLS) \
  if (DIM == dim) { \
    cuda_static_assert<DIM == 4 * ROWS * COLS>(); \
    cuda_static_assert<COLS <= MAX_THREAD_DIM>(); \
    measure_one_unrolled_kernel<ROWS, COLS><<<size, COLS>>>( \
        (uchar4 *) probe.data, \
        (uchar4 *) points.data, \
        squared_distances.data); \
  } else

  /*
  kernel( 128, 1, 32)
  kernel( 256, 1, 64)
  */
  kernel( 384, 3, 32)
  kernel( 512, 2, 64)
  kernel( 768, 3, 64)
  kernel(1024, 4, 64)
  kernel(1152, 9, 32)
  kernel(2048, 8, 64)

#undef kernel

  TODO("implement measure_one_tasks_kernel<...> for dim = " << dim);
}

//----( tasked version )----

// XXX this sometimes segfaults
template<int num_rows, int num_cols, int tasks>
__global__ void __launch_bounds__(num_cols) measure_one_tasks_kernel (
    const uchar4 * restrict probe,
    const uchar4 * restrict points,
    float * restrict squared_distances)
{
  // this does more work per thread, as suggested by Vasily Volkov in
  // http://www.cs.berkeley.edu/~volkov/volkov10-GTC.pdf

  cuda_static_assert<(num_cols == 32) or (num_cols == 64)>();

  typedef int Accum;

  const int dim = num_rows * num_cols;
  const int col = threadIdx.x;

  probe += col;
  points += tasks * blockIdx.x * dim + col;
  squared_distances += tasks * blockIdx.x;

  // sum rows (1 iteration per row)

  uchar4 q = probe[0];
  uchar4 p[tasks];

  Accum row_sum[tasks];
  {
    #pragma unroll
    for (int t = 0; t < tasks; ++t) {
      p[t] = points[dim * t];
    }

    #pragma unroll
    for (int t = 0; t < tasks; ++t) {
      row_sum[t] = dist_squared<Accum>(q, p[t]);
    }
  }

  #pragma unroll
  for (int row = 1; row < num_rows; ++row) {

    q = probe[row * num_cols];

    #pragma unroll
    for (int t = 0; t < tasks; ++t) {
      p[t] = points[dim * t + row * num_cols];
    }

    #pragma unroll
    for (int t = 0; t < tasks; ++t) {
      row_sum[t] += dist_squared<Accum>(q, p[t]);
    }
  }

  // reduce

  __shared__ volatile Accum row_sums[tasks][num_cols];

  #pragma unroll
  for (int t = 0; t < tasks; ++t) {
    row_sums[t][col] = row_sum[t];
  }

  int reduce_col = num_cols / 2;
  if (col >= reduce_col) return;

  if (num_cols > 32) {
    __syncthreads();

    #pragma unroll
    for (int t = 0; t < tasks; ++t) {
      row_sums[t][col] += row_sums[t][col + reduce_col];
    }

    reduce_col /= 2;
  }

  #pragma unroll
  for (size_t i = 0; i < 5; ++i) {

    #pragma unroll
    for (int t = 0; t < tasks; ++t) {
      row_sums[t][col] += row_sums[t][col + reduce_col];
    }

    reduce_col /= 2;
  }

  cuda_static_assert<tasks <= num_cols>();
  if (col < tasks) {
    int t = col;

    squared_distances[t] = row_sums[t][0];
  }
}

void measure_one_tasks (
    const CudaVector<uint8_t> & probe,
    const CudaVector<uint8_t> & points,
    CudaVector<float> & squared_distances)
{
  int dim = probe.size;
  int size = squared_distances.size;
  ASSERT_SIZE(points, dim * size);

  ASSERT_DIVIDES(4 * THREADS_PER_WARP, dim);

#define kernel(DIM, ROWS, COLS, TASKS) \
  if (DIM == dim) { \
    ASSERT_DIVIDES(TASKS, size); \
    ASSERT_LE(int(size / TASKS), MAX_GRID_DIM); \
    cuda_static_assert<DIM == 4 * ROWS * COLS>(); \
    cuda_static_assert<COLS <= MAX_THREAD_DIM>(); \
    measure_one_tasks_kernel<ROWS, COLS, TASKS><<<size/TASKS, COLS>>>( \
        (uchar4 *) probe.data, \
        (uchar4 *) points.data, \
        squared_distances.data); \
  } else

  /*
  kernel( 128, 1, 32, 4)
  kernel( 256, 1, 64, 4)
  */
  kernel( 384, 3, 32, 4)
  kernel( 512, 2, 64, 4)
  kernel( 768, 3, 64, 4)
  kernel(1024, 4, 64, 4)
  kernel(1152, 9, 32, 4)
  kernel(2048, 8, 64, 4)

#undef kernel

  TODO("implement measure_one_tasks_kernel<...> for dim = " << dim);
}

void measure_one (
    const CudaVector<uint8_t> & probe,
    const CudaVector<uint8_t> & points,
    CudaVector<float> & squared_distances)
{
  // this is slow
  measure_one_64

  // this is sometimes buggy for size = 2048. WTF?
  //measure_one_unrolled

  // this is sometimes buggy for size = 2048. WTF?
  //measure_one_tasks

  (probe, points, squared_distances);
}

//----( many probes vs all points )----

// each of num_probes * num_points blocks
//   measures the distances between tile_size points and tile_size probes
// each of tile_size * tiles_size threads
//   measures the distance between one point and one probe
template<int dim, int tile_size>
__global__ void measure_batch_kernel (
    const uchar4 * restrict probes,
    const uchar4 * restrict points,
    float * restrict squared_distances)
{
  cuda_static_assert<(tile_size > 1)>();
  cuda_static_assert<dim % tile_size == 0>();
  cuda_static_assert<(tile_size == 16) or (tile_size == 32)>();

  typedef int Accum;

  const int P = tile_size * gridDim.y;
  {
    const int p0 = tile_size * blockIdx.y;
    const int q0 = tile_size * blockIdx.x;

    probes += dim * q0;
    points += dim * p0;
    squared_distances += P * q0 + p0;
  }

  const int i = threadIdx.y;
  const int x = threadIdx.x;

  probes += dim * i + x;
  points += dim * i + x;

  // copy a tile of probes data into shared memory

  __shared__ uchar4 shared_probes[tile_size][tile_size];

  shared_probes[i][x] = probes[0];
  __syncthreads();

  uchar4 point = points[0];

  // sum data

  Accum register_sum[tile_size];

  #pragma unroll
  for (int j = 0; j < tile_size; ++j) {

    register_sum[j] = dist_squared<Accum>(shared_probes[j][x], point);
  }
  __syncthreads();

  #pragma unroll 2
  for (int x0 = tile_size; x0 < dim; x0 += tile_size) {

    // load data

    shared_probes[i][x] = probes[x0];
    __syncthreads();

    uchar4 point = points[x0];

    #pragma unroll
    for (int j = 0; j < tile_size; ++j) {

      register_sum[j] += dist_squared<Accum>(shared_probes[j][x], point);
    }
    __syncthreads();
  }

  // reduce

  __shared__ volatile Accum shared_sum[tile_size][tile_size][tile_size];

  #pragma unroll
  for (int j = 0; j < tile_size; ++j) {
    shared_sum[j][i][x] = register_sum[j];
  }

  if (x < tile_size / 2) {
    #pragma unroll
    for (int j = 0; j < tile_size; ++j) {

      if (tile_size > 16) {
        shared_sum[j][i][x] += shared_sum[j][i][x + 16];
      }

      shared_sum[j][i][x] += shared_sum[j][i][x + 8];
      shared_sum[j][i][x] += shared_sum[j][i][x + 4];
      shared_sum[j][i][x] += shared_sum[j][i][x + 2];
      shared_sum[j][i][x] += shared_sum[j][i][x + 1];
    }
  }

  __syncthreads();

  if (x < tile_size) {
    const int p = x;
    const int q = i;

    squared_distances[P * q + p] = shared_sum[q][p][0];
  }
}

void measure_batch (
    const CudaVector<uint8_t> & probes,
    const CudaVector<uint8_t> & points,
    CudaVector<float> & squared_distances,
    size_t num_probes)
{
  ASSERT_DIVIDES(num_probes, probes.size);
  const size_t dim = probes.size / num_probes;

  ASSERT_DIVIDES(dim, points.size);
  const size_t num_points = points.size / dim;
  ASSERT_SIZE(squared_distances, num_probes * num_points);

#define kernel(DIM, TILE_SIZE) \
  if (dim == DIM) { \
    cuda_static_assert<DIM % TILE_SIZE == 0>(); \
    ASSERT_DIVIDES(TILE_SIZE, num_probes); \
    ASSERT_DIVIDES(TILE_SIZE, num_points); \
    dim3 grid(num_probes / TILE_SIZE, num_points / TILE_SIZE); \
    dim3 block(TILE_SIZE, TILE_SIZE); \
    measure_batch_kernel \
        <DIM/4, TILE_SIZE> \
        <<<grid, block>>>( \
        (uchar4 *) probes.data, \
        (uchar4 *) points.data, \
        squared_distances.data); \
  } else

  /*
  kernel( 128, 16)
  kernel( 192, 16)
  kernel( 256, 16)
  */
  kernel( 384, 16)
  kernel( 512, 16)
  kernel( 768, 16)
  kernel(1024, 16)
  kernel(1152, 16)
  kernel(2048, 16)

#undef kernel

  TODO("implement measure_batch<" << dim << ",... >");
}

//----( quantization )--------------------------------------------------------

struct GetShiftsOp
{
  int size;
  float sd_scale;
  const float * restrict sd;
  float * restrict sd_shifts;

  typedef float Accum;

  __device__ Accum load (int p, int block = 0) { return sd[p]; }
  __device__ Accum reduce () const { return INFINITY; }
  __device__ Accum reduce (Accum a1, Accum a2) const { return min(a1, a2); }
  __device__ void store (int block, Accum accum)
  {
    sd_shifts[block] = sd_scale * accum;
  }
};

struct QuantizeOp
{
  int size;

  float sd_scale;
  const float * restrict sd_shift;

  const float * restrict sd;
  float * restrict likes;

  float * restrict like_scales;
  float * restrict entropy;
  float * restrict energy;
  float * restrict energy2;
  float * restrict energy_cold;

  struct Accum { float Z, U, U2, Zcold, Ucold; };

  __device__ Accum load (int p, int block = 0)
  {
    const float u = sd_scale * sd[p];
    const float z = expf(sd_shift[block] - u);

    likes[p] = z;

    Accum accum = {z, z * u, z * u * u, z * z, z * z * u};
    return accum;
  }

  __device__ Accum reduce () const
  {
    Accum accum = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    return accum;
  }
  __device__ Accum reduce (Accum a1, Accum a2) const
  {
    Accum accum = {
      a1.Z + a2.Z,
      a1.U + a2.U,
      a1.U2 + a2.U2,
      a1.Zcold + a2.Zcold,
      a1.Ucold + a2.Ucold
    };
    return accum;
  }

  __device__ void store (int block, Accum accum)
  {
    const float Z = accum.Z;
    const float inv_Z = 1.0f / Z;
    const float U = accum.U * inv_Z;
    const float U2 = accum.U2 * inv_Z;
    const float Ucold = accum.Ucold / accum.Zcold;
    const float H = logf(Z) + U - sd_shift[block];

    like_scales[block] = size * inv_Z;
    energy[block] = U;
    energy2[block] = U2;
    energy_cold[block] = Ucold;
    entropy[block] = H;
  }
};

struct ScaleLikesOp
{
  int size;
  const float * restrict like_scales;
  float * restrict likes;

  __device__ void load_map_store (int p, int block = 0) const
  {
    likes[p] *= like_scales[block];
  }
};

QuantizeStats quantize_batch (
    float radius,
    const CudaVector<float> & squared_distances,
    CudaVector<float> & likes,
    CudaVector<float> & dev_work,
    Vector<float> & host_work,
    size_t num_probes)
{
  ASSERT_DIVIDES(num_probes, squared_distances.size)
  const size_t num_points = squared_distances.size / num_probes;

  ASSERT_SIZE(dev_work, num_probes * 6);
  CudaVector<float> sd_shifts   = dev_work.block(num_probes, 0);
  CudaVector<float> like_scales = dev_work.block(num_probes, 1);
  CudaVector<float> entropy     = dev_work.block(num_probes, 2);
  CudaVector<float> energy      = dev_work.block(num_probes, 3);
  CudaVector<float> energy2     = dev_work.block(num_probes, 4);
  CudaVector<float> energy_cold = dev_work.block(num_probes, 5);

  float sd_scale = 0.5f / sqr(radius);

  GetShiftsOp get_shifts_op = {
      num_points,
      sd_scale,
      squared_distances.data,
      sd_shifts};
  reduce(get_shifts_op, num_probes);

  QuantizeOp quantize_op = {
      num_points,
      sd_scale,
      sd_shifts,
      squared_distances.data,
      likes.data,
      like_scales,
      entropy,
      energy,
      energy2,
      energy_cold};
  reduce(quantize_op, num_probes);

  ScaleLikesOp scale_likes_op = {
      num_points,
      like_scales,
      likes};
  map(scale_likes_op, num_probes);

  ASSERT1_LE(0, min(likes)); // really checks for NAN

  copy_dev_to_host(
      host_work.data,
      dev_work.data + num_probes * 2,
      num_probes * 4);

  ASSERT_SIZE(host_work, num_probes * 4);
  Vector<float> host_entropy     = host_work.block(num_probes, 0);
  Vector<float> host_energy      = host_work.block(num_probes, 1);
  Vector<float> host_energy2     = host_work.block(num_probes, 2);
  Vector<float> host_energy_cold = host_work.block(num_probes, 3);

  float H = sum(host_entropy);
  float U = sum(host_energy);
  float U2 = sum(host_energy2);
  float Ucold = sum(host_energy_cold);

  return QuantizeStats(H, U, U2, Ucold);
}

//----( vq construction )-----------------------------------------------------

// each of dim/threads blocks sums num_points terms
// each of threads threads computes one components of the sum
template<int threads, bool batch>
__global__ void vq_construct_kernel (
    const int num_points,
    const float * restrict likes,
    const uint8_t * restrict points,
    uint8_t * restrict recon,
    const float tol)
{
  const int dim = gridDim.x * blockDim.x;
  const int t = threadIdx.x;
  const int x = blockDim.x * blockIdx.x + threadIdx.x;

  // we assume mean(likes) = 1
  const float inv_sum_likes = 1.0f / num_points;

  if (batch) {
    const int i_probe = blockIdx.y;
    likes += num_points * i_probe;
    recon += dim * i_probe;
  }

  __shared__ float shared_likes[threads];

  float sum = 0;

  int i_point;
  for (i_point = 0; i_point + (threads - 1) < num_points; i_point += threads) {

    __syncthreads();
    if (i_point + t < num_points) shared_likes[t] = likes[i_point + t];
    __syncthreads();

    #pragma unroll 32
    for (int j_point = 0; j_point < threads; ++j_point) {

      const float like = shared_likes[j_point];
      if (like > tol) {

        sum += like * points[dim * (i_point + j_point) + x];
      }
    }
  }

  __syncthreads();
  if (i_point + t < num_points) shared_likes[t] = likes[i_point + t];
  __syncthreads();

  #pragma unroll 32
  for (int j_point = 0; j_point < num_points - i_point; ++j_point) {

    const float like = shared_likes[j_point];
    if (like > tol) {

      sum += like * points[dim * (i_point + j_point) + x];
    }
  }

  recon[x] = float_to_uchar(inv_sum_likes * sum);
}

void vq_construct_one (
    const CudaVector<float> & likes,
    const CudaVector<uint8_t> & points,
    CudaVector<uint8_t> & recon,
    float tol)
{
  const size_t dim = recon.size;
  const size_t num_points = likes.size;
  ASSERT_SIZE(points, dim * num_points);

  ASSERT_DIVIDES(32, dim);
  vq_construct_kernel
      <32, false>
      <<<dim / 32, 32>>>
      (num_points, likes, points, recon, tol);
}

void vq_construct_batch (
    const CudaVector<float> & likes,
    const CudaVector<uint8_t> & points,
    CudaVector<uint8_t> & recon,
    float tol,
    size_t num_probes)
{
  ASSERT_DIVIDES(num_probes, recon.size);
  const size_t dim = recon.size / num_probes;
  ASSERT_DIVIDES(num_probes, likes.size);
  const size_t num_points = likes.size / num_probes;
  ASSERT_SIZE(points, dim * num_points);

  ASSERT_DIVIDES(32, dim);
  int threads = 32;
  int blocks_x = dim / threads;

  // target as many threads as possible
  while ((blocks_x & 1) == 0 and threads < 1024) {
    threads *= 2;
    blocks_x /= 2;
  }

  dim3 blocks(blocks_x, num_probes);

#define kernel(THREADS) \
  if (threads == THREADS) { \
    vq_construct_kernel \
        <THREADS, true> \
        <<<blocks, threads>>> \
        (num_points, likes, points, recon, tol); \
  } else

  kernel(32)
  kernel(64)
  kernel(128)
  kernel(256)
  kernel(512)
  kernel(1024)

#undef kernel

  ERROR("invalid number of threads: " << threads);
}

//----( sparse version )----

struct SparseVectorEntry { int index; float value; };

__global__ void sparsify_likes_kernel (
    int dense_size,
    const float * dense_likes,
    float thresh,
    int * sparse_size,
    SparseVectorEntry * sparse_entries)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < dense_size) {

    const float likes_i = dense_likes[i];

    if (likes_i > thresh) {

      float prob_i = likes_i / dense_size;
      SparseVectorEntry entry = { i, prob_i };
      sparse_entries[ atomicAdd(sparse_size, 1) ] = entry;
    }
  }
}

__global__ void vq_construct_one_sparse_kernel (
    const int sparse_size,
    const SparseVectorEntry * sparse_entries,
    const uint8_t * points,
    uint8_t * recon)
{
  const int dim = gridDim.x * blockDim.x;
  const int x = blockDim.x * blockIdx.x + threadIdx.x;

  points += x;
  recon += x;

  float sum = 0;

  #pragma unroll 32
  for (int j = 0; j < sparse_size; ++j) {

    const SparseVectorEntry entry = sparse_entries[j];

    sum += entry.value * points[dim * entry.index];
  }

  * recon = float_to_uchar(sum);
}

int vq_construct_one (
    const CudaVector<float> & likes,
    const CudaVector<uint8_t> & points,
    CudaVector<uint8_t> & recon,
    CudaVector<float> & work,
    float tol)
{
  const size_t dim = recon.size;
  const size_t num_points = likes.size;
  ASSERT_SIZE(points, dim * num_points);
  ASSERT_SIZE(work, num_points * 2);
  ASSERTW_LT(0, tol);

  static CudaVector<int> sparse_size_data(1); // WARNING not thread safe
  SparseVectorEntry * sparse_entries = (SparseVectorEntry *) work.data;

  {
    sparse_size_data.zero();

    int threads = 32;
    int blocks = (num_points + 31) / 32;
    while (blocks > 128 and threads < 1024) {
      threads *= 2;
      blocks /= 2;
    }

    sparsify_likes_kernel
        <<<blocks, threads>>>
        (num_points, likes, tol, sparse_size_data, sparse_entries);
  }

  int sparse_size = sparse_size_data.get(0);
  ASSERT_LT(0, sparse_size);

  {
    ASSERT_DIVIDES(32, dim);
    vq_construct_one_sparse_kernel
        <<<dim / 32, 32>>>
        (sparse_size, sparse_entries, points, recon);
  }

  return sparse_size;
}

//----( with derivative )----

// This computes the reconstruction
// (letting p = point, q = probe, r = reconstruction)
//
//              sum p. exp(-beta ||p-q||^2) p
//   r = E[p] = -----------------------------
//               sum p. exp(-beta ||p-q||^2)
//
// and the partial derivative
//
//   dr/dbeta = Cov[q-p, ||q-p||^2]
//
//            = E[(q-p) ||q-p||^2] + (q - r) E[||q-p||^2]
//
// and saves two statistics of the partial derivative
//
//       info = < dr/dbeta | dr/dbeta >
//   surprise = < dr/dbeta | (q - r) >
//
// By aggregating these statistics over all probe,recon pairs,
// minimize reconstruction error py performing
// a Gauss-Newton update in some coordinate system, eg
//
//   d(beta) = surprise / info
//
//                  surprise
//   d(log beta) = ---------
//                 beta info
//
//                     radius^2 surprise
//   d(log radius) = - -----------------     where beta = 1 / (2 radius^2)
//                           info

// each of dim/(4 threads_per_block) x num_probes blocks sums num_points terms
// each of threads_per_block threads computes 4 components of the sum
template<int dim>
__global__ void vq_construct_deriv_kernel (
    const int num_points,
    const float * restrict likes,
    const float * restrict squared_distances,
    const uchar4 * restrict probes,
    const uchar4 * restrict points,
    uchar4 * restrict recon,
    float * restrict stats,
    const float tol)
{
  {
    const int i_probe = blockIdx.x;
    likes += num_points * i_probe;
    squared_distances += num_points * i_probe;
    probes += dim * i_probe;
    recon += dim * i_probe;
  }
  const int t = threadIdx.x;

  const float4 probe = uchar4_to_float4(probes[t]);

  float mean_sd = 0;
  float4 mean_qp = {0,0,0,0};
  float4 mean_sd_qp = {0,0,0,0};

  #pragma unroll 16
  for (int i_point = 0; i_point < num_points; ++i_point) {

    const float like = likes[i_point];

    if (like > tol) {

      const float sd = squared_distances[i_point];
      const float like_sd = like * sd;
      const uchar4 point = points[dim * i_point + t];

      const float4 qp = make_float4(
          probe.x - point.x,
          probe.y - point.y,
          probe.z - point.z,
          probe.w - point.w);

      mean_sd += like_sd;

      mean_qp.x += like * qp.x;
      mean_qp.y += like * qp.y;
      mean_qp.z += like * qp.z;
      mean_qp.w += like * qp.w;

      mean_sd_qp.x += like_sd * qp.x;
      mean_sd_qp.y += like_sd * qp.y;
      mean_sd_qp.z += like_sd * qp.z;
      mean_sd_qp.w += like_sd * qp.w;
    }
  }

  // rescale
  {
    // we assume mean(likes) = 1
    const float inv_sum_likes = 1.0f / num_points;

    mean_sd *= inv_sum_likes;

    mean_qp.x *= inv_sum_likes;
    mean_qp.y *= inv_sum_likes;
    mean_qp.z *= inv_sum_likes;
    mean_qp.w *= inv_sum_likes;

    mean_sd_qp.x *= inv_sum_likes;
    mean_sd_qp.y *= inv_sum_likes;
    mean_sd_qp.z *= inv_sum_likes;
    mean_sd_qp.w *= inv_sum_likes;
  }

  // compute moments for statistics

  __shared__ float error[dim];
  __shared__ float info[dim];
  __shared__ float surprise[dim];

  mean_sd_qp.x -= mean_sd * mean_qp.x;
  mean_sd_qp.y -= mean_sd * mean_qp.y;
  mean_sd_qp.z -= mean_sd * mean_qp.z;
  mean_sd_qp.w -= mean_sd * mean_qp.w;
  // whereafter mean_sd_qp = Cov[q-p,||q-p||^2]

  error[t] = mean_qp.x * mean_qp.x
           + mean_qp.y * mean_qp.y
           + mean_qp.z * mean_qp.z
           + mean_qp.w * mean_qp.w;

  info[t] = mean_sd_qp.x * mean_sd_qp.x
          + mean_sd_qp.y * mean_sd_qp.y
          + mean_sd_qp.z * mean_sd_qp.z
          + mean_sd_qp.w * mean_sd_qp.w;

  surprise[t] = mean_sd_qp.x * mean_qp.x
              + mean_sd_qp.y * mean_qp.y
              + mean_sd_qp.z * mean_qp.z
              + mean_sd_qp.w * mean_qp.w;

  // store recon

  recon[t] = float4_to_uchar4(make_float4(
      probe.x - mean_qp.x,
      probe.y - mean_qp.y,
      probe.z - mean_qp.z,
      probe.w - mean_qp.w));

  // reduce inner products

  const int exponent = static_log2i<2 * dim - 1>::value;
  const int even_dim = 1 << exponent;
  cuda_static_assert<( even_dim/2 < dim and dim <= even_dim )>();

  __syncthreads();

  int T = even_dim / 2;
  if (t < T and t + T < dim) {
    error[t] += error[t + T];
    info[t] += info[t + T];
    surprise[t] += surprise[t + T];
  }

  __syncthreads();

  #pragma unroll
  for (int i = 1; i < exponent; ++i) {

    T /= 2;
    if (t < T) {
      error[t] += error[t + T];
      info[t] += info[t + T];
      surprise[t] += surprise[t + T];
    }

    __syncthreads();
  }

  if (t == 0) {
    atomicAdd(stats + 0, error[0]);
    atomicAdd(stats + 1, info[0]);
    atomicAdd(stats + 2, surprise[0]);
  }
}

ConstructStats vq_construct_deriv (
    const CudaVector<float> & likes,
    const CudaVector<float> & squared_distances,
    const CudaVector<uint8_t> & probes,
    const CudaVector<uint8_t> & points,
    CudaVector<uint8_t> & recon,
    float tol,
    size_t num_probes)
{
  ASSERT_EQ(likes.size, squared_distances.size);
  ASSERT_DIVIDES(num_probes, likes.size);
  const size_t num_points = likes.size / num_probes;

  ASSERT_DIVIDES(num_probes, probes.size);
  const size_t dim = probes.size / num_probes;
  ASSERT_SIZE(recon, dim * num_probes);

  ASSERT_SIZE(points, dim * num_points);

  ASSERT_DIVIDES(4, dim);
  const int threads = dim / 4;
  const int blocks = num_probes;

  static CudaVector<float> dev_stats(3);
  dev_stats.zero();

#define kernel(DIM) \
  if (dim == DIM) { \
    vq_construct_deriv_kernel<DIM/4><<<blocks, threads>>>( \
        num_points, \
        likes, \
        squared_distances, \
        (uchar4 *) probes.data, \
        (uchar4 *) points.data, \
        (uchar4 *) recon.data, \
        dev_stats, \
        tol); \
  } else

  kernel( 384)
  kernel( 512)
  kernel( 768)
  kernel(1024)
  kernel(1152)
  kernel(2048)

#undef kernel

  TODO("implement vq_construct_deriv_kernel<" << dim << ">");

  static Vector<float> stats(3);
  stats = dev_stats;
  return ConstructStats(stats[0], stats[1], stats[2]);
}

//----( fitting )-------------------------------------------------------------

// This computes the update
//
//   p += sum q. rate_q (q - p)
//
// where p is the point and q is the observation

// each of num_points blocks attracts one point towards num_probes probes
// each of dim/4 thread attracts 4 components of one point
//   towards num_probes probes
__global__ void fit_points_to_obs_kernel_128 (
    int num_probes,
    float rate_thresh,
    const uchar4 * restrict probes,
    const float * restrict rates,
    const float * restrict jitters,
    uchar4 * restrict points)
{
  const int num_points = gridDim.x;
  const int dim = blockDim.x;
  const int i_point = blockIdx.x;
  const int t = threadIdx.x;

  uchar4 * restrict point = points + dim * i_point;

  // copy rates to shared memory (not coalesced)

  __shared__ float shared_rates[128];
  for (int i_probe = t; i_probe < num_probes; i_probe += dim) {
    shared_rates[i_probe] = rates[num_points * i_probe + i_point];
  }
  __syncthreads();

  // accumulate floating point update

  float4 p = uchar4_to_float4(point[t]);
  float4 dp = {0,0,0,0};

  for (int i_probe = 0; i_probe < num_probes; ++i_probe) {

    const float rate = shared_rates[i_probe];
    if (rate > rate_thresh) {

      const uchar4 q = probes[dim * i_probe + t];

      dp.x += rate * (q.x - p.x);
      dp.y += rate * (q.y - p.y);
      dp.z += rate * (q.z - p.z);
      dp.w += rate * (q.w - p.w);
    }
  }

  // perform update

  const float jitter = jitters[i_point];

  dp.x += p.x + jitter;
  dp.y += p.y + jitter;
  dp.z += p.z + jitter;
  dp.w += p.w + jitter;

  point[t] = float4_to_uchar4(dp);
}

void fit_points_to_obs (
    float rate_thresh,
    const CudaVector<uint8_t> & probes,
    const CudaVector<float> & rates,
    CudaVector<curandState> & curand_states,
    CudaVector<float> & jitters,
    CudaVector<uint8_t> & points)
{
  const int num_points = jitters.size;
  ASSERT_SIZE(curand_states, num_points);

  curand_unif(-0.5f, 0.5f, curand_states, jitters);

  ASSERT_DIVIDES(num_points, points.size);
  const int dim = points.size / num_points;

  ASSERT_DIVIDES(dim, probes.size);
  const int num_probes = probes.size / dim;
  ASSERT_SIZE(rates, num_points * num_probes);

  ASSERT_DIVIDES(4, dim);
  int threads = dim / 4;

  ASSERT_LE(num_probes, 128);

  fit_points_to_obs_kernel_128<<<num_points, threads>>>(
      num_probes,
      rate_thresh,
      (const uchar4 *) probes.data,
      rates,
      jitters,
      (uchar4 *) points.data);
}

//----( reconstruction fitting )----------------------------------------------

// This computes the update
//
//                    radius^2 (q - r) + <p-r|q-r> (q - p)
//   p += sum q. rate ------------------------------------
//                       sqrt( radius^4 + <p-r|q-r>^2 )
//
// where p is the point, q is the observation, and r is the reconstruction

// each of num_points blocks attracts one point towards num_probes probes
// each of dim/4 threads
//   attracts 4 components of one point towards num_probes probes
template<int dim>
__global__ void fit_points_to_recon_kernel_128 (
    const int num_probes,
    const float sqr_radius,
    const float rate_thresh,
    const uchar4 * restrict probe,
    const uchar4 * restrict recon,
    const float * restrict rates,
    const float * restrict jitters,
    uchar4 * restrict points)
{
  const int exponent = static_log2i<2 * dim - 1>::value;
  const int even_dim = 1 << exponent;
  cuda_static_assert<( even_dim/2 < dim and dim <= even_dim )>();

  const int num_points = gridDim.x;
  const int i_point = blockIdx.x;
  const int t = threadIdx.x;

  uchar4 * restrict point = points + dim * i_point;

  // copy rates to shared memory (not coalesced)

  __shared__ float shared_rates[128];
  for (int i_probe = t; i_probe < num_probes; i_probe += dim) {
    shared_rates[i_probe] = rates[num_points * i_probe + i_point];
  }
  __syncthreads();

  // accumulate floating point update

  const float4 p = uchar4_to_float4(point[t]);
  float4 dp = {0,0,0,0};

  for (int i_probe = 0; i_probe < num_probes; ++i_probe) {

    const float rate = shared_rates[i_probe];
    if (rate > rate_thresh) {

      const float4 q = uchar4_to_float4(probe[dim * i_probe + t]);
      const float4 r = uchar4_to_float4(recon[dim * i_probe + t]);

      float4 q_minus_r;
      q_minus_r.x = q.x - r.x;
      q_minus_r.y = q.y - r.y;
      q_minus_r.z = q.z - r.z;
      q_minus_r.w = q.w - r.w;

      // compute inner product <p-r|q-r>

      __shared__ float shared_ip[dim];

      shared_ip[t] = (p.x - r.x) * q_minus_r.x
                   + (p.y - r.y) * q_minus_r.y
                   + (p.z - r.z) * q_minus_r.z
                   + (p.w - r.w) * q_minus_r.w;

      __syncthreads();

      int T = even_dim / 2;
      if (t < T and t + T < dim) shared_ip[t] += shared_ip[t + T];

      __syncthreads();

      #pragma unroll
      for (int i = 1; i < exponent; ++i) {

        T /= 2;
        if (t < T) shared_ip[t] += shared_ip[t + T];

        __syncthreads();
      }

      // compute update

      const float ip = shared_ip[0];
      const float scale = rate / sqrtf(sqr_radius * sqr_radius + ip * ip);
      const float qr_rate = scale * sqr_radius;
      const float qp_rate = scale * ip;

      dp.x += qr_rate * q_minus_r.x + qp_rate * (q.x - p.x);
      dp.y += qr_rate * q_minus_r.y + qp_rate * (q.y - p.y);
      dp.z += qr_rate * q_minus_r.z + qp_rate * (q.z - p.z);
      dp.w += qr_rate * q_minus_r.w + qp_rate * (q.w - p.w);
    }
  }

  // perform total update

  const float jitter = jitters[i_point];

  dp.x += p.x + jitter;
  dp.y += p.y + jitter;
  dp.z += p.z + jitter;
  dp.w += p.w + jitter;

  point[t] = float4_to_uchar4(dp);
}

void fit_points_to_recon (
    float radius,
    float rate_thresh,
    const CudaVector<uint8_t> & probes,
    const CudaVector<uint8_t> & recons,
    const CudaVector<float> & rates,
    CudaVector<curandState> & curand_states,
    CudaVector<float> & jitters,
    CudaVector<uint8_t> & points)
{
  const size_t num_points = jitters.size;
  ASSERT_SIZE(curand_states, num_points);

  curand_unif(-0.5f, 0.5f, curand_states, jitters);

  ASSERT_DIVIDES(num_points, rates.size);
  const size_t num_probes = rates.size / num_points;

  ASSERT_DIVIDES(num_probes, probes.size);
  const size_t dim = probes.size / num_probes;
  ASSERT_SIZE(recons, dim * num_probes);
  ASSERT_SIZE(points, dim * num_points);

  ASSERT_LE(num_probes, 128);

#define kernel(DIM) \
  if (dim == DIM) { \
    cuda_static_assert<DIM % 4 == 0>(); \
    fit_points_to_recon_kernel_128 \
      <DIM/4> \
      <<<num_points, DIM/4>>> \
      (num_probes, \
       sqr(radius), \
       rate_thresh, \
       (uchar4 *) probes.data, \
       (uchar4 *) recons.data, \
       rates, \
       jitters, \
       (uchar4 *) points.data); \
  } else

  /*
  kernel( 128)
  kernel( 256)
  */
  kernel( 384)
  kernel( 512)
  kernel( 768)
  kernel(1024)
  kernel(1152)
  kernel(2048)

#undef kernel

  TODO("implement fit_points_to_recon_kernel_128<" << (dim/4) << ">");
}

} // namespace Gpu

