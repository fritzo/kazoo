
#include "gpu.cu.h"
#include "gpu_test.h"
#include "vectors.cu.h"
#include "cloud_kernels.cu.h"

//----( mapping )-------------------------------------------------------------

void test_iadd (
    const Vector<float> & values,
    float shift,
    Vector<float> & shifted,
    size_t test_iters)
{
  const size_t size = values.size;
  ASSERT_SIZE(shifted, size);

  CudaVector<float> dev_values(values.size);
  CudaVector<float> dev_shifted(shifted.size);

  dev_values = shifted;

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {
    dev_shifted = dev_values;
    dev_shifted += shift;
  }

  shifted = dev_shifted;

  float rate = size * test_iters / timer.elapsed();
  float gflops = rate / 1e9;
  LOG(" performed " << (rate/1e6) << " operations/ms ("
      << (1e3/rate) << " ms/call, " << gflops << " gflops)");
}

//----( reduction )-----------------------------------------------------------

float test_reduce_min (
    size_t size,
    const float * val,
    size_t test_iters)
{
  CudaVector<float> dev_temp(size);
  CudaVector<float> dev_val(size);
  CudaVector<float> dev_min_val(1);

  float min_val = NAN;

  Gpu::copy_host_to_dev(dev_temp.data, val, size);
  Gpu::copy_host_to_dev(dev_val.data, val, size);

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {
    min_val = min(dev_val);
  }

  float rate = size * test_iters / timer.elapsed();
  LOG(" reduced " << (rate/1e9) << " elements/ns");

  return min_val;
}

float test_reduce_max (
    size_t size,
    const float * val,
    size_t test_iters)
{
  CudaVector<float> dev_temp(size);
  CudaVector<float> dev_val(size);
  CudaVector<float> dev_min_val(1);

  float max_val = NAN;

  Gpu::copy_host_to_dev(dev_temp.data, val, size);
  Gpu::copy_host_to_dev(dev_val.data, val, size);

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {
    max_val = max(dev_val);
  }

  float rate = size * test_iters / timer.elapsed();
  LOG(" reduced " << (rate/1e9) << " elements/ns");

  return max_val;
}

float test_reduce_sum (
    size_t size,
    const float * val,
    size_t test_iters)
{
  CudaVector<float> dev_temp(size);
  CudaVector<float> dev_val(size);
  CudaVector<float> dev_sum_val(1);

  float sum_val = NAN;

  Gpu::copy_host_to_dev(dev_temp.data, val, size);
  Gpu::copy_host_to_dev(dev_val.data, val, size);

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {
    sum_val = sum(dev_val);
  }

  float rate = size * test_iters / timer.elapsed();
  LOG(" reduced " << (rate/1e9) << " elements/ns");

  return sum_val;
}

void test_parallel_reduce_min (
    size_t size,
    size_t parallel,
    const float * val,
    float * result,
    size_t test_iters)
{
  CudaVector<float> dev_val(size * parallel);
  CudaVector<float> dev_result(parallel);

  Gpu::copy_host_to_dev(dev_val.data, val, size * parallel);

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {
    parallel_min(dev_val, dev_result);
  }

  Gpu::copy_dev_to_host(result, dev_result.data, parallel);

  float rate = parallel * size * test_iters / timer.elapsed();
  LOG(" reduced " << (rate/1e9) << " elements/ns");
}

//----( clouds )--------------------------------------------------------------

void test_measure_one_gpu (
    size_t dim,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances,
    size_t test_iters)
{
  const size_t num_probes = probes.size / dim;
  const size_t num_points = points.size / dim;
  ASSERT_SIZE(squared_distances, num_probes * num_points);

  CudaVector<uint8_t> dev_probe(dim);
  CudaVector<uint8_t> dev_points(points.size);
  CudaVector<float> dev_distances(squared_distances.size);
  dev_distances.set(NAN);

  dev_points = points;

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {

    for (size_t i_probe = 0; i_probe < num_probes; ++i_probe) {

      dev_probe = probes.block(dim, i_probe);

      CudaVector<float> dev_distances_block
        = dev_distances.block(num_points, i_probe);

      Gpu::measure_one(dev_probe, dev_points, dev_distances_block);
    }
  }

  squared_distances = dev_distances;

  const size_t num_pairs = num_probes * test_iters;
  float pair_rate = num_pairs * num_points / timer.elapsed();
  float call_rate = num_pairs / timer.elapsed();
  float gflops = 3 * dim * pair_rate / 1e9;
  LOG(" measured " << (pair_rate/1e6) << " pairs/ms ("
      << (1e3/call_rate) << " ms/call, " << gflops << " gflops)");
}

void test_measure_batch_gpu (
    size_t dim,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances,
    size_t test_iters)
{
  const size_t num_probes = probes.size / dim;
  const size_t num_points = points.size / dim;

  ASSERT_SIZE(squared_distances, num_probes * num_points);

  CudaVector<uint8_t> dev_probes(probes.size);
  CudaVector<uint8_t> dev_points(points.size);
  CudaVector<float> dev_distances(squared_distances.size);

  dev_points = points;
  dev_distances.set(NAN);

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {

    dev_probes = probes;

    Gpu::measure_batch(dev_probes, dev_points, dev_distances, num_probes);
  }

  squared_distances = dev_distances;

  float pair_rate = num_probes * num_points * test_iters / timer.elapsed();
  float call_rate = test_iters / timer.elapsed();
  float gflops = 3 * dim * pair_rate / 1e9;
  LOG(" measured " << (pair_rate/1e6) << " pairs/ms ("
      << (1e3/call_rate) << " ms/call, " << gflops << " gflops)");
}

void test_vq_construct_one_gpu (
    size_t dim,
    const Vector<float> & likes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    float tol,
    size_t test_iters)
{
  const size_t num_probes = recons.size / dim;
  const size_t num_points = points.size / dim;
  ASSERT_SIZE(likes, num_probes * num_points);

  CudaVector<float> dev_likes(likes.size);
  CudaVector<uint8_t> dev_points(points.size);
  CudaVector<uint8_t> dev_recons(recons.size);
  CudaVector<float> dev_work(Gpu::vq_construct_one_work_size(num_points));

  dev_likes = likes;
  dev_points = points;
  dev_recons.zero();

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {
    for (size_t i_probe = 0; i_probe < num_probes; ++i_probe) {

      CudaVector<float> dev_likes_block = dev_likes.block(num_points, i_probe);
      CudaVector<uint8_t> dev_recon = dev_recons.block(dim, i_probe);

      Gpu::vq_construct_one(
          dev_likes_block,
          dev_points,
          dev_recon,
          dev_work,
          tol);
    }
  }

  recons = dev_recons;

  float rate = num_probes * test_iters / timer.elapsed();
  float call_rate = test_iters / timer.elapsed();
  float gflops = 2 * dim * num_points * rate / 1e9;
  LOG(" computed " << (rate/1e3) << " means/ms ("
      << (1e3/call_rate) << " ms/call, " << gflops << " gflops)");

  //PRINT2(min(dev_recon), max(dev_recon)); // DEBUG
  //PRINT2(min(recon), max(recon)); // DEBUG
}

void test_vq_construct_batch_gpu (
    size_t dim,
    const Vector<float> & likes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    float tol,
    size_t test_iters)
{
  const size_t num_probes = recons.size / dim;
  const size_t num_points = points.size / dim;
  ASSERT_SIZE(likes, num_probes * num_points);

  CudaVector<float> dev_likes(likes.size);
  CudaVector<uint8_t> dev_points(points.size);
  CudaVector<uint8_t> dev_recons(recons.size);

  dev_likes = likes;
  dev_points = points;
  dev_recons.zero();

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {

    Gpu::vq_construct_batch(
        dev_likes,
        dev_points,
        dev_recons,
        tol,
        num_probes);
  }

  recons = dev_recons;

  float rate = num_probes * test_iters / timer.elapsed();
  float call_rate = test_iters / timer.elapsed();
  float gflops = 2 * dim * num_points * rate / 1e9;
  LOG(" computed " << (rate/1e3) << " means/ms ("
      << (1e3/call_rate) << " ms/call, " << gflops << " gflops)");

  //PRINT2(min(dev_recon), max(dev_recon)); // DEBUG
  //PRINT2(min(recon), max(recon)); // DEBUG
}

void test_vq_construct_deriv_gpu (
    const Vector<float> & likes,
    const Vector<float> & squared_distances,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    float tol,
    size_t num_probes,
    size_t test_iters)
{
  const size_t dim = probes.size / num_probes;
  const size_t num_points = points.size / dim;
  ASSERT_SIZE(likes, num_probes * num_points);

  CudaVector<float> dev_likes(likes.size);
  CudaVector<float> dev_sd(squared_distances.size);
  CudaVector<uint8_t> dev_probes(probes.size);
  CudaVector<uint8_t> dev_points(points.size);
  CudaVector<uint8_t> dev_recons(recons.size);

  dev_likes = likes;
  dev_sd = squared_distances;
  dev_probes = probes;
  dev_points = points;
  dev_recons.zero();

  Cloud::ConstructStats stats;

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {

    stats = Gpu::vq_construct_deriv(
        dev_likes,
        dev_sd,
        dev_probes,
        dev_points,
        dev_recons,
        tol,
        num_probes);
  }

  recons = dev_recons;

  float rate = num_probes * test_iters / timer.elapsed();
  float call_rate = test_iters / timer.elapsed();
  //float gflops = ??? * dim * num_points * rate / 1e9;
  LOG(" computed " << (rate/1e3) << " means/ms ("
      << (1e3/call_rate) << " ms/call)");

  PRINT2(stats.info, stats.surprise);

  //PRINT2(min(dev_recon), max(dev_recon)); // DEBUG
  //PRINT2(min(recon), max(recon)); // DEBUG
}

