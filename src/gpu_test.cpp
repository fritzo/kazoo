
#include "gpu_test.h"
#include "cloud_kernels.h"
#include "args.h"
#include <climits>

//----( clouds )--------------------------------------------------------------

void test_measure_one_cpu (
    size_t dim,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances,
    size_t test_iters)
{
  const size_t num_probes = probes.size / dim;
  const size_t num_points = points.size / dim;
  ASSERT_SIZE(squared_distances, num_probes * num_points);

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {

    for (size_t i_probe = 0; i_probe < num_probes; ++i_probe) {

      Vector<uint8_t> probe = probes.block(dim, i_probe);
      Vector<float> sd_block = squared_distances.block(num_points, i_probe);

      Cpu::measure_one(probe, points, sd_block);
    }
  }

  const size_t num_pairs = num_probes * test_iters;
  float pair_rate = num_pairs * num_points / timer.elapsed();
  float call_rate = num_pairs / timer.elapsed();
  float gflops = 3 * dim * pair_rate / 1e9;
  LOG(" measured " << (pair_rate/1e6) << " pairs/ms ("
      << (1e3/call_rate) << " ms/call, " << gflops << " gflops)");
}

void test_measure_batch_cpu (
    size_t dim,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances,
    size_t test_iters)
{
  const size_t num_probes = probes.size / dim;
  const size_t num_points = points.size / dim;
  ASSERT_SIZE(squared_distances, num_probes * num_points);

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {

    Cpu::measure_batch(probes, points, squared_distances, num_probes);
  }

  const size_t num_pairs = num_probes * test_iters;
  float pair_rate = num_pairs * num_points / timer.elapsed();
  float call_rate = num_pairs / timer.elapsed();
  float gflops = 3 * dim * pair_rate / 1e9;
  LOG(" measured " << (pair_rate/1e6) << " pairs/ms ("
      << (1e3/call_rate) << " ms/call, " << gflops << " gflops)");
}

void test_vq_construct_one_cpu (
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

  Vector<float> work(Cpu::vq_construct_work_size(dim));

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {

    for (size_t i_probe = 0; i_probe < num_probes; ++i_probe) {

      Vector<float> likes_block = likes.block(num_points, i_probe);
      Vector<uint8_t> recon = recons.block(dim, i_probe);

      Cpu::vq_construct_one(likes_block, points, recon, work, tol);
    }
  }

  float rate = num_probes * test_iters / timer.elapsed();
  float call_rate = test_iters / timer.elapsed();
  float gflops = 2 * dim * num_points * rate / 1e9;
  LOG(" computed " << (rate/1e3) << " means/ms ("
      << (1e3/call_rate) << " ms/call, " << gflops << " gflops)");
}

void test_vq_construct_batch_cpu (
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

  Vector<float> work(Cpu::vq_construct_work_size(dim, num_probes));

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {

    Cpu::vq_construct_batch(likes, points, recons, work, tol, num_probes);
  }

  float rate = num_probes * test_iters / timer.elapsed();
  float call_rate = test_iters / timer.elapsed();
  float gflops = 2 * dim * num_points * rate / 1e9;
  LOG(" computed " << (rate/1e3) << " means/ms ("
      << (1e3/call_rate) << " ms/call, " << gflops << " gflops)");
}

void test_vq_construct_deriv_cpu (
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

  Vector<float> work(Cpu::vq_construct_deriv_work_size(dim, num_probes));

  Cloud::ConstructStats stats;

  Timer timer;
  for (size_t iter = 0; iter < test_iters; ++iter) {

    stats = Cpu::vq_construct_deriv(
        likes,
        squared_distances,
        probes,
        points,
        recons,
        work,
        tol,
        num_probes);
  }

  float rate = num_probes * test_iters / timer.elapsed();
  float call_rate = test_iters / timer.elapsed();
  //float gflops = ??? * dim * num_points * rate / 1e9;
  LOG(" computed " << (rate/1e3) << " means/ms ("
      << (1e3/call_rate) << " ms/call)");

  PRINT2(stats.info, stats.surprise);
}

//----( tools )---------------------------------------------------------------

void randomize (Vector<float> & x)
{
  for (size_t i = 0; i < x.size; ++i) {
    x[i] = random_std();
  }
}

void randomize (Vector<uint8_t> & x)
{
  for (size_t i = 0; i < x.size; ++i) {
    x[i] = random_choice(256);
  }
}

//----( tests )---------------------------------------------------------------

void run_info (Args & args)
{
  Gpu::print_gpu_info();
}

void run_measure (Args & args)
{
  size_t iters = args.pop(1);
  size_t num_probes = args.pop(64);
  size_t num_points = args.pop(64000);
  size_t dim = args.pop(1152);
  PRINT4(iters, num_probes, num_points, dim);

  Vector<uint8_t> probes(dim * num_probes);
  Vector<uint8_t> points(dim * num_points);

  randomize(probes);
  randomize(points);

  // distance results
  Vector<float> cpu_one(num_probes * num_points);
  Vector<float> cpu_batch(num_probes * num_points);
  Vector<float> gpu_one(num_probes * num_points);
  Vector<float> gpu_batch(num_probes * num_points);

  bool use_gpu = true;
  if (not Gpu::cuda_device_count()) {
    WARN("no cuda devices were found");
    use_gpu = false;
  }

  if (use_gpu) {
  LOG("\nGpu::measure_batch");
  test_measure_batch_gpu(
      dim,
      probes,
      points,
      gpu_batch,
      20 * iters);

  LOG("\nGpu::measure_one");
  test_measure_one_gpu(
      dim,
      probes,
      points,
      gpu_one,
      10 * iters);
  }

  LOG("\nCpu::measure_batch");
  test_measure_batch_cpu(
      dim,
      probes,
      points,
      cpu_batch,
      iters);

  LOG("\nCpu::measure_one");
  test_measure_one_cpu(
      dim,
      probes,
      points,
      cpu_one,
      iters);

  LOG("");
  PRINT(max_dist(cpu_one, cpu_batch));

  if (use_gpu) {
  PRINT(max_dist(gpu_one, gpu_batch));
  PRINT(max_dist(cpu_one, gpu_one));
  PRINT(max_dist(cpu_batch, cpu_batch));
  }
}

void run_profile_measure_one (Args & args)
{
  ASSERT(Gpu::cuda_device_count(), "no cuda devices were found");

  size_t iters = args.pop(100);
  const int max_size = 64000;
  size_t size = args.pop(max_size);
  size_t dim = args.pop(1152);
  PRINT3(iters, size, dim);

  Vector<uint8_t> probes(dim);
  randomize(probes);

  Vector<uint8_t> points(dim * size);
  randomize(points);

  Vector<float> dist_cpu(size);
  Vector<float> dist_gpu(size);

  test_measure_one_gpu(dim, probes, points, dist_gpu, iters);
}

void run_profile_measure_batch (Args & args)
{
  ASSERT(Gpu::cuda_device_count(), "no cuda devices were found");

  size_t iters = args.pop(100);
  size_t num_probes = args.pop(16);
  size_t num_points = args.pop(64000);
  size_t dim = args.pop(1152);
  PRINT4(iters, num_probes, num_points, dim);

  Vector<uint8_t> probes(dim * num_probes);
  Vector<uint8_t> points(dim * num_points);

  randomize(probes);
  randomize(points);

  Vector<float> dist_batch(num_probes * num_points);
  Vector<float> dist_sequential(num_probes * num_points);

  LOG("Batch distances");
  test_measure_batch_gpu(dim, probes, points, dist_batch, iters);
}

void run_vq_construct (Args & args)
{
  ASSERT(Gpu::cuda_device_count(), "no cuda devices were found");

  size_t iters = args.pop(10);
  size_t num_probes = args.pop(64);
  size_t num_points = args.pop(8192);
  size_t dim = args.pop(1152);
  float tol = args.pop(4e-3f);
  PRINT5(iters, num_probes, num_points, dim, tol);

  Vector<float> likes(num_probes * num_points);
  Vector<uint8_t> points(dim * num_points);
  Vector<uint8_t> mean_gpu_batch(dim * num_probes);
  Vector<uint8_t> mean_gpu_one(dim * num_probes);
  Vector<uint8_t> mean_cpu_batch(dim * num_probes);
  Vector<uint8_t> mean_cpu_one(dim * num_probes);

  randomize(points);
  randomize(likes);
  for (size_t i = 0, I = likes.size; i < I; ++i) {
    likes[i] = expf(4 * likes[i]);
  }
  likes *= num_points / sum(likes);

  PRINT2(min(likes), max(likes))

  bool use_gpu = true;
  if (not Gpu::cuda_device_count()) {
    WARN("no cuda devices were found");
    use_gpu = false;
  }

  if (use_gpu) {
  LOG("\nGpu::vq_construct_batch");
  test_vq_construct_batch_gpu(dim, likes, points, mean_gpu_batch, tol, iters);

  LOG("\nGpu::vq_construct_one");
  test_vq_construct_one_gpu(dim, likes, points, mean_gpu_one, tol, iters);
  }

  LOG("\nCpu::vq_construct_batch");
  test_vq_construct_batch_cpu(dim, likes, points, mean_cpu_batch, tol, iters);

  LOG("\nCpu::vq_construct_one");
  test_vq_construct_one_cpu(dim, likes, points, mean_cpu_one, tol, iters);

  LOG("");
  PRINT(max_dist(mean_cpu_batch, mean_cpu_one));

  if (use_gpu) {
  PRINT(max_dist(mean_gpu_batch, mean_cpu_batch));
  PRINT(max_dist(mean_gpu_batch, mean_gpu_one));
  PRINT(max_dist(mean_gpu_one, mean_cpu_one));
  }
}

void run_vq_construct_deriv (Args & args)
{
  ASSERT(Gpu::cuda_device_count(), "no cuda devices were found");

  size_t iters = args.pop(10);
  size_t num_probes = args.pop(64);
  size_t num_points = args.pop(64000);
  size_t dim = args.pop(1152);
  PRINT4(iters, num_probes, num_points, dim);

  const float radius = sqrt(0.1 * 255 * dim);
  Vector<uint8_t> points(dim * num_points);
  Vector<uint8_t> probes(dim * num_probes);
  Vector<float> squared_distances(num_probes * num_points);
  Vector<float> likes(num_probes * num_points);
  Vector<uint8_t> recons_gpu(dim * num_probes);
  Vector<uint8_t> recons_cpu(dim * num_probes);

  randomize(points);
  randomize(probes);
  Cpu::measure_batch(probes, points, squared_distances, num_probes);
  Cpu::quantize_batch(radius, squared_distances, likes, num_probes);

  PRINT3(min(likes), mean(likes), max(likes));

  float tol = 0;
  PRINT(tol)

  LOG("Gpu::vq_construct_deriv");
  test_vq_construct_deriv_gpu(
      likes,
      squared_distances,
      probes,
      points,
      recons_gpu,
      tol,
      num_probes,
      iters);

  LOG("Cpu::vq_construct_deriv");
  test_vq_construct_deriv_cpu(
      likes,
      squared_distances,
      probes,
      points,
      recons_cpu,
      tol,
      num_probes,
      iters);

  PRINT(max_dist(recons_gpu, recons_cpu));

  PRINT2(min(recons_gpu), min(recons_cpu));
  PRINT2(mean(recons_gpu), mean(recons_cpu));
  PRINT2(max(recons_gpu), max(recons_cpu));
}

void run_reduce (Args & args)
{
  ASSERT(Gpu::cuda_device_count(), "no cuda devices were found");

  const int max_size = 64000;
  size_t size = args.pop(max_size);
  size_t iters = args.pop(10000);
  PRINT2(size, iters);

  Vector<float> val(size);
  randomize(val);

  float gpu_min = test_reduce_min(size, val, iters);
  float cpu_min = min(val);
  ASSERT_EQ(gpu_min, cpu_min);

  float gpu_max = test_reduce_max(size, val, iters);
  float cpu_max = max(val);
  ASSERT_EQ(gpu_max, cpu_max);

  float gpu_sum = test_reduce_sum(size, val, iters);
  float cpu_sum = sum(val);

  LOG("min error = 0");
  LOG("max error = 0");
  LOG("sum error = " << (gpu_sum - cpu_sum));
}

void run_parallel_reduce (Args & args)
{
  ASSERT(Gpu::cuda_device_count(), "no cuda devices were found");

  const int max_size = 64000;
  size_t size = args.pop(max_size);
  const int parallel = args.pop(64);
  size_t iters = args.pop(10000);
  PRINT3(size, parallel, iters);

  Vector<float> val(parallel * size);
  randomize(val);

  Vector<float> result(parallel);

  test_parallel_reduce_min(size, parallel, val, result, iters);

  for (int p = 0; p < parallel; ++p) {

    float gpu_min = result[p];
    float cpu_min = min(val.block(size, p));
    ASSERT_EQ(gpu_min, cpu_min);
  }
}

void run_profile (Args & args)
{
  args
    .case_("measure_one", run_profile_measure_one)
    .case_("measure_batch", run_profile_measure_batch)
    .default_error();
}

//----( harness )-------------------------------------------------------------

const char * help_message =
"Usage: gpu_test COMMAND [OPTIONS]"
"\nCommands:"
"\n  info"
"\n  measure [ITERS = 1] [probes = 64] [POINTS = 64000] [DIM = 1152]"
"\n  construct [ITERS = 10] [probes = 64] [POINTS = 8192] [DIM = 1152] [TOL]"
"\n  con-deriv [ITERS = 10] [probes = 64] [POINTS = 64000] [DIM = 1152]"
"\n  reduce [POINTS = 64000] [ITERS = 1000]"
"\n  preduce [POINTS = 64000] [PARALLEL = 64] [ITERS = 1000]"
"\n  profile"
"\n    measure_one [ITERS = 100] [POINTS = 64000] [DIM = 1152]"
"\n    measure_batch [ITERS = 100] [PROBES = 16] [POINTS = 64000] [DIM = 1152]"
;

int main (int argc, char * * argv)
{
  Args(argc, argv, help_message)
    .case_("info", run_info)
    .case_("measure", run_measure)
    .case_("construct", run_vq_construct)
    .case_("con-deriv", run_vq_construct_deriv)
    .case_("profile", run_profile)
    .case_("reduce", run_reduce)
    .case_("preduce", run_parallel_reduce)
    .default_error();

  return 0;
}

