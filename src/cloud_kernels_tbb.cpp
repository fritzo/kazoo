
#include "cloud_kernels.h"
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

using Cloud::QuantizeStats;
using Cloud::ConstructStats;

namespace Cpu
{

//----( distance measurement )------------------------------------------------

void measure_one_notbb (
    const Vector<uint8_t> & probe,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances);

namespace
{
struct MeasureOne
{
  const Vector<uint8_t> * probe;
  const Vector<uint8_t> * points;
  Vector<float> * squared_distances;

  void operator() (const tbb::blocked_range<size_t> & range) const
  {
    const size_t dim = probe->size;
    const size_t num_points = range.end() - range.begin();

    Vector<uint8_t> po(dim * num_points, points->data + dim * range.begin());
    Vector<float> sd(num_points, squared_distances->data + range.begin());

    measure_one_notbb(* probe, po, sd);
  }
};
} // anonymous namespace

void measure_one (
    const Vector<uint8_t> & probe,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances)
{
  MeasureOne tasks = { & probe, & points, & squared_distances };

  static tbb::task_scheduler_init init;
  tbb::blocked_range<size_t> range(0, squared_distances.size);
  tbb::parallel_for(range, tasks);
}

//----( batch )----

namespace
{
struct MeasureBatch
{
  const size_t dim;
  const size_t num_points;
  const Vector<uint8_t> * probes;
  const Vector<uint8_t> * points;
  Vector<float> * squared_distances;

  void operator() (const tbb::blocked_range<size_t> & range) const
  {
    for (size_t i_probe = range.begin(); i_probe != range.end(); ++i_probe) {

      Vector<uint8_t> probe = probes->block(dim, i_probe);
      Vector<float> sd = squared_distances->block(num_points, i_probe);

      measure_one_notbb(probe, * points, sd);
    }
  }
};
} // anonymous namespace

void measure_batch (
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances,
    size_t num_probes)
{
  ASSERT_DIVIDES(num_probes, probes.size);
  const size_t dim = probes.size / num_probes;

  ASSERT_DIVIDES(dim, points.size);
  const size_t num_points = points.size / dim;

  MeasureBatch tasks = {
      dim,
      num_points,
      & probes,
      & points,
      & squared_distances};

  static tbb::task_scheduler_init init;
  tbb::blocked_range<size_t> range(0, num_probes);
  tbb::parallel_for(range, tasks);
}

//----( quantization )--------------------------------------------------------

namespace
{
struct QuantizeBatchData
{
  const float radius;
  const size_t num_points;
  const Vector<float> * squared_distances;
  Vector<float> * likes;
};
struct QuantizeBatch : public QuantizeBatchData
{
  QuantizeStats stats;

  QuantizeBatch (const QuantizeBatchData & data) : QuantizeBatchData(data) {}
  QuantizeBatch (QuantizeBatch & other, tbb::split)
    : QuantizeBatchData(other)
  {}
  void join (const QuantizeBatch & other) { stats += other.stats; }

  void operator() (const tbb::blocked_range<size_t> & range)
  {
    for (size_t i_probe = range.begin(); i_probe < range.end(); ++i_probe) {

      const Vector<float> sd_i = squared_distances->block(num_points, i_probe);
      Vector<float> likes_i = likes->block(num_points, i_probe);

      stats += quantize_one(radius, sd_i, likes_i);
    }
  }
};
} // anonymous namespace

QuantizeStats quantize_batch (
    float radius,
    const Vector<float> & squared_distances,
    Vector<float> & likes,
    size_t num_probes)
{
  ASSERT_DIVIDES(num_probes, squared_distances.size);
  const size_t num_points = squared_distances.size / num_probes;
  ASSERT_SIZE(likes, num_probes * num_points);

  QuantizeBatchData data = {
      radius,
      num_points,
      & squared_distances,
      & likes};
  QuantizeBatch tasks(data);

  static tbb::task_scheduler_init init;
  tbb::blocked_range<size_t> range(0, num_probes);
  tbb::parallel_reduce(range, tasks);

  return tasks.stats;
}

//----( vq construction )-----------------------------------------------------

namespace
{
struct ConstructBatch
{
  const size_t dim;
  const size_t num_points;
  const float tol;

  const Vector<float> * likes;
  const Vector<uint8_t> * points;
  Vector<uint8_t> * recons;
  Vector<float> * work;

  void operator() (const tbb::blocked_range<size_t> & range) const
  {
    // reuse a single work to improve cache locality
    Vector<float> work_i = work->block(dim, range.begin());

    for (size_t i_probe = range.begin(); i_probe < range.end(); ++i_probe) {

      const Vector<float> likes_i = likes->block(num_points, i_probe);
      Vector<uint8_t> recon_i = recons->block(dim, i_probe);

      vq_construct_one(likes_i, * points, recon_i, work_i, tol);
    }
  }
};
} // anonymous namespace

void vq_construct_batch (
    const Vector<float> & likes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    Vector<float> & work,
    float tol,
    size_t num_probes)
{
  ASSERT_DIVIDES(num_probes, recons.size);
  const size_t dim = recons.size / num_probes;
  ASSERT_DIVIDES(dim, points.size);
  const size_t num_points = points.size / dim;
  ASSERT_SIZE(likes, num_points * num_probes);
  ASSERT_SIZE(work, dim * num_probes);

  ConstructBatch tasks = {
      dim,
      num_points,
      tol,
      & likes,
      & points,
      & recons,
      & work};

  static tbb::task_scheduler_init init;
  tbb::blocked_range<size_t> range(0, num_probes);
  tbb::parallel_for(range, tasks);
}

ConstructStats vq_construct_deriv (
    const Vector<float> & likes,
    const Vector<float> & squared_distances,
    const Vector<uint8_t> & probe,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recon,
    Vector<float> & work,
    float tol);

namespace
{
struct ConstructDerivData
{
  const size_t dim;
  const size_t num_points;
  const float tol;

  const Vector<float> * likes;
  const Vector<float> * sd;
  const Vector<uint8_t> * probes;
  const Vector<uint8_t> * points;
  Vector<uint8_t> * recons;
  Vector<float> * work;
};
struct ConstructDeriv : public ConstructDerivData
{
  ConstructStats stats;

  ConstructDeriv (const ConstructDerivData & data)
    : ConstructDerivData(data)
  {}
  ConstructDeriv (ConstructDeriv & other, tbb::split)
    : ConstructDerivData(other)
  {}
  void join (const ConstructDeriv & other) { stats += other.stats; }

  void operator() (const tbb::blocked_range<size_t> & range)
  {
    // reuse a single work to improve cache locality
    Vector<float> work_i = work->block(2 * dim, range.begin());

    for (size_t i_probe = range.begin(); i_probe < range.end(); ++i_probe) {

      const Vector<float> likes_i = likes->block(num_points, i_probe);
      const Vector<float> sd_i = sd->block(num_points, i_probe);
      const Vector<uint8_t> probe_i = probes->block(dim, i_probe);
      Vector<uint8_t> recon_i = recons->block(dim, i_probe);

      stats += vq_construct_deriv(
          likes_i,
          sd_i,
          probe_i,
          * points,
          recon_i,
          work_i,
          tol);
    }
  }
};
} // anonymous namespace

ConstructStats vq_construct_deriv (
    const Vector<float> & likes,
    const Vector<float> & squared_distances,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    Vector<float> & work,
    float tol,
    size_t num_probes)
{
  ASSERT_DIVIDES(num_probes, probes.size);
  const size_t dim = probes.size / num_probes;
  ASSERT_SIZE(recons, dim * num_probes);
  ASSERT_SIZE(work, 2 * dim * num_probes);
  ASSERT_DIVIDES(dim, points.size);
  const size_t num_points = points.size / dim;
  ASSERT_SIZE(likes, num_points * num_probes);
  ASSERT_SIZE(squared_distances, num_points * num_probes);

  ConstructDerivData data = {
      dim,
      num_points,
      tol,
      & likes,
      & squared_distances,
      & probes,
      & points,
      & recons,
      & work};
  ConstructDeriv tasks(data);

  static tbb::task_scheduler_init init;
  tbb::blocked_range<size_t> range(0, num_probes);
  tbb::parallel_reduce(range, tasks);

  return tasks.stats;
}

//----( fitting )-------------------------------------------------------------

void fit_point_to_obs (
    float rate_thresh,
    const Vector<uint8_t> & probes,
    const size_t rate_offset,
    const size_t rate_stride,
    const Vector<float> & rates,
    Vector<uint8_t> & point,
    Vector<float> & work);

namespace
{
struct FitPointToObs
{
  const size_t dim;
  const size_t num_points;
  const size_t num_probes;
  const float rate_thresh;

  const Vector<uint8_t> * probes;
  const Vector<float> * rates;
  Vector<uint8_t>  * points;
  Vector<float> * work;

  void operator() (const tbb::blocked_range<size_t> & range) const
  {
    if (range.begin() == range.end()) return;

    // reuse a single work to improve cache locality
    Vector<float> work_i = work->block(dim, range.begin());

    for (size_t i_point = range.begin(); i_point < range.end(); ++i_point) {

      Vector<uint8_t> point = points->block(dim, i_point);

      fit_point_to_obs(
          rate_thresh,
          * probes,
          i_point,
          num_points,
          * rates,
          point,
          work_i);
    }
  }
};
} // anonymous namespace

void fit_points_to_obs (
    float rate_thresh,
    const Vector<uint8_t> & probes,
    const Vector<float> & rates,
    Vector<uint8_t> & points,
    Vector<float> & work,
    size_t num_probes)
{
  ASSERT_DIVIDES(num_probes, probes.size);
  const size_t dim = probes.size / num_probes;
  ASSERT_DIVIDES(dim, points.size);
  const size_t num_points = points.size / dim;
  ASSERT_SIZE(rates, num_points * num_probes);
  ASSERT_SIZE(work, dim * num_points);

  FitPointToObs tasks = {
      dim,
      num_points,
      num_probes,
      rate_thresh,
      & probes,
      & rates,
      & points,
      & work};

  static tbb::task_scheduler_init init;
  tbb::blocked_range<size_t> range(0, num_points);
  tbb::parallel_for(range, tasks);
}

//----( reconstruction fitting )----

void fit_point_to_recon (
    float radius,
    float rate_thresh,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & recons,
    const size_t rate_offset,
    const size_t rate_stride,
    const Vector<float> & rates,
    Vector<uint8_t> & point,
    Vector<float> & work);

namespace
{
struct FitPointToRecon
{
  size_t dim;
  size_t num_points;
  size_t num_probes;
  float radius;
  float rate_thresh;

  const Vector<uint8_t> * probes;
  const Vector<uint8_t> * recons;
  const Vector<float> * rates;
  Vector<uint8_t>  * points;
  Vector<float> * work;

  void operator() (const tbb::blocked_range<size_t> & range) const
  {
    if (range.begin() == range.end()) return;

    // reuse a single work to improve cache locality
    Vector<float> work_i = work->block(dim, range.begin());

    for (size_t i_point = range.begin(); i_point < range.end(); ++i_point) {

      Vector<uint8_t> point = points->block(dim, i_point);

      fit_point_to_recon(
          radius,
          rate_thresh,
          * probes,
          * recons,
          i_point,
          num_points,
          * rates,
          point,
          work_i);
    }
  }
};
} // anonymous namespace

void fit_points_to_recon (
    float radius,
    float rate_thresh,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & recons,
    const Vector<float> & rates,
    Vector<uint8_t> & points,
    Vector<float> & work,
    size_t num_probes)
{
  ASSERT_DIVIDES(num_probes, probes.size);
  const size_t dim = probes.size / num_probes;
  ASSERT_SIZE(recons, dim * num_probes);
  ASSERT_DIVIDES(dim, points.size);
  const size_t num_points = points.size / dim;
  ASSERT_SIZE(rates, num_points * num_probes);
  ASSERT_SIZE(work, dim * num_points);

  FitPointToRecon tasks = {
      dim,
      num_points,
      num_probes,
      radius,
      rate_thresh,
      & probes,
      & recons,
      & rates,
      & points,
      & work};

  static tbb::task_scheduler_init init;
  tbb::blocked_range<size_t> range(0, num_points);
  tbb::parallel_for(range, tasks);
}

} // namespace Cpu

