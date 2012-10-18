
#include "cloud_points.h"
#include "cloud_kernels.cu.h"

namespace Cloud
{

//----( gpu point sets )------------------------------------------------------

class GpuPointSet : public PointSet
{
  CudaVector<uint8_t> m_dev_points;
  CudaVector<float> m_dev_prior;

  CudaVector<double> m_dev_prior_accum;
  size_t m_prior_accum_count;

  mutable CudaVector<uint8_t> m_dev_probes_batch;
  mutable CudaVector<uint8_t> m_dev_recons_batch;
  mutable CudaVector<float> m_dev_squared_distances_batch;
  mutable CudaVector<float> m_dev_likes_batch;
  mutable CudaVector<float> m_dev_fit_rates_batch;

  mutable CudaVector<float> m_dev_fit_rate_scales;
  mutable CudaVector<float> m_dev_jitters;
  mutable CudaVector<curandState> m_dev_curand_states;

  mutable CudaVector<float> m_dev_work;
  mutable Vector<float> m_host_work;

public:

  GpuPointSet (size_t dim, size_t size, Rectangle shape);
  virtual ~GpuPointSet () {}

  virtual void get_point (size_t p, Point & point) const;
  virtual void set_point (size_t p, const Point & point);

  virtual void quantize (const Point & point, Vector<float> & likes) const;
  virtual void quantize_batch (
      const Point & points,
      Vector<float> & likes) const;
  virtual void construct (const Vector<float> & likes, Point & point) const;

protected:

  virtual void update_fit_rates ();
  virtual void init_prior_accum ();
  virtual void update_prior_accum ();

  virtual void measure (const Point & point) const;
  virtual void measure (size_t p) const;

  virtual void accum_stats (const Point & probes, size_t num_probes);
  virtual void fit_points (const Point & probes, size_t num_probes);

  virtual void purturb_points (size_t group_size, float purturb_factor);

private:

  uint8_t * get_point (size_t p) { return m_dev_points + dim * p; }
  const uint8_t * get_point (size_t p) const { return m_dev_points + dim * p; }

  inline void measure_one () const;
  inline void measure_batch (size_t num_probes) const;

  void quantize_one () const;
  QuantizeStats quantize_batch (size_t num_probes = 1) const;

  void accum_prior (size_t num_probes);

  inline void construct_one () const;
  inline void construct_batch (size_t num_probes) const;
  inline ConstructStats construct_deriv (size_t num_probes) const;
};

PointSet * PointSet::create_gpu (size_t dim, size_t size, Rectangle shape)
{
  return new GpuPointSet(dim, size, shape);
}

GpuPointSet::GpuPointSet (size_t d, size_t s, Rectangle r)
  : PointSet(d,s,r),

    m_dev_points(dim * size),
    m_dev_prior(size),

    m_dev_prior_accum(size),
    m_prior_accum_count(0),

    m_dev_probes_batch(dim * max_batch_size),
    m_dev_recons_batch(dim * max_batch_size),
    m_dev_squared_distances_batch(size * max_batch_size),
    m_dev_likes_batch(size * max_batch_size),
    m_dev_fit_rates_batch(size * max_batch_size),

    m_dev_fit_rate_scales(size),
    m_dev_jitters(size),
    m_dev_curand_states(size),

    m_dev_work(max(Gpu::quantize_dev_work_size(max_batch_size),
                   Gpu::vq_construct_one_work_size(size))),
    m_host_work(Gpu::quantize_host_work_size(max_batch_size))
{
  LOG("Building GpuPointSet with dim = " << dim << ", size = " << size);

  // this does not seem to help
  //LOG(" setting cuda to spin while waiting for device calls to return");
  //cudaSetDeviceFlags(cudaDeviceScheduleSpin);

  const double max_dimension = floor(double(INT_MAX) / sqr(255));
  ASSERT_LT(dim, max_dimension);

  Gpu::curand_init(m_dev_curand_states);
}

void GpuPointSet::get_point (size_t p, Point & point) const
{
  ASSERT1_LT(p, size);

  Gpu::copy_dev_to_host(point.data, get_point(p), dim);
}

void GpuPointSet::set_point (size_t p, const Point & point)
{
  ASSERT1_LT(p, size);

  Gpu::copy_host_to_dev(get_point(p), point.data, dim);
}

struct UpdateFitRateScalesOp
{
  int size;
  float fit_rate;
  const float * restrict prior;
  float * restrict fit_rate_scales;

  __device__ void load_map_store (int p) const
  {
    fit_rate_scales[p] = fit_rate / prior[p];
  }
};

void GpuPointSet::update_fit_rates ()
{
  float fit_rate = get_fit_rate() / size;

  m_dev_prior = m_prior;

  UpdateFitRateScalesOp op = {
      size,
      fit_rate,
      m_dev_prior,
      m_dev_fit_rate_scales};
  Gpu::map(op);

  ASSERT_LT(0, min(m_dev_fit_rate_scales));
}

void GpuPointSet::init_prior_accum ()
{
  m_dev_prior_accum.zero();
  m_prior_accum_count = 0;
}

struct UpdatePriorOp
{
  int size;
  double scale;
  float fit_rate;
  float * restrict prior;
  const double * restrict accum;
  float * restrict fit_rate_scales;

  __device__ void load_map_store (int p) const
  {
    float scaled_prior = scale * accum[p];
    prior[p] = scaled_prior;
    fit_rate_scales[p] = fit_rate / scaled_prior;
  }
};

void GpuPointSet::update_prior_accum ()
{
  ASSERT_LT(0, m_prior_accum_count);

  double scale = 1.0 / m_prior_accum_count;
  float fit_rate = get_fit_rate() / size;

  UpdatePriorOp op = {
      size,
      scale,
      fit_rate,
      m_dev_prior,
      m_dev_prior_accum,
      m_dev_fit_rate_scales};
  Gpu::map(op);

  ASSERT_LT(0, min(m_dev_fit_rate_scales));

  m_prior = m_dev_prior;
}

//----( measure )----

inline void GpuPointSet::measure_one () const
{
  CudaVector<uint8_t> dev_probe(dim, m_dev_probes_batch);
  CudaVector<float> dev_sd(size, m_dev_squared_distances_batch);

  Gpu::measure_one(dev_probe, m_dev_points, dev_sd);
}

inline void GpuPointSet::measure_batch (size_t num_probes) const
{
  // Gpu::measure_batch expects blocks to be a multiple of 16.
  int blocks = (num_probes + 15) / 16 * 16;
  ASSERT_LE(blocks, max_batch_size);

  CudaVector<uint8_t> dev_probes(dim * blocks, m_dev_probes_batch);
  CudaVector<float> dev_sd(size * blocks, m_dev_squared_distances_batch);

  Gpu::measure_batch(dev_probes, m_dev_points, dev_sd, blocks);
}

void GpuPointSet::measure (const Point & probe) const
{
  ASSERT_SIZE(probe, dim);

  CudaVector<uint8_t> dev_probe(dim, m_dev_probes_batch);
  CudaVector<float> dev_sd(size, m_dev_squared_distances_batch);
  Vector<float> sd(size, m_temp_squared_distances);

  dev_probe = probe;

  Gpu::measure_one(dev_probe, m_dev_points, dev_sd);

  sd = dev_sd;
}

void GpuPointSet::measure (size_t p) const
{
  ASSERT_LT(p, size);

  CudaVector<uint8_t> dev_probe = m_dev_points.block(dim, p);
  CudaVector<float> dev_sd(size, m_dev_squared_distances_batch);
  Vector<float> sd(size, m_temp_squared_distances);

  Gpu::measure_one(dev_probe, m_dev_points, dev_sd);

  sd = dev_sd;
}

//----( quantize )----

inline void GpuPointSet::quantize_one () const
{
  CudaVector<float> dev_likes(size, m_dev_likes_batch);
  CudaVector<float> dev_sd(size, m_dev_squared_distances_batch);

  CudaVector<float> dev_work(Gpu::quantize_dev_work_size(), m_dev_work);
  Vector<float> host_work(Gpu::quantize_host_work_size(), m_host_work);

  Gpu::quantize_one(
      get_radius(),
      dev_sd,
      dev_likes,
      dev_work,
      host_work);

  ASSERT1_LE(0, min(dev_likes)); // really checks for NAN
}

inline QuantizeStats GpuPointSet::quantize_batch (size_t num_probes) const
{
  CudaVector<float> dev_likes(size * num_probes, m_dev_likes_batch);
  CudaVector<float> dev_sd(size * num_probes, m_dev_squared_distances_batch);

  CudaVector<float> dev_work(
      Gpu::quantize_dev_work_size(num_probes),
      m_dev_work);
  Vector<float> host_work(
      Gpu::quantize_host_work_size(num_probes),
      m_host_work);

  QuantizeStats stats = Gpu::quantize_batch(
      get_radius(),
      dev_sd,
      dev_likes,
      dev_work,
      host_work,
      num_probes);

  ASSERT1_LE(0, min(dev_likes)); // really checks for NAN

  return stats;
}

void GpuPointSet::quantize (const Point & probe, Vector<float> & likes) const
{
  ASSERT_SIZE(probe, dim);
  ASSERT_SIZE(likes, size);

  CudaVector<uint8_t> dev_probe(dim, m_dev_probes_batch);
  CudaVector<float> dev_likes(size, m_dev_likes_batch);

  dev_probe = probe;

  measure_one();

  quantize_one();

  likes = dev_likes;
}

void GpuPointSet::quantize_batch (
    const Point & probes,
    Vector<float> & likes) const
{
  ASSERT_DIVIDES(dim, probes.size);
  const size_t num_probes = probes.size / dim;
  ASSERT_SIZE(likes, size * num_probes);
  ASSERT_LE(num_probes, max_batch_size);

  CudaVector<uint8_t> dev_probes(dim * num_probes, m_dev_probes_batch);
  CudaVector<float> dev_likes(size * num_probes, m_dev_likes_batch);

  dev_probes = probes;

  measure_batch(num_probes);
  quantize_batch(num_probes);

  likes = dev_likes;
}

//----( construct )----

inline void GpuPointSet::construct_one () const
{
  CudaVector<float> dev_likes(size, m_dev_likes_batch);
  CudaVector<uint8_t> dev_recons(dim, m_dev_recons_batch);
  CudaVector<float> dev_work(Gpu::vq_construct_one_work_size(size), m_dev_work);

  ASSERT1_LE(sqr(mean(dev_likes) - 1.0f), 1e-8f);
  const float sum_likes = size;
  const float tol = get_construct_tol() * sum_likes;

  m_construct_one_stats_count += 1;
  m_construct_one_stats_total += Gpu::vq_construct_one(
      dev_likes,
      m_dev_points,
      dev_recons,
      dev_work,
      tol);
}

inline void GpuPointSet::construct_batch (size_t num_probes) const
{
  CudaVector<float> dev_likes(size * num_probes, m_dev_likes_batch);
  CudaVector<uint8_t> dev_recons(dim * num_probes, m_dev_recons_batch);

  ASSERT1_LE(sqr(mean(dev_likes) - 1.0f), 1e-8f);
  const float sum_likes = size;
  const float tol = get_construct_tol() * sum_likes;

  Gpu::vq_construct_batch(dev_likes, m_dev_points, dev_recons, tol, num_probes);
}

inline ConstructStats GpuPointSet::construct_deriv (size_t num_probes) const
{
  CudaVector<float> dev_likes(size * num_probes, m_dev_likes_batch);
  CudaVector<float> dev_sd(size * num_probes, m_dev_squared_distances_batch);
  CudaVector<uint8_t> dev_probes(dim * num_probes, m_dev_probes_batch);
  CudaVector<uint8_t> dev_recons(dim * num_probes, m_dev_recons_batch);

  ASSERT1_LE(sqr(mean(dev_likes) - 1.0f), 1e-8f);
  const float sum_likes = size;
  const float tol = get_construct_deriv_tol() * sum_likes;

  return Gpu::vq_construct_deriv(
      dev_likes,
      dev_sd,
      dev_probes,
      m_dev_points,
      dev_recons,
      tol,
      num_probes);
}

void GpuPointSet::construct (const Vector<float> & likes, Point & point) const
{
  ASSERT_SIZE(likes, size);
  ASSERT_SIZE(point, dim);

  CudaVector<float> dev_likes(size, m_dev_likes_batch);
  CudaVector<uint8_t> dev_recon(dim, m_dev_recons_batch);

  dev_likes = likes;

  construct_one();

  point = dev_recon;
}

//----( accum prior )----

struct AccumPriorOp
{
  int size;
  int num_probes;
  const float * restrict likes;
  double * restrict prior_accum;

  __device__ void load_map_store (int p) const
  {
    float like = 0;

    #pragma unroll 16
    for (int i_probe = 0; i_probe < num_probes; ++i_probe) {
      like += likes[size * i_probe + p];
    }

    prior_accum[p] += like;
  }
};

void GpuPointSet::accum_prior (size_t num_probes)
{
  AccumPriorOp op = {
      size,
      num_probes,
      m_dev_likes_batch,
      m_dev_prior_accum};
  Gpu::map(op);

  m_prior_accum_count += num_probes;
}

//----( accum stats )----

void GpuPointSet::accum_stats (const Point & probes, size_t num_probes)
{
  ASSERT_SIZE(probes, dim * num_probes);
  ASSERT_LE(num_probes, max_batch_size);

  CudaVector<uint8_t> dev_probes(probes.size, m_dev_probes_batch);
  CudaVector<float> dev_sd(size * num_probes, m_dev_squared_distances_batch);

  dev_probes = probes;

  measure_batch(num_probes);

  m_quantize_stats_accum += quantize_batch(num_probes);
  m_construct_stats_accum += construct_deriv(num_probes);
  m_count_stats_accum += num_probes;

  accum_prior(num_probes);
}

//----( fit point )----

struct UpdateFitRatesOp
{
  int size;
  float max_fit_rate;

  const float * restrict likes;
  const float * restrict fit_rate_scales;
  float * restrict fit_rates;

  __device__ void load_map_store (int p, int block = 0) const
  {
    int p0 = p - size * block; // fit_rate_scales are the same for each block

    fit_rates[p] = min(max_fit_rate, fit_rate_scales[p0] * likes[p]);
  }
};

void GpuPointSet::fit_points (const Point & probes, size_t num_probes)
{
  ASSERT_SIZE(probes, dim * num_probes);
  ASSERT_LE(num_probes, max_batch_size);

  float max_fit_rate = 1.0f / num_probes;
  float min_fit_rate = get_fit_rate_tol() * get_fit_rate();

  CudaVector<uint8_t> dev_probes(probes.size, m_dev_probes_batch);
  CudaVector<uint8_t> dev_recons(probes.size, m_dev_recons_batch);
  CudaVector<float> dev_sd(size * num_probes, m_dev_squared_distances_batch);
  CudaVector<float> dev_likes(size * num_probes, m_dev_likes_batch);
  CudaVector<float> dev_fit_rates(size * num_probes, m_dev_fit_rates_batch);

  dev_probes = probes;

  measure_batch(num_probes);

  m_quantize_stats_accum += quantize_batch(num_probes);
  m_construct_stats_accum += construct_deriv(num_probes);
  m_count_stats_accum += num_probes;

  accum_prior(num_probes);

  UpdateFitRatesOp op = {
      size,
      max_fit_rate,
      m_dev_likes_batch,
      m_dev_fit_rate_scales,
      m_dev_fit_rates_batch};
  Gpu::map(op, num_probes);

  if (not fitting_points_to_recon()) {

    Gpu::fit_points_to_obs(
        min_fit_rate,
        dev_probes,
        dev_fit_rates,
        m_dev_curand_states,
        m_dev_jitters,
        m_dev_points);

  } else {

    Gpu::fit_points_to_recon(
        get_radius(),
        min_fit_rate,
        dev_probes,
        dev_recons,
        dev_fit_rates,
        m_dev_curand_states,
        m_dev_jitters,
        m_dev_points);
  }
}

//----( purturbation )----

struct PurturbPointsOp
{
  int size;
  const int8_t * restrict noise;
  uint8_t * restrict points;

  __device__ void load_map_store (int p, int block = 0) const
  {
    int p0 = p - size * block; // noise is the same for each block

    points[p] = max(0, min(255, int(points[p]) + int(noise[p0])));
  }
};

void GpuPointSet::purturb_points (size_t group_size, float purturb_factor)
{
  ASSERT_DIVIDES(group_size, size);
  const size_t blocks = size / group_size;
  const size_t block_size = dim * group_size;

  float sigma = purturb_factor * get_radius() / sqrt(dim);

  LOG("purturbing point groups of size " << group_size << " by " << sigma);

  Vector<int8_t> noise(block_size);
  generate_noise(noise, sigma);

  CudaVector<int8_t> dev_noise(block_size);
  dev_noise = noise;

  PurturbPointsOp op = { block_size, dev_noise, m_dev_points };
  Gpu::map(op, blocks);
}

} // namespace Cloud

