
#include "cloud_points.h"
#include "cloud_kernels.h"
#include <cstring>
#include <climits>

namespace Cloud
{

//----( cpu point sets )------------------------------------------------------

class CpuPointSet : public PointSet
{
  Vector<uint8_t> m_points;

  Vector<double> m_prior_accum;
  size_t m_prior_accum_count;

  Vector<float> m_fit_rate_scales;
  Vector<float> m_fit_rates_batch;

  mutable Point m_temp_probes_batch;
  mutable Point m_temp_recons_batch;
  mutable Vector<float> m_temp_squared_distances_batch;
  mutable Vector<float> m_temp_likes_batch;

  mutable Vector<float> m_work;

public:

  CpuPointSet (size_t d, size_t c, Rectangle r);
  virtual ~CpuPointSet () {}

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

  uint8_t * get_point (size_t p) { return m_points + dim * p; }
  const uint8_t * get_point (size_t p) const { return m_points + dim * p; }

  void accum_prior (size_t num_probes);
  void construct (size_t num_probes) const;
  ConstructStats construct_deriv (size_t num_probes) const;
};

PointSet * PointSet::create_cpu (size_t dim, size_t size, Rectangle shape)
{
  return new CpuPointSet(dim, size, shape);
}

CpuPointSet::CpuPointSet (size_t d, size_t s, Rectangle r)
  : PointSet(d,s,r),
    m_points(dim * size),

    m_prior_accum(size),
    m_prior_accum_count(0),

    m_fit_rate_scales(size),
    m_fit_rates_batch(size * max_batch_size),

    m_temp_probes_batch(dim * max_batch_size),
    m_temp_recons_batch(dim * max_batch_size),
    m_temp_squared_distances_batch(size * max_batch_size),
    m_temp_likes_batch(size * max_batch_size),

    m_work(max(Cpu::vq_construct_work_size(dim, max_batch_size),
           max(Cpu::vq_construct_deriv_work_size(dim, max_batch_size),
               Cpu::fit_points_work_size(dim, size))))
{
  LOG("Building CpuPointSet with dim = " << dim << ", size = " << size);

  const double max_dimension = floor(double(INT_MAX) / sqr(255));
  ASSERT_LT(dim, max_dimension);
}

void CpuPointSet::get_point (size_t p, Point & point) const
{
  ASSERT1_LT(p, size);

  memcpy(point, get_point(p), dim);
}

void CpuPointSet::set_point (size_t p, const Point & point)
{
  ASSERT1_LT(p, size);

  memcpy(get_point(p), point, dim);
}

void CpuPointSet::update_fit_rates ()
{
  float * restrict prior = m_prior;
  float * restrict fit_rate_scales = m_fit_rate_scales;

  const float fit_rate = get_fit_rate() / size;

  for (size_t p = 0, P = size; p < P; ++p) {
    fit_rate_scales[p] = fit_rate / prior[p];
  }

  ASSERT_LT(0, min(m_fit_rate_scales));
}

void CpuPointSet::init_prior_accum ()
{
  m_prior_accum_count = 0;
  m_prior_accum.zero();
}

void CpuPointSet::update_prior_accum ()
{
  ASSERT_LT(0, m_prior_accum_count);
  const double scale = 1.0 / m_prior_accum_count;

  float * restrict prior = m_prior;
  const double * restrict accum = m_prior_accum;

  float * restrict fit_rate_scales = m_fit_rate_scales;
  const float fit_rate = get_fit_rate() / size;

  for (size_t p = 0, P = size; p < P; ++p) {

    float scaled_prior = scale * accum[p];
    prior[p] = scaled_prior;
    fit_rate_scales[p] = fit_rate / scaled_prior;
  }

  ASSERT_LT(0, min(m_fit_rate_scales));
}

//----( measure )----

void CpuPointSet::measure (const Point & probe) const
{
  ASSERT_SIZE(probe, dim);

  Cpu::measure_one(probe, m_points, m_temp_squared_distances);
}

void CpuPointSet::measure (size_t p) const
{
  ASSERT_LT(p, size);

  const Point probe = m_points.block(dim, p);

  Cpu::measure_one(probe, m_points, m_temp_squared_distances);
}

//----( quantize )----

void CpuPointSet::quantize (
    const Point & probe,
    Vector<float> & likes) const
{
  ASSERT_SIZE(probe, dim);
  ASSERT_SIZE(likes, size);

  measure(probe);
  Vector<float> & sd = m_temp_squared_distances;

  Cpu::quantize_one(get_radius(), sd, likes);
}

void CpuPointSet::quantize_batch (
    const Point & probes,
    Vector<float> & likes) const
{
  ASSERT_DIVIDES(dim, probes.size);
  const size_t num_probes = probes.size / dim;
  ASSERT_SIZE(likes, size * num_probes);
  ASSERT_LE(num_probes, max_batch_size);

  Vector<float> sd(size * num_probes, m_temp_squared_distances_batch);

  Cpu::measure_batch(probes, m_points, sd, num_probes);
  Cpu::quantize_batch(get_radius(), sd, likes, num_probes);
}

//----( vq construction )----

void CpuPointSet::construct (size_t num_probes) const
{
  Vector<float> likes(size * num_probes, m_temp_likes_batch);
  Vector<uint8_t> recons(dim * num_probes, m_temp_recons_batch);
  Vector<float> work(Cpu::vq_construct_work_size(dim, num_probes), m_work);

  ASSERT1_LE(sqr(mean(likes) - 1.0f), 1e-8f);
  const float sum_likes = size;
  const float tol = get_construct_tol() * sum_likes;

  Cpu::vq_construct_batch(likes, m_points, recons, work, tol, num_probes);
}

ConstructStats CpuPointSet::construct_deriv (size_t num_probes) const
{
  Vector<float> sd(size * num_probes, m_temp_squared_distances_batch);
  Vector<float> likes(size * num_probes, m_temp_likes_batch);
  Vector<uint8_t> probes(dim * num_probes, m_temp_probes_batch);
  Vector<uint8_t> recons(dim * num_probes, m_temp_recons_batch);
  Vector<float> work(
      Cpu::vq_construct_deriv_work_size(dim, num_probes), m_work);

  ASSERT1_LE(sqr(mean(likes) - 1.0f), 1e-8f);
  const float sum_likes = size;
  const float tol = get_construct_deriv_tol() * sum_likes;

  return Cpu::vq_construct_deriv(
      likes,
      sd,
      probes,
      m_points,
      recons,
      work,
      tol,
      num_probes);
}

void CpuPointSet::construct (const Vector<float> & likes, Point & point) const
{
  ASSERT_SIZE(likes, size);
  ASSERT_SIZE(point, dim);

  Vector<float> work(Cpu::vq_construct_work_size(dim), m_work);

  ASSERT1_LE(sqr(mean(likes) - 1.0f), 1e-8f);
  const float sum_likes = size;
  const float tol = get_construct_tol() * sum_likes;

  m_construct_one_stats_count += 1;
  m_construct_one_stats_total += Cpu::vq_construct_one(
      likes,
      m_points,
      point,
      work,
      tol);
}

//----( accum prior )----

void CpuPointSet::accum_prior (size_t num_probes)
{
  Vector<float> likes(size * num_probes, m_temp_likes_batch);

  for (size_t i_probe = 0; i_probe < num_probes; ++i_probe) {
    m_prior_accum += likes.block(size, i_probe);
  }
  m_prior_accum_count += num_probes;
}

//----( accum stats )----

void CpuPointSet::accum_stats (const Point & probes, size_t num_probes)
{
  ASSERT_SIZE(probes, dim * num_probes);
  ASSERT_LE(num_probes, max_batch_size);

  Vector<float> sd(size * num_probes, m_temp_squared_distances_batch);
  Vector<float> likes(size * num_probes, m_temp_likes_batch);

  Cpu::measure_batch(probes, m_points, sd, num_probes);

  m_quantize_stats_accum += Cpu::quantize_batch(
      get_radius(),
      sd,
      likes,
      num_probes);

  m_construct_stats_accum += construct_deriv(num_probes);
  m_count_stats_accum += num_probes;

  accum_prior(num_probes);
}

//----( fit point )----

void CpuPointSet::fit_points (const Point & probes, size_t num_probes)
{
  ASSERT_SIZE(probes, dim * num_probes);
  ASSERT_LE(num_probes, max_batch_size);

  const size_t num_points = size;

  float max_fit_rate = 1.0f / num_probes;
  float min_fit_rate = get_fit_rate_tol() * get_fit_rate();

  Vector<uint8_t> recons(probes.size, m_temp_recons_batch);
  Vector<float> sd(size * num_probes, m_temp_squared_distances_batch);
  Vector<float> likes(size * num_probes, m_temp_likes_batch);
  Vector<float> fit_rates(size * num_probes, m_fit_rates_batch);
  Vector<float> work(Cpu::fit_points_work_size(dim, size), m_work);

  Cpu::measure_batch(probes, m_points, sd, num_probes);

  m_quantize_stats_accum += Cpu::quantize_batch(
      get_radius(),
      sd,
      likes,
      num_probes);

  m_construct_stats_accum += construct_deriv(num_probes);
  m_count_stats_accum += num_probes;

  accum_prior(num_probes);

  for (size_t i_probe = 0; i_probe < num_probes; ++i_probe) {

    const float * restrict fit_rate_scales = m_fit_rate_scales;
    const float * restrict likes = m_temp_likes_batch + num_points * i_probe;
    float * restrict fit_rates = m_fit_rates_batch + num_points * i_probe;

    for (size_t p = 0; p < num_points; ++p) {
      fit_rates[p] = min(max_fit_rate, fit_rate_scales[p] * likes[p]);
    }
  }

  if (not fitting_points_to_recon()) {

    Cpu::fit_points_to_obs(
        min_fit_rate,
        probes,
        fit_rates,
        m_points,
        work,
        num_probes);

  } else {

    Cpu::fit_points_to_recon(
        get_radius(),
        min_fit_rate,
        probes,
        recons,
        fit_rates,
        m_points,
        work,
        num_probes);
  }
}

//----( purturbation )----

void CpuPointSet::purturb_points (size_t group_size, float purturb_factor)
{
  ASSERT_DIVIDES(group_size, size);
  const size_t num_probes = size / group_size;
  const size_t block_size = dim * group_size;

  float sigma = purturb_factor * get_radius() / sqrt(dim);

  LOG("purturbing point groups of size " << group_size << " by " << sigma);

  Vector<int8_t> noise(block_size);
  generate_noise(noise, sigma);

  int8_t * restrict dx = noise;

  for (size_t i_probe = 0; i_probe < num_probes; ++i_probe) {

    uint8_t * restrict x = m_points + block_size * i_probe;

    for (size_t i = 0; i < block_size; ++i) {

      x[i] = bound_to(0, 255, int(x[i]) + int(dx[i]));
    }
  }
}

} // namespace Cloud

