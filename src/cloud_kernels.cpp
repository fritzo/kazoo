
#include "cloud_kernels.h"
#include <climits>

using Cloud::QuantizeStats;
using Cloud::ConstructStats;

namespace Cpu
{

//----( distance measurement )------------------------------------------------

void measure_one_notbb (
    const Vector<uint8_t> & probe,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances)
{
  const size_t dim = probe.size;
  const size_t size = squared_distances.size;

  const double max_dim = floor(double(INT_MAX) / sqr(255));
  ASSERT_LT(dim, max_dim);

  typedef int Accum;

  const uint8_t * restrict x = probe;

  for (size_t p = 0; p < size; ++p) {

    const uint8_t * restrict y = points + dim * p;

    Accum d2 = 0;
    for (size_t i = 0; i < dim; ++i) {
      d2 += sqr(Accum(x[i]) - Accum(y[i]));
    }

    squared_distances[p] = d2;
  }
}

//----( quantization )--------------------------------------------------------

QuantizeStats quantize_one (
    float radius,
    const Vector<float> & squared_distances,
    Vector<float> & likes)
{
  const size_t num_points = squared_distances.size;
  ASSERT_SIZE(likes, num_points);

  // find sd_shift and sd_scale

  const float sd_scale = 0.5f / sqr(radius);
  const float sd_shift = sd_scale * min(squared_distances);

  // compute statistics (most importantly Z)

  const float * restrict sd = squared_distances.data;
  float * restrict pr = likes.data;

  float Z = 0;
  float U = 0;
  float U2 = 0;
  float Zcold = 0;
  float Ucold = 0;
  for (size_t p = 0; p < num_points; ++p) {

    float u = sd_scale * sd[p];
    float z = pr[p] = expf(sd_shift - u);

    Z += z;
    U += z * u;
    U2 += z * u * u;
    Zcold += z * z;
    Ucold += z * z * u;
  }
  ASSERT_LT(0, Z);
  ASSERT_LT(0, Zcold);

  likes *= num_points / Z;
  ASSERT1_LE(0, min(likes)); // really checks for NAN

  U *= 1 / Z;
  U2 *= 1 / Z;
  Ucold *= 1 / Zcold;
  float H = logf(Z) + U - sd_shift;

  return QuantizeStats(H, U, U2, Ucold);
}

//----( vq construction )-----------------------------------------------------

// These compute the reconstruction
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

int vq_construct_one (
    const Vector<float> & likes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recon,
    Vector<float> & work,
    float tol)
{
  const size_t dim = recon.size;
  const size_t num_points = likes.size;
  ASSERT_SIZE(points, dim * num_points);

  float * restrict mean_ = work;
  uint8_t * restrict recon_ = recon;

  work.zero();
  int num_terms = 0;

  for (size_t i_point = 0; i_point < num_points; ++i_point) {

    const float like = likes[i_point];

    if (like > tol) {

      const uint8_t * restrict point = points + dim * i_point;

      for (size_t i = 0; i < dim; ++i) {
        mean_[i] += like * point[i];
      }

      ++num_terms;
    }
  }

  // we assume mean(likes) = 1
  float inv_sum_likes = 1.0f / num_points;

  for (size_t i = 0; i < dim; ++i) {
    recon_[i] = bound_to(0, 255, roundi(mean_[i] * inv_sum_likes));
  }

  return num_terms;
}

ConstructStats vq_construct_deriv (
    const Vector<float> & likes,
    const Vector<float> & squared_distances,
    const Vector<uint8_t> & probe,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recon,
    Vector<float> & work,
    float tol)
{
  const size_t dim = probe.size;
  const size_t num_points = points.size / dim;

  const float * restrict likes_ = likes;
  const float * restrict sd_ = squared_distances;
  const uint8_t * restrict probe_ = probe;
  uint8_t * restrict recon_ = recon;

  float mean_sd = 0;
  float * restrict mean_qp = work;
  float * restrict mean_sd_qp = work + dim;
  work.zero();

  for (size_t i_point = 0; i_point < num_points; ++i_point) {

    const float like = likes_[i_point];

    if (like > tol) {

      const float sd = sd_[i_point];
      const float like_sd = like * sd;
      const uint8_t * restrict point_ = points + dim * i_point;

      mean_sd += like_sd;

      for (size_t x = 0; x < dim; ++x) {

        const float qp = probe_[x] - point_[x];

        mean_qp[x] += like * qp;
        mean_sd_qp[x] += like_sd * qp;
      }
    }
  }

  // we assume mean(likes) = 1
  const float inv_sum_likes = 1.0f / num_points;

  mean_sd *= inv_sum_likes;

  float error = 0;
  float info = 0;
  float surprise = 0;

  for (size_t x = 0; x < dim; ++x) {

    const float qr = mean_qp[x] * inv_sum_likes;
    const float cov_sd_qp = mean_sd_qp[x] * inv_sum_likes - qr * mean_sd;

    error += qr * qr;
    info += cov_sd_qp * cov_sd_qp;
    surprise += cov_sd_qp * qr;

    recon_[x] = bound_to(0, 255, roundi(probe_[x] - qr));
  }

  return ConstructStats(error, info, surprise);
}

//----( fitting )-------------------------------------------------------------

// This computes the update
//
//   p += sum q. rate_q (q - p)
//
// where p is the point and q is the observation

void fit_point_to_obs (
    float rate_thresh,
    const Vector<uint8_t> & probes,
    const size_t rate_offset,
    const size_t rate_stride,
    const Vector<float> & rates,
    Vector<uint8_t> & point,
    Vector<float> & work)
{
  const size_t dim = point.size;
  ASSERT_SIZE(work, dim);
  ASSERT_DIVIDES(dim, probes.size);
  const size_t num_probes = probes.size / dim;

  uint8_t * restrict p = point;
  float * restrict dp = work;

  work.zero();

  for (size_t i_probe = 0; i_probe < num_probes; ++i_probe) {

    const float rate = rates[rate_offset + rate_stride * i_probe];
    if (rate > rate_thresh) {

      const uint8_t * restrict q = probes + dim * i_probe;

      for (size_t i = 0; i < dim; ++i) {

        const float qi = q[i];
        const float pi = p[i];

        dp[i] += rate * (qi - pi);
      }
    }
  }

  const float jitter = random_01() - 0.5f;
  for (size_t i = 0; i < dim; ++i) {
    p[i] = bound_to(0, 255, p[i] + roundi(dp[i] + jitter));
  }
}

// This computes the update
//
//                    radius^2 (q - r) + <p-r|q-r> (q - p)
//   p += sum q. rate ------------------------------------
//                       sqrt( radius^4 + <p-r|q-r>^2 )
//
// where p is the point, q is the observation, and r is the reconstruction

void fit_point_to_recon (
    float radius,
    float rate_thresh,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & recons,
    const size_t rate_offset,
    const size_t rate_stride,
    const Vector<float> & rates,
    Vector<uint8_t> & point,
    Vector<float> & work)
{
  const size_t dim = point.size;
  ASSERT_SIZE(work, dim);
  ASSERT_DIVIDES(dim, probes.size);
  const size_t num_probes = probes.size / dim;
  const float sqr_radius = sqr(radius);

  uint8_t * restrict p = point;
  float * restrict dp = work;
  work.zero();

  for (size_t i_probe = 0; i_probe < num_probes; ++i_probe) {

    const float rate = rates[rate_offset + rate_stride * i_probe];
    if (rate > rate_thresh) {

      const uint8_t * restrict q = probes + dim * i_probe;
      const uint8_t * restrict r = recons + dim * i_probe;

      float ip = 0;

      for (size_t i = 0; i < dim; ++i) {

        const float qi = q[i];
        const float pi = p[i];
        const float ri = r[i];

        ip += (pi - ri) * (qi - ri);
      }

//#define DEBUG_FIT_POINT_TO_RECON
#ifdef DEBUG_FIT_POINT_TO_RECON

      const float safety = 0.5f;
      const float norm = sqrtf(sqr(sqr_radius) + sqr(ip));
      const float qr_rate = rate * ( safety * 0
                                   + (1-safety) * sqr_radius / norm );
      const float qp_rate = rate * ( safety * 1
                                   + (1-safety) * ip / norm );

#else // DEBUG_FIT_POINT_TO_RECON

      const float norm = sqrtf(sqr(sqr_radius) + sqr(ip));
      const float qr_rate = rate * sqr_radius / norm;
      const float qp_rate = rate * ip / norm;

#endif // DEBUG_FIT_POINT_TO_RECON

      for (size_t i = 0; i < dim; ++i) {

        const float qi = q[i];
        const float pi = p[i];

        dp[i] += qr_rate * (qi - r[i])
               + qp_rate * (qi - pi);
      }
    }
  }

  const float jitter = random_01() - 0.5f;
  for (size_t i = 0; i < dim; ++i) {
    p[i] = bound_to(0, 255, p[i] + roundi(dp[i] + jitter));
  }
}

} // namespace Cpu

