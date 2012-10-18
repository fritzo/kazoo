#ifndef KAZOO_CLOUD_KERNELS_H
#define KAZOO_CLOUD_KERNELS_H

#include "common.h"
#include "cloud_stats.h"
#include "vectors.h"

namespace Cpu
{

//----( distance measurement )------------------------------------------------

void measure_one (
    const Vector<uint8_t> & probe,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances);

void measure_batch (
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances,
    size_t num_probes);

//----( quantization )--------------------------------------------------------

Cloud::QuantizeStats quantize_one (
    float radius,
    const Vector<float> & squared_distances,
    Vector<float> & likes);

Cloud::QuantizeStats quantize_batch (
    float radius,
    const Vector<float> & squared_distances,
    Vector<float> & likes,
    size_t num_probes);

//----( vq construction )-----------------------------------------------------

inline size_t vq_construct_work_size (size_t dim, size_t num_probes = 1)
{
  return dim * num_probes;
}

int vq_construct_one (
    const Vector<float> & likes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recon,
    Vector<float> & work,
    float tol);

void vq_construct_batch (
    const Vector<float> & likes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    Vector<float> & work,
    float tol,
    size_t num_probes = 1);

inline size_t vq_construct_deriv_work_size (size_t dim, size_t num_probes = 1)
{
  return 2 * dim * num_probes;
}

Cloud::ConstructStats vq_construct_deriv (
    const Vector<float> & likes,
    const Vector<float> & squared_distances,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    Vector<float> & work,
    float tol,
    size_t num_probes);

//----( fitting )-------------------------------------------------------------

inline size_t fit_points_work_size (size_t dim, size_t num_points)
{
  return dim * num_points;
}

void fit_points_to_obs (
    float rate_thresh,
    const Vector<uint8_t> & probes,
    const Vector<float> & rates,
    Vector<uint8_t> & points,
    Vector<float> & work,
    size_t num_probes);

void fit_points_to_recon (
    float radius,
    float rate_thresh,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & recons,
    const Vector<float> & rates,
    Vector<uint8_t> & points,
    Vector<float> & work,
    size_t num_probes);

} // namespace Cpu

#endif // KAZOO_CLOUD_KERNELS_H
