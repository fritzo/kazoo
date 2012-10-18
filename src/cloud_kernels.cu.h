#ifndef KAZOO_CLOUD_KERNELS_CU_H
#define KAZOO_CLOUD_KERNELS_CU_H

#include "common.h"
#include "cloud_stats.h"
#include "gpu.cu.h"
#include "vectors.cu.h"

namespace Gpu
{

//----( distance measurement )------------------------------------------------

void measure_one (
    const CudaVector<uint8_t> & probe,
    const CudaVector<uint8_t> & points,
    CudaVector<float> & squared_distances);

void measure_batch (
    const CudaVector<uint8_t> & probes,
    const CudaVector<uint8_t> & points,
    CudaVector<float> & squared_distances,
    size_t num_probes);

//----( quantization )--------------------------------------------------------

inline size_t quantize_dev_work_size (size_t num_probes = 1)
{
  return 6 * num_probes;
}
inline size_t quantize_host_work_size (size_t num_probes = 1)
{
  return 4 * num_probes;
}

Cloud::QuantizeStats quantize_batch (
    float radius,
    const CudaVector<float> & squared_distances,
    CudaVector<float> & likes,
    CudaVector<float> & work,
    Vector<float> & host_work,
    size_t batch_size);

inline void quantize_one (
    float radius,
    const CudaVector<float> & squared_distances,
    CudaVector<float> & likes,
    CudaVector<float> & work,
    Vector<float> & host_work)
{
  quantize_batch(radius, squared_distances, likes, work, host_work, 1);
}

//----( vq construction )-----------------------------------------------------

inline size_t vq_construct_one_work_size (size_t num_points)
{
  return 2 * num_points;
}

int vq_construct_one (
    const CudaVector<float> & likes,
    const CudaVector<uint8_t> & points,
    CudaVector<uint8_t> & recon,
    CudaVector<float> & work,
    float tol);

// this version uses no work
void vq_construct_one (
    const CudaVector<float> & likes,
    const CudaVector<uint8_t> & points,
    CudaVector<uint8_t> & recon,
    float tol);

void vq_construct_batch (
    const CudaVector<float> & likes,
    const CudaVector<uint8_t> & points,
    CudaVector<uint8_t> & recon,
    float tol,
    size_t num_probes);

Cloud::ConstructStats vq_construct_deriv (
    const CudaVector<float> & likes,
    const CudaVector<float> & squared_distances,
    const CudaVector<uint8_t> & probes,
    const CudaVector<uint8_t> & points,
    CudaVector<uint8_t> & recon,
    float tol,
    size_t num_probes);

//----( fitting )-------------------------------------------------------------

void fit_points_to_obs (
    float rate_thresh,
    const CudaVector<uint8_t> & probes,
    const CudaVector<float> & rates,
    CudaVector<curandState> & curand_states,
    CudaVector<float> & jitters,
    CudaVector<uint8_t> & points);

void fit_points_to_recon (
    float radius,
    float rate_thresh,
    const CudaVector<uint8_t> & probes,
    const CudaVector<uint8_t> & recons,
    const CudaVector<float> & rates,
    CudaVector<curandState> & curand_states,
    CudaVector<float> & jitters,
    CudaVector<uint8_t> & points);

} // namespace Gpu

#endif // KAZOO_CLOUD_KERNELS_CU_H
