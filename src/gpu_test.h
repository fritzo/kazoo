#ifndef KAZOO_GPU_TEST_H
#define KAZOO_GPU_TEST_H

#include "gpu.h"
#include "vectors.h"

//----( reduction )-----------------------------------------------------------

float test_reduce_min (
    size_t size,
    const float * val,
    size_t test_iters = 10000);

float test_reduce_max (
    size_t size,
    const float * val,
    size_t test_iters = 10000);

float test_reduce_sum (
    size_t size,
    const float * val,
    size_t test_iters = 10000);

void test_parallel_reduce_min (
    size_t size,
    size_t parallel,
    const float * val,
    float * result,
    size_t test_iters = 10000);

//----( clouds )--------------------------------------------------------------

//----( measurement )----

void test_measure_one_cpu (
    size_t dim,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances,
    size_t test_iters);

void test_measure_batch_cpu (
    size_t dim,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances,
    size_t test_iters);

void test_measure_one_gpu (
    size_t dim,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances,
    size_t test_iters);

void test_measure_batch_gpu (
    size_t dim,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<float> & squared_distances,
    size_t test_iters);

//----( averaging )----

void test_vq_construct_one_cpu (
    size_t dim,
    const Vector<float> & likes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    float tol,
    size_t test_iters);

void test_vq_construct_batch_cpu (
    size_t dim,
    const Vector<float> & likes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    float tol,
    size_t test_iters);

void test_vq_construct_one_gpu (
    size_t dim,
    const Vector<float> & likes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    float tol,
    size_t test_iters);

void test_vq_construct_batch_gpu (
    size_t dim,
    const Vector<float> & likes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    float tol,
    size_t test_iters);

void test_vq_construct_deriv_cpu (
    const Vector<float> & likes,
    const Vector<float> & squared_distances,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    float tol,
    size_t num_probes,
    size_t test_iters);

void test_vq_construct_deriv_gpu (
    const Vector<float> & likes,
    const Vector<float> & squared_distances,
    const Vector<uint8_t> & probes,
    const Vector<uint8_t> & points,
    Vector<uint8_t> & recons,
    float tol,
    size_t num_probes,
    size_t test_iters);

#endif // KAZOO_GPU_TEST_H
