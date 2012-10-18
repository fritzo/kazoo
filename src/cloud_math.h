#ifndef KAZOO_CLOUD_MATH_H
#define KAZOO_CLOUD_MATH_H

#include "common.h"
#include "vectors.h"
#include "eigen.h"
#include "probability.h"
#include "cloud_stats.h"

namespace Cloud
{

typedef Vector<uint8_t> Point;

//----( random numbers )------------------------------------------------------

void randomize (Vector<float> x, float sigma);
inline void randomize (VectorXf & x, float sigma = 1)
{
  randomize(as_vector(x), sigma);
}
inline void randomize (MatrixXf & x, float sigma = 1)
{
  randomize(as_vector(x), sigma);
}

size_t random_index (const Vector<float> & likes);
size_t random_index (const VectorXf & likes);
int random_index (const VectorSf & likes);

void generate_noise (Vector<int8_t> & noise, float sigma);

//----( normalization )-------------------------------------------------------

inline void normalize_l1 (Vector<float> & x, float tot = 1)
{
  x *= tot / sum(x);
}
void normalize_l1 (VectorXf & x, float tot = 1);
void normalize_l1 (VectorSf & x, float tot = 1);
void normalize_l1 (MatrixXf & x, float tot = 1);
void normalize_l1 (MatrixSf & x, float tot = 1);

void normalize_rows_l1 (size_t I, size_t J, Vector<float> & A);
void normalize_columns_l1 (MatrixXf & A);
void normalize_columns_l1 (MatrixSf & A);

//----( sparse tools )--------------------------------------------------------

//              mean(x)^2
// density(x) = ---------
//              mean(x^2)

double density (const VectorXf & x);
float density (const VectorSf & x);
double density (const MatrixXf & x);
double density (const MatrixSf & x);

float sparsify_size (VectorSf & sparse, int size);
float sparsify_size (MatrixSf & sparse, int size);

float sparsify_absolute (
    const VectorXf & dense,
    VectorSf & sparse,
    float thresh);

float sparsify_absolute (
    const VectorSf & dense,
    VectorSf & sparse,
    float thresh);

float sparsify_absolute (
    const VectorXf & dense,
    VectorSf & sparse,
    const Vector<float> & thresh);

void sparsify_absolute (
    const MatrixXf & dense,
    MatrixSf & sparse,
    float thresh);

void sparsify_soft_relative_to_row_col_max (
    const MatrixXf & dense,
    MatrixSf & sparse,
    float relthresh,
    bool ignore_diagonal = false);

void sparsify_hard_relative_to_row_col_max (
    const MatrixXf & dense,
    MatrixSf & sparse,
    float relthresh,
    int max_entries,
    bool ignore_diagonal = false);

float max_entries_heuristic (MatrixXf & dense);

//----( joint probabilities )-------------------------------------------------

double likelihood_entropy (const VectorXf & likes);
double likelihood_entropy (const VectorSf & likes);
double likelihood_entropy_rate (const MatrixSf & joint_likes);
double likelihood_mutual_info (const MatrixSf & joint_likes);

void constrain_marginals_bp (
    MatrixXf & joint,
    const VectorXf & prior_dom,
    const VectorXf & prior_cod,
    VectorXf & temp_dom,
    VectorXf & temp_cod,
    float tol,
    size_t max_steps = 40,
    bool logging = false);

//----( distances )-----------------------------------------------------------

float squared_distance (const Point & x, const Point & y);

} // namespace Cloud

#endif // KAZOO_CLOUD_MATH_H
