#ifndef KAZO_EIGEN_GPU
#define KAZO_EIGEN_GPU

#include "common.h"
#include "eigen.h"
#include "vectors.h"

struct cusparseMatDescr;
typedef struct cusparseMatDescr * cusparseMatDescr_t;

namespace Gpu
{

//----( dense matrix product )------------------------------------------------

// WARNING currently only trans_A = trans_B = false works
// C = alpha A B + beta C
void matrix_multiply (
    const MatrixXf & A,  // rect(I,J)
    bool trans_A,
    const MatrixXf & B,  // rect(J,K)
    bool trans_B,
    MatrixXf &  C,       // rect(I,K)
    float alpha = 1,
    float beta = 0);

//----( sparse matrix product )-----------------------------------------------

class SparseMultiplier
{
public:

  SparseMultiplier (const MatrixSf & sparse, size_t batch_size = 64);
  ~SparseMultiplier ();

  int rows () const { return m_rows; }
  int cols () const { return m_cols; }

  // WARNING it is always faster when transpose = true
  void left_imul (VectorXf & rhs, bool transpose);
  void left_mul (const VectorXf & rhs, VectorXf & result, bool transpose);
  void left_fma (const VectorXf & rhs, VectorXf & result, bool transpose);
  void left_imul (MatrixXf & rhs, bool transpose);
  void left_mul (const MatrixXf & rhs, MatrixXf & result, bool transpose);
  void left_fma (const MatrixXf & rhs, MatrixXf & result, bool transpose);

private:

  void matrix_vector (bool transpose, bool add_inplace);
  void matrix_matrix (bool transpose, bool add_inplace);

  const int m_rows;
  const int m_cols;

  const size_t m_batch_size;
  size_t m_buffer_size;

  float * const m_dev_sparse_csrVal;
  int * const m_dev_sparse_csrRowPtr;
  int * const m_dev_sparse_csrColInd;

  cusparseMatDescr_t m_sparse_descr;

  const float * m_dense_in;
  float * m_dense_out;
  float * const m_dev_dense_cols;
  float * const m_dev_dense_rows;
};

} // namespace Gpu

#endif // KAZO_EIGEN_GPU
