
#include "eigen_gpu.h"
#include "gpu.h"
#include <cublas.h>
#include <cusparse.h>

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#define ASSERT_CUBLAS(info) \
  ASSERT(info == CUBLAS_STATUS_SUCCESS, \
      "cublas error: " << print_cublas_error(info))

#define ASSERT_CUSPARSE(info) \
  ASSERT(info == CUSPARSE_STATUS_SUCCESS, \
      "cusparse error: " << print_cusparse_error(info))

namespace Gpu
{

//----( cublas misc )---------------------------------------------------------

// WARNING there is disagreement between cublas.h and
//   version 4.0 of the cublas reference manual; we follow cublas.h

static const char * print_cublas_error (cublasStatus_t info)
{
  switch (info) {
    case CUBLAS_STATUS_SUCCESS: return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "cublas not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "alloc failed";
    case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "arch mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR: return "mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
    default: return "unrecognized cublas error";
  }
}

void init_cublas ()
{
  static bool initialized = false;
  if (not initialized) {

    ASSERT_CUBLAS(cublasInit());

    initialized = true;

    int version = 0;
    ASSERT_CUBLAS(cublasGetVersion(& version));
    LOG("using CUBLAS version " << (version * 1e-3f));
  }
}

//----( cusparse misc )-------------------------------------------------------

static const char * print_cusparse_error (cusparseStatus_t info)
{
  switch (info) {
    case CUSPARSE_STATUS_SUCCESS: return "success";
    case CUSPARSE_STATUS_NOT_INITIALIZED: return "cusparse not initialized";
    case CUSPARSE_STATUS_ALLOC_FAILED: return "alloc failed";
    case CUSPARSE_STATUS_INVALID_VALUE: return "invalid value";
    case CUSPARSE_STATUS_ARCH_MISMATCH: return "arch mismatch";
    case CUSPARSE_STATUS_MAPPING_ERROR: return "mapping error";
    case CUSPARSE_STATUS_EXECUTION_FAILED: return "execution failed";
    case CUSPARSE_STATUS_INTERNAL_ERROR: return "internal error";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "matrix type not supported";
    default: return "unrecognized cusparse error";
  }
}

static cusparseHandle_t g_cusparseHandle = NULL;

void init_cusparse ()
{
  if (g_cusparseHandle == NULL) {

    ASSERT_CUSPARSE(cusparseCreate(& g_cusparseHandle));

    int version = 0;
    ASSERT_CUSPARSE(cusparseGetVersion(g_cusparseHandle, & version));
    LOG("using CUSPARSE version " << (version * 1e-3f));
  }
}

//----( dense matrix product )------------------------------------------------

void matrix_multiply (
    const MatrixXf & A,  // rect(I,J)
    bool trans_A,
    const MatrixXf & B,  // rect(J,K)
    bool trans_B,
    MatrixXf & C,        // rect(I,K)
    float alpha,
    float beta)
{
  ASSERT(not trans_A, "XXX transpose(A) is not working yet");
  ASSERT(not trans_B, "XXX transpose(B) is not working yet");

  init_cublas();

  // TODO buffer tiles to reduce gpu memory footprint

  ASSERT_EQ(C.rows(), trans_A ? A.cols() : A.rows());
  ASSERT_EQ(C.cols(), trans_B ? B.rows() : B.cols());
  ASSERT_EQ(trans_A ? A.rows() : A.cols(), trans_B ? B.cols() : B.rows());

  int m = C.rows();
  int n = C.cols();
  int k = trans_A ? A.rows() : A.cols();

  float * dev_A = (float *) cuda_malloc(A.size() * sizeof(float));
  float * dev_B = (float *) cuda_malloc(B.size() * sizeof(float));
  float * dev_C = (float *) cuda_malloc(C.size() * sizeof(float));

  cuda_memcpy_h2d(dev_A, A.data(), A.size() * sizeof(float));
  cuda_memcpy_h2d(dev_B, B.data(), B.size() * sizeof(float));
  if (beta != 0) {
    cuda_memcpy_h2d(dev_C, C.data(), C.size() * sizeof(float));
  }

  cublasSgemm(
      trans_A ? CUBLAS_OP_T : CUBLAS_OP_N,
      trans_B ? CUBLAS_OP_T : CUBLAS_OP_N,
      m, n, k,
      alpha,
      dev_A, A.rows(),
      dev_B, B.rows(),
      beta,
      dev_C, C.rows());

  ASSERT_CUBLAS(cublasGetError());

  cuda_memcpy_d2h(C.data(), dev_C, C.size() * sizeof(float));

  cuda_free(dev_A);
  cuda_free(dev_B);
  cuda_free(dev_C);
}

//----( sparse matrix product )-----------------------------------------------

SparseMultiplier::SparseMultiplier (
    const MatrixSf & sparse,
    size_t batch_size)

  : m_rows(sparse.rows()),
    m_cols(sparse.cols()),

    m_batch_size(batch_size),
    m_buffer_size(0),

    m_dev_sparse_csrVal(
        (float *) cuda_malloc(sparse.nonZeros() * sizeof(float))),
    m_dev_sparse_csrRowPtr(
        (int *) cuda_malloc((1 + sparse.outerSize()) * sizeof(int))),
    m_dev_sparse_csrColInd(
        (int *) cuda_malloc(sparse.nonZeros() * sizeof(int))),

    m_sparse_descr(NULL),

    m_dense_in(NULL),
    m_dense_out(NULL),
    m_dev_dense_cols(
        (float *) cuda_malloc(sparse.cols() * batch_size * sizeof(float))),
    m_dev_dense_rows(
        (float *) cuda_malloc(sparse.rows() * batch_size * sizeof(float)))
{
  init_cusparse();
  ASSERT_CUSPARSE(cusparseCreateMatDescr(& m_sparse_descr));

  // DEBUG
  //PRINT(m_sparse_csrVal);
  //PRINT(m_sparse_csrRowPtr);
  //PRINT(m_sparse_csrColInd);

  cuda_memcpy_h2d(
      m_dev_sparse_csrVal,
      sparse._valuePtr(),
      sparse.nonZeros() * sizeof(float));
  cuda_memcpy_h2d(
      m_dev_sparse_csrRowPtr,
      sparse._outerIndexPtr(),
      (1 + sparse.outerSize()) * sizeof(int));
  cuda_memcpy_h2d(
      m_dev_sparse_csrColInd,
      sparse._innerIndexPtr(),
      sparse.nonZeros() * sizeof(int));
}

SparseMultiplier::~SparseMultiplier ()
{
  ASSERT_CUSPARSE(cusparseDestroyMatDescr(m_sparse_descr));

  cuda_free(m_dev_sparse_csrVal);
  cuda_free(m_dev_sparse_csrRowPtr);
  cuda_free(m_dev_sparse_csrColInd);
  cuda_free(m_dev_dense_cols);
  cuda_free(m_dev_dense_rows);
}

void SparseMultiplier::left_imul (
    VectorXf & rhs,
    bool transpose)
{
  ASSERT_EQ(rows(), cols());
  ASSERT_EQ(rhs.size(), rows());

  m_dense_in = rhs.data();
  m_dense_out = rhs.data();

  matrix_vector(transpose, false);
}

void SparseMultiplier::left_mul (
    const VectorXf & rhs,
    VectorXf & result,
    bool transpose)
{
  if (transpose) {
    ASSERT_EQ(rhs.size(), rows());
    ASSERT_EQ(result.size(), cols());
  } else {
    ASSERT_EQ(rhs.size(), cols());
    ASSERT_EQ(result.size(), rows());
  }

  m_dense_in = rhs.data();
  m_dense_out = result.data();

  matrix_vector(transpose, false);
}

void SparseMultiplier::left_fma (
    const VectorXf & rhs,
    VectorXf & result,
    bool transpose)
{
  if (transpose) {
    ASSERT_EQ(rhs.size(), rows());
    ASSERT_EQ(result.size(), cols());
  } else {
    ASSERT_EQ(rhs.size(), cols());
    ASSERT_EQ(result.size(), rows());
  }

  m_dense_in = rhs.data();
  m_dense_out = result.data();

  matrix_vector(transpose, true);
}

void SparseMultiplier::left_imul (
    MatrixXf & rhs,
    bool transpose)
{
  ASSERT_EQ(rows(), cols());
  ASSERT_EQ(rhs.rows(), rows());

  m_buffer_size = rhs.cols();
  m_dense_in = rhs.data();
  m_dense_out = rhs.data();

  matrix_matrix(transpose, false);
}

void SparseMultiplier::left_mul (
    const MatrixXf & rhs,
    MatrixXf & result,
    bool transpose)
{
  if (transpose) {
    ASSERT_EQ(rhs.rows(), rows());
    ASSERT_EQ(result.rows(), cols());
  } else {
    ASSERT_EQ(rhs.rows(), cols());
    ASSERT_EQ(result.rows(), rows());
  }
  ASSERT_EQ(rhs.cols(), result.cols());

  m_buffer_size = rhs.cols();
  m_dense_in = rhs.data();
  m_dense_out = result.data();

  matrix_matrix(transpose, false);
}

void SparseMultiplier::left_fma (
    const MatrixXf & rhs,
    MatrixXf & result,
    bool transpose)
{
  if (transpose) {
    ASSERT_EQ(result.rows(), rows());
    ASSERT_EQ(result.rows(), cols());
  } else {
    ASSERT_EQ(rhs.rows(), rows());
    ASSERT_EQ(rhs.rows(), cols());
  }
  ASSERT_EQ(rhs.cols(), result.cols());

  m_buffer_size = rhs.cols();
  m_dense_in = rhs.data();
  m_dense_out = result.data();

  matrix_matrix(transpose, true);
}

void SparseMultiplier::matrix_vector (bool transpose, bool add_inplace)
{
  int size_in = transpose ? rows() : cols();
  int size_out = transpose ? cols() : rows();
  float * dev_dense_in = transpose ? m_dev_dense_rows : m_dev_dense_cols;
  float * dev_dense_out = transpose ? m_dev_dense_cols : m_dev_dense_rows;

  cuda_memcpy_h2d(dev_dense_in, m_dense_in, size_in * sizeof(float));
  if (add_inplace) {
    cuda_memcpy_h2d(dev_dense_out, m_dense_out, size_out * sizeof(float));
  }

  // Eigen uses CSC format; cusparse uses CSR format
  cusparseOperation_t transA = transpose ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                         : CUSPARSE_OPERATION_TRANSPOSE ;
  int m = cols();
  int n = rows();

  float alpha = 1.0f;
  const float * x = dev_dense_in;
  float beta = add_inplace ? 1.0f : 0.0f;
  float * y = dev_dense_out;

  ASSERT_CUSPARSE(cusparseScsrmv(
      g_cusparseHandle,
      transA,
      m, n,
      alpha,
      m_sparse_descr,
      m_dev_sparse_csrVal,
      m_dev_sparse_csrRowPtr,
      m_dev_sparse_csrColInd,
      x,
      beta,
      y));

  cuda_memcpy_d2h(m_dense_out, dev_dense_out, size_out * sizeof(float));

  m_dense_in = NULL;
  m_dense_out = NULL;
}

void SparseMultiplier::matrix_matrix (bool transpose, bool add_inplace)
{
  int size_in = transpose ? rows() : cols();
  int size_out = transpose ? cols() : rows();
  float * dev_dense_in = transpose ? m_dev_dense_rows : m_dev_dense_cols;
  float * dev_dense_out = transpose ? m_dev_dense_cols : m_dev_dense_rows;

  // Eigen uses CSC format; cusparse uses CSR format
  cusparseOperation_t transA = transpose ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                         : CUSPARSE_OPERATION_TRANSPOSE ;
  int m = cols();
  int k = rows();

  float alpha = 1.0f;
  const float * B = dev_dense_in;
  int ldb = size_in;
  float beta = add_inplace ? 1.0f : 0.0f;
  float * C = dev_dense_out;
  int ldc = size_out;

  const size_t batch_size = m_batch_size;
  while (m_buffer_size) {

    int n = min(batch_size, m_buffer_size);

    cuda_memcpy_h2d(dev_dense_in, m_dense_in, size_in * n * sizeof(float));
    if (add_inplace) {
      cuda_memcpy_h2d(dev_dense_out, m_dense_out, size_out * n * sizeof(float));
    }

    ASSERT_CUSPARSE(cusparseScsrmm(
        g_cusparseHandle,
        transA,
        m, n, k,
        alpha,
        m_sparse_descr,
        m_dev_sparse_csrVal,
        m_dev_sparse_csrRowPtr,
        m_dev_sparse_csrColInd,
        B, ldb,
        beta,
        C, ldc));

    cuda_memcpy_d2h(m_dense_out, dev_dense_out, size_out * n * sizeof(float));

    m_buffer_size -= n;
    m_dense_in += size_in * n;
    m_dense_out += size_out * n;
  }

  m_dense_in = NULL;
  m_dense_out = NULL;
}

} // namespace Gpu

