
#include "eigen_gpu.h"
#include "args.h"

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Eigen>
#include <Eigen/Sparse>

//----( randomization )-------------------------------------------------------

inline float random_real () { return tanf(3 * (random_std() - 0.5f)); }

void randomize (Vector<float> & x)
{
  for (size_t i = 0; i < x.size; ++i) {
    x[i] = random_real();
  }
}

void randomize (VectorXf & x)
{
  for (int i = 0; i < x.size(); ++i) {
    x[i] = random_real();
  }
}

void randomize (MatrixXf & x)
{
  for (int j = 0; j < x.cols(); ++j) {
  for (int i = 0; i < x.rows(); ++i) {
    x(i,j) = random_real();
  }}
}

void randomize (MatrixSf & mat, float density)
{
  ASSERT_LT(0, density);
  ASSERT_LT(density, 1);

  Eigen::DynamicSparseMatrix<float> temp(mat.rows(), mat.cols());

  const size_t X = mat.rows();
  const size_t Y = mat.cols();

  size_t num_entries = ceil(density * mat.rows() * mat.cols());
  for (size_t e = 0; e < num_entries; ++e) {

    int x,y;
    do {
      int xy = random_choice(X*Y);
      x = xy % X;
      y = xy / X;
    } while (temp.coeff(x,y) > 0);

    temp.coeffRef(x,y) = random_real();
  }

  mat = temp;
}

//----( tests )---------------------------------------------------------------

void test_dense_matrix_matrix (
    int rows,
    int cols,
    int width,
    size_t iters,
    bool trans_A,
    bool trans_B)
{
  LOG("\nTesting dense matrix-matrix product"
      << " (op(A) = " << (trans_A ? "A" : "A'")
      << ", op(B) = " << (trans_B ? "B" : "B'")
      << ")");

  MatrixXf lhs(rows, cols);   randomize(lhs);
  MatrixXf rhs(cols, width);  randomize(rhs);

  if (trans_A) lhs.transposeInPlace();
  if (trans_B) rhs.transposeInPlace();

  //DEBUG("")
  //PRINT4(rows, cols, lhs.rows(), lhs.cols());

  MatrixXf cpu_prod(rows, width);
  MatrixXf gpu_prod(rows, width);

  float speedup = 1;
  {
    Timer timer;

    for (size_t i = 0; i < iters; ++i) {
      if (trans_A) {
        if (trans_B) cpu_prod = lhs.transpose() * rhs.transpose();
        else         cpu_prod = lhs.transpose() * rhs;
      } else {
        if (trans_B) cpu_prod = lhs * rhs.transpose();
        else         cpu_prod = lhs * rhs;
      }
    }

    float rate = iters / timer.elapsed();
    float gflops = rate * rows * cols * width * 1e-9f;
    LOG("Cpu performed " << rate << " mul/sec = " << gflops << " gflops");
    speedup /= rate;
  }

  {
    Timer timer;

    for (size_t i = 0; i < iters; ++i) {
      Gpu::matrix_multiply(lhs, trans_A, rhs, trans_B, gpu_prod);
    }

    float rate = iters / timer.elapsed();
    float gflops = rate * rows * cols * width * 1e-9f;
    LOG("Gpu performed " << rate << " mul/sec = " << gflops << " gflops");
    speedup *= rate;
    PRINT(speedup);
  }

  //PRINT(as_vector(lhs));
  //PRINT(as_vector(rhs));
  //PRINT(as_vector(cpu_prod));
  //PRINT(as_vector(gpu_prod));

  float rms_lhs = lhs.norm() / sqrtf(lhs.size());
  float rms_rhs = rhs.norm() / sqrtf(rhs.size());
  float rms_error = (cpu_prod - gpu_prod).norm() / sqrtf(cpu_prod.size());
  PRINT(rms_error);
  ASSERTW_LT(rms_error, 1e-6f * rms_lhs * rms_rhs);
}

void test_sparse_matrix_vector (
    int rows,
    int cols,
    float density,
    size_t iters,
    bool trans)
{
  LOG("\nTesting sparse matrix-vector product"
      << (trans ? " (transposed)" : ""));

  MatrixSf sparse(rows, cols);
  randomize(sparse, density);

  Gpu::SparseMultiplier multiply(sparse, 1);

  VectorXf rhs(trans ? rows : cols);
  randomize(rhs);

  VectorXf cpu_prod(trans ? cols : rows);
  VectorXf gpu_prod(trans ? cols : rows);

  float speedup = 1;
  {
    Timer timer;

    for (size_t i = 0; i < iters; ++i) {
      if (trans) cpu_prod = sparse.transpose() * rhs;
      else cpu_prod = sparse * rhs;
    }

    float rate = iters / timer.elapsed();
    float gflops = rate * rows * cols * density * 1e-9f;
    LOG("Cpu performed " << rate << " mul/sec = " << gflops << " gflops");
    speedup /= rate;
  }

  {
    Timer timer;

    for (size_t i = 0; i < iters; ++i) {
      multiply.left_mul(rhs, gpu_prod, trans);
    }

    float rate = iters / timer.elapsed();
    float gflops = rate * rows * cols * density * 1e-9f;
    LOG("Gpu performed " << rate << " mul/sec = " << gflops << " gflops");
    speedup *= rate;
    PRINT(speedup);
  }

  float rms_sparse = sparse.norm() / sqrtf(sparse.size());
  float rms_rhs = rhs.norm() / sqrtf(rhs.size());
  float rms_error = (cpu_prod - gpu_prod).norm() / sqrtf(cpu_prod.size());
  PRINT(rms_error);
  ASSERTW_LT(rms_error, 1e-6f * rms_sparse * rms_rhs);
}

void test_sparse_matrix_matrix (
    int rows,
    int cols,
    int width,
    float density,
    size_t iters,
    bool trans)
{
  LOG("\nTesting sparse matrix-matrix product"
      << (trans ? " (transposed)" : ""));

  MatrixSf sparse(rows, cols);
  randomize(sparse, density);

  Gpu::SparseMultiplier multiply(sparse, 64);

  MatrixXf rhs(trans ? rows : cols, width);
  randomize(rhs);

  MatrixXf cpu_prod(trans ? cols : rows, width);
  MatrixXf gpu_prod(trans ? cols : rows, width);

  float speedup = 1;
  {
    Timer timer;

    for (size_t i = 0; i < iters; ++i) {
      if (trans) cpu_prod = sparse.transpose() * rhs;
      else cpu_prod = sparse * rhs;
    }

    float rate = iters / timer.elapsed();
    float gflops = rate * rows * cols * width * density * 1e-9f;
    LOG("Cpu performed " << rate << " mul/sec = " << gflops << " gflops");
    speedup /= rate;
  }

  {
    Timer timer;

    for (size_t i = 0; i < iters; ++i) {
      multiply.left_mul(rhs, gpu_prod, trans);
    }

    float rate = iters / timer.elapsed();
    float gflops = rate * rows * cols * width * density * 1e-9f;
    LOG("Gpu performed " << rate << " mul/sec = " << gflops << " gflops");
    speedup *= rate;
    PRINT(speedup);
  }

  float rms_sparse = sparse.norm() / sqrtf(sparse.size());
  float rms_rhs = rhs.norm() / sqrtf(rhs.size());
  float rms_error = (cpu_prod - gpu_prod).norm() / sqrtf(cpu_prod.size());
  PRINT(rms_error);
  ASSERTW_LT(rms_error, 1e-6f * rms_sparse * rms_rhs);
}

//----( main )----------------------------------------------------------------

const char * help_message =
"Usage: eigen_gpu_test [ROWS] [COLS] [WIDTH] [DENSITY] [ITERS]"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);
  LOG(help_message);

  int rows = args.pop(2048);
  int cols = args.pop(4096);
  int width = args.pop(1024);
  float density = args.pop(0.01f);
  size_t iters = args.pop(1);
  PRINT5(rows, cols, width, density, iters);
  // Test results below were generated with
  // rows = 2048, cols = 4096, density = 0.01, iters = 1

  test_sparse_matrix_vector(rows, cols, density, iters, false);
  // Cpu performed 4830.92 mul/sec = 0.405247 gflops
  // Gpu performed 242.131 mul/sec = 0.0203114 gflops
  // rms_error = 0.0110587

  test_sparse_matrix_vector(rows, cols, density, iters, true);
  // Cpu performed 5780.35 mul/sec = 0.484891 gflops
  // Gpu performed 14705.9 mul/sec = 1.23362 gflops
  // rms_error = 0.00113682

  test_sparse_matrix_matrix(rows, cols, width, density, iters, false);
  // Cpu performed 0.700048 mul/sec = 0.0601337 gflops
  // Gpu performed 1.75981 mul/sec = 0.151167 gflops
  // rms_error = 0.0506754

  test_sparse_matrix_matrix(rows, cols, width, density, iters, true);
  // Cpu performed 0.643097 mul/sec = 0.0552416 gflops
  // Gpu performed 34.4816 mul/sec = 2.96194 gflops
  // rms_error = 0.00686571

  test_dense_matrix_matrix(rows, cols, width, iters, false, false);
  // Cpu performed 3.5864 mul/sec = 30.807 gflops
  // Gpu performed 27.9431 mul/sec = 240.029 gflops
  // rms_error = 21.362

  // TODO get these working
  //test_dense_matrix_matrix(rows, cols, width, iters, true, false);
  //test_dense_matrix_matrix(rows, cols, width, iters, false, true);
  //test_dense_matrix_matrix(rows, cols, width, iters, true, true);

  LOG("");
  return 0;
}

