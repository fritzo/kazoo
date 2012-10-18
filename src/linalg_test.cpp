
#include "linalg.h"

#define TOL (1e-6)
#define DEFAULT_VECTOR_SIZE (2)

#define ASSERT_CLOSE(x,y) \
  ASSERT(fabs((x)-(y))<TOL, \
      "expected " #x " close to " #y "; actual difference: " << ((x)-(y)))

using namespace LinAlg;

void randomize (Vector<float> & x)
{
  for (size_t i = 0; i < x.size; ++i) {
    x[i] = random_std();
  }
}

//----( unit tests )----------------------------------------------------------

void test_matrix_multiply (
    int I = DEFAULT_VECTOR_SIZE,
    int J = DEFAULT_VECTOR_SIZE + 1,
    int K = DEFAULT_VECTOR_SIZE + 2)
{
  LOG("\ntesting matrix multiply");

  Vector<float> A(I * J);
  Vector<float> B(J * K);
  Vector<float> AB(I * K);

  randomize(A);
  PRINT_MAT(I, J, A);

  randomize(B);
  PRINT_MAT(J, K, B);

  matrix_multiply(I,J,K, A,false, B,false, AB);
  PRINT_MAT(I, K, AB);

  for (int i = 0; i < I; ++i) {
    for (int k = 0; k < K; ++k) {
      float truth = 0;
      for (int j = 0; j < J; ++j) {
        truth += A[J * i + j] * B[K * j + k];
      }
      ASSERT_CLOSE(AB[K * i + k], truth);
    }
  }

  LOG("passed");
}

void test_outer_prod (
    int m = DEFAULT_VECTOR_SIZE,
    int n = DEFAULT_VECTOR_SIZE + 1)
{
  LOG("\ntesting outer product");

  Vector<float> A(m * n);
  Vector<float> AAt(m * m);

  randomize(A);
  PRINT_MAT(m, n, A);

  outer_prod (m,n, A,false, AAt);
  PRINT_MAT(m, m, AAt);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      float truth = 0;
      for (int t = 0; t < n; ++t) {
        truth += A[n * i + t] * A[n * j + t];
      }
      ASSERT_CLOSE(AAt[m * i + j], truth);
    }
  }

  LOG("passed");
}

void test_cholesky (
    bool trans,
    int m = DEFAULT_VECTOR_SIZE)
{
  LOG("\ntesting cholesky decomposition");

  Vector<float> A(m * m);
  Vector<float> cholA(m * m);
  Vector<float> cAcAt(m * m);

  randomize(A);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < i; ++j) {
      A[m * j + i] = A[m * i + j];
    }
    A[m * i + i] += 2 * m; // diagonally dominant
  }
  PRINT_MAT(m, m, A);

  cholA = A;
  cholesky(m, cholA, trans);
  PRINT_MAT(m, m, cholA);

  outer_prod(m, m, cholA,trans, cAcAt);
  PRINT_MAT(m, m, cAcAt);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      ASSERT_CLOSE(cAcAt[m * i + j], A[m * i + j]);
    }
  }

  LOG("passed");
}

void test_symmetric_invert (
    int n = DEFAULT_VECTOR_SIZE)
{
  LOG("\ntesting symmetric inverse");

  Vector<float> A(n * n);
  Vector<float> Ai(n * n);
  Vector<float> AAi(n * n);
  Vector<float> AiA(n * n);

  randomize(A);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      A[n * j + i] = A[n * i + j];
      A[n * i + i] += 2 * n; // diagonally dominant
    }
  }
  PRINT_MAT(n, n, A);

  Ai = A;
  symmetric_invert(n, Ai);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      A[n * j + i] = A[n * i + j];
      Ai[n * j + i] = Ai[n * i + j];
    }
  }
  PRINT_MAT(n, n, Ai);

  matrix_multiply(n,n,n, A,false, Ai,false, AAi);
  PRINT_MAT(n, n, AAi);

  matrix_multiply(n,n,n, Ai,false, A,false, AiA);
  PRINT_MAT(n, n, AiA);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float delta_ij = i == j ? 1 : 0;
      ASSERT_CLOSE(AAi[n * i + j], delta_ij);
      ASSERT_CLOSE(AiA[n * i + j], delta_ij);
    }
  }

  LOG("passed");
}

void test_symmetric_solve (
    int m = DEFAULT_VECTOR_SIZE,
    int n = DEFAULT_VECTOR_SIZE + 1)
{
  LOG("\ntesting symmetric solve");

  Vector<float> A(m * m);
  Vector<float> BA(n * m);
  Vector<float> B(n * m);
  Vector<float> BA2(n * m);

  randomize(A);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < i; ++j) {
      A[m * j + i] = A[m * i + j];
    }
    A[m * i + i] += 2 * m; // diagonally dominant
  }
  PRINT_MAT(m, m, A);

  randomize(BA);
  PRINT_MAT(n, m, BA);

  symmetric_solve(m,n, A, BA,false, B);
  PRINT_MAT(n, m, B);

  matrix_multiply(n,m,m, B,false, A,false, BA2);
  PRINT_MAT(n, m, BA2);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      ASSERT_CLOSE(BA[m * i + j], BA2[m * i + j]);
    }
  }

  LOG("passed");
}

void test_least_squares (
    int m = DEFAULT_VECTOR_SIZE * 2 + 1,
    int n = DEFAULT_VECTOR_SIZE)
{
  LOG("\ntesting least squares (idempotence)");

  Vector<float> A(m * n);
  Vector<float> Ax(m);
  Vector<float> x(n);
  Vector<float> Ax2(m);
  Vector<float> x2(n);

  randomize(A);
  PRINT_MAT(m, n, A);

  randomize(Ax);
  PRINT_MAT(1, m, Ax);

  least_squares(A,false, Ax, x);
  PRINT_MAT(1, n, x);

  matrix_multiply(m,n,1, A,false, x,false, Ax2);
  PRINT_MAT(1, m, Ax2);

  least_squares(A,false, Ax2, x2);
  PRINT_MAT(1, n, x2);

  for (int i = 0; i < n; ++i) {
    ASSERT_CLOSE(x[i], x2[i]);
  }

  LOG("passed");
}

//----( test harness )--------------------------------------------------------

int main ()
{
  LOG("Testing linear algebra functions");

  test_matrix_multiply();
  test_outer_prod();
  test_cholesky(true);
  test_cholesky(false);
  test_symmetric_invert();
  test_symmetric_solve();
  test_least_squares();

  LOG("\nAll tests passed!");
}

