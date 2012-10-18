
#include "linalg.h"

namespace Image
{

void transpose (
    const size_t width,
    const size_t height,
    const float * restrict source,
    float * restrict destin);

} // namespace Image

namespace LinAlg
{

string print_vector (const Vector<float> & x)
{
  std::ostringstream o;
  o << " array([";
  for (size_t i = 0; i < x.size; ++i) {
    o << x[i] << ", ";
  }
  o << "])";
  return o.str();
}

string print_matrix (int m, int n, const Vector<float> & A)
{
  ASSERT_SIZE(A, m * n);

  std::ostringstream o;
  o << " array([";
  for (int i = 0; i < m; ++i) {
    o << "\n  [";
    for (int j = 0; j < n; ++j) {
      o << A[n * i + j] << ", ";
    }
    o << "],";
  }
  o << "\n  ])";
  return o.str();
}

//----( fortran declarations )------------------------------------------------

typedef int & LapackChar;
typedef int & LapackInt;
typedef int * LapackInts;
typedef float & LapackFloat;
typedef float * LapackVectorFloat;

//----( blas )----

extern "C" void sgemm_ (
    LapackChar transa,
    LapackChar transb,
    LapackInt m,
    LapackInt n,
    LapackInt k,
    LapackFloat alpha,
    LapackVectorFloat a,
    LapackInt lda,
    LapackVectorFloat b,
    LapackInt ldb,
    LapackFloat beta,
    LapackVectorFloat c,
    LapackInt ldc);

extern "C" void ssyrk_ (
    LapackChar uplo,
    LapackChar trans,
    LapackInt n,
    LapackInt k,
    LapackFloat alpha,
    LapackVectorFloat a,
    LapackInt lda,
    LapackFloat beta,
    LapackVectorFloat c,
    LapackInt ldc);

//----( lapack )----

extern "C" void spotrf_ (
    LapackChar uplo,
    LapackInt n,
    LapackVectorFloat a,
    LapackInt lda,
    LapackInt info);

extern "C" void spotri_ (
    LapackChar uplo,
    LapackInt n,
    LapackVectorFloat a,
    LapackInt lda,
    LapackInt info);

extern "C" void spotrs_ (
    LapackChar uplo,
    LapackInt n,
    LapackInt nrhs,
    LapackVectorFloat a,
    LapackInt lda,
    LapackVectorFloat b,
    LapackInt ldb,
    LapackInt info);

extern "C" void sgels_ (
    LapackChar trans,
    LapackInt m,
    LapackInt n,
    LapackInt nrhs,
    LapackVectorFloat a,
    LapackInt lda,
    LapackVectorFloat b,
    LapackInt ldb,
    LapackVectorFloat work,
    LapackInt lwork,
    LapackInt info);

//----( fortran wrappers )----------------------------------------------------

//----( blas )----

void transpose (
    int m,
    int n,
    const Vector<float> & A,
    Vector<float> & At)
{
  ASSERT_SIZE(A, m * n);
  ASSERT_SIZE(At, n * m);
  ASSERT_NE(A.data, At.data);

  Image::transpose(m, n, A, At);

  //for (int i = 0; i < m; ++i) {
  //  for (int j = 0; j < n; ++j) {
  //    At[n * i + j] = A[m * i + j];
  //  }
  //}
}

void matrix_multiply (
    int I,
    int J,
    int K,
    const Vector<float> & A,  // rect(I,J)
    bool trans_A,
    const Vector<float> & B,  // rect(J,K)
    bool trans_B,
    Vector<float> & AB,       // rect(I,K)
    float alpha,
    float beta)
{
  ASSERT_SIZE(A, I * J);
  ASSERT_SIZE(B, J * K);
  ASSERT_SIZE(AB, I * K);

  // swap a <-> b to account for C/FORTRAN transpose difference

  int transa = trans_B ? 'T' : 'N';
  int transb = trans_A ? 'T' : 'N';

  int m = K;
  int k = J;
  int n = I;

  int lda = trans_B ? J : K;
  int ldb = trans_A ? I : J;
  int ldc = K;

  float * a = const_cast<float *>(B.data);
  float * b = const_cast<float *>(A.data);
  float * c = AB.data;

  sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void outer_prod (
    int n,
    int k,
    const Vector<float> & A,  // rect(n,k)
    bool trans_A,
    Vector<float> & AAt)      // sym(n,n)
{
  ASSERT_SIZE(A, n * k);
  ASSERT_SIZE(AAt, n * n);

  int uplo = 'U'; // lower-diag in C indexing
  int trans = trans_A ? 'N' : 'T';

  int lda = trans_A ? n : k;
  int ldc = n;

  float alpha = 1;
  float beta = 0;

  float * a = const_cast<float *>(A.data);
  float * c = AAt.data;

  ssyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);

  // copy upper to lower idagonal
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      AAt[n * j + i] = AAt[n * i + j];
    }
  }
}

float symmetric_norm (
    const Vector<float> & x,  // vect(n)
    const Vector<float> & A)  // sym(n)
{
  size_t n = x.size;
  ASSERT_SIZE(A, sqr(n));

  float result = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j) {
      result += 2 * A[n * i + j] * x[i] * x[j];
    }
    result += A[n * i + i] * sqr(x[i]);
  }
  return result;
}

//----( lapack )----

void cholesky (
    int n,
    Vector<float> & A, //sym(A,trans)
    bool trans)
{
  ASSERT_SIZE(A, n * n);

  int uplo = trans ? 'L' : 'U'; // upper : lower -diag in C indexing
  int lda = n;
  float * a = A.data;

  int info;
  spotrf_(uplo, n, a, lda, info);
  ASSERT(info == 0, "spotrf_ error, info = " << info);

  // zero out off-triangle
  if (trans) {
    for (int i = 0, I = n; i < I; ++i) {
      for (int j = 0; j < i; ++j) {
          A[I * i + j] = 0;
      }
    }
  } else {
    for (int i = 0, I = n; i < I; ++i) {
      for (int j = i + 1; j < I; ++j) {
          A[I * i + j] = 0;
      }
    }
  }
}

void symmetric_invert (
    int n,
    Vector<float> & A)  // sym(n)
{
  ASSERT_SIZE(A, n * n);

  int uplo = 'U'; // lower-diag in C indexing
  int lda = n;

  float * a = A.data;

  int info;
  spotrf_(uplo, n, a, lda, info);
  ASSERT(info == 0, "spotrf_ error, info = " << info);

  spotri_ (uplo, n, a, lda, info);
  ASSERT(info == 0, "spotri_ error, info = " << info);

  // copy upper to lower idagonal
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      A[n * j + i] = A[n * i + j];
    }
  }
}

void symmetric_solve (
    int m,
    int n,
    const Vector<float> & A,  // sym(m)
    const Vector<float> & BA, // rect(n,m)
    bool trans_B,
    Vector<float> & B)        // rect(n,m)
{
  ASSERT_SIZE(A, m * m);
  ASSERT_SIZE(BA, n * m);
  ASSERT_SIZE(B, n * m);

  int uplo = 'U'; // lower-diag in C indexing
  int nrhs = n;
  int lda = m;
  int ldb = m;

  Vector<float> cholA(m * m);
  cholA = A;
  cholesky(m, cholA);
  float * a = cholA.data;

  if (trans_B) transpose(m,n, BA, B); else B = BA;
  float * b = B.data;

  int info;
  spotrs_(uplo, m, nrhs, a, lda, b, ldb, info);
  ASSERT(info == 0, "spotrs_ error, info = " << info);

  if (trans_B) {
    Vector<float> Bt(m * n);
    Bt = B;
    transpose(n,m, BA, B);
  }
}

void least_squares (
    const Vector<float> & A,
    bool trans_A,
    const Vector<float> & Ax,
    Vector<float> & x)
{
  int m = x.size;
  int n = Ax.size;
  ASSERT_SIZE(A, n * m);

  int trans = trans_A ? 'N' : 'T';
  int nrhs = 1;
  int lda = trans_A ? n : m;
  int ldb = n;

  Vector<float> Acopy(n * m);
  Acopy = A;
  float * a = Acopy.data;

  Vector<float> b(n);
  b = Ax;

  int lwork = -1;
  float query[1];

  int info;
  sgels_(trans, m, n, nrhs, a, lda, b, ldb, query, lwork, info);
  ASSERT(info == 0, "sgels_ query error, info = " << info);

  lwork = static_cast<int>(query[0]);
  Vector<float> work(lwork);

  sgels_(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
  ASSERT(info == 0, "sgels_ error, info = " << info);

  copy_float(b,x,m);
}

//----( rotation )------------------------------------------------------------

void set_identity (size_t size, float * restrict matrix)
{
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[size * i + j] = i == j;
    }
  }
}

void orthonormalize (size_t size, size_t count, float * restrict basis)
{
  for (size_t m = 0; m < count; ++m) {
    float * restrict x = basis + size * m;

    for (size_t n = 0; n < m; ++n) {
      float * restrict y = basis + size * n;

      float sum_xy = 0;
      for (size_t i = 0; i < size; ++i) {
        sum_xy += x[i] * y[i];
      }

      float shift = -sum_xy;
      for (size_t i = 0; i < size; ++i) {
        x[i] += shift * y[i];
      }
    }

    float sum_x2 = 0;
    for (size_t i = 0; i < size; ++i) {
      sum_x2 += sqr(x[i]);
    }

    float scale = 1.0f / sqrtf(sum_x2);
    for (size_t i = 0; i < size; ++i) {
      x[i] *= scale;
    }
  }
}

void Orientation3D::drag (float dx, float dy)
{
  Vector<float> x = coord(0);
  Vector<float> y = coord(1);
  Vector<float> z = coord(2);

  multiply_add(dx, z, x);
  multiply_add(dy, z, y);
  orthonormalize();
}

void Orientation4D::drag1 (float dx, float dy)
{
  Vector<float> x = coord(0);
  Vector<float> y = coord(1);
  Vector<float> z = coord(2);

  multiply_add(dx, z, x);
  multiply_add(dy, z, y);
  orthonormalize();
}

void Orientation4D::drag2 (float dx, float dy)
{
  Vector<float> x = coord(0);
  Vector<float> y = coord(1);
  Vector<float> z = coord(3);

  multiply_add(dx, z, x);
  multiply_add(dy, z, y);
  orthonormalize();
}

} // namespace LinAlg

