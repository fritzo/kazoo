
#include "sparse.h"

#define USE_EIGEN
//#define USE_SUITESPARSE

#ifdef USE_EIGEN
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
//#define EIGEN_CHOLMOD_SUPPORT
//#define EIGEN_UMFPACK_SUPPORT
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/SparseExtra>
using namespace Eigen;
#endif  // USE_EIGEN

#ifdef USE_SUITESPARSE
#include <suitesparse/SuiteSparseQR.hpp>
#endif // USE_SUITESPARSE

namespace LinAlg
{

//----( sparse solvers )------------------------------------------------------

#ifdef USE_EIGEN

// conversions between k nearest neighbor & Eigen sparse matrix formats

void knn_to_eigen (
    const size_t dim,
    const size_t degree,
    const Vector<uint16_t> & nbhd,
    const Vector<float> & knn_A,
    SparseMatrix<double> & eigen_A)
{
  DynamicSparseMatrix<double> A(dim,dim);

  for (size_t i = 0; i < dim; ++i) {
    for (size_t n = 0; n < degree; ++n) {
      size_t knn_pos = degree * i + n;
      size_t j = nbhd[knn_pos];

      A.coeffRef(i,j) = knn_A[knn_pos];
    }
  }

  eigen_A = A;
}

void symmetric_knn_to_eigen (
    const size_t dim,
    const size_t degree,
    const Vector<uint16_t> & nbhd,
    const Vector<float> & knn_A,
    SparseMatrix<double> & eigen_A)
{
  DynamicSparseMatrix<double> A(dim,dim);

  for (size_t i = 0; i < dim; ++i) {
    for (size_t n = 0; n < degree; ++n) {
      size_t knn_pos = degree * i + n;
      size_t j = nbhd[knn_pos];

      A.coeffRef(i,j) = A.coeffRef(j,i) = knn_A[knn_pos];
    }
  }

  eigen_A = A;
}

void eigen_to_knn (
    const size_t dim,
    const size_t degree,
    const Vector<uint16_t> & nbhd,
    const SparseMatrix<double> & eigen_A,
    Vector<float> & knn_A)
{
  for (size_t i = 0; i < dim; ++i) {
    for (size_t n = 0; n < degree; ++n) {
      size_t knn_pos = degree * i + n;
      size_t j = nbhd[knn_pos];

      knn_A[knn_pos] = eigen_A.coeff(i,j);
    }
  }
}

void sparse_symmetric_solve (
    const size_t dim,
    const size_t degree,
    const Vector<uint16_t> & nbhd,
    const Vector<float> & knn_Pxx,
    const Vector<float> & knn_Pxy,
    Vector<float> & knn_Fxy,
    double tol,
    bool debug)
{
  ASSERT_SIZE(nbhd, dim * degree);
  ASSERT_SIZE(knn_Pxx, dim * degree);
  ASSERT_SIZE(knn_Pxy, dim * degree);
  ASSERT_SIZE(knn_Fxy, dim * degree);

  LOG("solving sparse " << dim << " x " << dim << " matrix problem");
  SparseMatrix<double> Pxx;
  symmetric_knn_to_eigen(dim, degree, nbhd, knn_Pxx, Pxx);

  //typedef SparseLLT<SparseMatrix<double>, Cholmod> Solver;
  typedef SparseLLT<SparseMatrix<double> > Solver;
  Solver solver(Pxx);

  VectorXd Fxy_i(dim);
  VectorXd Pxy_i(dim);
  float residual = 0;

  for (size_t i = 0; i < dim; ++i) {

    Pxy_i.setZero();
    for (size_t n = 0; n < degree; ++n) {
      size_t knn_pos = degree * i + n;
      Pxy_i(nbhd[knn_pos]) = knn_Pxy[knn_pos];
    }

    Fxy_i = Pxy_i;
    solver.solveInPlace(Fxy_i);
    // eigen does not set the succeeded flag, otherwise we could check this
    //ASSERT(solver.succeeded(), "sparse linear solver failed");

    for (size_t n = 0; n < degree; ++n) {
      size_t knn_pos = degree * i + n;
      knn_Fxy[knn_pos] = Fxy_i(nbhd[knn_pos]);
    }

    if (debug) {
      Pxy_i -= Pxx * Fxy_i;
      residual += Pxy_i.norm();
    }
  }

  if (debug) {

    LOG(" 2-norm of dense residual = " << residual);

    SparseMatrix<double> Pxy, Fxy;
    knn_to_eigen(dim, degree, nbhd, knn_Pxy, Pxy);
    knn_to_eigen(dim, degree, nbhd, knn_Fxy, Fxy);

    Pxy -= Pxx * Fxy;
    LOG(" 2-norm of sparse residual = " << Pxy.norm());
  }
}

#endif // USE_EIGEN

//----------------------------------------------------------------------------

#ifdef USE_SUITESPARSE

// This uses the GLP's SuiteSparseQR function
// http://www.cise.ufl.edu/research/sparse/SPQR/SPQR/Doc/spqr_user_guide.pdf

// conversions between k nearest neighbor & cholmod formats for sparse matrices

static cholmod_sparse * knn_to_cholmod (
    const size_t dim,
    const size_t degree,
    const Vector<uint16_t> & nbhd,
    const Vector<float> & A,
    cholmod_common * cc)
{
  TODO("implement conversion");
}

static cholmod_sparse * symmetric_knn_to_cholmod (
    const size_t dim,
    const size_t degree,
    const Vector<uint16_t> & nbhd,
    const Vector<float> & A,
    cholmod_common * cc)
{
  TODO("implement conversion");
}

static void cholmod_to_knn (
    const size_t dim,
    const size_t degree,
    const Vector<uint16_t> & nbhd,
    const cholmod_sparse * a,
    Vector<float> & A,
    cholmod_common * cc)
{
  TODO("implement conversion");
}

void sparse_symmetric_solve (
    const size_t dim,
    const size_t degree,
    const Vector<uint16_t> & nbhd,
    const Vector<float> & Pxx,
    const Vector<float> & Pxy,
    Vector<float> & Fxy,
    double tol,
    bool debug)
{
  ASSERT_SIZE(Pxx, dim * degree);
  ASSERT_SIZE(Pxy, dim * degree);
  ASSERT_SIZE(Fxy, dim * degree);

  TODO("implement sparse symmetric linear solver");

  LOG("solving sparse " << dim << " x " << dim << " matrix problem");

  // start CHOLMOD
  cholmod_common Common, *cc = &Common;
  cholmod_l_start(cc);

  cholmod_sparse * A = symmetric_knn_to_cholmod(dim, degree, nbhd, Pxx, cc);
  cholmod_sparse * B = knn_to_cholmod(dim, degree, nbhd, Pxy, cc);

  // matlab equivalent: X = A\B
  int ordering = CHOLMOD_NATURAL;
  cholmod_sparse * X = SuiteSparseQR<double>(ordering, tol, A, B, cc);

  cholmod_to_knn(dim, degree, nbhd, X, Fxy, cc);

  if (debug) {
    cholmod_sparse * B2 = cholmod_l_ssmult(A, B, 0, true, true, cc);

    Vector<float> Pxy2(Pxy.size);
    cholmod_to_knn(dim, degree, nbhd, B2, Pxy2, cc);

    cholmod_l_free_sparse(&B2, cc);

    LOG(" 2-norm of residual = " << dist_squared(Pxy, Pxy2));
  }

  // free everything and finish CHOLMOD
  cholmod_l_free_sparse(&A, cc);
  cholmod_l_free_sparse(&B, cc);
  cholmod_l_free_sparse(&X, cc);
  cholmod_l_finish(cc);
}

#endif // USE_SUITESPARSE

} // namespace LinAlg

