#ifndef KAZOO_EIGEN_H
#define KAZOO_EIGEN_H

// This file forward-declares some types in Eigen2,
// as suggested in this eigen3 bug report:
//   http://eigen.tuxfamily.org/bz/show_bug.cgi?id=269
//
// Some of this information can be found in:
//   Eigen/src/Core/util/ForwardDeclarations.h
//   Eigen/src/Core/util/Constants.h
//   Eigen/src/Core/Matrix.h
//   Eigen/src/Sparse/SparseUtil.h
//
// Forward declarations are needed for compilation with NVCC,
// since nvcc is not smart enough to handle eigen's template magic.

#include "vectors.h"

#define KAZOO_EIGEN_VERSION 3
#if KAZOO_EIGEN_VERSION == 3

//----( eigen 3 )-------------------------------------------------------------

namespace Eigen
{

//----( dense matrices )----

#define ColMajor 0
#define AutoAlign 0
#define Dynamic -1

template<
    class _Scalar,
    int _Rows,
    int _Cols,
    int _Options, // = ColMajor | AutoAlign
    int _MaxRows, // = _Rows
    int _MaxCols> // = _Cols
class Matrix;

#define Options (ColMajor | AutoAlign)
typedef Matrix<float, Dynamic, Dynamic, Options, Dynamic, Dynamic> MatrixXf;
typedef Matrix<double, Dynamic, Dynamic, Options, Dynamic, Dynamic> MatrixXd;
typedef Matrix<float, Dynamic, 1, Options, Dynamic, 1> VectorXf;
typedef Matrix<double, Dynamic, 1, Options, Dynamic, 1> VectorXd;
#undef Options

#undef ColMajor
#undef AutoAlign
#undef Dynamic

//----( sparse matrices )----

#define DefaultSparseFlags 0

template<class _Scalar, int _Options, class _Index> class SparseMatrix;
typedef SparseMatrix<float, DefaultSparseFlags, int> MatrixSf;
typedef SparseMatrix<double, DefaultSparseFlags, int> MatrixSd;

template<class _Scalar, int _Options, class _Index> class SparseVector;
typedef SparseVector<float, DefaultSparseFlags, int> VectorSf;
typedef SparseVector<double, DefaultSparseFlags, int> VectorSd;

#undef DefaultSparseFlags

} // namespace Eigen

#elif KAZOO_EIGEN_VERSION == 2

//----( eigen 2 )-------------------------------------------------------------

namespace Eigen
{

//----( dense matrices )----

#define ColMajor 0
#define AutoAlign 0x2
#define Dynamic 10000

template<
    class _Scalar,
    int _Rows,
    int _Cols,
    int _Options, // = ColMajor | AutoAlign
    int _MaxRows, // = _Rows
    int _MaxCols> // = _Cols
class Matrix;

typedef Matrix<float, Dynamic, Dynamic, ColMajor | AutoAlign, Dynamic, Dynamic>
MatrixXf;
typedef Matrix<double, Dynamic, Dynamic, ColMajor | AutoAlign, Dynamic, Dynamic>
MatrixXd;

typedef Matrix<float, Dynamic, 1, ColMajor | AutoAlign, Dynamic, 1>
VectorXf;
typedef Matrix<double, Dynamic, 1, ColMajor | AutoAlign, Dynamic, 1>
VectorXd;

#undef ColMajor
#undef AutoAlign
#undef Dynamic

//----( sparse matrices )----

#define DefaultSparseFlags 0

template<class _Scalar, int _Flags> class SparseMatrix;
typedef SparseMatrix<float, DefaultSparseFlags> MatrixSf;
typedef SparseMatrix<double, DefaultSparseFlags> MatrixSd;


#undef DefaultSparseFlags
} // namespace Eigen

#endif // KAZOO_EIGEN_VERSION

//----( utilities )-----------------------------------------------------------

using Eigen::VectorXf;
using Eigen::VectorXd;
using Eigen::VectorSf;
using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::MatrixSf;
using Eigen::MatrixSd;

template<class Scalar_>
struct Eigen_;

template<>
struct Eigen_<float>
{
  typedef float Scalar;
  typedef VectorXf VectorX;
  typedef MatrixXf MatrixX;
  typedef MatrixSf MatrixS;
};

template<>
struct Eigen_<double>
{
  typedef double Scalar;
  typedef VectorXd VectorX;
  typedef MatrixXd MatrixX;
  typedef MatrixSd MatrixS;
};

Vector<float> as_vector (const VectorXf & x);
Vector<float> as_vector (const MatrixXf & x);
Vector<double> as_vector (const VectorXd & x);
Vector<double> as_vector (const MatrixXd & x);

void write_to_python (const VectorXf & x, ostream & os);
void save_to_python (const MatrixXf & A, string filename);

#endif // KAZOO_EIGEN_H
