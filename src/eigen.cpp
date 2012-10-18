
#include "eigen.h"
#include <fstream>

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Eigen>
#include <Eigen/Sparse>

Vector<float> as_vector (const VectorXf & x)
{
  return Vector<float>(x.size(), const_cast<float *>(x.data()));
}

Vector<float> as_vector (const MatrixXf & x)
{
  return Vector<float>(x.size(), const_cast<float *>(x.data()));
}

Vector<double> as_vector (const VectorXd & x)
{
  return Vector<double>(x.size(), const_cast<double *>(x.data()));
}

Vector<double> as_vector (const MatrixXd & x)
{
  return Vector<double>(x.size(), const_cast<double *>(x.data()));
}

void write_to_python (const VectorXf & x, ostream & os)
{
  const int entries_per_line = 8;

  os << "[";
  for (int i = 0; i < x.size(); ++i) {
    os << x[i] << ", ";
    if ((i + 1) % entries_per_line == 0) os << "\n  ";
  }
  os << "]";
}

void save_to_python (const MatrixXf & A, string filename)
{
  LOG("saving MatrixXf to " << filename);

  std::ofstream file(filename);

  file << "[";
  for (int i = 0; i < A.rows(); ++i) {
    file << "\n  [";
    for (int j = 0; j < A.cols(); ++j) {
      file << A(i,j) << ", ";
    }
    file << "],";
  }
  file << "\n]\n";
}

