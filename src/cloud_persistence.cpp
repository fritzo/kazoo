
#include "cloud_persistence.h"
#include <fstream>

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#define TOK(expected_tok) \
  file >> tok; \
  ASSERT(not file.fail(), \
      "failed to read expected token '" expected_tok "'"); \
  ASSERT_EQ(tok, expected_tok);

#define PARSE(type, variable) \
  type variable; \
  file >> variable; \
  ASSERT(not file.fail(), \
      "failed to parse variable '" #variable "' of type " #type);

namespace Cloud
{

//----( persistence )---------------------------------------------------------

static const int FILE_VERSION = 6;

void write_matrix (const MatrixSf & A, ostream & o)
{
  LOG("  writing " << A.nonZeros() << " entries of "
      << A.rows() << " x " << A.cols() << " sparse matrix");

  o << "\n  SparseMatrix<float>"
    << "\n   shape = " << A.rows() << " " << A.cols()
    << "\n   " << A.nonZeros() << " entries (row,col,value)";

  for (int i = 0; i < A.outerSize(); ++i) {

    for (MatrixSf::InnerIterator iter(A,i); iter; ++iter) {

      o << "\n   " << iter.row() << " " << iter.col() << " " << iter.value();
    }
  }
}

void read_matrix (MatrixSf & A, istream & file)
{
  string tok;

  TOK("SparseMatrix<float>")
  TOK("shape") TOK("=") PARSE(int, I) PARSE(int, J);
  PARSE(int, N); TOK("entries"); TOK("(row,col,value)");

  LOG("  reading " << N << " entries of "
      << I << " x " << J << " sparse matrix");

  A.resize(I,J);
  for (int n = 0; n < N; ++n) {

    PARSE(int, row)
    PARSE(int, col)
    PARSE(float, value)

    A.insert(row, col) = value;
  }

  A.finalize();
}

void Persistent::save (string filename) const
{
  LOG("saving to file " << filename);
  filestem = filename;

  std::ofstream o(filename);

  o << "kazoo cloud file";

  o << "\n""version " << FILE_VERSION;

  write(o);

  o << "\n"
    << "\n""end"
    << "\n";
}

void Persistent::load (string filename)
{
  LOG("loading from file " << filename);

  std::ifstream file(filename);
  ASSERT(file, "failed to open " << filename);

  string tok;

  TOK("kazoo") TOK("cloud") TOK("file")

  TOK("version") PARSE(int, version)
  ASSERT_EQ(version, FILE_VERSION);

  read(file);

  TOK("end")
}

} // namespace Cloud

