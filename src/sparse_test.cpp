
#include "sparse.h"
#include "linalg.h"
#include "args.h"

void test_ring (Args & args)
{
  const size_t dim = args.pop(10);
  const size_t radius = 2;
  const size_t degree = radius + 1 + radius;

  Vector<uint16_t> nbhd(dim * degree);
  Vector<float> Pxx(dim * degree);
  Vector<float> Pxy(dim * degree);
  Vector<float> Fxy(dim * degree);

  for (size_t i = 0; i < dim; ++i) {
    for (size_t n = 0; n < degree; ++n) {
      size_t knn_pos = degree * i + n;

      float dx = n - 2.0f;

      nbhd[knn_pos] = (i + dim + n - radius) % dim;
      Pxx[knn_pos] = expf(-sqr(dx));
      Pxy[knn_pos] = expf(-sqr(dx - 0.5) / 2);
    }
  }

  PRINT_MAT(dim, degree, Pxx);
  PRINT_MAT(dim, degree, Pxy);

  LinAlg::sparse_symmetric_solve(dim, degree, nbhd, Pxx, Pxy, Fxy);

  PRINT_MAT(dim, degree, Fxy);
}

const char * help_message =
"Usage: sparse_test COMMAND"
"\n:Commands:"
"\n  ring [DIM = 10]"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);

  args
    .case_("ring", test_ring)
    .default_error();

  return 0;
}

