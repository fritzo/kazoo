
#include "matching.h"

using namespace Matching;

//----( testing )-------------------------------------------------------------

inline float noise () { return random_std() / BIG; }

void test_pair (size_t iters)
{
  LOG("\nTest case for pair");

  HardMatching matching;
  matching.clear();

  matching.add1(1.0 + noise());
  matching.add2(1.0 + noise());
  matching.add12(0, 0, 0.0 + noise());

  matching.validate_problem();

  matching.print_prior();
  matching.solve(iters);
  matching.print_post();

  matching.validate_solution();
}

void test_complete (size_t iters, size_t size_1 = 3, size_t size_2 = 4)
{
  LOG("\nTest case for complete matching");

  HardMatching matching;
  matching.clear();

  for (size_t i = 0; i < size_1; ++i) {
    matching.add1(2.0 + noise());
  }
  for (size_t j = 0; j < size_2; ++j) {
    matching.add2(2.0 + noise());
  }
  for (size_t i = 0; i < size_1; ++i) {
    for (size_t j = 0; j < size_2; ++j) {
      matching.add12(i, j, (i == j ? -1.0 : 1.0) + noise());
    }
  }

  matching.validate_problem();

  matching.print_prior();
  matching.solve(iters);
  matching.print_post();

  matching.validate_solution();
}

void test_ladder (size_t iters, size_t size = 4)
{
  LOG("\nTest case for /\\/\\/\\/\\/ scenario");
  HardMatching matching;
  matching.clear();

  for (size_t i = 0; i < size; ++i) {
    matching.add1(3.0 + noise());
    matching.add2(3.0 + noise());
    matching.add12(i, i, 0.0 + noise());
  }
  for (size_t i = 0, j = 1; j < size; ++i, ++j) {
    matching.add12(i, j, -1.0 + noise());
  }

  matching.validate_problem();

  matching.print_prior();
  matching.solve(iters);
  matching.print_post();

  matching.validate_solution();
}

int main (int argc, char ** argv)
{
  size_t iters = 4;

  if (argc > 1) iters = atoi(argv[1]);

  test_pair(iters);
  test_complete(iters);
  test_ladder(iters);

  return 0;
}

