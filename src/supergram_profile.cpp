
#include "common.h"
#include "reassign.h"

void profile_supergram (size_t size_exp = 10,
                        size_t factor_exp = 3,
                        size_t num_iterations = 1000,
                        bool fwd = true,
                        bool bwd = true)
{
  Supergram super(size_exp,factor_exp);

  Vector<complex> input(super.small_size());
  Vector<float>     output(super.large_size());

  for (size_t i = 0; i < input.size; ++i) {
    input[i] = complex(random_std(), random_std());
  }
  for (size_t i = 0; i < output.size; ++i) {
    output[i] = random_std();
  }

  size_t num_samples = super.small_size() * num_iterations;
  LOG( "Running a Supergram(" << size_exp << "," << factor_exp << ") on "
      << num_samples << " samples");

  float total_time = -get_elapsed_time();

  for (size_t i = 0; i < num_iterations; ++i) {
    if (fwd) super.transform_fwd(input, output);
    if (bwd) super.transform_bwd(output, input);
  }

  total_time += get_elapsed_time();
  LOG(" processed " << (num_samples / total_time) << " samples/sec");
}

int main ()
{
  profile_supergram(9,2);
  profile_supergram(9,3);
  profile_supergram(10,2);
  profile_supergram(10,3);

  return 0;
}

