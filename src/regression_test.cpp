
#include "regression.h"
#include "linalg.h"

struct LineFit : public VectorFunction
{
  std::vector<float> data;

  virtual size_t size_in () const { return 2; }
  virtual size_t size_out () const { return data.size(); }
  virtual void operator () (const Vector<float> & input, Vector<float> & output)
  {
    ASSERT_SIZE(input, size_in());
    ASSERT_SIZE(output, size_out());

    float a = exp(input[0]);
    float b = exp(input[1]);

    for (size_t i = 0; i < size_out(); ++i) {
      output[i] = a * i + b - data[i];
    }
  }
};

int main (int argc, char ** argv)
{
  ASSERT(argc >= 3, "need at least two datapoints");

  LineFit fit;
  for (int i = 1; i < argc; ++i) {
    fit.data.push_back(atof(argv[i]));
  }

  Vector<float> mean(2);
  mean.zero();
  Vector<float> sigma(2);
  sigma[0] = 1;
  sigma[1] = 1;

  Regression::FunctionWithPrior fun(fit, mean, sigma);

  float chi2_dof = Regression::nonlinear_least_squares(fun, mean, fun.cov);
  PRINT_VECT(mean);
  PRINT(chi2_dof);

  return 0;
}

