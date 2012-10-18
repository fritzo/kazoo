#ifndef KAZOO_REGRESSION_H
#define KAZOO_REGRESSION_H

#include "common.h"
#include "vectors.h"
#include <vector>

#define REGRESSION_MAX_ITERS            (8)
#define REGRESSION_TOLERANCE            (1e-2f)

namespace Regression
{

class FunctionWithPrior : public VectorFunction
{
  VectorFunction & m_function;
  const Vector<float> & m_prior_mean;
  const Vector<float> & m_prior_sigma;

public:

  Vector<float> cov;

  FunctionWithPrior (
    VectorFunction & function,
    const Vector<float> & prior_mean,
    const Vector<float> & prior_sigma)
  : m_function(function),
    m_prior_mean(prior_mean),
    m_prior_sigma(prior_sigma),
    cov(sqr(prior_sigma.size))
  {
    ASSERT_EQ(prior_mean.size, prior_sigma.size);
    cov.zero();
    for (size_t i = 0, I = prior_sigma.size; i < I; ++i) {
      cov[I * i + i] = sqr(prior_sigma[i]);
    }
  }

  virtual size_t size_in () const { return m_function.size_in(); }
  virtual size_t size_out () const
  {
    return m_function.size_out() + m_prior_mean.size;
  }
  virtual void operator() (const Vector<float> & input, Vector<float> & output)
  {
    Vector<float> fun_out(m_function.size_out(), output.data);
    Vector<float> prior_out(m_function.size_in(), output.data + fun_out.size);
    m_function(input, fun_out);
    prior_out = input;
    prior_out -= m_prior_mean;
    prior_out /= m_prior_sigma;
  }
};

//----( nonlinear least squares )---------------------------------------------

/** Gauss-newton algorithm using unscented function sampling.

  Returns chi2/dof of solution.
*/
float nonlinear_least_squares (
    VectorFunction & function,
    Vector<float> & mean,
    Vector<float> & cov,
    int max_iters = REGRESSION_MAX_ITERS,
    float tol      = REGRESSION_TOLERANCE);

//----( online pca )----------------------------------------------------------

class OnlinePca
{
  const size_t m_size;
  const size_t m_count;

  Vector<float> m_basis;

public:

  OnlinePca (size_t size, size_t count);
  virtual ~OnlinePca () {}

  Vector<float> component (size_t n) { return m_basis.block(m_size, n); }

  void add_sample (const Vector<float> & sample, float dt);

private:

  void standardize ();
};

//----( online autoregression )-----------------------------------------------

/** Online Autoregressive model using Truncated SVD

  TODO implement online autoregression

  References:
  (R1) "Fast online SVD revisions for lightweight recommender systems"
    -Mathew Brand, MERL tech report TR-2003-14
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.3.1001
*/
class OnlineAR
{
  const size_t m_size_in;
  const size_t m_size_out;

  size_t m_num_components;
  size_t m_max_components;
  float m_component_thresh;

  Vector<float> m_inputs;
  Vector<float> m_outputs;
  size_t m_num_datapoints;

public:
  OnlineAR (
      size_t size_in,
      size_t size_out,
      size_t max_components = 0,
      float components_thres = 0);

  size_t size_in () const { return m_size_in; }
  size_t size_out () const { return m_size_out; }
  size_t num_components () const { return m_num_components; }

  void set_max_components (size_t max_components);
  void set_component_thresh (float component_thresh);

  void add_point (const Vector<float> & input, const Vector<float> & output);
  void map_point (const Vector<float> & input, Vector<float> & output) const;
};

} // namespace Regression

#endif // KAZOO_REGRESSION_H
