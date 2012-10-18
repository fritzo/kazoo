
#include "probability.h"
#include "optim.h"

//----( chi^2 )---------------------------------------------------------------

/**
The single-parameter chi^2 distribution has density

           (x/2)^(d/2-1) exp(-x/2)
  p(x|d) = -----------------------
                2 Gamma(d/2)

We use a two-parameter chi^2 distribution

             (x/(2 r^2))^(d/2-1) exp(-x/(2 r^2))
  p(x|d,r) = -----------------------------------
                     2 r^2 Gamma(d/2)

  -log p(x|d,r) = (1 - d/2) log(x/(2 r^2))                                  (1)
                + x / (2 r^2)
                + log(2 r^2)
                + log(Gamma(d/2))

which has sufficient statistics E[log(x)] and E[x].
To estimate ML values of d = dof and r = radius, we compute partial derivatives

  d/dr (-log p(x|d,r)) = -x / r^3 + d / r

                                                Gamma'(d/2)
  d/dd (-log p(x|d,r)) = -1/2 log(x/(2 r^2)) + ------------
                                               2 Gamma(d/2)

At stationary points, the sufficient statistics satisfy

  E[x] = d r^2                                                              (2)

              Gamma'(d/2)
  E[log x] = ------------ - log(2 r^2)                                      (3)
             2 Gamma(d/2)

           = 1/2 digamma(d/2) - log(2 r^2)

(where expectation is over the sample).
Equation (3) is difficult to solve, so we solve (2)

  r^2 = E[x] / d                                                            (4)

substitute (4) into (1), and optimize numerically.

(R1) "Analysis of Minimum Distances in High-Dimensional Musical Spaces"
  -Michael Casey, Christophe Rhodes, Malcolm Slaney
  http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.134.3952
  http://www.doc.gold.ac.uk/~mas01cr/papers/taslp2008/04544816.pdf
*/

namespace
{

class ChiSquaredMLObjective : public Function
{
  const float m_mean_x;
  const float m_mean_log_x;

public:

  ChiSquaredMLObjective (float x, float l) : m_mean_x(x), m_mean_log_x(l) {}
  virtual ~ChiSquaredMLObjective () {}

  virtual float value (float log_d) const
  {
    float d = expf(log_d);
    ASSERT_FINITE(d);
    float r2 = m_mean_x / d;

    //PRINT2(d, r2);

    return (1 - d/2) * (m_mean_log_x - logf(2 * r2))
         + m_mean_x / (2 * r2)
         + logf(2 * r2)
         + lgammaf(d/2);
  }
};

} // anonymous namespace

ChiSquared::Estimator::Estimator (
    Statistic<float> observed,
    float predicted_dof,
    float predicted_radius)
{
  // Estimate maximum-likelihood values of (dof,radius), following (R1).

  //PRINT2(predicted_dof, predicted_radius);
  ASSERT_LT(0, predicted_dof);
  ASSERT_LT(0, predicted_radius);

  float observed_mean = observed.get_mean();
  float observed_mean_log = observed.get_mean_log();
  //PRINT2(observed_mean, exp(observed_mean_log));

  ASSERT_LT(0, observed_mean);
  ASSERT_FINITE(observed_mean_log);

  ChiSquaredMLObjective fun(observed_mean, observed_mean_log);
  float log_d = logf(predicted_dof);
  float xjump = 0.5f;
  float xtol = 1e-4f;

  log_d = minimize_grid_search(fun, log_d, xjump);
  log_d = minimize_bisection_search(fun, log_d - xjump, log_d + xjump, xtol);

  updated_dof = expf(log_d);
  updated_radius = sqrtf(observed_mean / updated_dof) * predicted_radius;
}

ChiSquared::RadiusEstimator::RadiusEstimator (
    Statistic<float> observed,
    float dof,
    float predicted_radius)
{
  // Estimate maximum-likelihood value of radius.

  //PRINT2(dof, predicted_radius);
  ASSERT_LT(0, dof);
  ASSERT_LT(0, predicted_radius);

  float observed_mean = observed.get_mean();
  //PRINT(observed_mean);
  ASSERT_LT(0, observed_mean);

  updated_radius = sqrtf(observed_mean / dof) * predicted_radius;
}

ChiSquared::DofEstimator::DofEstimator (
    Statistic<float> observed,
    float predicted_dof)
{
  // Estimate maximum-likelihood value of dof.

  //PRINT(predicted_dof);
  ASSERT_LT(0, predicted_dof);

  float observed_mean = observed.get_mean();
  float observed_mean_log = observed.get_mean_log();
  //PRINT2(observed_mean, exp(observed_mean_log));

  ASSERT_LT(0, observed_mean);
  ASSERT_FINITE(observed_mean_log);

  TODO("updated_dof = ???;");
}

