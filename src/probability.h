#ifndef KAZOO_PROBABILITY_H
#define KAZOO_PROBABILITY_H

#include "common.h"

/** Probability, Likelihood, and Energy.

  In probabilistic inference algorithms,
  the prior information on a discrete variable consists of
  a set of weights -as- likelihoods, or energies -as- log-likelihoods.
  This file contains utilities to compute these prior likelihoods and energies.

  Example 1:
    Consider three possibilities:
      a variable x is e0-surprisingly not observed at all
      a variable x is observed with e1-surprising value z1
      a variable x is observed with e2-surprising value z2
    Suppose the prior probabilities of observing x are p0,e0, resp.
    Suppose the prior distribution of z is standard normal.
    Then we compute energies
      e0 = -log(p0 / p1)     # energy_gap = -log(likelihood_ratio)
      e1 = z1^2 / 2 - 1      # free_energy = -log(likelihood)
      e2 = z2^2 / 2 - 1      # free_energy = -log(likelihood)

  Defintion:

    likelihood_ratio  =  p / (1 - p)  =  exp(-energy_gap)

    energy_gap  =  log(1 - p) - log(p)  =  -log(likelihood_ratio)

    free_energy  =  -log(probability) - entropy
*/

template<class T>
inline T entropy_term (T prob) { return prob > 0 ? -prob * std::log(prob) : 0; }

//----( bernoulli distributions )---------------------------------------------

inline float bernoulli_entropy (float p)
{
  return entropy_term(p) + entropy_term(1 - p);
}

inline float bernoulli_free_energy (float p)
{
  float q = 1 - p;
  if (not (p > 0)) return INFINITY;
  if (not (q > 0)) return 0;
  return q * logf(q / p);
}

inline float bernoulli_energy_gap (float p)
{
  float q = 1 - p;
  if (not (p > 0)) return INFINITY;
  if (not (q > 0)) return -INFINITY;
  return logf(q / p);
}

inline float bernoulli_likelihood_ratio (float p)
{
  float q = 1 - p;
  if (not (q > 0)) return INFINITY;
  return p / q;
}

//----( exponential distributions )-------------------------------------------

/** Exponential likelihood ratio
  Properties:
   * at fixed scale parameter lambda,
     result(t/lambda) is proportional to exponential density P(t;lambda)
   * for all lambda, t, E[log result(t/lambda)] = 0
*/

inline float exponential_free_energy (float t) { return t - 1; }
inline float exponential_likelihood (float t) { return exp(1 - t); }

/** Exponential Cumulative Distribution Function

  The exponential CDF defines a Bernoulli variable with probability parmeter

  cdf(t) = int s:[0,t]. exp(-t)
         = exp(0) - exp(-t)
         = 1 - exp(-t)
*/

inline float exponential_cdf (float t) { return 1 - expf(-t); }

inline float exponential_cdf_likelihood_ratio (float t)
{
  return bernoulli_likelihood_ratio(exponential_cdf(t));
}

inline float exponential_cdf_energy_gap (float t)
{
  return bernoulli_energy_gap(exponential_cdf(t));
}

//----( gaussian distributions )----------------------------------------------

inline float gaussian_free_energy (float chi2, int dof = 1)
{
  return 0.5f * (chi2 - dof);
}

inline float gaussian_likelihood (float chi2, int dof = 1)
{
  return expf(-0.5f * (chi2 - dof));
}

//----( normal distributions )------------------------------------------------

inline float normal_free_energy (float x, float mu, float sigma)
{
  return 0.5f * (sqr((x - mu) / sigma) - 1);
}

inline float normal_likelihood (float x, float mu, float sigma)
{
  return expf(-0.5f * (1 - sqr((x - mu) / sigma)));
}

//----( gaussian )------------------------------------------------------------

struct Gaussian
{
  template<class T>
  struct Statistic
  {
    T sum_1;
    T sum_x;
    T sum_x2;

    void zero ()  { sum_1 = 0; sum_x = 0; sum_x2 = 0; }

    //void set (T x) { sum_1 = 1; sum_x = x; sum_x2 = x * x; }
    void set (T s1, T sx, T sx2) { sum_1 = s1; sum_x = sx; sum_x2 = sx2; }

    //void add (T x) { sum_1 += 1; sum_x += x; sum_x2 += x * x; }
    void add (T s1, T sx, T sx2) { sum_1 += s1; sum_x += sx; sum_x2 += sx2; }

    template<class S>
    void operator+= (const Statistic<S> & other)
    {
      sum_1 += other.sum_1;
      sum_x += other.sum_x;
      sum_x2 += other.sum_x2;
    }

    void shift (T dx)
    {
      sum_x2 += (2 * sum_x + sum_1) * dx * sum_1;
      sum_x += dx * sum_1;
    }
    void scale (T factor)
    {
      sum_x *= factor;
      sum_x2 *= factor * factor;
    }

    T get_mean () const { return sum_x / sum_1; }
    T get_var () const { return sum_x2 / sum_1 - sqr(get_mean()); }

    template<class S>
    Statistic<S> cast ()
    {
      Statistic<S> result;
      result.set(sum_1, sum_x, sum_x2);
      return result;
    }
  };
};

//----( chi^2 )---------------------------------------------------------------

// We use a two-parameter chi^2 distribution
//
//              (x/r^2)^(d/2-1) exp(-x/(2 r^2))
//   p(x|d,r) = -------------------------------
//                  2^(d/2) Gamma(d/2) r^2
//
// which has sufficient statistics E[log(x)] and E[x].

struct ChiSquared
{
  template<class T>
  struct Statistic
  {
    T sum_1;
    T sum_x;
    T sum_log_x;

    void zero () { sum_1 = 0; sum_x = 0; sum_log_x = 0; }

    //void set (T x) { sum_1 = 1; sum_x = x; sum_log_x = std::log(x); }
    void set (T s1, T sx, T sl) { sum_1 = s1; sum_x = sx; sum_log_x = sl; }

    //void add (T x) { sum_1 += 1; sum_x += x; sum_log_x += std::log(x); }
    void add (T s1, T sx, T sl) { sum_1 += s1; sum_x += sx; sum_log_x += sl; }

    template<class S>
    void operator+= (const Statistic<S> & other)
    {
      sum_1 += other.sum_1;
      sum_x += other.sum_x;
      sum_log_x += other.sum_log_x;
    }

    // shifting does not commute with accumulation, but scaling does
    //void shift (T dx);
    void scale (T factor)
    {
      sum_x *= factor;
      sum_log_x += sum_1 * std::log(factor);
    }

    T get_mean () const { return sum_x / sum_1; }
    T get_mean_log () const { return sum_log_x / sum_1; }

    template<class S>
    Statistic<S> cast ()
    {
      Statistic<S> result;
      result.set(sum_1, sum_x, sum_log_x);
      return result;
    }
  };

  struct Estimator
  {
    float updated_dof;
    float updated_radius;

    Estimator (
        Statistic<float> observed,
        float predicted_dof,
        float predicted_radius = 1);
  };

  struct RadiusEstimator
  {
    float updated_radius;

    RadiusEstimator (
        Statistic<float> observed,
        float dof,
        float predicted_radius = 1);
  };

  struct DofEstimator
  {
    float updated_dof;

    DofEstimator (
        Statistic<float> observed,
        float predicted_dof);
  };
};

#endif // KAZOO_PROBABILITY_H
