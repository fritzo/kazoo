
#include "regression.h"
#include "linalg.h"

#define LOG1(message) LOG(message)

namespace Regression
{

using namespace LinAlg;

//----( line search )---------------------------------------------------------

/** Line Search.
  Search in given direction,
    retracting to half as far until objective function decreases.

  Returns chi^2/dof
*/
float line_search(
    Vector<float> & x0,
    Vector<float> & dx,
    VectorFunction & function)
{
  size_t I = function.size_in();
  size_t J = function.size_out();
  ASSERT_SIZE(x0, I);
  ASSERT_SIZE(dx, I);

  Vector<float> x1(I);
  Vector<float> fx(J);
  float y0, y1;

  add(x0, dx, x1);
  function(x0, fx); y0 = norm_squared(fx);
  function(x1, fx); y1 = norm_squared(fx);

  while (y1 > y0) {
    LOG("  retracting by factor of 2,  y1/y0 = " << (y1/y0));
    dx *= 0.5f;
    add(x0, dx, x1);
    function(x1, fx);
    y1 = norm_squared(fx);
  } while (y1 > y0);

  x0 = x1;
  return y1 / (J - I);
}

//----( nonlinear least squares )---------------------------------------------

inline void sample_unscented (
    const Vector<float> & mean,
    const Vector<float> & cov,
    Vector<float> & samples)
{
  size_t I = mean.size;
  ASSERT_SIZE(cov, I * I);
  ASSERT_SIZE(samples, 2 * I * I);

  Vector<float> chol_cov(I * I);
  chol_cov = cov;
  cholesky(I, chol_cov, true);

  for (size_t i = 0; i < I; ++i) {
    Vector<float> direction = chol_cov.block(I,i);
    Vector<float> pos = samples.block(I, 2 * i);
    Vector<float> neg = samples.block(I, 2 * i + 1);

    for (size_t j = 0; j < I; ++j) {
      pos[j] = mean[j] + direction[j];
      neg[j] = mean[j] - direction[j];
    }
  }
}

float nonlinear_least_squares (
    VectorFunction & function,
    Vector<float> & Ex,
    Vector<float> & Vxx,
    int max_iters,
    float tol)
{
  size_t I = function.size_in();
  size_t J = function.size_out();
  ASSERT_SIZE(Ex, I);
  ASSERT_SIZE(Vxx, I * I);

  size_t S = 2 * I;
  Vector<float> sample_x(S * I);
  Vector<float> sample_fx(S * J);
  Vector<float> Efx(J);
  Vector<float> Vyx(J * I);
  Vector<float> F(J * I);
  Vector<float> dx(I);
  float chi2_dof = INFINITY;

  LOG("Solving " << I << " x " << J << " nonlinear least squares problem:");
  for (int iters = 0; iters < max_iters; ++iters) {

    LOG1(" sampling function");
    sample_unscented(Ex, Vxx, sample_x);
    Efx.zero();
    for (size_t s = 0; s < S; ++s) {
      Vector<float> x  = sample_x.block(I,s);
      Vector<float> fx = sample_fx.block(J,s);
      function(x,fx);
      Efx += fx;
    }
    Efx *= 1.0f / S;

    LOG1(" constructing linearization");
    for (size_t s = 0; s < S; ++s) {
      Vector<float> x  = sample_x.block(I,s);   x -= Ex;
      Vector<float> fx = sample_fx.block(J,s);  fx -= Efx;
    }
    matrix_multiply(J, S, I, sample_fx,true, sample_x,false, Vyx, 0.5f);
    symmetric_solve(I, J, Vxx, Vyx,false, F);

    LOG1(" solving least-squares problem");
    least_squares(F,false, Efx, dx);
    dx *= -1.0f;

    chi2_dof = line_search(Ex, dx, function);

    outer_prod(I, J, F,true, Vxx);
    float residual = sqrtf(symmetric_norm(dx, Vxx)); // Vxx is inverted here
    symmetric_invert(I, Vxx);

//#define DEBUG_REGRESSION
#ifdef DEBUG_REGRESSION
    for (size_t i = 0; i < I; ++i) {
      LOG("  x(" << i << ") = " << Ex[i] << "\t+- " << sqrtf(Vxx[I * i + i]));
    }
#endif // DEBUG_REGRESSION

    PRINT2(residual, chi2_dof);
    if (residual < tol) {
      LOG(" converged.");
      break;
    }
  }

  return chi2_dof;
}

//----( online pca )----------------------------------------------------------

OnlinePca::OnlinePca (size_t size, size_t count)
  : m_size(size),
    m_count(count),
    m_basis(size * count)
{
  ASSERTW_DIVIDES(4, size);

  for (size_t i = 0; i < size * count; ++i) {
    m_basis[i] = random_std() + random_std() + random_std();
  }

  standardize();
}

void OnlinePca::add_sample (const Vector<float> & y, float dt)
{
  ASSERT_LT(0, dt);
  float drift = sqrtf(dt) / norm_squared(y);

  for (size_t m = 0; m < m_count; ++m) {
    Vector<float> x = component(m);
    multiply_add(drift * dot(x,y), y, x);
  }

  standardize();
}

void OnlinePca::standardize ()
{
  for (size_t m = 0; m < m_count; ++m) {
    Vector<float> x = component(m);
    x -= mean(x);

    for (size_t n = 0; n < m; ++n) {
      Vector<float> y = component(n);
      multiply_add(-dot(x,y), y, x);
    }

    x *= powf(norm_squared(x), -0.5f);
  }
}

} // namespace Regression

