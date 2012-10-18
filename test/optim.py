
from numpy import *
from scipy import *

def unscented_sample (mean, cov):
  size = len(mean)
  assert cov.shape == (size,size)
  L = linalg.cholesky(cov)
  sample = zeros((2 * size, size))
  for i in range(size):
    sample[2*i+0,:] = mean + L[i,:]
    sample[2*i+1,:] = mean - L[i,:]
  return sample

def linesearch_step (x0, dx, f):
  f0 = linalg.norm(f(x0))

  while linalg.norm(dx) > 0:
    x1 = x0 + dx
    f1 = linalg.norm(f(x1))

    if f1 <= f0:
      break
    else:
      ' retracting'
      dx /= 2

  x0 += dx

def nonlinear_minimize (f, x0, sigma, iters = 1, tol = 1e-6):

  I = len(x0)

  print 'solving %i-dimensional nonlinear optimization problem' % I

  print ' x = %s' % x0

  Dx = diag(sigma)

  for iter in range(iters):

    for i in range(I):

      dx = Dx[i,:]
      xp = x0 + dx
      xn = x0 - dx

      f0 = f(x0)
      fp = f(xp)
      fn = f(xn)

      slope = max(fp,fn) - f0
      if slope < tol:
        continue
      abs_dx = 0.5 * ((f0 - min(fp,fn)) / slope + 1)

      if fp < fn:
        x0 += dx * abs_dx
      else:
        x0 -= dx * abs_dx

      #if abs_dx > 0.75:
      #  #print 'widening dimension %i' % i
      #  sigma[i] *= 2.0
      if abs_dx < 0.25:
        #print 'narrowing dimension %i' % i
        sigma[i] *= 0.5

def nonlinear_least_squares (f, x0, cov, iters = 1, min_cov = None):

  f0 = f(x0)
  I = len(x0)
  J = len(f0)
  S = 2 * I

  print 'solving %i x %i nonlinear least squares problem' % (I,J)

  print ' x = %s' % x0

  for iter in range(iters):

    sample_x = unscented_sample(x0, cov)
    sample_fx = zeros((S,J))
    for s in range(S):
      sample_fx[s,:] = f(sample_x[s,:])
    Efx = sum(sample_fx, axis=0) / S
    for s in range(S):
      sample_x[s,:] -= x0
      sample_fx[s,:] -= Efx

    Vyx = dot(sample_fx.T, sample_x)
    F = dot(Vyx, linalg.inv(cov))
    cov[:,:] = linalg.inv(dot(F.T, F))
    if min_cov is not None:
      cov += min_cov
    dx,residues,rank,singuar_values = linalg.lstsq(F, Efx)

    dx *= -1.0
    linesearch_step(x0, dx, f)
    chi2_dof = linalg.norm(f(x0)) ** 2 / (J - I)

    print '  |dx| = %g' % linalg.norm(dx)
    print '  chi^2/dof = %g' % chi2_dof
    print ' x = %s' % x0

  return chi2_dof

def fun_with_prior (mean, cov, fun):
  chol_inv = linalg.cholesky(linalg.inv(cov)).T
  print chol_inv
  def new_fun (param):
    y1 = dot(chol_inv, param - mean)
    y2 = fun(param)
    y = concatenate((y1, y2))
    return y
  return new_fun

