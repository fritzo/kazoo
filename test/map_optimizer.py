#!/usr/bin/python

from math import *
from scipy import *
#from matplotlib import pyplot
#from scipy import linalg
import main

#----( commands )-------------------------------------------------------------

@main.command
def project_unif (X = 3, Y = 2):
  'projects dJ matrix via inclusion-exclusion, WRT J metric'

  J = exp(randn(Y,X))
  J /= sum(J)
  print('J = %s' % J)

  p = dot(ones(Y), J)
  q = dot(J, ones(X))
  Z = sum(J)
  print('p = %s' % p)
  print('q = %s' % q)
  print('Z = %s' % Z)

  print('\nBefore projecting:')

  dJ = randn(Y,X)
  print('dJ = %s' % dJ)

  dp = dot(ones(Y), dJ)
  dq = dot(dJ, ones(X))
  dZ = sum(dJ)
  print('dp = %s' % dp)
  print('dq = %s' % dq)
  print('dZ = %s' % dZ)

  print('\nAfter projecting:')

  dJ -= outer(ones(Y) / Y, dp)
  dJ -= outer(dq, ones(X) / X)
  dJ += dZ / (X * Y)

  dp = dot(ones(Y), dJ)
  dq = dot(dJ, ones(X))
  dZ = sum(dJ)
  print('dp = %s' % dp)
  print('dq = %s' % dq)
  print('dZ = %s' % dZ)

@main.command
def project_log (X = 3, Y = 2):
  'projects dJ matrix via inclusion-exclusion WRT log(J) metric (incorrectly)'

  J = exp(randn(Y,X))
  J /= sum(J)
  print('J = %s' % J)

  p = dot(ones(Y), J)
  q = dot(J, ones(X))
  Z = sum(J)
  print('p = %s' % p)
  print('q = %s' % q)
  print('Z = %s' % Z)

  print('\nBefore projecting:')

  dJ = randn(Y,X)
  dJ *= J # convert gradient to descent direction in log(J) metric
  print('dJ = %s' % dJ)

  dp = dot(ones(Y), dJ)
  dq = dot(dJ, ones(X))
  dZ = sum(dJ)
  print('dp = %s' % dp)
  print('dq = %s' % dq)
  print('dZ = %s' % dZ)

  print('\nAfter projecting:')

  dJ -= J * outer(ones(Y), dp / p)
  dJ -= J * outer(dq / q, ones(X))
  dJ += J * (dZ / Z)

  dp = dot(ones(Y), dJ)
  dq = dot(dJ, ones(X))
  dZ = sum(dJ)
  print('dp = %s' % dp)
  print('dq = %s' % dq)
  print('dZ = %s' % dZ)

@main.command
def project_iter (tol = 1e-12, X = 3, Y = 2):
  'projects dJ matrix iteratively WRT log(J) metric'

  '''
  Each of the projections 1-dp/p, 1-dq/q has eigenvalues in {0,1}.
  The sum 1 - (dp/p + dq/q) has eigenvalues in [-1,0] u {1}.
  The optimal iterative approximation is 1 - 2/3 * (dp/p + dq/q),
  which has eigenvalues in [-1/3,1/3] u {1}.
  '''

  logging = (X + Y < 8)

  from matplotlib import pyplot

  J = exp(randn(Y,X))
  J /= sum(J)
  if logging:
    print('J = %s' % J)

  p = dot(ones(Y), J)
  q = dot(J, ones(X))
  Z = sum(J)
  if logging:
    print('p = %s' % p)
    print('q = %s' % q)
    print('Z = %s' % Z)

  print('\nBefore projecting:')

  dJ = randn(Y,X)
  dJ *= J # convert gradient to descent direction in log(J) metric
  if logging:
    print('dJ = %s' % dJ)

  scale = 2 / 3.0
  iters = []
  errors = []
  for i in range(100):

    dp = scale * dot(ones(Y), dJ)
    dq = scale * dot(dJ, ones(X))

    dJ -= J * outer(ones(Y), dp / p)
    dJ -= J * outer(dq / q, ones(X))

    iter = 1 + i
    error = sqrt((sum(dq * dq) + sum(dp * dp)) / (X + Y))
    iters.append(iter)
    errors.append(error)

    if error < tol:
      print('projection converged after %i steps (expected %g)' \
          % (iter, -log(tol) / log(3.0)))
      break

  print('\nAfter projecting:')

  dp = dot(ones(Y), dJ)
  dq = dot(dJ, ones(X))
  dZ = sum(dJ)
  if logging:
    print('dp/tol = %s' % (dp / tol))
    print('dq/tol = %s' % (dq / tol))
    print('dZ/tol = %s' % (dZ / tol))

  pyplot.plot(iters, errors, 'ko')
  pyplot.yscale('log')
  pyplot.xlabel('iteration')
  pyplot.ylabel('rms error')
  pyplot.show()

if __name__ == '__main__': main.main()

