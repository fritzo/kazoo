#!/usr/bin/python

from math import *
from numpy import *
from matplotlib import pyplot
from scipy import linalg
import main

def read_array (name):
  return array(eval(open('data/temp.' + name + '.py').read()))

def stats (A):
  density = sum(abs(A)) / sum(A*A) / len(A)
  return 'min = %s, max = %s, density = %s' % (A.min(), A.max(), density)

def stats2 (A):
  A_mean = A.mean()
  A_var = ((A - A_mean)**2).mean()
  return '\n  '.join([
      'min = %s, max = %s' % (A.min(), A.max()),
      'mean = %s, std.dev. = %s' % (A_mean, sqrt(A_var)),
      ])

#----( commands )-------------------------------------------------------------

@main.command
def hh_stats ():
  'prints statistics of the |head><head| matrix'
  HH = read_array('m_head_head')

  print 'HH: ' + stats(HH)

  evals = linalg.eigvalsh(HH)
  print 'HH: condition number = %s' % (evals.max() / evals.min())
  print 'HH: evals = %s' % evals

  invHH = linalg.inv(HH)
  print 'inv(HH): ' + stats(invHH)

@main.command
def ht_stats ():
  'prints statistics of the |head><tail| matrix'
  HH = read_array('m_head_head')
  HT = read_array('m_head_tail')

  print 'HH: ' + stats(HH)
  print 'HT: ' + stats(HT)

  cholHH = linalg.cholesky(HH)
  F = linalg.cho_solve((cholHH, False), HT)
  print 'F: ' + stats(F)
  print 'residual = %s' % sum((dot(HH, F) - HT)**2)

  Fsums = F.sum(0);
  print 'F sums: ' + stats2(Fsums)

  B = linalg.cho_solve((cholHH, True), HT.transpose())
  print 'B: ' + stats(B)
  print 'residual = %s' % sum((dot(HH, B) - HT.transpose())**2)

  Bsums = B.sum(0);
  print 'B sums: ' + stats2(Bsums)

if __name__ == '__main__': main.main()

