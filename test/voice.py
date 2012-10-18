#!/usr/bin/python

from numpy import *
from matplotlib import pyplot
import main

@main.command
def plot_spectral_params (w0=30.0, w1=5e3, tau=0.2, I=512):
  'plots error of spectral parameters'

  assert w0 < w1

  t_01 = arange(0,1,0.001)

  rho0 = I / log(w1/w0)
  rho = rho0 * 8 ** (2 * t_01 - 1)

  error = rho * log((tau * w1 + rho) / (tau * w0 + rho)) - I

  print min(rho), max(rho)
  print min(error), max(error)

  pyplot.figure()
  pyplot.plot(rho, error, 'r-')
  pyplot.plot(rho, 0 * error, 'k--')
  pyplot.xlabel('rho')
  pyplot.ylabel('I - I_rho')
  pyplot.xscale('log')
  pyplot.show()

if __name__ == '__main__': main.main()

