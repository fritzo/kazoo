#!/usr/bin/python

import os
import subprocess
from numpy import *
from scipy.special import gamma
import main

def chi2_pdf (x, dof):
  dof2 = 0.5 * dof
  return x ** (dof2 - 1) * exp(-0.5 * x) / (2 ** dof2 * gamma(dof2))

class ChiSquaredMLObjective:

  def __init__ (self, mean_x, mean_log_x):
    self.mean_x = mean_x
    self.mean_log_x = mean_log_x

  def __call__ (self, d):
    r2 = self.mean_x / d;

    return ( (1 - d/2) * (self.mean_log_x - log(2 * r2))
           + self.mean_x / (2 * r2)
           + log(2 * r2 * gamma(d/2))
           )

def learn (args):
  info = os.system

#----( commands )-------------------------------------------------------------

@main.command
def small_multi (dom_size = 256, cod_size = 128, grids = 4):
  'builds a multigrid voice + gloves + gv controller'

  voice_file = 'data/voice/multi-%s.cloud' % grids
  gloves_file = 'data/gloves/multi-%s.cloud' % grids
  map_file = 'data/gloves/multi-%s.map' % grids

  dom_grids = ['-g', str(grids)] + [str(dom_size * 2**i) for i in range(grids)]
  cod_grids = ['-g', str(grids)] + [str(cod_size * 2**i) for i in range(grids)]

  if not os.path.exists(voice_file):
    voice = subprocess.Popen(
        ['../src/learn', 'voice', 'new'] +
        dom_grids +
        ['-o', voice_file] +
        ['data/voice/05  Amy Winehouse - Back To Black.avi'])
    voice.wait()

  if not os.path.exists(gloves_file):
    gloves = subprocess.Popen(
        ['../src/learn', 'gloves', 'new'] +
        cod_grids +
        ['-o', gloves_file])
    gloves.wait()

  if not os.path.exists(map_file):
    gv = subprocess.Popen(
        ['../src/learn', 'gv', 'new',
          '-d', voice_file,
          '-c', gloves_file,
          '-o', map_file])
    gv.wait()

@main.command
def gloves_multi (size = 128, grids = 5):
  'builds a multigrid gloves cloud'

  gloves_file = 'data/gloves/multi-%s.cloud' % grids

  gloves_grids = ['-g', str(grids)] + [str(size * 2**i) for i in range(grids)]

  gloves = subprocess.Popen(
      ['../src/learn', 'gloves', 'new'] +
      gloves_grids +
      ['-o', gloves_file])
  gloves.wait()

@main.command
def plot_chi2 (max_dof = 20, num_samples = 1000):
  'plots chi^2 likelihood for various DOF values'

  from matplotlib import pyplot

  X = 2.0 * max_dof / num_samples * array(list(range(1 + num_samples)))

  for dof in range(1, 1 + max_dof):
    Y = chi2_pdf(X, dof)
    Y *= 1 / max(Y)
    pyplot.plot(X,Y, 'r-')

  pyplot.title('Chi^2 likelihood functions for dof in [1,%i]' % max_dof)

  pyplot.show()

@main.command
def plot_chi2_objective (
    observed_mean = 2.51,
    observed_mean_log = 0.795,
    predicted_dof = 33.9):
  'plots -log P(sufficient statistics | dof, ML radius) vs dof'

  from matplotlib import pyplot

  fun = ChiSquaredMLObjective(observed_mean, observed_mean_log)

  X = exp(arange(log(predicted_dof) - 5, log(predicted_dof) + 2, 0.1))
  Y = fun(X)

  pyplot.loglog(X,Y)
  pyplot.xlabel('dof')
  pyplot.ylabel('-log P(obs|dof)')
  pyplot.title('Objective function for Chi^2 optimization')
  pyplot.show()

@main.command
def plot_hist (filename = 'data/gloves/histogram.py'):
  'plots histograms of squared distances'

  from matplotlib import pyplot

  histograms = eval(open(filename).read())

  bins_per_unit = 50
  grids = len(histograms)

  for g,(size,radius,hist) in enumerate(histograms):

    if g == 0:
      ax = pyplot.subplot(grids, 1, 1+g)
      pyplot.title('Histogram of squared distances from %s' % filename)
    else:
      pyplot.subplot(grids, 1, 1+g, sharex = ax, sharey = ax)

    x_scale = radius
    y_scale = bins_per_unit / x_scale
    chi2_scale = 1.0 / radius

    X = array([x_scale * x for x,y in hist])
    Y = array([y_scale * y for x,y in hist])

    pyplot.plot(X,Y, 'r.', label = '%i points, radius = %0.1f' % (size, radius))

    #dof = ???
    #chi2 = chi2_scale * chi2_pdf(X / radius, dof)
    #pyplot.plot(X,chi2, 'k--', label = 'chi^2 pdf, dof = %0.1f' % dof)

    pyplot.legend()

    if 1+g == len(histograms):
      pyplot.xlabel('squared distance')

  pyplot.show()

@main.command
def plot_prior (filename = 'data/gloves/prior.py'):
  'plots priors with staying probabilities'

  from matplotlib import pyplot

  stats = eval(open(filename).read())

  prior = array(stats['prior'])
  stay0 = array(stats['stay0'])
  stay1 = array(stats['stay1'])
  stay2 = array(stats['stay2'])
  stay3 = array(stats['stay3'])
  stay4 = array(stats['stay4'])

  R = 1000 * sqrt(prior)

  stay = [stay0, stay1, stay2, stay3, stay4]
  def prob_to_like (p):
    return p / (1 - p)

  pyplot.figure()
  for i in range(4):
    ax = pyplot.subplot(2,2,1+i)

    if i == 0:
      pyplot.title('Dwell times in %s' % filename)

    X = prob_to_like(stay[i])
    Y = prob_to_like(stay[i+1])
    stay_no_i = stay[:i] + stay[1+i:]
    color = list(zip(*tuple(stay_no_i)))

    c = pyplot.scatter(X, Y, s=R, c=color, alpha=0.3)

    pyplot.xlabel('likelihood of T^%d(x|x)' % (2**i))
    pyplot.ylabel('likelihood of T^%d(x|x)' % (2**(i+1)))

    ax.loglog(basex=2, basey=2)
    ax.set_xlim(min(X) / sqrt(2), max(X) * sqrt(2))
    ax.set_ylim(min(Y) / sqrt(2), max(Y) * sqrt(2))

  pyplot.show()

@main.command
def plot_mix (filename = 'data/gloves/prior.py'):
  'plots mixing rates'

  from matplotlib import pyplot

  stats = eval(open(filename).read())

  prior = array(stats['prior'])
  stay0 = array(stats['stay0'])
  stay1 = array(stats['stay1'])
  stay2 = array(stats['stay2'])
  stay3 = array(stats['stay3'])
  stay4 = array(stats['stay4'])

  R = 1000 * sqrt(prior)

  stay = [stay0, stay1, stay2, stay3, stay4]
  def prob_to_time (p):
    return 1 / (1 - p)

  pyplot.figure()
  for i in range(4):
    ax = pyplot.subplot(2,2,1+i)

    if i == 0:
      pyplot.title('Dwell times in %s' % filename)

    X = prob_to_time(stay[i])
    Y = prob_to_time(stay[i+1])
    stay_no_i = stay[:i] + stay[1+i:]
    color = list(zip(*tuple(stay_no_i)))

    c = pyplot.scatter(X, Y, s=R, c=color, alpha=0.3)

    pyplot.xlabel('dwell times of T^%d(x|x)' % (2**i))
    pyplot.ylabel('dwell times of T^%d(x|x)' % (2**(i+1)))

    ax.loglog(basex=2, basey=2)
    ax.set_xlim(min(X) / sqrt(2), max(X) * sqrt(2))
    ax.set_ylim(min(Y) / sqrt(2), max(Y) * sqrt(2))

  pyplot.show()

if __name__ == '__main__': main.main()

