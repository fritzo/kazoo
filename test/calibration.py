#!/usr/bin/python

import sys, os, re
from numpy import *
from matplotlib import pyplot
import optim
import main

def lognormal_prob (x, mean = 0, sigma = 1):
  return exp(-((mean-log(x)) / sigma) ** 2 / 2)

def read_points (filename):
  file = open(filename)
  points = []
  for line in file:
    try:
      x,y,z = line.split()
      points.append((float(x), float(y), float(z) ** 0.5))
    except ValueError:
      pass
  return points

def read_edges (filename):
  file = open(filename)
  edges = []
  for line in file:
    try:
      i,j,w,v = line.split()
      edges.append(Edge(int(i), int(j), float(w) / 10, bool(int(v))))
    except ValueError:
      pass
  return edges

class Edge:
  def __init__ (self, i,j, weight, vertical):
    self.i = i
    self.j = j
    self.weight = weight
    self.vertical = vertical

def find_edges (points):
  print('finding edges among %i points' % len(points))
  size = len(points)

  N = {}
  E = {}
  S = {}
  W = {}

  def update (D,i1,i2,like12,weight):
    try:
      i3,like13,_ = D[i1]
      if like12 > like13:
        D[i1] = i2,like12,weight
    except KeyError:
      D[i1] = i2,like12,weight

  for i1,(x1,y1,z1) in enumerate(points):
    for i2,(x2,y2,z2) in enumerate(points[:i1]):
      dx = x2 - x1
      dy = y2 - y1

      r2 = dx ** 2 + dy ** 2
      weight = z1 * z2 * lognormal_prob(r2)
      like = weight * (dy ** 2 - dx ** 2) / r2 ** 1.5

      vertical = like > 0
      if vertical:
        if dy > 0:
          update(N,i1,i2,like,weight)
          update(S,i2,i1,like,weight)
        else:
          update(S,i1,i2,like,weight)
          update(N,i2,i1,like,weight)
      else:
        like *= -1
        if dx > 0:
          update(E,i1,i2,like,weight)
          update(W,i2,i1,like,weight)
        else:
          update(W,i1,i2,like,weight)
          update(E,i2,i1,like,weight)

  edges = {}
  def add_edges (D, vertical, reverse):
    for i1,(i2,_,weight) in D.items():
      if reverse:
        i1,i2 = i2,i1
      try:
        edges[i1,i2].weight += weight
      except KeyError:
        edges[i1,i2] = Edge(i1,i2, weight, vertical)

  add_edges(N, True, False)
  add_edges(E, False, False)
  add_edges(S, True, True)
  add_edges(W, False, True)

  print(' found %i edges' % len(edges))

  return list(edges.values())

class Calibration:
  def __init__ (self):
    self.param = array([ 0.0, 0.0, 0.2, 0.2, 0.0, 1.5, 2.0 ])
    self.sigma = array([ 0.3, 0.3, 0.1, 0.1, 0.5, 1.0, 2.0 ])
    self.p0 = 0
    self.t0 = 0

  def __call__ (self, x,y, param = None):
    if param is None:
      param = self.param
    u0,v0,k2,k4,theta,p1,t1 = param

    u = x - u0
    v = y - v0

    r2 = u ** 2 + v ** 2
    s = 1 + r2 * (k2 + r2 * k4)

    x = u0 + s * u
    y = v0 + s * v

    c = 1 - theta ** 2 / 2
    s = theta

    rot_x = c * x + s * y;
    rot_y = c * y - s * x;

    x = self.p0 + exp(p1) * rot_x;
    y = self.t0 + exp(t1) * rot_y;

    return x,y

  def map_points (self, points, param = None):
    return [self(x,y,param)+(z,) for (x,y,z) in points]

  def fit (self, verts, em_iters = 1, nls_iters = 1):
    print('fitting grid to %i points' % len(verts))
    I = len(self.param)
    points = self.map_points(verts)

    def f (param):
      ps = self.map_points(verts, param)
      result = zeros(2 * E + I)
      for j,e in enumerate(edges):
        p0 = ps[e.i]
        p1 = ps[e.j]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if e.vertical:
          dy -= 1.0
        else:
          dx -= 1.0
        result[2*j+0] = dx * e.weight
        result[2*j+1] = dy * e.weight

      result[2*E:] = (param - self.param) / self.sigma
      return result

    for em_iter in range(em_iters):
      edges = find_edges(points)
      E = len(edges)

      cov = diag(self.sigma)
      optim.nonlinear_least_squares(f, self.param, cov, nls_iters)

  def align (self, verts):
    print('aligning grid')
    points = self.map_points(verts)
    phase = zeros(4)
    total = 0.0
    for x,y,z in points:
      phase[0] += z * cos(2 * pi * x)
      phase[1] += z * sin(2 * pi * x)
      phase[2] += z * cos(2 * pi * y)
      phase[3] += z * sin(2 * pi * y)
      total += z
    phase /= total

    self.p0 -= arctan2(phase[1], phase[0]) / (2 * pi)
    self.t0 -= arctan2(phase[3], phase[2]) / (2 * pi)

#----( main commands )--------------------------------------------------------

@main.command
def fit (filename = '../grid.text', em_iters = 0, nls_iters = 1, align = False):
  'Run calibration algorithm, and display calibrated gridpoints'
  verts = read_points(filename)

  cal = Calibration()

  if em_iters and nls_iters:
    cal.fit(verts, em_iters, nls_iters)

  align = bool(align)
  if align:
    cal.align(verts)

  points = cal.map_points(verts)

  edges = find_edges(points)

  pyplot.figure()
  pyplot.axis('equal')

  X = [x for x,y,z in points]
  Y = [y for x,y,z in points]
  pyplot.plot(X,Y, 'k.')

  for e in edges:
    pyplot.plot(
        [points[e.i][0], points[e.j][0]],
        [points[e.i][1], points[e.j][1]],
        'r-' if e.vertical else 'g-',
        linewidth = e.weight)

  pyplot.show()

@main.command
def show (points_filename, edges_filename = ''):
  'Display calibration solution'

  pyplot.figure()
  pyplot.axis('equal')

  points = read_points(points_filename)
  X = [x for x,y,z in points]
  Y = [y for x,y,z in points]
  #pyplot.plot(X,Y, 'k.')
  z_scale = 1 / max(z for x,y,z in points)
  for x,y,z in points:
    w = 1 - z_scale * z
    pyplot.plot([x],[y],
        linestyle = ' ',
        marker = 'o',
        color = (w,w,w))

  if edges_filename:
    edges = read_edges(edges_filename)
    w_scale = 4 / max(e.weight for e in edges)
    for e in edges:
      pyplot.plot(
          [points[e.i][0], points[e.j][0]],
          [points[e.i][1], points[e.j][1]],
          'r-' if e.vertical else 'g-',
          linewidth = w_scale * e.weight)

  pyplot.title( '%i vertical (red) + %i horizontal (green)'
              % ( sum(e.vertical for e in edges),
                  sum(not e.vertical for e in edges) ) )
  pyplot.show()

if __name__ == '__main__': main.main()

