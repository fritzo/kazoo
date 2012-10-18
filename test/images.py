#!/usr/bin/python

import Image
from numpy import *
from scipy import *
import main
import os

def soft_clip (x):
  y = arctan(pi * x) / pi + 0.5
  y -= y.min()
  y /= y.max()
  return y

def quad_kernel (x):
  x = fabs(x)
  return ( (x <= 0.5) * (0.75 - x**2)
         + (0.5 < x) * (x <= 1.5) * 0.5 * (1.5 - x) ** 2
         )

@main.command
def plot_kernel (*scales):
  'Plots sum of quadratic kernels at specified scales'

  from matplotlib import pyplot

  T = arange(-2,2,0.01)
  K = 0 * T
  for scale in scales:
    scale = float(scale)
    K += quad_kernel(T / scale) / scale
  pyplot.plot(T,K)
  pyplot.show()

@main.command
def hdr (filename, extent = 0.75, gamma = 0.5):
  'Applies fake HDR transform to image'
  extent = float(extent)
  gamma = float(gamma)

  import kazoo
  K = kazoo.transforms

  im = Image.open(filename)
  rgb = misc.fromimage(im)
  J,I = im.size
  I8 = (I+7)/8*8
  J8 = (J+7)/8*8

  def get_channel (i):
    c0 = K.Reals(I8,J8)
    c0[:] = -1
    noisy = rgb[:,:,i] + random.random((I,J)) - 0.5
    c0[:I,:J] += 2.0 / 255 * noisy

    for i in range(I8-I): c0[I+i,:] = c0[I+i-1,:]
    for j in range(J8-J): c0[:,J+j] = c0[:,J+j-1]

    c1 = K.Reals(I8,J8)
    c1[:] = c0

    return c0,c1

  r0,r1= get_channel(0)
  g0,g1= get_channel(1)
  b0,b1= get_channel(2)

  K.hdr_real_color(I8,J8, r1,g1,b1)

  def set_channel (i, c0,c1):
    c = (1 - extent) * c0[:I,:J] + extent * c1[:I,:J]
    rgb[:,:,i] = 255 * soft_clip(gamma * c)
  set_channel(0,r0,r1)
  set_channel(1,g0,g1)
  set_channel(2,b0,b1)

  im = misc.toimage(rgb)
  filename = filename.rstrip('.png')
  im.save(filename + '.hdr.png')
  #im.show()

if __name__ == '__main__': main.main()

