#!/usr/bin/python

import kazoo.transforms as K
from numpy import *
from matplotlib import pyplot
import main

#----( wide bandpass filter )-------------------------------------------------

@main.command
def plot_fourier (width = 0.5):
  'Plots fourier energy envelopes for octave-pass filter'
  width = float(width)

  def lognormal (w):
    return exp(-(log(w) / width) ** 2 / 2)
  def rational (w):
    return 2 / (1 + ((w-1) / width)**2) / (1 + w**-2)

  W = arange(-1,3,0.01)
  pyplot.plot(W, lognormal(W), label='lognormal')
  pyplot.plot(W, rational(W), label='rational')
  pyplot.xlabel('frequency')
  pyplot.ylabel('energy')
  pyplot.legend()
  pyplot.title('Octave-pass energy envelopes (fourier)')
  pyplot.show()

@main.command
def plot_z_trans (w0 = 0.1, width = 0.5):
  'Plots z-transform energy envelopes for octave-pass filter'
  w0 = float(w0)
  width = float(width)

  def lognormal (w):
    return exp(-(log(w/w0) / width) ** 2 / 2)
  def H (z):
    return ( ((z - 1) / (z - exp(-w0))) ** 2
           / (z - exp((-width + 1.j) * w0))
           / (z - exp((-width - 1.j) * w0))
           )
  def h (w):
    return abs(H(exp(1.j * w)))
  def rational (w):
    return h(w) / h(w0)

  W = arange(-1,3,0.01) * w0
  pyplot.plot(W, lognormal(W), label='lognormal')
  pyplot.plot(W, rational(W), label='rational')
  pyplot.xlabel('frequency')
  pyplot.ylabel('energy')
  pyplot.legend()
  pyplot.title('Octave-pass energy envelopes (z-transform)')
  pyplot.show()

#----( shepard tones )--------------------------------------------------------

def shepard_steps (num_octaves):
  '1 2 1 4 1 2 1 8 ...'

  size = 2 << num_octaves
  sound = K.Complexes(size)
  for t in range(size):
    sound[t] = ((t ^ (t + 1)) + 1) / 2

  return sound

def interpolate (t, seq):
  assert 0 <= t
  assert t <= len(seq)
  t0 = int(t - floor(t))
  t1 = 1 + t0
  return (t1 - t) * seq[t0] + (t - t0) * seq[t1]

def shepard_steps_balanced (num_octaves):
  '1 -2 1 4 1 -2 1 -8 ...'

  size = 2 << num_octaves
  sound = K.Complexes(size)
  for t in range(size):
    s = ((t ^ (t + 1)) + 1) / 2
    if s & 0xaaaaaaaa:
      s *= -1
    sound[t] = s

  return sound

@main.command
def play_shepard_raw (num_octaves = 12, cycles = 100):
  'Plays raw Shepard steps'

  sound = shepard_steps(num_octaves)

  audio = K.Audio(sound.size, reading = False)
  audio.start()
  for i in range(cycles):
    audio.write(sound)
    print('.', end=' ')
  audio.stop()

@main.command
def play_shepard_balanced (num_octaves = 12, cycles = 100):
  'Plays balanced Shepard steps'

  sound = shepard_steps_balanced(num_octaves)

  audio = K.Audio(sound.size, reading = False)
  audio.start()
  for i in range(cycles):
    audio.write(sound)
    print('.', end=' ')
  audio.stop()

class Filter:
  def __init__ (self, freq, bandwidth = 0.25):
    w0 = 2.0 * pi / 4800 * freq
    z0 = exp(1.0j * w0)

    self.z = exp((1.0j - bandwidth) * w0)
    self.a = 1.0 / (z0 - self.z)

    self.state = 0.0j

  def __call__ (self, actuate):
    self.state = actuate + self.z * self.state
    return self.a * self.state

def shepard_add (fine_freq = 440, coarse_freq = 100, size = 1<<10):
  coarse_pitch = log(coarse_freq) / log(2)
  fine_pitch = log(fine_freq) / log(2)

  fine_pitch -= floor(fine_pitch - coarse_pitch)
  assert coarse_pitch <= fine_pitch
  assert fine_pitch < 1 + coarse_pitch

  pitch0 = floor(fine_pitch)
  pitch1 = floor(fine_pitch + 1)
  energy0 = pitch1 - fine_pitch
  energy1 = fine_pitch - pitch0
  freq0 = 2 ** pitch0
  freq1 = 2 ** pitch1

  T = 1.0j * pi * array(list(range(size)))
  sound = K.Complexes(size)
  sound += energy0 * exp(pitch0)
  sound += energy1 * exp(pitch1)

  return sound

@main.command
def play_shepard_add (cycles = 100):
  'Plays additively synthesized Shepard tone'

  sound = shepard_add(100, 220, 1<<12)

  audio = K.Audio(sound.size, reading = False)
  audio.start()
  for i in range(cycles):
    audio.write(sound)
  audio.stop()

#----( main )-----------------------------------------------------------------

if __name__ == '__main__': main.main()

