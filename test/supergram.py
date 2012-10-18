#!/usr/bin/python

import kazoo as K
from kazoo import formats
from numpy import *
import main

@main.command
def show_file (size_exponent = 10, time_exponent = 2, freq_exponent = 3):
  'Shows supergram of an audio file'

  s = K.Supergram(size_exponent, time_exponent, freq_exponent)
  size = s.size
  small_size = s.small_size
  large_size = s.super_size
  length = s.super_size #arbitrary

  print "reading sound file"
  sound = formats.read_wav('test.wav', small_size*length, small_size)
  image = K.transforms.Reals(length, large_size)

  print "transforming data"
  for i in range(length):
    s.transform_fwd(sound[i,:], image[i,:])
  #del sound

  print "saving image"
  image = K.util.energy_to_loudness(image + 1e-5)
  formats.write_image(image, 'test.png').show()

def show_signal (size_exponent, factor_exponent, signal, filename):
  size   = 1 << size_exponent
  factor = 1 << factor_exponent
  length_windows = size / 2
  length_samples = size * length_windows

  print "creating signal"
  input = K.transforms.Complexes(length_samples)

  for i in range(length_samples):
    t = (0.5 + i) / length_samples
    input[i] = signal(t)
    #print signal(t), #DEBUG

  print "transforming signal"
  s = K.Supergram(size_exponent, factor_exponent)

  length_frames = length_samples / s.small_size
  input = input.reshape((length_frames, s.small_size))
  output = K.transforms.Reals(length_frames, s.super_size)

  for i in range(length_frames):
    s.transform_fwd(input[i,:], output[i,:])

  print "writing supergram to %s" % filename

  #XXX this fails due to NaNs and Infs in output XXX
  #output[~isfinite(output)] = 0 #DEBUG

  output = K.util.energy_to_loudness(output, 0.1)
  formats.write_image(output, filename).show()

@main.command
def show_chirp (size_exponent = 7, factor_exponent = 5):
  'Shows supergram of a chirp'

  fft_size = 1 << size_exponent
  nyquist_freq = (fft_size / 2)**2
  sigma = 0.25
  def signal (t):
    return t * (1-t) * sin( 6 * t * (t**2/3 - t/2) * nyquist_freq * pi )

  show_signal(size_exponent, factor_exponent, signal, 'super.png')

@main.command
def show_impulse (size_exponent = 7, factor_exponent = 5):
  'Shows supergram of an impulse'
  fft_size = 1 << size_exponent
  nyquist_freq = (fft_size / 2)**2
  sigma = 0.25
  def signal (t):
    return exp(-((t-0.5) * nyquist_freq)**2)

  show_signal(size_exponent, factor_exponent, signal, 'super.png')

@main.command
def show_clicks (size_exponent = 7, factor_exponent = 5):
  'Shows supergram of clicks'

  fft_size = 1 << size_exponent
  nyquist_freq = (fft_size / 2)**2
  sigma = 0.25
  def signal (t):
    return fmod(2 * t, 1) - t

  show_signal(size_exponent, factor_exponent, signal, 'super.png')

if __name__ == '__main__': main.main()

