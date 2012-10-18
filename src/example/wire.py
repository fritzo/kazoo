
import kazoo as K
import numpy

def spectrogram_wire (num_seconds = 5):
  exponent = 10
  size = 1 << exponent

  audio = K.Audio(size)
  spec  = K.Spectrogram(exponent)
  wire  = K.Wire()

  sound1   = K.Complexes()
  sound2   = K.Complexes()
  freq_in  = K.Reals()
  freq_out = K.Reals()

  audio.reading(sound1)
  spec.stream_fwd(sound1, freq_out)
  wire.stream(freq_in, freq_out)
  spec.stream_bwd(freq_in, sound2)
  audio.writing(sound2)

  K.validate()

  audio.run_until_input()

def supergram_wire (size_exponent = 9, factor_exponent = 2):
  
  spec  = K.Supergram(size_exponent, factor_exponent)
  audio = K.Audio(spec.small_size)

  sound1 = K.Complexes()
  sound2 = K.Complexes()
  energy = K.Reals()

  audio.reading(sound1)
  spec.stream_fwd(sound1, energy)
  spec.stream_bwd(energy, sound2)
  audio.writing(sound2)

  K.validate()

  audio.run_until_input()

def pitch_bend (size_exponent = 10, factor_exponent = 2, factor=0.5):

  spec  = K.Supergram(size_exponent, factor_exponent)
  audio = K.Audio(spec.small_size)

  size = spec.super_size
  fun = K.Reals(size)
  for i in range(size):
    fun[i] = (0.5 + i) / size * factor
  bend = K.Spline(size, size, fun)

  sound1 = K.Complexes()
  sound2 = K.Complexes()
  energy = K.Reals()
  bent   = K.Reals()

  audio.reading(sound1)
  spec.stream_fwd(sound1, energy)
  bend.stream_fwd(energy, bent)
  spec.stream_bwd(bent, sound2)
  audio.writing(sound2)

  K.validate()

  audio.run_until_input()

def octave_lower (size = (1<<10)):

  audio = K.Audio(size)
  lower = K.OctaveLower(size)

  sound1 = K.Complexes()
  sound2 = K.Complexes()

  audio.reading(sound1)
  lower.stream(sound1, sound2)
  audio.writing(sound2)

  K.validate()

  audio.run_until_input()

if __name__ == '__main__':
  #spectrogram_wire()
  #supergram_wire()
  #pitch_bend()
  octave_lower()

