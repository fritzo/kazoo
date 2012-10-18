
import kazoo as K
from kazoo import formats

def record_spectrogram (exponent = 10):

  width = 1 << exponent
  height = width
  size = height * width

  sound = K.Complexes(width)
  image = K.Reals(height, width/2)

  s = K.Spectrogram(exponent)
  a = K.Audio(width)

  time = size * a.rate / 60.0
  print "recording & transforming sound for %g seconds..." % time
  a.start()
  for i in range(height):
    a.read(sound)
    s.transform_fwd(sound,image[i,:])
    a.write(sound) #HACK
  a.stop()

  print "saving image"
  image = formats.energy_to_loudness(image)
  formats.write_image(image, 'test.png')

if __name__ == '__main__':
  record_spectrogram()

