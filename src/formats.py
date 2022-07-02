
"""
Requirements:
  madplay
  mencoder

TODO test mp4 output
TODO add audio mp3 track to mp4 output
"""

import wave, struct, numpy

def TODO (*args): raise NotImplementedError(*args)

def quote (filename):
  return '"' + filename + '"'
  #return "'" + filename.replace("'","\\'") + "'"

#----( audio files )----------------------------------------------------------

def read_wav (filename, size, width):
  import wave, struct, numpy

  print("reading stereo 16bit wave file from %s" % filename)

  #see http://bugs.python.org/issue4913
  file = wave.Wave_read(filename)
  assert file.getsampwidth() == 2
  assert file.getnchannels() == 2
  assert file.getframerate() == DEFAULT_SAMPLE_RATE

  n = file.getnframes()
  if n < size:
    print('WARNING: %i frames requested, only returning %i' % (size,n))
  size = min(n, size)

  data = file.readframes(size)
  data = struct.unpack("<%uh" % (2 * size), data)
  data = numpy.array(data).astype(numpy.float32)
  data /= 1  << 15
  return data.view(numpy.complex64).reshape((size / width, width))

def write_wav (filename, samples):
  import wave, struct, numpy

  print("writing stereo 16bit wave file to %s" % filename)

  file = wave.Wave_write(filename)
  file.setsamplewidth(2)
  file.setnchannels(2)
  file.setframerate(44100)

  data = (samples.view(numpy.float32) * (1 << 15)).astype(int16)
  data = struct.pack("<%uh" % (2 * size), data)
  file.writeframes(data)

def mp3_to_wav (filename):
  "use madplay to convert mp3 --> wav"
  import os

  if not os.path.exists('temp'):
    os.mkdir('temp')

  cmd = "madplay -q %s -o temp/mp3_to_wav.wav" % quote(filename)
  print(cmd)
  if os.system(cmd):
    raise IOError("madplay failed with command\n%" % cmd)

  return 'temp/mp3_to_wav.wav'

def read_mp3 (filename, *args):
  return read_wav(mp3_to_wav(filename), *args)

class AudioFile:

  def __init__ (self, filename, size_out = 1024, max_size = None):
    "filename must be either .wav or .mp3"
    import wave

    print(( 'opening AudioFile %s with window size %i' % (filename, size_out)
         + (', max_size %i' % max_size if max_size else '') ))

    if filename[-4:] == '.mp3':
      filename = mp3_to_wav(filename)
    assert filename[-4:] == '.wav', "unknown file type: %s" % filename

    self.file = wave.Wave_read(filename)
    assert self.file.getsampwidth() == 2
    assert self.file.getnchannels() == 2
    assert self.file.getframerate() == DEFAULT_SAMPLE_RATE

    self.size = self.file.getnframes()
    if max_size is not None:
      self.size = min(self.size, max_size)
    self.position = 0
    self.size_out = size_out

  def __read_data (self, size):
    data = self.file.readframes(size)
    data = struct.unpack("<%uh" % (2 * size), data)
    data = numpy.array(data).astype(numpy.float32)
    data /= 1 << 15
    return data.view(numpy.complex64)

  def stopped (self): return self.position >= self.size

  def read (self, sound_out=None):
    "writes zero-padded data; returns true until EOF"

    if sound_out is None: return self.size_out

    assert isinstance(sound_out, numpy.ndarray), \
        "sound_out must be a numpy.ndarray, but was was a %s" \
        % sound_out.__class__
    assert sound_out.dtype == numpy.complex64, \
        "sound_out must have dtype complex64, but had dtype %s" \
        % sound_out.dtype
    assert sound_out.shape == (self.size_out,), \
        "sound_out must have shape (%i,) but had shape %s" \
        % (self.size_out, sound_out.shape)

    if self.stopped(): return False

    size = min(self.size_out, self.size - self.position)
    sound_out[:size] = self.__read_data(size)
    sound_out[size:] = 0
    self.position += size

    return True

#----( raster images )--------------------------------------------------------

#see http://www.cygwin.com/ml/cygwin/2007-01/msg00498.html
# namely, on cygwin use the command
# $ rebase -b 0x1000000000 /bin/tk84.dll
# before trying to build PIL via
# $ python setup.py build

def write_color_image (image, filename, tol=1e-8, transpose=True):
  import Image, numpy

  assert image.min() > -tol, "image out of bounds: min = %g" % image.min()
  assert image.max() < 1+tol, "image out of bounds: max = %g" % image.max()

  colored = numpy.zeros(image.shape + (3,), dtype = numpy.uint8)

  #colormap via quartic bernstein polynomials
  #see http://en.wikipedia.org/wiki/Bernstein_polynomial
  x = image
  w = x**4
  colored[:,:,0] = 255 * (w + 4 * x**3 * (1-x) )
  colored[:,:,1] = 255 * (w + 6 * x**2 * (1-x)**2 )
  colored[:,:,2] = 255 * (w + 4 * x * (1-x)**3 )

  im = Image.fromarray(colored)
  if transpose:
    im = im.transpose(Image.ROTATE_90)
  print("saving color image to %s" % filename)
  im.save(filename)
  return im

def write_gray_image (image, filename, tol=1e-8, transpose=True):
  import Image, numpy

  print("writing image to %s" % filename)
  assert image.min() > -tol, "image out of bounds: min = %g" % image.min()
  assert image.max() < 1+tol, "image out of bounds: max = %g" % image.max()

  image *= 255

  im = Image.fromarray(image.astype(numpy.uint8))
  if transpose:
    im = im.transpose(Image.ROTATE_90)
  print("saving gray image to %s" % filename)
  im.save(filename)
  return im

def write_image (image, filename, tol=1e-8, transpose=True, color=True):
  if color: return write_color_image(image, filename, tol, transpose)
  else:     return write_gray_image (image, filename, tol, transpose)

def concat_images (file_prefix, filetype='jpg', mode='RGB', width=None):
  import Image, re, os
  file_pattern = re.compile('^%s.+\.%s$' % (re.escape(file_prefix), filetype))
  filenames = [f for f in os.listdir('.') if file_pattern.match(f)]
  filenames.sort()
  print('concatenating images:\n  ' + '\n  '.join(filenames))
  h,w = Image.open(filenames[0]).size
  n = len(filenames)
  if width is None: width = n * w
  print('total image size: %i x %i' % (width,h))
  result = Image.new('RGB', (width, h))
  for i,filename in enumerate(filenames):
    result.paste(Image.open(filename), (i * w, 0))
  filename = file_prefix + '.' + filetype
  print('saving composite image ' + filename)
  result.save(filename)

#----( mp4 videos )-----------------------------------------------------------

def images_to_mp4 (filename, framerate=24, filetype='jpg'):
  """Calls mencoder to assemble images into an mp4 file.
  (get mencoder at http://www.mplayerhq.hu/design7/dload.html)

  Inputs:
    'filename'              Stem for input/output, eg myanimation
    'filetype'              Defaults to 'jpg', but also try 'png'
    <fielname>*.<filetype>  Input frames in working directory

  Outputs:
    myanimation.mp4   Output animation, in working directory
  """
  import os

  cmd = ' '.join([
    'mencoder',
    'mf://%s*.%s' % (filename, filetype),
    '-mf fps=%i:type=%s' % (framerate, filetype),
    #'-oac mp3lame -lamepots abr:br=128 -srate 44100',
    '-ovc x264',
    '-of lavf -o %s.mp4' % filename,
  ])
  print(cmd)

  if os.system(cmd):
    raise IOError("mencoder failed with command\n%" % cmd)

