
"""
Invertible time-series transforms.

  see doc/framework.text

Belief propagation.
  Often transforms have state depending on input and output.
  In the backward phase, estimated streams can be Bayesian-fused downstream,
  so that actual fused output differs from suggested output.
  To correctly update local state with fused output,
  an extra argument previous_value is often added to transform_bwd.

TODO run each of Multigram's spectrograms in a separate thread
"""

import math, numpy
from _kazoo import *
from _transforms import *
import formats
from formats import AudioFile

#----( simple transforms )----------------------------------------------------

class Wire:
  "copies input to output"
  #def transform_fwd (self, data_in, data_out): data_out[:] = data_in
  #def transform_bwd (self, data_in, data_out): data_out[:] = data_in
  def transform     (self, data_in, data_out): data_out[:] = data_in

class Null:
  "ignores input"
  def write (self, data_in): pass

class Splitter:
  "copies input to two outputs"
  def transform (self, data_in, (left_out, right_out)):
    left_out[:] = data_in
    right_out[:] = data_in

class Mixer:
  "mixes results of multiple inputs"
  def __init__ (self, coeffs = None):
    self.coeffs = coeffs
  def transform (self, multi_in, single_out):
    single_out[:] = 0
    if self.coeffs is None:
      for data_in in multi_in:
        single_out += data_in
      single_out /= len(multi_in)
    else:
      for coeff,data_in in zip(self.coeffs,multi_in):
        data_in *= coeff
        single_out += data_in

class Concat:
  "concatenates multiple inputs to single output"
  def transform_fwd (self, inputs, output):
    offset = 0
    for i in inputs:
      size = len(i)
      output[offset:offset+size] = i
      offset += size
  def transform_bwd (self, input, outputs):
    offset = 0
    for o in outputs:
      size = len(o)
      o[:] = input[offset:offset+size]
      offset += size

class Concat2D:
  "reshapes and concatenates multiple inputs to single 2D output"
  def __init__ (self, *shapes_in):
    self.shapes_in = shapes_in
    self.shape_out = (sum(x for x,y in shapes),
                      max(y for x,y in shapes))

  def transform (self, inputs, output):
    o = output.reshape(self.shape_out)
    offset = 0
    for i,shape in zip(inputs, self.shapes_in):
      w,h = shape
      o[offset:offset+w,:h] = i.reshape(shape)
      offset += w

class Recorder:
  "saves data in a 2D array"
  def __init__ (self, size_in, num_frames=1, allocator = Reals):
    print 'building Recorder with %i frames' % num_frames
    self.size_in = size_in
    self.num_frames = num_frames
    self.data = allocator(num_frames, size_in)
    self.__position = 0

  def write (self, data_in=None):

    if data_in is None: return self.size_in

    assert isinstance(data_in, numpy.ndarray), \
        "data_in must be a numpy.ndarray, but was was a %s" \
        % data_in.__class__
    assert data_in.dtype == self.data.dtype, \
        "data_in must have dtype %s, but had dtype %s" \
        % (self.data.dtype, data_in.dtype)
    assert data_in.shape == (self.size_in,), \
        "data_in must have shape (%i,) but had shape %s" \
        % (self.size_in, data_in.shape)

    self.data[self.__position,:] = data_in
    self.__position = (1 + self.__position) % self.num_frames

#----( formatting tools )-----------------------------------------------------

class ImageBuffer:
  "buffers [0,1]-valued data, and saves to png images"
  def __init__ (self, file_prefix, size_in,
                      num_frames = None,
                      color = True,
                      filetype = 'jpg'):
    self.__size_in = int(size_in)
    self.__num_frames = int(size_in if num_frames is None else num_frames)
    assert self.size_in > 0
    assert self.num_frames > 0

    self.__data = Reals(self.num_frames, self.size_in)
    self.__position = 0
    self.__image_num = 0
    self.__width = 0

    import os
    file_prefix = os.path.split(file_prefix)[-1]
    self.__file_prefix = file_prefix
    self.__filetype = filetype
    self.__filename = file_prefix + '%06i.' + filetype
    self.__color = color

  # diagnostics
  @property
  def size_in (self): return self.__size_in
  @property
  def num_frames (self): return self.__num_frames

  def __write (self):
    "writes current data to file"
    import formats
    filename = self.__filename % self.__image_num
    self.__image_num += 1
    self.__width += self.__position
    formats.write_image(self.__data, filename, color=self.__color)

  def flush (self):
    "if any data remains, flushes zero-padded data to a file"
    if self.__position > 0:
      if self.__position < self.num_frames:
        print "flushing ImageBuffer (%i/%i full)" \
            % (self.__position, self.num_frames)
        self.__data[self.__position:,:] = 0
      self.__write()
      self.__position = 0

  def write (self, data_in=None):

    if data_in is None: return self.size_in

    assert isinstance(data_in, numpy.ndarray), \
        "data_in must be a numpy.ndarray, but was was a %s" \
        % data_in.__class__
    assert data_in.dtype == numpy.float32, \
        "data_in must have dtype numpy.float32, but had dtype %s" \
        % data_in.dtype
    assert data_in.shape == (self.size_in,), \
        "data_in must have shape (%i,) but had shape %s" \
        % (self.size_in, data_in.shape)

    self.__data[self.__position, :] = data_in
    self.__position += 1
    if self.__position == self.num_frames:
      self.__write()
      self.__position = 0

  def assemble (self):
    self.flush()
    formats.concat_images(self.__file_prefix,
                         self.__filetype,
                         width = self.__width)

#----( higher-level transforms )----------------------------------------------

def Bernstein_polynomials (number, size, sharpness=1.0, mapping = None):
  i = numpy.arange(size, dtype=numpy.float32)
  t = ((0.5 + i) / size)
  if mapping is not None:
    t = mapping(t)
  def binomial_coeff (n,m):
    return ( numpy.product(range(1+m,1+n))
           / numpy.product(range(1,1+n-m)) )
  s = 1 - t
  basis = [ binomial_coeff(number,n) * pow(t,n) * pow(s,number-n)
            for n in range(number+1) ]
  if sharpness != 1.0:
    basis = [x**sharpness for x in basis]
    total = sum(basis)
    for x in basis: x /= total
  return basis

class Multigram:
  """\
  Multigram(small_exponent,
            large_exponent,
            sample_rate = DEFAULT_SAMPLE_RATE) -> object

  Multiscale supersampled time-frequency reassigned spectrogram.
  The number of scales is 1 + large_exponent - small_exponent.\
  """

  def __init__ (self, small_exponent,
                      large_exponent,
                      sample_rate = None,
                      blend_gamma = 3.0,
                      sharpness = 2.0,
                      single_thread = True):

    assert isinstance(small_exponent, int)
    assert isinstance(large_exponent, int)
    assert MIN_EXPONENT <= small_exponent
    assert small_exponent <= large_exponent
    assert large_exponent <= MAX_EXPONENT

    if sample_rate is None: sample_rate = DEFAULT_SAMPLE_RATE

    print "building Multigram(%i, %i, %g)" % (small_exponent,
                                              large_exponent,
                                              sample_rate)

    self.__small_exponent = small_exponent
    self.__large_exponent = large_exponent
    self.__sample_rate = float(sample_rate)
    self.__size_in = 1 << (small_exponent - 1)
    self.__size_out = 1 << large_exponent

    self.__supergrams = []
    for size_exponent in range(small_exponent, 1+large_exponent):
      time_exponent = 1 + size_exponent - small_exponent
      freq_exponent = 1 + large_exponent - size_exponent

      supergram = Supergram(size_exponent,
                            time_exponent,
                            freq_exponent,
                            self.sample_rate)

      assert supergram.small_size == self.size_in
      assert supergram.super_size == self.size_out
      self.__supergrams.append(supergram)

    # validate
    sizes = [s.transform_fwd() for s in self.__supergrams]
    for size in sizes:
      assert size == sizes[0], \
          "supergram sizes disagree: %s != %s" % (size, sizes[0])

    # Note: There are three time delays in the Supergram transformation:
    #   max delay = width for Fourier transform time window
    #             + width for reassignment accumulator
    #             + ??? WTF ???
    width = large_exponent - small_exponent
    max_delay = 3 << width
    self.__delays = [max_delay - (3 << w) for w in range(width+1)]
    self.__history = [Complexes(self.size_in) for _ in range(max_delay)]

    # Note: these ad hoc coefficients do not scale correctly with exponents
    self.__coeffs = Bernstein_polynomials(len(self), self.size_out,
                        sharpness = sharpness,
                        mapping = (lambda t: 1 - (1-t) ** blend_gamma))
    self.__coeffs.reverse()

    # behavior depends on number of threads
    self.__single_thread = single_thread
    if self.__single_thread:
      self.__super_part = Reals(self.size_out)
    else:
      import network
      self.__super_parts = [Reals(self.size_out) for _ in sizes]
      self.__switches = [network.Toggle() for _ in sizes]
      for s,t,d,x in zip(self.__switches,
                         self.__supergrams,
                         self.__delays,
                         self.__super_parts):
        def thread_loop ():
          network.DEBUG("starting Supergram thread")
          try:
            while network.threads_alive():
              s.wait(False)
              if network.threads_alive():
                network.DEBUG("running Supergram")
                t.transform_fwd(self.__history[d], x)
              s.toggle()
          except Exception, e:
            network.stop_threads()
            raise e
          finally:
            network.DEBUG("stopping Supergram thread")
        network.new_thread(thread_loop)

  # scale conversions
  def freq_scale (self, *args): return self.__supergrams[0].freq_scale(*args)
  freq_scale.__doc__ = Supergram.freq_scale.__doc__

  def pitch_scale (self, *args): return self.__supergrams[0].pitch_scale(*args)
  pitch_scale.__doc__ = Supergram.pitch_scale.__doc__

  # diagnostics
  def __len__ (self): return len(self.__supergrams)
  @property
  def size_in (self): return self.__size_in
  @property
  def size_out (self): return self.__size_out
  @property
  def sample_rate (self): return self.__sample_rate

  # history management
  def __add_history (self, time_in):
    "cyclic shift & copy"
    self.__history = [self.__history[-1]] + self.__history[:-1]
    self.__history[0][:] = time_in

  # these can operate in parallel
  def transform_fwd (self, time_in, super_out):
    "Averages over multiscale Supergrams to transform : time --> freq"

    self.__add_history(time_in)

    # behavior depends on number of threads
    if self.__single_thread:
      super_out[:] = 0
      for t,d,c in zip(self.__supergrams, self.__delays, self.__coeffs):
        t.transform_fwd(self.__history[d], self.__super_part)
        super_out += c * self.__super_part

    else:
      for s in self.__switches:
        s.toggle()
      super_out[:] = 0
      for s,c,x in zip(self.__switches, self.__coeffs, self.__super_parts):
        s.wait(True)
        super_out += c * x

  def transform_bwd (self, super_in, time_out):
    "Uses fastest Supergram to transform : freq --> time"

    self.__supergrams[0].transform_bwd(super_in, time_out)

