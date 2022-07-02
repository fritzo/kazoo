#!/usr/bin/python

import main
from numpy import *
from matplotlib import pyplot

def smooth (x, radius, iters = 1):
  if iters == 0:
    return x
  ix = cumsum(x, 0)
  sx = (ix[radius:,:] - ix[:-radius]) / radius
  return smooth(sx, radius, iters - 1)

#----( beat functions )-------------------------------------------------------

def beat_fun_exp (angle):
  return exp(sin(angle))

def beat_fun_sqr (angle):
  'a factor of 2/3 allows the largest 1:1 stair to be about 1 octave wide'
  return 2.0 / 3.0 * (sin(angle) + 1) ** 2
def beat_fun_sqr_weak (angle):
  'a factor of 1/2 allows the largest 1:1 stair to be about 1/2 octave wide'
  return 0.5 * (sin(angle) + 1) ** 2
def beat_fun_sqr_weaker (angle):
  'a factor of 1/3 allows the largest 1:1 stair to be about 1/4 octave wide'
  return 1.0 / 3.0 * (sin(angle) + 1) ** 2

def beat_fun_pow (exponent):
  def fun (angle):
    return 2 * sqrt(exponent) * (sin(angle) / 2 + 0.5) ** exponent
  return fun

def beat_fun_box (angle, radius):
  'box beat functions have many tongues'
  return abs(angle - pi / 2) < radius
def beat_fun_box_highpass (angle, radius):
  'box beat functions have many tongues'
  return (abs(angle - pi / 2) < radius) - (radius / pi)
def beat_fun_box_strong (strength = 10):
  'used to determine strength for given tongue width'
  def fun (angle):
    return sqrt(strength) * (abs(angle - pi / 2) < pi / 4)
  return fun

'''
Empirically, each strength unit absorbs
  ~1/2 semitone = log(2) / 24 radius around 1:1
'''
beat_fun_box_tempo = beat_fun_box_strong(10)
beat_fun_box_pitch = beat_fun_box_strong(5)
beat_fun = beat_fun_box_pitch

'''
Pitch and tempo have different acuity, hence different radius
'''
tempo_radius_circ = pi * 3 / 8
pitch_radius_circ = pi * 1 / 6
def beat_fun_circ (
    angle1,
    angle2,
    radius = pitch_radius_circ,
    strength = 1.5,
    ):
  '''
  Circular beat regions have even more tongues than square regions,
  but are not separable
  '''
  return ( strength
         / radius
         * ( (angle1 - radius) ** 2
           + (angle2 - radius) ** 2
           < radius ** 2
           )
         )

'''
The piecewise biquadratic beta function acts much like the circular beat fun,
but is separable and continuous.
'''
def beat_fun_biquad (t):
  return (t < 2) * (1 - (1 - t) ** 2)

tempo_radius_biquad = pi / 3
pitch_radius_biquad = pi / 7

def beat_fun_pair_biquad (
    angle1,
    angle2,
    radius = pitch_radius_biquad,
    strength = 2.5,
    ):
  return ( strength
         / radius
         * beat_fun_biquad(angle1 / radius)
         * beat_fun_biquad(angle2 / radius)
         )

def beat_fun_pair_biquad_highpass (
    angle1,
    angle2,
    radius = pitch_radius_biquad,
    strength = 2.0,
    ):
  return ( strength
         / radius
         * ( beat_fun_biquad(angle1 / radius)
           - radius * 4 / 3 / (2 * pi)
           )
         * beat_fun_biquad(angle2 / radius)
         )

tempo_radius_bilin = pi / 2
pitch_radius_bilin = pi / 4

def beat_fun_bilin (angle, radius = pitch_radius_bilin):
  return maximum(0, 1 - abs(angle / radius))
def beat_fun_pair_bilin (
    angle1,
    angle2,
    radius = pitch_radius_bilin,
    strength = 8.0,
    ):
  return ( strength
         / radius
         * beat_fun_bilin(angle1, radius)
         * beat_fun_bilin(angle2, radius)
         )

def beat_fun_bicos (angle, radius = pitch_radius_biquad):
  return maximum(0, cos(angle) - cos(radius)) / (1 - cos(radius))
def beat_fun_bicos_highpass (angle, radius = pitch_radius_biquad):
  return ( ( maximum(0, cos(angle) - cos(radius))
           - (2 * sin(radius) - 2 * radius * cos(radius)) / (2 * pi)
           )
         / (1 - cos(radius))
         )
def beat_fun_pair_bicos (
    angle1,
    angle2,
    radius = pitch_radius_biquad,
    strength = 2.0,
    ):
  return ( strength
         / radius
         * beat_fun_bicos(angle1, radius)
         * beat_fun_bicos(angle2, radius)
         )
def beat_fun_pair_bicos_highpass (
    angle1,
    angle2,
    radius = pitch_radius_biquad,
    strength = 2.0,
    ):
  return ( strength
         / radius
         * beat_fun_bicos_highpass(angle1, radius)
         * beat_fun_bicos(angle2, radius)
         )

def beat_fun_pair_box (
    angle1,
    angle2,
    radius = pitch_radius_biquad,
    strength = 1.0,
    ):
  return ( strength
         / radius
         * beat_fun_box_highpass(angle1, radius)
         * beat_fun_box(angle2, radius)
         )

def beat_fun_bicos1 (angle, radius = pitch_radius_biquad):
  def cos1 (x):
    return (1 + cos(x)) ** 2
  return maximum(0, (cos1(angle) - cos1(radius)) / (cos1(0) - cos1(radius)))

def beat_fun_bicos2 (angle, radius = pitch_radius_biquad, tol=1e-10):
  x = maximum(0, cos(angle))
  y = sin(angle)
  x2 = x**2
  y2 = y**2
  r2 = x2 + y2
  u2 = x2 / (r2 + tol**2)
  U2 = cos(radius)**2
  return maximum(0, (u2 - U2) / (1 - U2))

tempo_radius = tempo_radius_biquad
pitch_radius = pitch_radius_biquad
beat_fun_pair = beat_fun_pair_bicos

@main.command
def beats ():
  'Plots example separable beat functions'
  T = arange(0, 4 * pi, 0.01)

  pyplot.figure()
  for fun in [
      beat_fun_exp,
      beat_fun_bilin,
      beat_fun_bicos,
      beat_fun_bicos_highpass,
      beat_fun_bicos1,
      beat_fun_bicos2,
      ]:
    pyplot.plot(T, fun(T), label = fun.__name__)
  pyplot.title('Beat functions')
  pyplot.xlabel('phase')
  pyplot.ylabel('syncrhonizing')
  pyplot.legend()
  pyplot.savefig('beats.pdf')
  pyplot.show()

@main.command
def beat_response (ratio = 64.0, samples = 2000):
  'Plots convolution of two beat functions'

  param = (array(list(range(samples))) + 0.5) / samples

  u = pow(ratio, param - 0.5)
  v = 1 / u

  periods = sqrt(samples)
  t = 2 * pi * param * periods

  pairs = [
      (beat_fun_bicos, beat_fun_bicos),
      (beat_fun_bicos, beat_fun_bicos_highpass),
      (beat_fun_bicos_highpass, beat_fun_bicos),
      (beat_fun_bicos_highpass, beat_fun_bicos_highpass),
      ]
  def name (f):
    return f.__name__.replace('beat_fun_','')

  pyplot.figure()
  for f,g in pairs:
    conv = sum(f(outer(u,t)) * g(outer(v,t)), 1) / samples
    assert conv.shape == (samples,)
    pyplot.plot(u/v, conv, label = '%s * %s' % (name(f), name(g)))

  pyplot.title('Correlation of beat function pairs')
  pyplot.legend()
  pyplot.xlabel('frequency ratio')
  pyplot.ylabel('mean beat response')
  pyplot.xscale('log')
  pyplot.xlim(1 / ratio, ratio)
  pyplot.savefig('beat_response.pdf')
  pyplot.show()

#----( beat strength normalization )----

def max_bend (acuity):
  'returns optimal angle and bend strength at optimal angle'

  from scipy.optimize import fminbound

  U = 1 - (pi / acuity) ** 2 / 2 # approximating cos(pi / acuity)
  def f (angle):
    x = cos(angle)
    beat = (x - U) / (1 - U)
    return - beat ** 2 * sin(2 * angle)

  LB,UB = 0.0, pi / max(1.0,acuity)
  (angle,) = fminbound(f,LB,UB)

  return angle,-f(angle)

def beat_scale (acuity):
  _,bend = max_bend(acuity)
  return bend ** -0.5

@main.command
def beat_strength (min_acuity = 3, max_acuity = 7):
  'Plots beat strength for a pair of oscillators'

  pyplot.figure()

  for acuity in range(min_acuity, max_acuity+1):
    dangle = arange(0, 2*pi/acuity, 0.01)
    angle = dangle / 2
    x = cos(angle)
    U = cos(pi / acuity)
    beat = (x - U) / (1 - U)
    bend = beat * beat * sin(dangle)
    pyplot.plot(angle, bend, label='acuity = %i' % acuity)

    x0,y0 = max_bend(acuity)
    pyplot.plot([x0],[y0], 'ko')

  pyplot.title('Bend strength')
  pyplot.xlabel('angle difference')
  pyplot.ylabel('beat strength')
  pyplot.legend(loc = 'upper right')

  pyplot.savefig('beat_strength.png')
  pyplot.show()

@main.command
def beat_rescale (min_acuity = 1, max_acuity = 24):
  'Prints & plots beat rescaling coefficients for various acuities'

  print('acuity\tscale')
  print('-' * 32)
  acuity = list(range(min_acuity, max_acuity+1))
  scale = list(map(beat_scale, acuity))

  for a,s in zip(acuity,scale):
    print('%i\t%s' % (a,s))

  pyplot.title('Beat function rescaling function')
  pyplot.plot(acuity,scale)
  pyplot.xlabel('acuity')
  pyplot.ylabel('beat scale')
  pyplot.show()

@main.command
def beat_rescale_log2 (min_log2_acuity = -3, max_log2_acuity = 4, size = 20):
  'Prints & log-log-plots beat rescaling coefficients for various acuities'

  import scipy

  print('acuity\tscale')
  print('-' * 32)
  acuity = 0.5 * array(list(range(2 * min_log2_acuity, 1 + 2 * max_log2_acuity)))
  scale = log(list(map(beat_scale, exp(acuity)))) / log(2)

  for a,s in zip(acuity,scale):
    print('%s\t%s' % (a,s))

  pyplot.title('Beat function rescaling function')
  pyplot.plot(acuity, scale, 'o')
  pyplot.xlabel('log2(acuity)')
  pyplot.ylabel('log2(beat scale)')
  pyplot.show()

class BeatFun:
  def __init__ (self, acuity):

    a = pi / acuity
    cos_a = cos(a)
    sin_a = sin(a)

    # let f(theta) = max(0, cos(theta) - cos(a))
    Ef = (sin_a - a * cos_a) / pi
    Ef2 = (a - 3 * sin_a * cos_a + 2 * a * cos_a ** 2) / (2 * pi)
    Vf = Ef2 - Ef ** 2

    self.floor = cos_a
    self.shift = -Ef
    self.scale = 1 / sqrt(Vf)

  def __call__ (self, theta):
    return (maximum(0.0, cos(theta) - self.floor) + self.shift) * self.scale

@main.command
def standard_beat (min_acuity = 1.0, max_acuity = 20.0, count = 12):
  'Plots zero-mean unit-variance beat functions at various acuity'

  print('acuity\t\tmean\t\tvariance')
  print('-' * 8 * 6)

  angle = arange(0, 2*pi, 0.01)
  for i in range(count):
    acuity = min_acuity * pow(max_acuity / min_acuity, (i + 0.5) / count)
    fun = BeatFun(acuity)
    beat = fun(angle)

    Ef = sum(beat) / len(beat)
    Ef2 = sum(beat ** 2) / len(beat)
    Vf = Ef2 - Ef ** 2

    print('%g\t%g\t%g' % (acuity, Ef, Vf))

    pyplot.plot(angle, beat)

  pyplot.title('Beat function rescaling function')
  pyplot.xlabel('theta')
  pyplot.ylabel('beat(theta)')
  pyplot.show()


#----( frequency plots )------------------------------------------------------

@main.command
def sim_strength (size = 6, steps = 40000, period = 40):
  '''
  Plots frequencies of pitch-coupled set, varying coupling strength
  This motivates choice of strength = sqrt(2 / size),
  so that strength = 1 for a coupled pair.
  '''

  dt = 1.0 / period

  # intialize
  phase = random.uniform(0,1,size)

  golden = False
  if golden:
    phi = (sqrt(5) - 1) / 2
    freq0 = pow(phi, array(list(range(size))) - 0.5 * (size - 1))
  else:
    freq0 = exp(random.randn(size))

  # evolve through time
  freqs = []
  T = []
  for t in range(steps):
    param = (t + 0.5) / steps
    couple = pow(2, 2 * (param - 0.5))

    angle = 2 * pi * phase

    # energy = - sum i,j. cos(angle(i) - angle(j)) * exp-cos(i,j)
    diff = ( sin(angle.reshape(1,size) - angle.reshape(size,1))
           * beat_fun_pair(angle.reshape(1,size), angle.reshape(size,1))
           )
    force = sum(diff, 1)

    freq = freq0 * (1 + couple * force)
    phase += freq * dt
    phase -= floor(phase)

    freqs.append(freq)
    T.append(couple)

  mean_freqs = smooth(array(freqs), period, 4)
  T = T[2 * period : -2 * period]

  pyplot.figure()
  for i in range(size):
    pyplot.plot(T, mean_freqs[:,i])

  pyplot.title('Synchronization of %i oscillators' % size)
  pyplot.xlabel('coupling strength')
  pyplot.ylabel('frequency')
  pyplot.show()

@main.command
def sim_chord (*chord):
  '''
  Plots frequencies of pitch-coupled chord, varying one frequency
  Example: sim-chord 0 7 12
  '''
  if not chord:
    chord = [0,7,12]
  else:
    chord = [int(t) for t in chord]

  steps = 100000
  period = 100
  dt = 1.0 / period

  # intialize
  size = 1 + len(chord)
  phase = random.uniform(0,1,size)
  fixed_freqs = [2 ** (t / 12.0) for t in chord]

  # evolve through time
  freqs = []
  T = []
  for t in range(steps):
    param = (t + 0.5) / steps

    vary_freq = pow(8, 2 * param - 1)
    freq0 = array([vary_freq] + fixed_freqs)

    angle = 2 * pi * phase

    # energy = - sum i,j. cos(angle(i) - angle(j)) * exp-cos(i,j)
    diff = ( sin(angle.reshape(1,size) - angle.reshape(size,1))
           * beat_fun_pair(angle.reshape(1,size), angle.reshape(size,1))
           )
    force = sum(diff, 1)

    freq = freq0 * (1 + force)
    phase += freq * dt
    phase -= floor(phase)

    freqs.append(freq)
    T.append(vary_freq)

  mean_freqs = smooth(array(freqs), period, 4)
  T = T[2 * period : -2 * period]

  pyplot.figure()
  for i in range(size):
    pyplot.plot(T, mean_freqs[:,i])

  pyplot.title('Synchronization of %i oscillators' % size)
  pyplot.xlabel('varied frequency')
  pyplot.ylabel('frequency')
  pyplot.xscale('log')
  #pyplot.yscale('log')
  pyplot.show()

#----( arnold tongues )-------------------------------------------------------

@main.command
def staircases (size = 100, steps = 10000, period = 40):
  '''
  Plots devils staircases for a pitch-coupled pair of oscillators,
  at various coupling strengths, centered at strength = 1.
  This motivates the choice of strength = 1 for coupled pairs.
  '''

  param1 = (array(list(range(size))) + 0.5) / size
  param2 = (array(list(range(5))) + 0.5) / 5

  pyplot.figure()
  pyplot.title('Devil\'s staircases for pitch-coupled oscillators')
  for p in param2:

    couple = pow(2, 3 * (p - 0.5))

    freq1 = ones(size)
    freq2 = pow(2, 5 * (param1 - 0.5))
    phase1 = random.uniform(0,1,size)
    phase2 = random.uniform(0,1,size)

    total_phase1 = zeros(size)
    total_phase2 = zeros(size)

    for t in range(steps):

      angle1 = 2 * pi * phase1
      angle2 = 2 * pi * phase2
      force = ( couple
              * sin(angle2 - angle1)
              * beat_fun_pair(angle2, angle1)
              )

      dphase1 = freq1 * (1 + force)
      dphase2 = freq2 * (1 - force)

      phase1 += dphase1 / period;   phase1 -= floor(phase1)
      phase2 += dphase2 / period;   phase2 -= floor(phase2)

      total_phase1 += dphase1
      total_phase2 += dphase2

    pyplot.plot(freq2, total_phase1 / steps, 'k-')
    pyplot.plot(freq2, total_phase2 / steps, 'r-')

  pyplot.xscale('log')
  pyplot.yscale('log')
  pyplot.xlabel('natural frequency ratio')
  pyplot.ylabel('mean coupled frequency')
  pyplot.show()

@main.command
def lag (size = 200, steps = 40000, period = 40):
  '''
  Plots phase lag for pitch-locking oscillators.
  This motivates coupling peak falling 90deg after the downbeat.
  '''

  param1 = (array(list(range(size))) + 0.5) / size

  freq1 = ones(size)
  freq2 = pow(2, 6 * (param1 - 0.5))
  phase1 = random.uniform(0,1,size)
  phase2 = random.uniform(0,1,size)

  total_phase1 = zeros(size)
  total_phase2 = zeros(size)
  mean_lag = zeros(size, dtype=cfloat)
  peak_lag = zeros(size, dtype=cfloat)

  for t in range(steps):

    angle1 = 2 * pi * phase1
    angle2 = 2 * pi * phase2
    force = ( sin(angle2 - angle1)
            * beat_fun_pair(angle2, angle1)
            )

    lag = exp(1.0j * (angle2 - angle1))
    mean_lag += lag
    peak_lag += lag * exp(cos(angle1) + cos(angle2))

    dphase1 = freq1 * (1 + force)
    dphase2 = freq2 * (1 - force)

    phase1 += dphase1 / period;   phase1 -= floor(phase1)
    phase2 += dphase2 / period;   phase2 -= floor(phase2)

    total_phase1 += dphase1
    total_phase2 += dphase2

  freq_ratio = total_phase2 / total_phase1
  mean_angle = angle(mean_lag)
  max_angle = angle(peak_lag)

  fig = pyplot.figure()
  pyplot.subplots_adjust(hspace=0.001)

  ax = fig.add_subplot(2,1,1)
  ax.set_title('Phase lag for pitch-locking oscillators')
  ax.set_ylabel('mean phase lag')
  ax.set_xscale('log')
  ax.plot(freq_ratio, mean_angle, 'r-')
  ax.plot(freq_ratio, mean_angle, 'k.')
  pyplot.ylim(-pi,pi)
  pyplot.setp(ax.get_xticklabels(), visible=False)

  ax = fig.add_subplot(2,1,2)
  ax.set_xlabel('mean frequency ratio')
  ax.set_ylabel('peak phase lag')
  ax.set_xscale('log')
  ax.plot(freq_ratio, max_angle, 'g-')
  ax.plot(freq_ratio, max_angle, 'k.')
  pyplot.ylim(-pi,pi)

  pyplot.show()

@main.command
def stairs_lag (size = 400, steps = 20000, period = 40):
  '''
  Plots devils staircase & phase lags for a pitch-coupled pair.
  This motivates pitch-coupling rather than frequency coupling.
  '''

  param = (array(list(range(size))) + 0.5) / size

  freq1 = ones(size)
  freq2 = pow(10, 2 * (param - 0.5))
  phase1 = random.uniform(0,1,size)
  phase2 = random.uniform(0,1,size)

  total_phase1 = zeros(size)
  total_phase2 = zeros(size)
  mean_lag = zeros(size, dtype=cfloat)
  peak_lag = zeros(size, dtype=cfloat)

  for t in range(steps):

    angle1 = 2 * pi * phase1
    angle2 = 2 * pi * phase2
    force = ( sin(angle2 - angle1)
            * beat_fun_pair(angle2, angle1)
            )

    lag = exp(1.0j * (angle2 - angle1))
    mean_lag += lag
    peak_lag += lag * exp(cos(angle1) + cos(angle2))

    dphase1 = freq1 * (1 + force)
    dphase2 = freq2 * (1 - force)

    phase1 += dphase1 / period;   phase1 -= floor(phase1)
    phase2 += dphase2 / period;   phase2 -= floor(phase2)

    total_phase1 += dphase1
    total_phase2 += dphase2

  freq_ratio = total_phase2 / total_phase1
  mean_angle = angle(mean_lag)
  max_angle = angle(peak_lag)

  fig = pyplot.figure()
  pyplot.subplots_adjust(hspace=0.001)

  ax = fig.add_subplot(4,1,1)
  ax.set_title('Pitch and phase locking for coupled oscillators')
  ax.set_ylabel('coupled pitch')
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.plot(freq2, freq_ratio, 'k-')
  pyplot.setp(ax.get_xticklabels(), visible=False)

  ax = fig.add_subplot(4,1,2)
  ax.set_ylabel('pitch bend')
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.plot(freq2, freq_ratio / freq2, 'b-')
  pyplot.ylim(0.5,2.0)
  pyplot.setp(ax.get_xticklabels(), visible=False)

  ax = fig.add_subplot(4,1,3)
  ax.set_ylabel('mean phase lag')
  ax.set_xscale('log')
  ax.plot(freq2, mean_angle, 'r-')
  pyplot.ylim(-pi,pi)
  pyplot.setp(ax.get_xticklabels(), visible=False)

  ax = fig.add_subplot(4,1,4)
  ax.set_xlabel('natural frequency ratio')
  ax.set_ylabel('peak phase lag')
  ax.set_xscale('log')
  ax.plot(freq2, max_angle, 'g-')
  pyplot.ylim(-pi,pi)

  pyplot.show()

@main.command
def tongues (size = 100, steps = 4000, period = 40):
  '''
  Plots arnold tongues for a pitch-coupled pair of oscillators.
  This helps to choose a beat function achieving many wide tongues,
  and a coupling strength achieving a given 1:1 tongue width.
  '''

  param_x = ((array(list(range(size))) + 0.5) / size).reshape(1,size)
  param_y = ((array(list(range(size))) + 0.5) / size).reshape(size,1)

  couple = 2 * param_y    # ranging in [0,2]
  ratio = pow(2, param_x) # ranging in 1,2

  freq1 = 1 / sqrt(ratio)
  freq2 = sqrt(ratio)
  phase1 = zeros((size,size))
  phase2 = zeros((size,size))

  total_phase1 = zeros((size,size))
  total_phase2 = zeros((size,size))

  for t in range(steps):

    angle1 = 2 * pi * phase1
    angle2 = 2 * pi * phase2
    force = ( couple
            * sin(angle2 - angle1)
            * beat_fun_pair(angle2, angle1)
            )

    dphase1 = freq1 * (1 + force)
    dphase2 = freq2 * (1 - force)

    phase1 += dphase1 / period;   phase1 -= floor(phase1)
    phase2 += dphase2 / period;   phase2 -= floor(phase2)

    total_phase1 += dphase1
    total_phase2 += dphase2

  phase_ratio = total_phase2 / total_phase1
  decoupling = clip(phase_ratio[:,1:] - phase_ratio[:,:-1], 0, 2.0 / size)

  pyplot.figure()
  pyplot.imshow(
      decoupling,
      aspect = 'auto',
      origin = 'lower',
      extent = (0, 12, 0, 2),
      cmap = pyplot.cm.binary,
      )
  pyplot.title('Arnold tongues for pitch-coupled oscillators')
  pyplot.xlabel('pitch interval')
  pyplot.ylabel('coupling strength')
  pyplot.xlim(0,12)
  pyplot.ylim(0,2)
  pyplot.savefig('tongues.pdf')
  pyplot.show()

@main.command
def keys (size = 100, octaves = 1.0, steps = 4000, period = 40):
  '''
  Plots arnold keys for a pitch-coupled pair of oscillators.
  This helps to choose beat function radius.
  '''

  param_x = ((array(list(range(size))) + 0.5) / size).reshape(1,size)
  param_y = ((array(list(range(size))) + 0.5) / size).reshape(size,1)

  min_radius = pi / 12
  max_radius = pi / 2
  radius = min_radius * pow(max_radius / min_radius, param_y)
  ratio = pow(2, octaves * param_x) # ranging in 1,2

  freq1 = 1 / sqrt(ratio)
  freq2 = sqrt(ratio)
  phase1 = zeros((size,size))
  phase2 = zeros((size,size))

  total_phase1 = zeros((size,size))
  total_phase2 = zeros((size,size))

  for t in range(steps):

    angle1 = 2 * pi * phase1
    angle2 = 2 * pi * phase2
    force = ( sin(angle2 - angle1)
            * beat_fun_pair(angle2, angle1, radius)
            )

    dphase1 = freq1 * (1 + force)
    dphase2 = freq2 * (1 - force)

    phase1 += dphase1 / period;   phase1 -= floor(phase1)
    phase2 += dphase2 / period;   phase2 -= floor(phase2)

    total_phase1 += dphase1
    total_phase2 += dphase2

  phase_ratio = total_phase2 / total_phase1
  decoupling = clip(phase_ratio[:,1:] - phase_ratio[:,:-1], 0, 2.0 / size)

  pyplot.figure()
  pyplot.plot([0,12 * octaves],[log(pitch_radius)]*2, 'r-')
  pyplot.plot([0,12 * octaves],[log(tempo_radius)]*2, 'b-')
  pyplot.imshow(
      decoupling,
      aspect = 'auto',
      origin = 'lower',
      extent = (0, 12 * octaves, log(min_radius), log(max_radius)),
      cmap = pyplot.cm.binary,
      )
  pyplot.title('White keys for pitch-coupled oscillators')
  pyplot.xlabel('pitch interval')
  pyplot.ylabel('log(attractor radius)')
  pyplot.xlim(0,12 * octaves)
  pyplot.ylim(log(min_radius), log(max_radius))
  pyplot.savefig('keys.pdf')
  pyplot.show()

@main.command
def regions (size = 100, steps = 4000, period = 40):
  '''
  Plots mode-locking regions for 3 coupled oscillators.
  This helps to choose coupling strength, attractor radius,
  and scaling of coupling strength with number of oscillators.
  '''

  freq = zeros((3,size,size))
  phase = random.uniform(0,1,(3,size,size))
  force = zeros((3,size,size))
  total_phase = zeros((3,size,size))

  param_x = ((array(list(range(size))) + 0.5) / size).reshape(1,size)
  param_y = ((array(list(range(size))) + 0.5) / size).reshape(size,1)

  ratio01 = pow(2, param_x) # ranging in 1,2
  ratio12 = pow(2, param_y) # ranging in 1,2

  freq[0,:,:] = 1 / ratio01
  freq[1,:,:] = 1.0
  freq[2,:,:] = ratio12

  for t in range(steps):

    angle = 2 * pi * phase
    force[:,:,:] = 0

    for i in range(3):
      for j in range(i):
        force_ij = ( sin(angle[j,:,:] - angle[i,:,:])
                   * beat_fun_pair(angle[j,:,:], angle[i,:,:])
                   )
        force[i,:,:] += force_ij
        force[j,:,:] -= force_ij

    dphase = freq * (1 + force)
    phase += dphase / period;
    phase -= floor(phase)
    total_phase += dphase

  total_phase /= freq

  ratios = zeros((size,size,3))
  for i in range(3):
    ratios[:,:,i] = total_phase[i,:,:]
    r = ratios[:,:,i]
    r -= r.min()
    r /= r.max()

  pyplot.figure()
  pyplot.imshow(
      ratios,
      origin = 'lower',
      extent = (0, 12, 0, 12),
      )
  pyplot.title('Mode locking regions for 3 pitch-coupled oscillators')
  pyplot.xlabel('lower pitch interval')
  pyplot.ylabel('higher pitch interval')
  pyplot.xlim(0,12)
  pyplot.ylim(0,12)
  pyplot.savefig('regions.pdf')
  pyplot.show()

#----( synthesis )------------------------------------------------------------

@main.command
def synth (size = 5):
  '''
  Plots multiplicative synthesis signal for various sharpness values
  This motivates an exponential energy dependency
    signal = exp(mass * (sum i. cos(phase_i))) - 1
  rather than a linear dependency
    signal = mass * (exp(sum i. cos(phase_i)) - 1)
  '''

  T = arange(0, 4 * pi, 0.02)
  def synth (k):
    return exp(k * cos(T)) - 1

  pyplot.figure()
  for i in range(size):
    k = pow(2, (i - 0.5 * (size - 1)) / 2)
    pyplot.plot(T, synth(k), label='sharpness = %g' % k)
  pyplot.title('Multiplicative synthesis')
  pyplot.xlabel('t')
  pyplot.ylabel('exp(k cos(t)) - 1')
  pyplot.legend()
  pyplot.show()

#----( main )-----------------------------------------------------------------

if __name__ == '__main__': main.main()

