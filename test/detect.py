
import math, numpy
import Image, ImageDraw

def draw_circle (draw, center, radius, **kwds):
  x,y = center
  draw.ellipse((x-radius, y-radius, x+radius, y+radius), **kwds)

#----( image processing )-----------------------------------------------------

def square_blur (x, radius):
  'square blur kernel'

  I,J = x.shape
  assert radius < I and radius < J

  y = numpy.zeros(x.shape)
  for i in range(I):
    y[i,0] = radius * x[i,0] + sum(x[i,0:1+radius])
    for j in range(1,J):
      y[i,j] = y[i,j-1] - x[i,max(0,j-1-radius)] + x[i,min(J-1,j+radius)]
  y /= 1 + 2 * radius

  x = y
  y = numpy.zeros(x.shape)
  for j in range(J):
    y[0,j] = radius * x[0,j] + sum(x[0:1+radius,j])
    for i in range(1,I):
      y[i,j] = y[i-1,j] - x[max(0,i-1-radius),j] + x[min(I-1,i+radius),j]
  y /= 1 + 2 * radius

  return y

def poly_blur (x, radius, order = 3):

  radius = int(round(radius / math.sqrt(order)))

  for i in range(order):
    x = square_blur(x, radius)

  return x

def select_scale (x, radius):

  I,J = x.shape

  y = poly_blur(x, radius)

  z = numpy.zeros(x.shape)
  for i in range(I):
    for j in range(J):

      # uses a hexagonal kernel
      #    o o
      #   o x o
      #    o o

      di = int(round(radius / 2))
      dj = int(round(radius * math.sqrt(3) / 2))

      i0,i1,i2,i3 = max(0,i-radius),max(0,i-di),min(I-1,i+di),min(I-1,i+radius)
      j0,j1 = max(0,j-dj),min(J-1,j+dj)

      '''
      #this is a nonlinear point kernel
      center = 2 * y[i,j];
      points = - ( (y[i0,j] + y[i3,j] - center)
                  * (y[i1,j1] + y[i2,j0] - center)
                  * (y[i2,j1] + y[i1,j0] - center) ) * 8
      '''
      # this non-isotropic detector works better for fingertips
      fingers = ( 3 * y[i,j]
                  - y[i0,j]
                  - y[i1,j1]
                  - y[i2,j1]
                  + 2 * y[i3,j]
                  - y[i2,j0]
                  - y[i1,j0]
                  ) / 6.0
      '''
      laplacian = ( 6 * y[i,j]
                  - y[i0,j]
                  - y[i1,j1]
                  - y[i2,j1]
                  - y[i3,j]
                  - y[i2,j0]
                  - y[i1,j0]
                  ) / 6.0
      '''

      #z[i,j] = math.sqrt(max(0,fingers) * max(0,points)) * 4
      z[i,j] = fingers

  return z

def extract_peaks (x):

  I,J = x.shape

  peaks = [] # (value, i, j)

  for i in range(1,I-1):
    for j in range(1,J-1):
      value = x[i,j]
      if ( (value > 0) and (value > x[i-1,j-1])
                       and (value > x[i-1,j+0])
                       and (value > x[i-1,j+1])
                       and (value > x[i+0,j+1])
                       and (value > x[i+1,j+1])
                       and (value > x[i+1,j+0])
                       and (value > x[i+1,j-1])
                       and (value > x[i+0,j-1]) ):
        peaks.append((value, i, j))

  peaks.sort()

  return peaks

#----( testing )--------------------------------------------------------------

def hand ():

  im = Image.open('hand.png')
  draw = ImageDraw.Draw(im)
  print('opened %d x %d image' % im.size)

  return im,draw

def test1 ():

  im,draw = hand()
  im.show()

def test2 ():

  im,draw = hand()
  x = 1 - numpy.asarray(im)
  y = Image.fromarray(x, mode='L') # 'L' = Luminance = grayscale
  y.show()

def test3 (radius = 8):

  im,draw = hand()
  x = numpy.asarray(im) / 255.0

  x = poly_blur(x, radius)

  x *= 255.0
  im = Image.fromarray(x.astype(numpy.uint8), mode='L')
  im.show()

def test4 (radius = 16):

  im,draw = hand()
  x = numpy.asarray(im) / 255.0

  x = select_scale(-x, radius) / 2 + 0.5

  x *= 255.0
  im = Image.fromarray(x.astype(numpy.uint8), mode='L')
  im.show()

def test5 (radius = 16, scale = 30):

  im,draw = hand()
  x = numpy.asarray(im) / 255.0

  x = select_scale(-x, radius)
  peaks = extract_peaks(x)

  for p,i,j in peaks:
    radius = int(round(scale * p))
    draw_circle(draw, (j,i), radius)

  im.show()

if __name__ == '__main__':
  test5()

