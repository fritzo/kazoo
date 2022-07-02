#!/usr/bin/python

from matplotlib import pyplot
import main

@main.command
def layers (offset = 4, *positions):
  'Plots grid with offset positions'

  positions = set(int(p) for p in positions)
  positions.add(0)

  radius = 3 * 12
  R = list(range(radius+1))

  def marked (p,x,y):
    return (x + offset * y) % 12 == p

  pyplot.figure()
  #pyplot.axis('equal')
  pyplot.title('offset = %i, positions = %s' % (offset, list(positions)))

  for r in R:
    pyplot.plot([0,radius],[r,r], color = (0.9,)*3)
    pyplot.plot([r,r],[0,radius], color = (0.9,)*3)

  for p in positions:
    XY = [(x,y) for x in R for y in R if marked(p,x,y)]
    X = [x for x,y in XY]
    Y = [y for x,y in XY]
    pyplot.plot(X,Y, marker = '.', linestyle = 'none')

  pyplot.xlim(0,radius)
  pyplot.ylim(0,radius)

  if main.at_top():
    pyplot.show()

@main.command
def tunings ():
  'Plots a few example grids'
  layers(6, 3,4,7)
  layers(4, 3,4,7)

  pyplot.show()

if __name__ == '__main__': main.main()

