#!/usr/bin/python

# sudo apt-get install python-opencv
from opencv import *
from opencv.highgui import *
import main

#----( main commands )--------------------------------------------------------

@main.command
def show ():
  'Show camera captured image'
  camera = cvCreateCameraCapture(-1)

  cvNamedWindow('kazoo')

  while True:
    frame = cvQueryFrame(camera)
    cvShowImage('kazoo', frame)

if __name__ == '__main__': main.main()

