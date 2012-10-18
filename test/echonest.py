
#import pyechonest
#pyechonest.config.ECHO_NEST_API_KEY = os.environ['ECHO_NEST_API_KEY']

from echonest import audio
import networkx as nx

def test ():
  track = audio.LocalAudioFile('test.mp3')

if __name__ == '__main__':
  test()

