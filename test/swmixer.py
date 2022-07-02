# from http://code.google.com/p/pygalaxy/wiki/SWMixer


def mic_test(play_background=False):
    """records and plays back data from the microphone
    while playing a test sound in the background.

    (2009:12:10) XXX this has high latency"""

    import sys
    import swmixer
    import numpy

    swmixer.init(samplerate=44100, chunksize=1024, stereo=False, microphone=True)

    if play_background:
        snd = swmixer.Sound("test.wav")
        snd.play(loops=-1)

    micdata = []
    frame = 0

    while True:
        swmixer.tick()
        frame += 1
        if frame < 50:
            micdata = numpy.append(micdata, swmixer.get_microphone())
        if frame == 50:
            micsnd = swmixer.Sound(data=micdata)
            micsnd.play()
            micdata = []
            frame = 0


if __name__ == "__main__":
    mic_test()
