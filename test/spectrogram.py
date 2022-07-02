#!/usr/bin/python

import kazoo.transforms as K
from kazoo import formats, util
from numpy import *
import random
import main


def signal_test(exponent, signal, filename):
    width = 1 << exponent
    size = width * width

    print("creating signal")
    input = K.Complexes(size)

    for i in range(size):
        t = (i + 0.5) / size
        input[i] = signal(t)

    input = input.reshape((width, width))

    print("transforming signal")
    output = K.Reals(width, width / 2)

    s = K.Spectrogram(exponent)
    for i in range(width):
        s.transform_fwd(input[i, :], output[i, :])

    print("writing spactrogram to %s" % filename)
    output = util.energy_to_loudness(output)
    formats.write_image(output, filename)


@main.command
def play_chirp(exponent=8):
    "Plays chirp"

    size = (1 << exponent) ** 2
    freq = size / 2.0
    sigma = 1 / 2.0

    def signal(t):
        return exp(
            2 * pi * (t - 0.5) ** 2 * freq * 1.0j - ((t - 0.5) / (0.5 * sigma)) ** 2
        )

    signal_test(exponent, signal, "chirp.png")


@main.command
def play_vibrato(exponent=8):
    "Plays FM tone"

    size = (1 << exponent) ** 2
    sigma = 1 / 2.0
    freq = size / 4.0
    dfreq = size / 8.0
    modulate = 4.0

    def signal(t):
        return exp(
            1.0j
            * 2
            * pi
            * (t - 0.5)
            * (freq + dfreq * sin(modulate * 2 * pi * (t - 0.5)))
            - ((t - 0.5) / (0.5 * sigma)) ** 2
        )

    signal_test(exponent, signal, "vibrato.png")


@main.command
def test_inversion(exponent=4):

    "Tests spectrogram inversion"
    size = 1 << exponent

    s = K.Spectrogram(exponent)
    print("weights:")
    print(s.weights)

    time_in = K.Complexes(size)
    freq_io = K.Reals(size / 2)
    time_out = K.Complexes(size)

    # print time_in.ndim
    # print time_in.dtype
    # print time_in.shape
    # print time_in.strides

    for i in range(size):
        # time_in[i] = cos(2 * pi * (0.5 + i) / size)
        t = (0.5 + i) / size
        # time_in[i] = exp(2 * pi * 1.j * t)
        # time_in[i] = exp(2 * pi * 1.j * t) / (0.5 + 4 * t * (1-t))
        time_in[i] = exp(2 * pi * 1.0j * t) + random.normalvariate(0, 1)

    print("time input:")
    print(time_in)

    # Transform twice to stabilize fwd transform;
    #         thrice to stabilize bwd transform

    s.transform_fwd(time_in, freq_io)
    print("freq output:")
    print(freq_io)

    s.transform_bwd(freq_io, time_out)
    print("time output:")
    print(time_out)

    s.transform_fwd(time_in, freq_io)
    print("freq output:")
    print(freq_io)

    s.transform_bwd(freq_io, time_out)
    print("time output:")
    print(time_out)

    s.transform_fwd(time_in, freq_io)
    print("freq output:")
    print(freq_io)

    s.transform_bwd(freq_io, time_out)
    print("time output:")
    print(time_out)

    print("residual: = %g (should be small)" % linalg.norm(time_in - time_out))


if __name__ == "__main__":
    main.main()
