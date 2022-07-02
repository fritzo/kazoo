#!/usr/bin/python

import kazoo as K
from kazoo import formats
from numpy import *
from scipy import *
import main

DEFAULT_BANK_SIZE = 320
DEFAULT_BLOCK_SIZE = 128

# -----------------------------------------------------------------------------


@main.command
def show_file(bank_size=512, block_size=K.DEFAULT_FRAMES_PER_BUFFER):
    "Shows pitchgram of an audio file"

    size_in = block_size
    size_out = bank_size
    length = size_out * 16 / 9  # arbitrary

    pitchgram = K.Pitchgram(size_in, size_out)

    print("reading sound file")
    sound = formats.read_wav("test.wav", size_in * length, size_in)
    image = K.transforms.Reals(length, size_out)

    print("transforming data")
    for i in range(length):
        pitchgram.transform(sound[i, :], image[i, :])
    # del sound

    print("saving image")
    image = K.util.energy_to_loudness(image + 1e-5)
    formats.write_image(image, "test.png").show()


# -----------------------------------------------------------------------------


def show_signal(bank_size, block_size, signal):

    size_in = block_size
    size_out = bank_size

    num_blocks = bank_size * 16 / 9  # arbitrary
    duration = num_blocks * block_size

    min_freq = 0.25 / block_size
    max_freq = 0.25
    pitchgram = K.Pitchgram(size_in, size_out, min_freq, max_freq)

    print("creating signal")
    t = array(list(range(duration))) / block_size
    sound = K.transforms.Complexes(duration)
    sound[:] = signal(t)
    sound = sound.reshape((num_blocks, block_size))

    print("transforming signal")
    image = K.transforms.Reals(num_blocks, size_out)
    for i in range(num_blocks):
        pitchgram.transform(sound[i, :], image[i, :])

    print("writing pitchgram to pitchgram.png")
    # XXX this fails due to NaNs and Infs in image XXX
    # image[~isfinite(image)] = 0 #DEBUG
    image = K.util.energy_to_loudness(image, 0.1)
    formats.write_image(image, "pitchgram.png").show()


@main.command
def show_sine(bank_size=DEFAULT_BANK_SIZE, block_size=DEFAULT_BLOCK_SIZE):
    "Shows pitchgram of a simple sine wave"

    mid_freq = 0.25 * sqrt(block_size)

    def signal(t):
        return sin(2 * pi * mid_freq * t)

    show_signal(bank_size, block_size, signal)


@main.command
def show_fm(bank_size=DEFAULT_BANK_SIZE, block_size=DEFAULT_BLOCK_SIZE):
    "Shows pitchgram of a simple sine wave"

    mid_freq = 0.25 * sqrt(block_size)
    mod_freq = 1.0 / bank_size
    mod_amp = 0.1

    def signal(t):
        mod = exp(mod_amp * sin(2 * pi * mod_freq * t))
        return sin(2 * pi * mid_freq * t * mod)

    show_signal(bank_size, block_size, signal)


@main.command
def show_chirp(bank_size=DEFAULT_BANK_SIZE, block_size=DEFAULT_BLOCK_SIZE):
    "Shows pitchgram of a chirp"

    min_freq = 0.25
    max_freq = 0.25 * block_size

    def signal(t):
        return sin(2 * pi * t * (min_freq * (max_freq / min_freq) ** (t / bank_size)))

    show_signal(bank_size, block_size, signal)


@main.command
def show_step(bank_size=DEFAULT_BANK_SIZE, block_size=DEFAULT_BLOCK_SIZE):
    "Shows pitchgram of a step function"

    max_freq = 0.25

    def signal(t):
        return 2 * (t < 0.5 * bank_size) - 1

    show_signal(bank_size, block_size, signal)


@main.command
def show_noise(bank_size=DEFAULT_BANK_SIZE, block_size=DEFAULT_BLOCK_SIZE):
    "Shows pitchgram of white noise"

    max_freq = 0.25

    def signal(t):
        return randn(len(t))

    show_signal(bank_size, block_size, signal)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main.main()
