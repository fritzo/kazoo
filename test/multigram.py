import kazoo.transforms as K
from kazoo import formats, util
from numpy import *


def file_test(small_exponent=9, large_exponent=12):
    "multigram"
    s = K.Multigram(small_exponent, large_exponent)
    size_in = s.size_in
    large_size = s.size_out
    length = s.size_out / 2  # arbitrary

    print("reading sound file")
    sound = formats.read_wav("test.wav", size_in * length, size_in)
    image = K.Reals(length, large_size)

    print("transforming data")
    for i in range(length):
        s.transform_fwd(sound[i, :], image[i, :])
    del sound

    print("saving image")
    image = util.energy_to_loudness(image + 1e-5)
    formats.write_image(image, "test.png")


def signal_test(small_exponent, large_exponent, signal, filename):
    fft_size = 1 << ((small_exponent + large_exponent) / 2)
    length_windows = fft_size / 2
    length_samples = fft_size * length_windows

    print("creating signal")
    input = K.Complexes(length_samples)

    for i in range(length_samples):
        t = (0.5 + i) / length_samples
        input[i] = signal(t)

    print("transforming signal")
    s = K.Multigram(small_exponent, large_exponent)

    length_frames = length_samples / s.size_in
    input = input.reshape((length_frames, s.size_in))
    output = K.Reals(length_frames, s.size_out)

    for i in range(length_frames):
        s.transform_fwd(input[i, :], output[i, :])

    print("writing multigram to %s" % filename)

    output = util.energy_to_loudness(output)
    formats.write_image(output, filename)


def chirp_test(small_exponent=4, large_exponent=10):
    fft_size = 1 << ((small_exponent + large_exponent) / 2)
    nyquist_freq = (fft_size / 2) ** 2
    sigma = 0.25

    def signal(t):
        return t * (1 - t) * sin(6 * t * (t**2 / 3 - t / 2) * nyquist_freq * pi)

    signal_test(small_exponent, large_exponent, signal, "test.png")


def impulse_test(small_exponent=4, large_exponent=10):
    fft_size = 1 << ((small_exponent + large_exponent) / 2)
    nyquist_freq = (fft_size / 2) ** 2
    sigma = 0.25

    def signal(t):
        return exp(-(((t - 0.5) * nyquist_freq) ** 2))

    signal_test(small_exponent, large_exponent, signal, "test.png")


def click_test(small_exponent=4, large_exponent=10):
    fft_size = 1 << ((small_exponent + large_exponent) / 2)
    nyquist_freq = (fft_size / 2) ** 2
    sigma = 0.25

    def signal(t):
        return fmod(2 * t, 1) - t

    signal_test(small_exponent, large_exponent, signal, "test.png")


if __name__ == "__main__":
    file_test()
    # chirp_test()
    # impulse_test()
    # click_test()
