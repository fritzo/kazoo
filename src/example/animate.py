import kazoo as K
import math


def animate_spectrogram(exponent=10):

    size = 1 << exponent
    length = size / 2

    audio = K.Audio(size, writing=False)
    spec = K.Spectrogram(exponent)
    screen = K.Screen(size, length)

    sound = K.Complexes()
    freq = K.Reals()

    audio.reading(sound)
    spec.stream_fwd(sound, freq)
    screen.vertical_sweeping(K.energy_to_loudness(freq))
    spec.stream_bwd(freq, sound)

    K.validate()

    audio.run_until_input()


def animate_super(size_exponent=9, time_exponent=2, freq_exponent=2):

    screen = K.Screen(1024, 1024)
    spec = K.Supergram(size_exponent, time_exponent, freq_exponent, screen.height)
    audio = K.Audio(spec.small_size, writing=False)
    rate = audio.rate / spec.small_size
    loud = K.Loudness(spec.super_size, rate)

    sound = K.Complexes()
    energy = K.Reals()
    louds = K.Reals()

    audio.reading(sound)
    spec.stream_fwd(sound, energy)
    loud.stream_fwd(energy, louds)
    screen.vertical_sweeping(louds)

    K.validate()

    audio.run_until_input()


def pitch_bend(size_exponent=10, factor_exponent=2, factor=0.5, width=1600, height=800):

    spec = K.Supergram(size_exponent, factor_exponent)
    audio = K.Audio(spec.small_size)
    height = min(height, spec.super_size)
    screen = K.Screen(width, height)
    split = K.Splitter()

    sound1 = K.Complexes()
    sound2 = K.Complexes()
    sound3 = K.Complexes()
    size = spec.super_size
    fun = K.Reals()
    for i in range(size):
        fun[i] = (0.5 + i) / size * factor
    bend = K.Spline(size, size, fun)
    energy = K.Reals()
    bent = K.Reals()

    audio.reading(sound1)
    split.stream(sound1, (sound2, sound3))
    spec.stream_fwd(sound2, energy)
    # screen.vertical_sweeping(energy[:height] / energy.max())
    bend.stream_fwd(energy, bent)
    screen.vertical_sweeping(bent[:height] / bent.max())
    spec.stream_bwd(bent, sound)
    audio.writing(sound3)

    K.validate()

    audio.run_until_input()


def animate_freq(size_exponent=10, factor_exponent=2):

    screen = K.Screen(1024, 1024)
    spec = K.Supergram(size_exponent, factor_exponent)
    audio = K.Audio(spec.small_size, writing=False)
    scale = spec.freq_scale(screen.height, 4000)
    rate = audio.rate / spec.small_size
    loud = K.Loudness(scale.size_out, rate)

    sound = K.Complexes()
    energy = K.Reals()
    pitch = K.Reals()
    louds = K.Reals()

    audio.reading(sound)
    spec.stream_fwd(sound, energy)
    scale.stream_fwd(energy, pitch)
    loud.stream_fwd(pitch, louds)
    screen.vertical_sweeping(louds)
    # ...stream backward

    K.validate()

    audio.run_until_input()


def animate_pitch(size_exponent=10, factor_exponent=2):

    screen = K.Screen(1024, 1024)
    spec = K.Supergram(size_exponent, factor_exponent)
    audio = K.Audio(spec.small_size, writing=False)
    scale = spec.pitch_scale(screen.height)
    rate = audio.rate / spec.small_size
    loud = K.Loudness(scale.size_out, rate)

    sound = K.Complexes()
    energy = K.Reals()
    pitch = K.Reals()
    louds = K.Reals()

    audio.reading(sound)
    spec.stream_fwd(sound, energy)
    scale.stream_fwd(energy, pitch)
    loud.stream_fwd(pitch, louds)
    screen.vertical_sweeping(louds)
    # ...stream backward

    K.validate()

    audio.run_until_input()


def animate_correlogram(size_exponent=10, factor_exponent=2):

    screen = K.Screen(1024, 1024)
    spec = K.Supergram(size_exponent, factor_exponent)
    audio = K.Audio(spec.small_size, writing=False)
    scale = spec.freq_scale(screen.height / 2)
    rate = audio.rate / spec.small_size
    loud = K.Loudness(scale.size_out, rate)
    cgram = K.Correlogram(scale.size_out, math.exp(-1.0 / 2.0 / rate))

    assert cgram.size_out == 2 * cgram.size_in

    sound = K.Complexes()
    energy = K.Reals()
    pitch = K.Reals()
    louds = K.Reals()
    corr = K.Reals()

    audio.reading(sound)
    spec.stream_fwd(sound, energy)
    scale.stream_fwd(energy, pitch)
    loud.stream_fwd(pitch, louds)
    cgram.stream_fwd(louds, corr)
    screen.vertical_sweeping(corr)
    # ...stream backward

    K.validate()

    audio.run_until_input()


if __name__ == "__main__":
    # animate_spectrogram()
    animate_super()
    # pitch_bend()
    # animate_freq()
    # animate_pitch()
    # animate_correlogram()
