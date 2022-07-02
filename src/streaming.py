from . import network
from .network import stop_threads, validate
from . import transforms
import _kazoo

from _transforms import DEFAULT_EXPONENT
from _transforms import DEFAULT_SAMPLE_RATE
from _transforms import DEFAULT_FRAMES_PER_BUFFER
from _transforms import DEFAULT_MIN_FREQ
from _transforms import DEFAULT_MAX_FREQ
from _transforms import MAX_EXPONENT

# ----( channels )-------------------------------------------------------------


def Reals(*args, **kwds):
    "Factory for real vector channels."
    if args:
        return network.Channel(_kazoo.Reals(*args), **kwds)
    else:
        return network.Channel(allocator=_kazoo.Reals, **kwds)


def Complexes(*args, **kwds):
    "Factory for complex vector channels."
    if args:
        return network.Channel(transforms.Complexes(*args), **kwds)
    else:
        return network.Channel(allocator=transforms.Complexes, **kwds)


# ----( sources & sinks )------------------------------------------------------


class Audio(transforms.Audio):
    __doc__ = transforms.Audio.__doc__
    reading = network.switched(network.source)(transforms.Audio.read)
    writing = network.switched(network.sink)(transforms.Audio.write)

    def start(self):
        transforms.Audio.start(self)
        if self.am_reading:
            self.start_read()
        if self.am_writing:
            self.start_write()

    def stop(self):
        if self.am_reading:
            self.stop_read()
        if self.am_writing:
            self.stop_write()
        transforms.Audio.stop(self)

    def run_until_input(self):
        try:
            self.start()
            print("press ENTER to stop")
            input()
            self.stop()
        finally:
            stop_threads()


AudioFile = network.FiniteSource(transforms.AudioFile)
Null = network.Sink(transforms.Null)
Recorder = network.Sink(transforms.Recorder)
ImageBuffer = network.Sink(transforms.ImageBuffer)
Screen = network.Sink(transforms.Screen, "vertical_sweep", "vertical_sweeping")

# ----( transforms )-----------------------------------------------------------

Wire = network.Stream(transforms.Wire)
Splitter = network.Stream(transforms.Splitter)
Mixer = network.Stream(transforms.Mixer)
Concat = network.Stream(transforms.Concat)
Spectrogram = network.Stream(transforms.Spectrogram)
Supergram = network.Stream(transforms.Supergram)
Multigram = network.Stream(transforms.Multigram)
Phasogram = network.Stream(transforms.Phasogram)
Pitchgram = network.Stream(transforms.Pitchgram)
MultiScale = network.Stream(transforms.MultiScale)
HiLoSplitter = network.Stream(transforms.HiLoSplitter)
Shepard = network.Stream(transforms.Shepard)
Loudness = network.Stream(transforms.Loudness)
Sharpener = network.Stream(transforms.Sharpener)
OctaveLower = network.Stream(transforms.OctaveLower)
# PitchShift   = network.Stream(transforms.PitchShift)
Melodigram = network.Stream(transforms.Melodigram)
Rhythmgram = network.Stream(transforms.Rhythmgram)
Spline = network.Stream(transforms.Spline)


def wrap_spline_output(Class, method_name):
    "wraps method outputting transforms.Spline to streaming.Spline"
    old_method = getattr(Class, method_name)

    def new_method(*args):
        spline_in = old_method(*args)
        spline_out = Spline(2, 2)
        spline_out.swap(spline_in)
        return spline_out

    new_method.__name__ = old_method.__name__
    new_method.__doc__ = old_method.__doc__
    setattr(Class, method_name, new_method)


wrap_spline_output(Supergram, "freq_scale")
wrap_spline_output(Supergram, "pitch_scale")
wrap_spline_output(Multigram, "freq_scale")
wrap_spline_output(Multigram, "pitch_scale")

# ----( testing )--------------------------------------------------------------


def test_wire(size_exponent=10, factor_exponent=2):
    import time

    print("declaring transforms")
    s = Supergram(size_exponent, factor_exponent)
    a = Audio(s.small_size)
    w = Wire()
    # v = Screen(s.super_size)

    print("declaring channels")
    c1 = Complexes()
    c2 = Reals()
    c3 = Reals()
    c4 = Complexes()

    print("connecting transforms via channels")
    a.reading(c1)
    s.stream_fwd(c1, c2)
    w.stream(c2, c3)
    # v.vertical_sweeping(c2)
    s.stream_bwd(c3, c4)
    a.writing(c4)

    print("validating")
    validate()

    duration = 3
    print("streaming for %g seconds..." % duration)
    a.start()
    time.sleep(duration)
    a.stop()
    print("...done")

    stop_threads()


if __name__ == "__main__":
    test_wire()
