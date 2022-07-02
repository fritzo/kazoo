import math, numpy
from math import sqrt, pow, pi
from scipy import special
from numpy import array, zeros, arange, dot
from matplotlib import pyplot

# ----( convenience math )-----------------------------------------------------

inf = float("inf")
nan = float("nan")
if "isnan" in dir(math):

    def isnormal(x):
        return not (math.isnan(x) or math.isinf(x))

else:

    def isnormal(x):
        return x == x and x != inf and x != -inf


# ----( windows )--------------------------------------------------------------


def Hann(t):
    return 0.5 * (1 + numpy.cos(pi * t))


def BlackmanNuttall(t):
    return (
        0.3635819
        + 0.4891775 * numpy.cos(pi * t)
        + 0.1365995 * numpy.cos(2 * pi * t)
        + 0.0106411 * numpy.cos(3 * pi * t)
    )


def narrow_window(width_exponent):
    n = 0.75 * ((1 << (2 * width_exponent)) - 1)
    print("building synth window with w = %i, n = %g" % (width_exponent, n))

    def h(t):
        return ((t + 1) * (1 - t)) ** n

    return h


def plot_windows(max_exponent=5):

    pyplot.figure(figsize=(12, 6))

    T = arange(-1, 1, 0.002)
    for i in range(1, 1 + max_exponent):
        h = narrow_window(i)
        pyplot.plot(
            T, h(T), color="black", linestyle="-", label="Narrow" if i == 1 else None
        )

    pyplot.plot(T, Hann(T), color="red", linestyle="-.", label="Hann")
    pyplot.plot(
        T, BlackmanNuttall(T), color="green", linestyle="--", label="Blackman-Nuttall"
    )

    widths = ", ".join("1/%i" % (2 ** (i - 1)) for i in range(1, 1 + max_exponent))
    pyplot.title("Narrow synthesis windows for widths %s" % widths)
    pyplot.legend()
    pyplot.savefig("synth_windows.pdf")


if __name__ == "__main__":
    plot_windows()
