from math import *
from numpy import *

# ---( psychoacoustic curves )-------------------------------------------------

# ----------------------------------------------------------
# This is from
#  (R1) http://www.mathworks.com/matlabcentral/fileexchange/7028


def equalize_1987(w):
    "implements ISO 226:1987 equal-loudness standard"
    raise NotImplementedError()


# ------------------------------------------------------------------
# These are from:
# (R2) "Bark and ERB bilinear transforms"
#   -J.O.Smith 3, J.S.Abel
#   http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.113.5882


def bark_transform(sampling_rate_hz):
    gamma = [1.0674, 65.83, 0.1916]  # 1000x (R2)'s gamma_2, for Hz vs kHz
    f = sampling_rate_hz
    rho = max(0, gamma[0] * sqrt(2 / pi * atan(gamma[1] * f)) + gamma[2])

    def hz_to_bark(freq_hz):
        return NotImplementedError()


# ---------------------------------------------------------
# These are from:
# (R3) "Perceptual linear predictive (PLP) analysis of speech"
#   -Hynek Hermansky, 1989
#   http://seed.ucsd.edu/mediawiki/images/5/5c/PLP.pdf


def hz_to_bark_1(w_hz):
    "converts frequency from Hz to Bark scale"
    w = w_hz / (1200 * pi)
    return 6.0 * log(w + sqrt(1 + w**2))


def bark_to_hz_1(w_bark):
    "converts frequency from Bark scale to Hz"
    w = sinh(w_bark / 6.0)
    return 1200 * pi * w


def mask_kernel(dw_bark):
    "Bark-scale convolution kernel for tone masking"
    if dw_bark < -1.3:
        return 0.0
    elif dw_bark < -0.5:
        return 10.0 ** (2.5 * (dw_bark + 0.5))
    elif dw_bark < 0.5:
        return 1.0
    elif dw_bark < 2.5:
        return 10.0 ** (-1.0 * (dw_bark - 0.5))
    else:
        return 0.0


def equalize(w_hz):
    "energy equalizing factor for equal-loudness over frequencies"
    w2 = w_hz**2
    return (
        w2**2 * (w2 + 5.68e7) / (w2 + 6.3e6) ** 2 / (w2 + 3.8e8) / (w2**3 + 9.58e26)
    )  # optional term for cutoff above 5kHz


# -----------------------------------------------------------------------------
# These is from
# (R4) http://www.ling.su.se/staff/hartmut/bark.htm


def hz_to_bark_2(f):
    return (26.81 / (1.0 + 1960.0 / f)) - 0.53


def bark_to_hz_2(z):
    return 1960.0 / (26.81 / (z + 0.53) - 1.0)


# ----( testing )--------------------------------------------------------------


def test1():
    from matplotlib import pyplot

    w_min = 8.0
    w_max = 5000.0  # 22050.0

    log_W = arange(log(w_min), log(w_max), log(2) / 12)
    W = exp(log_W)

    if False:
        pyplot.figure()
        pyplot.title("equal-loudness curve")
        E = array([equalize(w) for w in W])
        pyplot.plot(log_W, E)

    if True:
        pyplot.figure()
        pyplot.xlabel("log(frequency) (Hz)")
        pyplot.ylabel("bark")
        b = array([hz_to_bark(w) for w in W])
        pyplot.plot(log_W, b)


if __name__ == "__main__":
    test1()
