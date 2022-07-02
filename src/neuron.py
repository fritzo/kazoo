from pylab import *
from scipy import *
from numpy import *
from scipy import special
from matplotlib import pyplot
from random import normalvariate


class Model:
    "sketch of abstract graphical model class"

    def __init__(self, shape):
        self.params = None
        self.state = None
        pass

    def gibbs_sample(self, num_steps, temperature=1.0, fixed=[]):
        pass

    def remember(self, rate, cd_steps=1):
        "contrastive divergence learning"
        pass

    def generate(self, cd_steps=100):
        pass


def test1():
    "2-d oscillator"

    def time_deriv(xxx_todo_changeme, omega, delta, beta, nu=0.1):
        (x, y) = xxx_todo_changeme
        rr = x**2 + y**2
        return (
            omega * y + (delta - rr) * x,
            -omega * (x - beta) + (delta - rr) * y + normalvariate(0, nu),
        )

    N = 8000
    dt = 0.05

    def omega(t):
        return 2 * pi

    def delta(t):
        return 2 * t - 1

    def beta(t):
        # return 2 * (t - 0.5)
        return sin(8 * pi * t)
        # return 0.01

    X = zeros(N)
    Y = zeros(N)
    for i in range(1, N):
        t = (i + 0.5) / N
        dx, dy = time_deriv((X[i - 1], Y[i - 1]), omega(t), delta(t), beta(t))
        X[i] = X[i - 1] + dt * dx
        Y[i] = Y[i - 1] + dt * dy

    figure()
    plot(X)
    # figure(); plot(X,Y)


def test2a():
    "oscilattor on scircle, exhibiting spiking behavior"

    def time_deriv(xxx_todo_changeme1, omega, theta, delta, nu=0.1):
        (x, y) = xxx_todo_changeme1
        x0 = sin(theta)
        y0 = cos(theta)
        return (
            omega * y - delta * (x - x0) + normalvariate(0, nu),
            -omega * x - delta * (y - y0) + normalvariate(0, nu),
        )

    N = 2000
    dt = 0.05

    def omega(t):
        return 2 * pi

    def delta(t):
        return 2 * pi * (2 * t - 1)

    def theta(t):
        return 0

    X = zeros(N)
    Y = zeros(N)
    for i in range(1, N):
        t = (i + 0.5) / N
        dx, dy = time_deriv((X[i - 1], Y[i - 1]), omega(t), theta(t), delta(t))
        x = X[i - 1] + dt * dx
        y = Y[i - 1] + dt * dy
        r = sqrt(x**2 + y**2)
        X[i] = x / r
        Y[i] = y / r

    figure()
    plot(X)


def test2b():
    "oscilattor on scircle, exhibiting spiking behavior"

    def time_deriv(xxx_todo_changeme2, omega, theta, delta, nu=0.1):
        (x, y) = xxx_todo_changeme2
        x0 = sin(theta)
        y0 = cos(theta)
        return (
            omega * y - delta * (x - x0) + normalvariate(0, nu),
            -omega * x - delta * (y - y0) + normalvariate(0, nu),
        )

    N = 2000
    dt = 0.05

    def omega(t):
        return 2 * pi

    def delta(t):
        return 2 * pi * t

    def theta(t):
        return 0

    def nu(t):
        return 1

    X = zeros(N)
    Y = zeros(N)
    for i in range(1, N):
        t = (i + 0.5) / N
        dx, dy = time_deriv((X[i - 1], Y[i - 1]), omega(t), theta(t), delta(t), nu(t))
        x = X[i - 1] + dt * dx
        y = Y[i - 1] + dt * dy
        r = sqrt(x**2 + y**2)
        X[i] = x / r
        Y[i] = y / r

    figure()
    plot(X)


def test3():
    "complex oscillator"

    def time_deriv(xxx_todo_changeme3, omega, beta, nu):
        (x, y) = xxx_todo_changeme3
        damp = 16 * (1 - (x**2 + y**2))
        return (
            omega * y + damp * x,
            -omega * x + damp * y + beta + normalvariate(0, nu),
        )

    N = 4000
    dt = 0.05

    def omega(t):
        return 2 * pi

    def beta(t):
        return 2 * pi * sin(4 * pi * t)
        # return 0.01

    def nu(t):
        return 0.1

    X = zeros(N)
    Y = zeros(N)
    for i in range(1, N):
        t = (i + 0.5) / N
        dx, dy = time_deriv((X[i - 1], Y[i - 1]), omega(t), beta(t), nu(t))
        X[i] = X[i - 1] + dt * dx
        Y[i] = Y[i - 1] + dt * dy

    figure()
    plot(X)
    # figure(); plot(X,Y)


def test_pdfs():
    "demonstrates that wrapped normal and vonmises are close for small variance"

    def normal_pdf(x, sigma):
        return exp(-((x / sigma) ** 2) / 2) / sqrt(2 * pi) / sigma

    def vonmises_pdf(x, sigma):
        return exp(cos(x) / sigma**2) / (2 * pi * special.iv(0, 1 / sigma**2))

    x = linspace(-pi, pi, 100)

    pyplot.plot(x, normal_pdf(x, 0.3))
    pyplot.plot(x, vonmises_pdf(x, 0.3))
