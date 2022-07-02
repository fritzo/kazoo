#!/usr/bin/python

from numpy import *
import main


@main.command
def plot_beta(min_alpha=1e-1, max_alpha=1e1, curves=10, samples=1000):
    "plots chi^2 likelihood for various DOF values"

    from matplotlib import pyplot

    a01 = (arange(curves) + 0.5) / curves
    alphas = min_alpha * pow(max_alpha / min_alpha, a01)

    X = (arange(samples) + 0.5) / samples
    for alpha in alphas:
        Y = pow(1 - X, alpha - 1) * alpha
        pyplot.plot(X, Y, "r-")

    pyplot.title("beta distributions used in the stick distribution")
    pyplot.ylim(0, max_alpha)
    pyplot.show()


if __name__ == "__main__":
    main.main()
