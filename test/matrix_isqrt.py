#!/usr/bin/python

from numpy import *
from matplotlib import pyplot
import main


def pade_sqrt(t, order=2):
    """
    Returns Pade approximation to square root, as described in:
      "A Pade Approximation Method For Square Roots
        Of Symmetric Positive Definite Matrices"
      -Ya Yan Lu
    """
    x = t - 1
    denom = 2.0 * order + 1
    result = 1
    for i in range(1, 1 + order):
        a = 2 / denom * sin(pi * i / denom) ** 2
        b = cos(pi * i / denom) ** 2
        result += a * x / (1 + b * x)
    return result


def pade_isqrt(t, order=2):
    return 1 / pade_sqrt(t, order)


def heron_isqrt(t, order=2):
    """
    Returns Heron's approximation of inverse square root.
    from:
    http://en.wikipedia.org/wiki/Square_root#Computation
    """
    x = 1.0
    for _ in range(1 + order):
        x = (x + 1 / (t * x)) / 2
    return x


def heron2_isqrt(t, order):
    """
    Returns inverse of Heron's approximation of square root.
    from:
    http://en.wikipedia.org/wiki/Square_root#Computation
    """
    x = 1.0
    for _ in range(1 + order):
        x = (x + t / x) / 2
    return 1 / x


@main.command
def plot_sqrt(max_order=4):
    "Plots first few Pade approximations of square root"
    max_order = int(max_order)

    pyplot.figure()
    T = arange(0.01, 10, 0.01)
    for order in range(max_order + 1):
        print(pade_sqrt(T, order))
        pyplot.plot(T, pade_sqrt(T, order))
    pyplot.xscale("log")
    pyplot.title("sqrt_4(0) = %g" % pade_sqrt(0, 4))
    pyplot.show()


@main.command
def plot_isqrt(method="pade", max_order=4):
    "Plots first few approximants of inverse square root"
    max_order = int(max_order)

    isqrt = {
        "pade": pade_isqrt,
        "heron": heron_isqrt,
        "heron2": heron2_isqrt,
    }[method]

    pyplot.figure()
    T = arange(0.01, 10, 0.01)
    pyplot.plot(T, 1 / sqrt(T), "k:")
    for order in range(max_order + 1):
        pyplot.plot(T, isqrt(T, order))
    pyplot.xscale("log")
    pyplot.yscale("log")
    pyplot.title(
        "isqrt(0): %s, ..."
        % ", ".join(["%g" % isqrt(0, i) for i in range(max_order + 1)])
    )
    pyplot.show()


def sym33_mult(xxx_todo_changeme, xxx_todo_changeme1):

    (a00, a01, a02, a11, a12, a22) = xxx_todo_changeme
    (b00, b01, b02, b11, b12, b22) = xxx_todo_changeme1
    c00 = a00 * b00 + a01 * b01 + a02 * b02
    c01 = a00 * b01 + a01 * b11 + a02 * b12
    c02 = a00 * b02 + a01 * b12 + a02 * b22
    # ...
    raise NotImplementedError("tedious")

    return array([c00, c01, c02, c11, c12, c22])


def sym33_inverse(xxx_todo_changeme2):
    """
    from:
    http://en.wikipedia.org/wiki/Matrix_inverse#Inversion_of_3.C3.973_matrices
    """
    (a00, a01, a02, a11, a12, a22) = xxx_todo_changeme2
    det = (
        a00 * a11 * a22
        - a00 * a12**2
        - a11 * a02**2
        - a22 * a01**2
        + 2 * a01 * a02 * a12
    )

    # TODO check this
    b00 = a00 * a11 - a12**2
    b01 = a02 * a12 - a00 * a01
    b02 = a01 * a12 - a02 * a11
    b11 = a00**2 - a02**2
    b12 = a01 * a02 - a00 * a12
    b22 = a00 * a11 - a01**2

    return array([b00, b01, b02, b11, b12, b22]) / det


def sym33_isqrt(a, order=3, tol=1e-2):
    a = a + tol
    x = array([1, 0, 0, 1, 0, 1])
    for _ in range(order):
        x = 0.5 * (x + sym33_mult(a, sym33_inverse(x)))
    return sym33_inverse(x)


# ----( code generation )------------------------------------------------------

fst = [0, 0, 0, 1, 1, 2]
snd = [0, 1, 2, 1, 2, 2]
pair = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]


@main.command
def print_sym33_mult():
    "Prints c code for symmetric 3x3 matrix multiply"

    print("inline void multiply (const Sym33 & a, const Sym33 & b, Sym33 & c)")
    print("{")
    print("  // produced with python test/matrix_isqrt.py print-sym33-mult")
    print("")
    for ij in range(6):
        i = fst[ij]
        j = snd[ij]
        terms = ["a[i%d%d] * b[i%d%d]" % (i, k, k, j) for k in range(3)]
        print("  c[i%d%d] = %s;" % (i, j, " + ".join(terms)))
    print("}")


@main.command
def print_sym33_inv():
    "Prints c code for symmetric 3x3 matrix inverse"

    print("inline void inverse (const Sym33 & a, Sym33 & b)")
    print("{")
    print("  // produced with python test/matrix_isqrt.py print-sym33-inv")
    print("")

    def p(t):
        return (t + 1) % 3

    def n(t):
        return (t + 2) % 3

    for ij in range(6):
        i = fst[ij]
        j = snd[ij]
        pos = "a[i%d%d] * a[i%d%d]" % (p(j), p(i), n(j), n(i))
        neg = "a[i%d%d] * a[i%d%d]" % (p(j), n(i), n(j), p(i))
        print("  b[i%d%d] = %s - %s;" % (i, j, pos, neg))
    print("")
    print("  b *= 1.0f / det(a);")
    print("}")


if __name__ == "__main__":
    main.main()
