#!/usr/bin/python

from numpy import *
from scipy.integrate import quad
from matplotlib import pyplot
import main


def TODO(message=""):
    raise NotImplementedError("TODO %s" % message)


# ----( concept )--------------------------------------------------------------


@main.command
def components(max_order=4, periods=2.0):
    "Plots Fourier basis functions up to given order"

    P = arange(0, periods, 0.01)
    T = 2 * pi * P

    pyplot.figure()

    pyplot.plot(P, ones(T.shape) / (2 * pi), color="k", linewidth=2)

    for n in range(1, 1 + max_order):
        pyplot.plot(P, cos(n * T))
        pyplot.plot(P, sin(n * T))

    pyplot.show()


@main.command
def deltas(max_order=8):
    "Plots approximations to delta function up to given order"

    pyplot.figure()

    P = arange(-0.5, 0.5, 0.002)
    T = 2 * pi * P
    D = ones(T.shape)

    pyplot.plot(P, D)
    pyplot.text(
        0,
        1.05,
        "N = 1",
        horizontalalignment="center",
        fontsize="x-small",
    )

    for n in range(1, 1 + max_order):
        D += cos(n * T)
        pyplot.plot(P, D)
        pyplot.text(
            0,
            1.05 + n,
            "%i" % (1 + 2 * n),
            horizontalalignment="center",
            fontsize="x-small",
        )

    pyplot.xlim(-0.5, 0.5)
    pyplot.ylim(-2, n + 2)
    pyplot.title("Approximate delta functions with N components")

    pyplot.show()


# ----( beating )--------------------------------------------------------------

"""
Fourier basis:
        1
  [ ----------, cos(t), sin(t), cos(2 t), sin(2 t), ..., cos(n t), sin(n t) ]
    sqrt(2 pi)

beat function
"""


def beat_integral(f, acuity, tol=1e-8):
    "Computes complex beat-weighted integral of a periodic function"

    r = pi / acuity

    def beat(t):
        return (cos(t) - cos(r)) / (1 - cos(r))

    def integrand_x(t):
        return f(t) * beat(t) * cos(t)

    def integrand_y(t):
        return f(t) * beat(t) * sin(t)

    int_x, error_x = quad(integrand_x, -r, r)
    int_y, error_y = quad(integrand_y, -r, r)

    assert error_x < tol
    assert error_y < tol
    assert (abs(int_x) < tol) or (abs(int_y) < tol)

    return int_x + 1.0j * int_y


def c2str(z, tol=1e-8):
    x = z.real
    y = z.imag
    if abs(y) < tol:
        return "%s" % x
    if abs(x) < tol:
        return "%sj" % y
    return string(z)


@main.command
def beat_mean(max_order=4, acuity=3.0):
    "Computes beat-mean integral vector in Fourier coords."
    print("\nBeat mean = <e_n|beat|")
    print("  (acuity = %g)" % acuity)
    print("m\tc\t\ts")
    print("-" * (8 * 5))
    for n in range(1 + max_order):
        print(
            "\t".join(
                str(x)
                for x in (
                    n,
                    c2str(beat_integral(lambda t: cos(n * t), acuity)),
                    c2str(beat_integral(lambda t: sin(n * t), acuity)),
                )
            )
        )


@main.command
def beat_response(max_order=4, acuity=3.0):
    "Computes beat response integral matrix in Fourier coords."

    print("\nBeat response = <e_m | -i beat d_theta | e_n>")
    print("  (acuity = %g)" % acuity)
    print("m\tn\tcc\t\tcs\t\tsc\t\tss")
    print("-" * (8 * 9))
    for m in range(1 + max_order):
        for n in range(1, 1 + max_order):
            print(
                "\t".join(
                    str(x)
                    for x in (
                        m,
                        n,
                        c2str(
                            1.0j
                            * beat_integral(
                                lambda t: cos(m * t) * n * sin(n * t), acuity
                            )
                        ),
                        c2str(
                            1.0j
                            * beat_integral(
                                lambda t: -cos(m * t) * n * cos(n * t), acuity
                            )
                        ),
                        c2str(
                            1.0j
                            * beat_integral(
                                lambda t: sin(m * t) * n * sin(n * t), acuity
                            )
                        ),
                        c2str(
                            1.0j
                            * beat_integral(
                                lambda t: -sin(m * t) * n * cos(n * t), acuity
                            )
                        ),
                    )
                )
            )


@main.command
def beat_code(order=4, acuity=3.0):
    "Generates C++ code for beat mean & beat response tables"

    N = list(range(1, 1 + order))

    def I(f):
        z = beat_integral(f, acuity)
        assert abs(z.real) * abs(z.imag) < 1e-16
        return z

    def real1(Z):
        return "{\n%s\n}" % ",\n".join(["  %s" % z.real for z in Z])

    def imag1(Z):
        return "{\n%s\n}" % ",\n".join(["  %s" % z.imag for z in Z])

    def real2(Z2):
        return "{%s}" % ",".join([real1(Z) for Z in Z2])

    def imag2(Z2):
        return "{%s}" % ",".join([imag1(Z) for Z in Z2])

    def let(eq):
        print("const Real %s;\n" % eq)

    def let_real(name, z):
        let("%s_x = %s" % (name, z.real))

    def let_real1(name, Z):
        let("%s_x[order] = %s;\n" % (name, real1(Z)))

    def let_imag1(name, Z):
        let("%s_y[order] = %s;\n" % (name, imag1(Z)))

    def let_real2(name, Z2):
        let("%s_x[order][order] = %s" % (name, real2(Z2)))

    def let_imag2(name, Z2):
        let("%s_y[order][order] = %s" % (name, imag2(Z2)))

    print("// This following code was automatically generated by")
    print("//   test/fourier_loop.py beat-code [order] [acuity]")
    print("")
    print("namespace Beat")
    print("{")
    print("")
    print("enum { order = %s };" % order)
    print("const Real acuity = %s;" % acuity)
    print("")

    print("//----( beat mean )----\n")

    let_real("mean_dc", I(lambda t: 1))
    let_real1("mean_cos", [I(lambda t: cos(n * t)) for n in N])
    let_imag1("mean_sin", [I(lambda t: sin(n * t)) for n in N])

    print("//----( beat response )----\n")

    let_imag1("response_dc_cos", [I(lambda t: -sin(n * t)) for n in N])
    let_real1("response_dc_sin", [I(lambda t: cos(n * t)) for n in N])

    let_real2(
        "response_cos_cos",
        [[1.0j * I(lambda t: cos(m * t) * n * sin(n * t)) for m in N] for n in N],
    )

    let_imag2(
        "response_cos_sin",
        [[1.0j * I(lambda t: cos(m * t) * n * (-cos(n * t))) for m in N] for n in N],
    )

    let_imag2(
        "response_sin_cos",
        [[1.0j * I(lambda t: sin(m * t) * n * sin(n * t)) for m in N] for n in N],
    )

    let_real2(
        "response_sin_sin",
        [[1.0j * I(lambda t: sin(m * t) * n * (-cos(n * t))) for m in N] for n in N],
    )

    print("} // namespace Beat")


# ----( main )-----------------------------------------------------------------

if __name__ == "__main__":
    main.main()
