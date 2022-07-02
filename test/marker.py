#!/usr/bin/python

import sys, os, re

import Image, ImageDraw

from numpy import *
from scipy import *
from matplotlib import pyplot

import optim
import main


def read_image(filename):
    "print reading marker from %s" % filename
    im = Image.open(filename)
    x = misc.fromimage(im)
    LB, UB = x.min(), x.max()
    return 1.0 / (UB - LB) * (x - LB)


# ----( marker drawing )-------------------------------------------------------

draw_shape = {}


def draw_(shape):
    def deco(fun):
        draw_shape[shape] = fun
        return fun

    return deco


@draw_("dot")
def draw_dot(size, radius, x0, y0, u=0, v=0):
    t = arange(size)
    o = ones(size)
    x2 = outer(o, (t - x0) ** 2)
    y2 = outer((t - y0) ** 2, o)
    # xy = outer(t - y0, t - x0) TODO add tilt
    r = sqrt(x2 + y2)
    f = r - radius
    return maximum(-1, minimum(2 * f, 1))


def _draw_ring(size, x0, y0, r_out, r_in):
    assert r_in < r_out
    t = arange(size)
    o = ones(size)
    x2 = (t - x0) ** 2
    y2 = (t - y0) ** 2
    r = sqrt(outer(o, x2) + outer(y2, o))
    f = (r_out - r) * (r - r_in) / (r_out - r_in)
    return maximum(-1, minimum(-2 * f, 1))


@draw_("ring")
def draw_ring(size, radius, x0, y0):
    return _draw_ring(size, x0, y0, radius, radius / 2)


@draw_("ring2")
def draw_ring2(size, radius, x0, y0):
    return _draw_ring(size, x0, y0, radius, radius * 2 / 3) + draw_dot(
        size, radius / 3, x0, y0
    )


@draw_("ring3")
def draw_ring3(size, radius, x0, y0):
    return _draw_ring(size, x0, y0, radius * 4 / 4, radius * 3 / 4) + _draw_ring(
        size, x0, y0, radius * 2 / 4, radius * 1 / 4
    )


@draw_("ring4")
def draw_ring4(size, radius, x0, y0):
    return (
        _draw_ring(size, x0, y0, radius * 5 / 5, radius * 4 / 5)
        + _draw_ring(size, x0, y0, radius * 3 / 5, radius * 2 / 5)
        + draw_dot(size, radius / 5, x0, y0)
    )


@draw_("dots3")
def draw_dots3(size, radius, x0, y0):
    r = 4.0
    dx = radius / sqrt(3)
    dy = radius / 2.0
    return (
        draw_dot(size, r, x0 + dx, y0)
        + draw_dot(size, r, x0 - dx / 2, y0 + dy)
        + draw_dot(size, r, x0 - dx / 2, y0 - dy)
    )


shapes = list(draw_shape.keys())


def drawing_error(draw, im):
    X, Y = im.shape
    assert X == Y
    XY = (X * Y,)

    def fun(param, show=False):
        log_r, x0, y0 = param
        r = exp(log_r)
        diff = im - draw(X, r, x0, y0)
        if show:
            misc.toimage(diff**2).resize((400, 400), Image.NEAREST).show()
        return linalg.norm(diff)

    return fun


# ----( main commands )--------------------------------------------------------

noise_level = 0.2


@main.command
def read(filename):
    "Display image from file"
    x = read_image(filename)
    x = 1 - x
    im = misc.toimage(x)
    im.show()


@main.command
def draw(shape, radius=50.0, noise_level=0.0, size=128):
    "Display picutre of shape"

    val = draw_shape[shape](size, radius, size / 2, size / 2)
    if noise_level:
        val += 2 * noise_level * randn(*val.shape)

    im = misc.toimage(val, mode="L")
    if noise_level:
        filename = "%s-noisy.png" % shape
    else:
        filename = "%s.png" % shape
    im.save(filename)
    im.show()

    return val


@main.command
def draw_all(radius=120, size=256):
    "Draws all shapes"

    for shape in shapes:
        draw(shape, radius, 0, size)


@main.command
def fit(shape, radius=20.0, iters=10):
    "Fit shape to noisy image, display pixel errors"
    draw = draw_shape[shape]

    size = int(3 * radius)
    im = draw(size, radius, 0.5 * size, 0.5 * size)
    im += 2 * noise_level * randn(*im.shape)

    truth = array(
        [
            log(radius),
            0.5 * size,
            0.5 * size,
        ]
    )
    sigma = array(
        [
            0.1,
            1.0,
            1.0,
        ]
    )
    mean = truth + sigma * randn(3)
    sigma
    fun = drawing_error(draw, im)

    def print_mean():
        r, x, y = exp(mean[0]), mean[1], mean[2]
        print("radius = %g, x = %g, y = %g" % (r, x, y))

    print("fitting dot radius of %i x %i image" % im.shape)
    print_mean()
    optim.nonlinear_minimize(fun, mean, sigma, iters)
    print_mean()

    if main.at_top():
        fun(mean, True)

    error = mean - truth
    print("error = %s" % error)
    return error


@main.command
def errors(shape, radius=20.0, trials=100):
    "Plot shape fitting errors for multiple MC trials"

    errors = []
    for i in range(trials):
        errors.append(fit(shape, radius))

    def safe(e):
        e = abs(e)
        if e < 1:
            return e
        else:
            return 1

    r_errors = array([safe(e[0]) for e in errors])
    xy_errors = array([safe(sqrt(e[2] ** 2 + e[2] ** 2)) for e in errors])
    r_mean = exp(sum(log(r_errors)) / trials)
    xy_mean = exp(sum(log(xy_errors)) / trials)

    if main.at_top():
        pyplot.figure()
        pyplot.loglog(r_errors, xy_errors, "r.")
        pyplot.loglog([r_mean], [xy_mean], "k+")
        pyplot.xlabel("radius error (px)")
        pyplot.ylabel("position error (px)")
        pyplot.title(
            "Precision of %g pixel %s markers (%i trials)"
            % (2 * radius, shape.capitalize(), trials)
        )
        pyplot.savefig("%s-%g-error.png" % (shape, radius))
        pyplot.savefig("%s-%g-error.pdf" % (shape, radius))
        pyplot.show()

    return r_mean, xy_mean


@main.command
def curves(*radii):
    "Plot geometric mean fitting error vs radius"
    radii = [float(r) for r in radii]
    radii.sort()

    r_shapes = dict((shape, []) for shape in shapes)
    xy_shapes = dict((shape, []) for shape in shapes)

    for radius in radii:
        for shape in shapes:
            r, xy = errors(shape, radius)
            r_shapes[shape].append(r)
            xy_shapes[shape].append(xy)

    if main.at_top():

        pyplot.figure()
        for s in shapes:
            pyplot.plot(radii, r_shapes[s], label=s)
        pyplot.xlabel("marker radius (px)")
        pyplot.ylabel("mean radius error (px)")
        pyplot.title("Precision of shapes: %s" % ", ".join(shapes))
        pyplot.legend()
        pyplot.savefig("radius-curves.png")
        pyplot.savefig("radius-curves.pdf")

        pyplot.figure()
        for s in shapes:
            pyplot.plot(radii, xy_shapes[s], label=s)
        pyplot.xlabel("marker radius (px)")
        pyplot.ylabel("position error (px, geometric mean)")
        pyplot.title("Precision of shapes: %s" % ", ".join(shapes))
        pyplot.legend()
        pyplot.savefig("position-curves.png")
        pyplot.savefig("position-curves.pdf")

        pyplot.show()


@main.command
def ranges(shape, min_rad=1.0, max_rad=20.0, steps=100):
    "Plot sampled fitting error vs radius"

    L0 = log(min_rad)
    L1 = log(max_rad)
    dL = (L1 - L0) / steps
    radii = exp(arange(L0, L1, dL))

    r_errors = []
    xy_errors = []
    for radius in radii:
        r, xy = errors(shape, radius, trials=1)
        r_errors.append(r)
        xy_errors.append(xy)
    r_errors = array(r_errors)
    xy_errors = array(xy_errors)

    if main.at_top():

        pyplot.figure()
        pyplot.loglog(2 * radii, 2 * r_errors, "k.")
        pyplot.xlabel("marker size (px)")
        pyplot.ylabel("size error (px)")
        pyplot.ylim(1e-3, 1)
        pyplot.title("%s marker radius precision" % shape.capitalize())
        pyplot.savefig("%s-radius-ranges.png" % shape)
        pyplot.savefig("%s-radius-ranges.pdf" % shape)

        pyplot.figure()
        pyplot.loglog(2 * radii, xy_errors, "k.")
        pyplot.xlabel("marker size (px)")
        pyplot.ylabel("position error (px)")
        pyplot.ylim(1e-3, 1)
        pyplot.title("%s marker position precision" % shape.capitalize())
        pyplot.savefig("%s-position-ranges.png" % shape)
        pyplot.savefig("%s-position-ranges.pdf" % shape)

        pyplot.show()

    return radii, r_errors, xy_errors


if __name__ == "__main__":
    main.main()
