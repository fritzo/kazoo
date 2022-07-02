# ----( loudness conversions )-------------------------------------------------


def log_m_to_loudness(image):
    import numpy

    image = numpy.exp(image * 2 / 3)
    image -= image.min()
    image /= image.max()
    return image


def energy_to_loudness(image, gamma=1.0):
    image = pow(image, gamma / 3)
    image -= image.min()
    image /= image.max()
    return image


def complex_to_loudness(image):
    image = pow(numpy.abs(image), 2.0 / 3)
    image -= image.min()
    image /= image.max()
    return image
