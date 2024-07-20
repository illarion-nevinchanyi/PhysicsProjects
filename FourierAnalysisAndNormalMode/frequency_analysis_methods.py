# Frequency analysis of a one-dimensional data set - methods
# TuGraz Computational physics - Assignment 2 Exercise 1
# Authors: Christoph Kircher, Gabriele Maschera, Illarion Nevinchanyi
# Date: 2023-11-06

import numpy as np


def fourier_transform(y, n=None, axis=-1):
    """
    The discrete fourier transform without prefactor and negative sign in exponent

    :param y: array-like data matrix
    :param n: length of the output matrix
    :param axis: axis length which is used as n
    :return: array-like fourier transformed matrix
    """
    if n is None:
        n = np.shape(y)[axis]

    Y = np.zeros(n, dtype='complex')

    i = np.linspace(0, n - 1, n)

    for k in range(n):
        Y[k] = np.sum(y * np.exp(-2j * np.pi * k * i) / n)
    return Y