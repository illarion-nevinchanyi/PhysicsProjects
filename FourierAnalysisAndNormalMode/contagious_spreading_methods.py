# Contagious_spreading methods
# TuGraz Computational physics - Assignment 2 Exercise 2
# Authors: Christoph Kircher, Gabriele Maschera, Illarion Nevinchanyi
# Date: 2023-11-06

import numpy as np
from Assignment_1.supercapacitor_methods import lu_composition, backward_substitution, forward_substitution


def fit_polynom(x, y, m=1, g=None):
    """
    Fits a given dataset (x, y) to a polynomial of grade m

    :param x: first values of data points
    :param y: second values of data points
    :param m: number of basis functions (grade)
    :param g: weighting functions lambda g(k)
    :return: fitted values, (A, a, beta)
    """
    polynomial = lambda variable, index: variable ** index
    return fit_function(x, y, polynomial, m=m, g=g)


def fit_exponential(x, y, m=1, g=None):
    """
    Fits a given dataset (x, y) to exponential of grade 1/m

    :param x: first values of data points
    :param y: second values of data points
    :param m: number of basis functions (grade)
    :param g: weighting functions lambda g(k)
    :return: fitted values, (A, a, beta)
    """
    exponential = lambda variable, index: 1 if index == 0 else 1 - np.exp(-variable / index)
    return fit_function(x, y, exponential, m=m, g=g)


def fit_function(x, y, basis_function, m=1, g=None):
    """
    Fit given data points (x, y) to polynomials with grade m

    :param x: first values of data points
    :param y: second values of data points
    :param basis_function: lambda expression f_i(x)=f(x, i) to calculate the
    :param m: number of basis functions (grade)
    :param g: weighting functions lambda g(k)
    :return: fitted values, (A, a, beta)
    """
    n = np.shape(x)[0]
    A, beta = get_fit_coefficient_equation(x, y, basis_function, m=m, g=g)
    a = solve_with_lu(A, beta)

    y_fit = np.zeros(n)
    for i in range(n):
        y_fit[i] = np.sum([a[k] * basis_function(x[i], k) for k in range(m)])

    return y_fit, (A, a, beta)


def get_fit_coefficient_equation(x, y, basis_function, m=1, g=None):
    """
    Creates the matrix A and the vector beta of Aa=beta to get the coefficients of the fitted function

    :param x: first values of data points
    :param y: second values of data points
    :param basis_function: lambda expression f_i(x)=f(x, i) to calculate the
    :param m: number of basis functions (grade)
    :param g: weighting functions
    :return: matrix A and vector beta of Aa=beta
    """
    n = np.shape(x)[0]
    if g is None:
        g = lambda index: 1

    beta = np.zeros(m)
    A = np.zeros([m, m])

    for i in range(m):
        beta[i] = np.sum([g(k) * y[k] * basis_function(x[k], i) for k in range(n)])

    for i in range(m):
        for j in range(m):
            A[i, j] = np.sum([g(k) * basis_function(x[k], i) * basis_function(x[k], j) for k in range(n)])

    return A, beta


def solve_with_lu(A, b):
    """
    Solves a given system of linear equations Ax=b with lu composition

    :param A: matrix of system of linear equations
    :param b: vector of system of linear equations
    :return: the solution of Ax=b
    """
    n = np.shape(b)[0]
    L, U = lu_composition(A)

    # Defining z see page 81
    y = np.zeros(n)
    z = np.zeros(n)  # y already in use, z is what in LU in script is y (solve z = Ux)
    for i in range(n):
        z[i] = U[i, i] * y[i]  # see eq (5.19)

    # for forward sub. z_1 = b_1, but yields bad results, therefore eq 5.19
    z = forward_substitution(z, b, L)
    y = backward_substitution(y, z, U)

    return y

def commit_succeed():
    '''Not relevant for the main code. Just to make sure, that this is the latest version of the code'''
    return None