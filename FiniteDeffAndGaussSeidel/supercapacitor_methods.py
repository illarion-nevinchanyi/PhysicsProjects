# Super capacitor energy storage (methods)
# TuGraz Computational physics - Assignment 1 Exercise 1
# Authors: Christoph Kircher, Gabriele Maschera, Illarion Nevinchanyi
# Date: 2023-10-25

import numpy as np


def get_matrix_A(N, delta_x, k):
    """
    Creates the matrix A used to solve the linear equation of the problem

    :param N: size of matrix
    :param delta_x: difference between to x values (step size)
    :param k: the constant given
    :return: NxN matrix which represents the differential equation
    """
    A = np.zeros((N, N))
    A[range(N), range(N)] = -2 - delta_x ** 2 * k ** 2  # main diagonal
    A[range(N - 1), range(1, N)] = 1  # upper secondary diagonal
    A[range(1, N), range(N - 1)] = 1  # lower secondary diagonal
    return A


def lu_composition(A):
    """
    Method doing the LU composition: A = LU
    L is a lower triangular matrix with 1 at the diagonal
    U is an upper triangular matrix

    :param A: the matrix to composite
    :return: matrices L and U
    """
    N = np.shape(A)[0]
    U = np.zeros((N, N))
    L = np.zeros((N, N))
    G = np.zeros((N, N))

    for j in range(N):
        L[j, j] = 1  # main diagonal of L is 1
        for i in range(j + 1):
            U[i, j] = A[i, j] - np.sum([L[i, k] * U[k, j] for k in range(i)])
        for i in range(j, N):
            G[i, j] = A[i, j] - np.sum([L[i, k] * U[k, j] for k in range(j)])
            U[j, j] = G[j, j]
            L[i, j] = G[i, j] / U[j, j]
    return L, U


def forward_substitution(z, b, L):
    """
    Doing the forward substitution of solving with LU composition

    :param z: vector z with boundary conditions
    :param b: the constant vector of Ax=b
    :param L: matrix L from A=LU
    :return: the calculated values for z
    """
    N = np.shape(L)[0]
    for i in range(1, N):
        z[i] = b[i] - np.sum([L[i, k] * z[k] for k in range(i)])
    return z


def backward_substitution(y, z, U):
    """
    Doing the backward substitution of solving with LU composition

    :param y: vector y with boundary conditions
    :param z: vector calculated at forward substitution
    :param U: matrix U from A=LU
    :return: the calculated values for y
    """
    N = np.shape(U)[0]
    for i in range(N - 2, 0, -1):
        y[i] = (1 / U[i, i]) * (z[i] - np.sum([U[i, k] * y[k] for k in range(i, N)]))
    return y
