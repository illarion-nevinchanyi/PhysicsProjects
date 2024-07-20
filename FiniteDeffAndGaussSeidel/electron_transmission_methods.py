# Electron transmission through a molecular transport system (methods)
# TuGraz Computational physics - Assignment 1 Exercise 2
# Authors: Christoph Kircher, Gabriele Maschera, Illarion Nevinchanyi
# Date: 2023-10-25

import numpy as np


def ring(N, E, t, delta, alpha, beta):
    """
    This function returns the (H_R - EI + i*delta) matrix

    :param N: size of square matrix
    :param E: value of the energy in diagonal
    :param t: value of the electrons hopping to neighbour
    :param delta: value of energy broadening
    :param alpha: index of the first connection of the ring
    :param beta: index of the second connection of the ring
    :return: complex NxN matrix H_R - EI + i*delta
    """

    M = np.zeros((N, N), dtype=complex)  # (H - EI + i * delta(E))
    M[range(N), range(N)] = complex(-E)
    M[alpha, alpha] += complex(0, delta)
    M[beta, beta] += complex(0, delta)
    M[0, N - 1] = t
    M[N - 1, 0] = t
    M[range(1, N), range(N - 1)] = t  # lower secondary diagonal
    M[range(0, N - 1), range(1, N)] = t  # upper secondary diagonal

    return M


def is_diagonal_dominant(A):
    """
    Tests a matrix A if it is diagonal dominant

    :param A: complex NxN matrix
    :return: True if the given matrix A is diagonal dominant otherwise False
    """

    N = np.shape(A)[0]

    A_diag_zero = np.abs(A)
    A_diag_zero[range(N), range(N)] = 0
    row_abs_sum = np.sum(A_diag_zero, 1)
    diag_row_diff = np.abs(A[range(N), range(N)]) - row_abs_sum
    return np.min(diag_row_diff) > 0


def diagonal_dominant(A):
    """
    calculates the maximum absolute difference between the value in the diagonal to the sum of the remaining row in
    matrix A

    :param A: complex NxN matrix
    :return: value k >= 0 with the maximum row difference diagonal and sum of row
    """

    if is_diagonal_dominant(A):
        return 0

    N = np.shape(A)[0]

    A_diag_zero = np.abs(A)
    A_diag_zero[range(N), range(N)] = 0
    row_abs_sum = np.sum(A_diag_zero, 1)
    k = np.max(row_abs_sum - np.min(np.abs(A[range(N), range(N)])))

    # Todo: Is weak diagonal dominance (>= vs >) enough?
    k = k + 0.1
    return k


def gauss_seidel(A, b, iterations=100, omega=1, target_error=1e-12, dtype='float64'):
    """
    Solves linear equation system Ax=b with the iterative gauss seidel method.
    changes the matrix A to be diagonal dominant first.

    :param A: complex matrix (must be diagonal dominant)
    :param b: constant vector
    :param iterations: maximum number of iterations
    :param omega: relaxation parameter
    :param target_error: error value at which the iteration stops
    :param dtype: data type of the gauss seidel method
    :return:
    """

    N = np.shape(A)[0]
    x = np.zeros(N, dtype=dtype)
    delta_x = np.zeros(N, dtype=dtype)

    error_values = np.zeros(iterations)

    for t in range(iterations):

        # Loop through all coefficients of vector x
        for i in range(N):
            # Attention: x[j] for x<j is from this iteration
            #            x[j] for x>=j is from last iteration
            delta_x[i] = x[i] + ((1 / A[i, i])
                                 * (np.sum([A[i, j] * x[j] for j in range(i)])
                                    + np.sum([A[i, j] * x[j] for j in range(i + 1, N)])
                                    - b[i]))
            x[i] = x[i] - omega * delta_x[i]

        delta_b = b - np.matmul(A, x)
        error_values[t] = np.sum(np.abs(delta_b)) / N

        # Stop at error less than target error
        if error_values[t] < target_error:
            # print("  Stopped at iteration " + str(t))
            error_values = error_values[:t]
            return x, (range(1, t + 1), error_values)

    # print("  Stopped at iteration " + str(iterations))
    return x, (range(1, iterations + 1), error_values)


def gs_not_dominant(A, b, k=None, it_gs=100, it_b=100, omega=1, target_error=1e-2, dtype='float64'):
    """
    Solves a linear equation system without a main diagonal dominant matrix A. The Gauss Seidel algorithm is used.

    :param A: Complex NxN matrix
    :param b: constant vector
    :param k: constant added to the diagonal to make matrix A diagonal dominat
    :param it_gs: maximum number of iterations in the gauss seidel algorithm
    :param it_b: maximum number of iterations in the gauss seidel algorithm
    :param omega: relaxation parameter
    :param target_error: error value at which the iteration stops
    :param dtype: data type of the gauss seidel method
    :return: calculated values of the solution x
    """

    N = np.shape(A)[0]
    x = np.zeros(N, dtype=dtype)

    # Test which k should be used for calculating
    if k is None:
        if not is_diagonal_dominant(A):
            k = diagonal_dominant(A)
        else:
            k = 0

    A_m = A + np.identity(N, dtype=dtype) * k
    b_m = b

    error_values = np.zeros(it_b)

    for t in range(it_b):
        x, (_, gs_err) = gauss_seidel(A_m, b_m, iterations=it_gs, omega=omega, target_error=target_error, dtype=dtype)
        # Use this line to calculate the linear equation system using the numpy library
        # x = np.linalg.solve(A_m, b_m)
        b_m = b + k * x

        delta_b = b - np.matmul(A, x)
        error_values[t] = np.sum(np.abs(delta_b)) / N

        # Stop at error less than target error
        if error_values[t] < target_error:
            # print("Stopped at iteration " + str(t))
            error_values = error_values[:t]
            return x, (range(1, t + 1), error_values)

    # print(np.sum(np.abs(np.matmul(A, x) - b)))
    # print("Stopped at iteration " + str(it_b))

    return x, (range(1, it_b + 1), error_values)
