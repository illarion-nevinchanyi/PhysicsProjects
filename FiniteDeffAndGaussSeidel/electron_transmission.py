# Electron transmission through a molecular transport system
# TuGraz Computational physics - Assignment 1 Exercise 2
# Authors: Christoph Kircher, Gabriele Maschera, Illarion Nevinchanyi
# Date: 2023-10-25

import matplotlib.pyplot as plt
from electron_transmission_methods import *


def test_gauss_seidl(N=100):
    """
    Method for testing the gauss seidel method with a large matrix

    :param N: Size of the matrix
    """

    print('Start testing the Gauss-Seidel Algorithm with a random matrix with size N={:d}'.format(N))

    A = np.random.rand(N, N)
    b = np.random.rand(N)

    # Make matrix A diagonal dominant. N is used as constant because np.random.rand returns values in interval (0, 1)
    A = A + np.identity(N) * N

    x, (plt_iteration, plt_errors) = gauss_seidel(A, b, iterations=50, target_error=0)

    plt.semilogy(plt_iteration, plt_errors)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Total Error of b')
    plt.title('Error vs Iterations for a matrix of size ' + str(N) + 'x' + str(N))
    plt.show()

    print('\n')


def electron_transmission(alpha, beta, N=6):
    """
    Method to calculate the electron transmission probability through the ring

    :param alpha: the first index the ring is connected to the wall
    :param beta: the second index the ring is connected to the wall
    :param N: Size of the matrix and therefore the ring
    """

    print('Start calculating electron transmission for alpha={} and beta={}'.format(alpha, beta))

    eltr_plots(alpha, beta, N=N)
    eltr_eigenvalues(alpha, beta, N=N)

    print('\n')


def eltr_eigenvalues(alpha, beta, N=6, t=-2.6):
    """
    Calculates the eigenvalues of the ring using the method numpy.linalg.eig

    :param alpha: the first index the ring is connected to the wall
    :param beta: the second index the ring is connected to the wall
    :param N: Size of the matrix and therefore the ring
    :param t: Represents the probability of transmission between atoms
    """

    A = ring(N=N, E=0, t=t, delta=0, alpha=alpha, beta=beta)
    print('The matrix A is:')
    print(A)

    (eigenvalues, _) = np.linalg.eig(A)
    print('\nThe eigenvalues of the matrix A are:')
    print(eigenvalues)


def eltr_plots(alpha, beta, N=6, t=-2.6, delta=0.5, build_in=False):
    """
    Calculates and plots the electron transmission probability through the ring

    :param alpha: the first index the ring is connected to the wall
    :param beta: the second index the ring is connected to the wall
    :param N: Size of the matrix and therefore the ring
    :param t: Represents the probability of transmission between atoms
    :param delta: Value of energy broadening of the electrons
    :param build_in: Use the build in method to solve the linear equation system (for debugging)
    """

    energies = np.linspace(-6, 6, 121, True)  # use num=61
    T_aa = np.zeros(len(energies))
    T_bb = np.zeros(len(energies))
    T_ab = np.zeros(len(energies))
    T_ba = np.zeros(len(energies))

    for i in range(len(energies)):
        print("Calculate for Energy E={:.2f}".format(energies[i]))
        A = ring(N=N, E=energies[i], t=t, delta=delta, alpha=alpha, beta=beta)

        # calculate a nice k with the given values of t
        k = complex(0, -2 * t) + 0.1j
        G = solve_G_matrix(A, N, k=k, build_in=build_in)
        T_aa[i] = np.square(np.abs(G[alpha, alpha]))
        T_bb[i] = np.square(np.abs(G[beta, beta]))
        T_ab[i] = np.square(np.abs(G[alpha, beta]))
        T_ba[i] = np.square(np.abs(G[beta, alpha]))

    plt.plot(energies, T_aa, '-r', label='T_aa')
    plt.plot(energies, T_bb, ':b', label='T_bb')
    plt.plot(energies, T_ab, '-g', label='T_ab')
    plt.plot(energies, T_ba, ':k', label='T_ba')
    plt.xlabel('Energy')
    plt.ylabel('Transmission probability')
    plt.legend(loc="upper left")
    plt.title('Electron transmission probability for alpha=' + str(alpha) + ' and beta=' + str(beta))
    plt.show()


def solve_G_matrix(A, N, k=None, build_in=False):
    """
    Calculates the G matrix which represents the probability of the electrons transmission

    :param A: Complex NxN matrix representing (H_R - EI + iD)
    :param N: The size of the matrix A
    :param k: The constant used to make matrix A main dominant
    :param build_in: Use the build in method to solve the linear equation system (for debugging)
    :return: calculated values of the matrix G
    """
    G = np.ones((N, N), dtype='complex')
    for col in range(N):
        b = np.zeros(N)
        b[col] = 1

        if build_in:
            x = np.linalg.solve(A, b)
        else:
            x, (_, _) = gs_not_dominant(A, b, k=k, dtype='complex')

        # Use this line to get an update of the calculated columns
        # print("  Colum {:d}/{:d}".format(col + 1, N))
        G[range(N), col] = x
    return G


test_gauss_seidl(N=100)
test_gauss_seidl(N=6)
electron_transmission(1, 3)
electron_transmission(1, 4)
