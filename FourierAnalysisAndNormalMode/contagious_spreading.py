# Contagious_spreading
# TuGraz Computational physics - Assignment 2 Exercise 2
# Authors: Christoph Kircher, Gabriele Maschera, Illarion Nevinchanyi
# Date: 2023-11-06


from contagious_spreading_methods import *
import numpy as np
import matplotlib.pyplot as plt


def test_polynom_fit(data, m_list=None):
    """
    Method for testing the polynom fit for a given dataset

    :param data: contains the data, where x=data[0] and y = data[1]
    :param m_list: integer array with the grades
    """

    if m_list is None:
        m_list = [2, 3, 4]

    t, y = data[::]
    plt.scatter(t, y, label='Data', color='r', marker='.')

    for m in m_list:
        y_fit, _ = fit_polynom(t, y, m=m)
        plt.plot(t, y_fit, label='m={:d}'.format(m))

    plt.axis([np.min(t), np.max(t), np.min(y), np.max(y) * 1.2])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Polynomial fit of data of different grades m')
    plt.legend()
    plt.savefig('Polynomial_fit.pdf')
    plt.show()


def test_exponential_fit(data, m_list=None):
    """
    Method for testing the exponential fit for a given dataset

    :param data: contains the data, where x=data[0] and y = data[1]
    :param m_list: integer array with the grades
    """

    if m_list is None:
        m_list = [2, 3, 4, 8]

    t, y = data[::]
    plt.scatter(t, y, label='Data', color='r', marker='.')
    u = 1 / y

    for m in m_list:
        # Fit the function for the inverse values and inverse them afterward
        u_fit, _ = fit_exponential(t, u, m=m)
        y_fit = 1 / u_fit
        plt.plot(t, y_fit, label='m={:d}'.format(m))

    plt.axis([np.min(t), np.max(t), np.min(y), np.max(y) * 1.2])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Exponential fit of data of different grades m')
    plt.legend()
    plt.savefig('Exponential_fit.pdf')
    plt.show()


def test_analytical_fit(data, m=4):
    """
    Method for comparing the exponential fit with an analytical solution

    :param data: contains the data, where x=data[0] and y = data[1]
    :param m: grade of exponential fit
    """

    t, y = data[::]
    plt.scatter(t, y, label='Data', color='r', marker='.')
    u = 1 / y

    u_fit, (A, a, beta) = fit_exponential(t, u, m=m)
    y_fit = 1 / u_fit
    plt.plot(t, y_fit, label='m={:d}'.format(m))

    # Plot the function with the calculated C and alpha from the coefficient vector
    U = np.linalg.inv(A)
    n = np.shape(U)[0]

    C = a[0] - 1
    sig_C = np.sqrt(U[0, 0])

    alpha = -(1 / C) * np.sum([a[j] / j for j in range(1, len(a))])
    sig_alpha = np.sqrt(np.abs(4 / (a[0] - 1)**2 * np.sum([U[0, j] / j for j in range(1, n)])
                               - 6 / (a[0] - 1)**3 * U[0, 0] * np.sum([a[j] / j for j in range(1, n)])))

    print('Exponential m={:d}:'.format(m))
    print('  C={:.0f}({:.0f})'.format(C, sig_C))
    print('  alpha={:.4f}({:.4f})'.format(alpha, sig_alpha))

    t_anl = np.linspace(np.min(t), np.max(t), 100)
    y_anl = 1 / (1 + C * np.exp(-alpha * t_anl))
    plt.plot(t_anl, y_anl, label='Analytically', color='k')

    plt.axis([np.min(t), np.max(t), np.min(y), np.max(y) * 1.2])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Analytical fit m={:d}, C={:.0f}({:.0f}), alpha={:.4f}({:.4f})'
              .format(m, C, sig_C, alpha, sig_alpha))
    plt.legend()
    plt.savefig('Analytical_fit.pdf')
    plt.show()


my_data = np.loadtxt('time_evolution.txt', delimiter=",", unpack=True)
test_polynom_fit(my_data)
test_exponential_fit(my_data, [2, 3, 4, 5])
test_analytical_fit(my_data, m=5)

def commit_succeed():
    '''Not relevant for the main code. Just to make sure, that this is the latest version of the code'''
    return None