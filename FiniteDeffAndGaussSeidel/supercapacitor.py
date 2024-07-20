# Super capacitor energy storage
# TuGraz Computational physics - Assignment 1 Exercise 1
# Authors: Christoph Kircher, Gabriele Maschera, Illarion Nevinchanyi
# Date: 2023-10-25

from supercapacitor_methods import *
import matplotlib.pyplot as plt


# input parameters:
N = 50  # size of vectors x and y

y_0 = -0.5
y_N = 0.5
d = 100  # [nm]
T = 300  # [K]
epsilon = 80
c0 = 0.006  # [nm^-3]

# definition of constants:
e0 = 1.6 * 10 ** (-19)  # [C]
epsilon0 = 8.85 * 10 ** (-21)  # [CV^-1nm^-1]
kB = 1.38 * 10 ** (-23)  # [J/K]

k = np.sqrt((2 * c0 * e0**2) / (epsilon * epsilon0 * kB * T))


def numerical_solution():
    """
    Calculates the numerical solution of the electrostatic potential differential equation

    :return: a vector with size N and the calculated potential values in x position
    """
    y = np.zeros(N)
    delta_x = d / (N - 1)  # space between two x values

    A = get_matrix_A(N, delta_x, k)
    b = np.zeros(N)
    L, U = lu_composition(A)

    # Boundary conditions
    z = np.zeros(N)
    y[0] = y_0
    y[N-1] = y_N
    z[0] = U[0, 0] * y_0

    # substitutions
    z = forward_substitution(z, b, L)
    y = backward_substitution(y, z, U)
    return y


def analytical_solution(x):
    """
    Calculate the electrostatic potential analytically by the given formula

    :param x: vector with the width values inside the super capacitor
    :return: a vector with size N and the calculated potential values in x position
    """
    y = np.zeros(N)
    y[0] = y_0
    y[N - 1] = y_N
    y = ((y[N - 1] * np.sinh(k * x)) + y[0] * np.sinh(k * (d - x))) / np.sinh(k * d)
    return y


def plot_electrostatic_potential():
    """
    Plots the numerical and analytical solution of electrostatic potential
    """
    x = np.linspace(0, d, N, True)  # endpoint is included x
    y_num = numerical_solution()
    y_anl = analytical_solution(x)

    plt.plot(x, y_anl, '-b', label="analytical")
    plt.plot(x, y_num, '-r', label="numerical")

    plt.legend(loc="lower right")
    plt.xlabel('x / nm')
    plt.ylabel('electrostatic potential')
    plt.title('Analytical vs. numerical Solution')
    plt.show()


def plot_capacitance_width():
    """
    Plots the capacitance of the super capacitor at different widths d
    """
    d_vec = np.linspace(1, 2 * d, N, True)
    C_dep_d = (2 * c0) / (k * np.sinh(k * d_vec)) * (np.cosh(k * d_vec) - 1)

    plt.plot(d_vec, C_dep_d, '-b', label="C_d")

    plt.legend(loc="lower right")
    plt.xlabel('d / nm')
    plt.ylabel('C / unit area')
    plt.title('Capacitance C_d vs. Width d')
    plt.show()


def plot_capacitance_debye_length():
    """
    Plots the capacitance of the super capacitor at different debye length
    """
    k_inv = np.linspace(0.2, 2 * (k - 0.2), N, True)
    k_vec = 1 / k_inv
    C_dep_k = (2 * c0) / (k_vec * np.sinh(k_vec * d)) * (np.cosh(k_vec * d) - 1)

    plt.plot(k_inv, C_dep_k, '-b', label="C_d")

    plt.legend(loc="lower right")
    plt.xlabel('k^-1 / nm')
    plt.ylabel('C / unit area')
    plt.title('Capacitance C_d vs. debye length k^-1')
    plt.show()


def plot_capacitance_temp():
    """
    Plots the capacitance of the super capacitor at different temperatures T
    """
    T_vec = np.linspace(100, 2 * T, N, True)
    k_vec = np.sqrt((2 * c0 * e0 ** 2) / (epsilon * epsilon0 * kB * T_vec))
    C_dep_T = (2 * c0) / (k_vec * np.sinh(k_vec * d)) * (np.cosh(k_vec * d) - 1)

    plt.plot(T_vec, C_dep_T, '-b', label="C_T")

    plt.legend(loc="lower right")
    plt.xlabel('T / K')
    plt.ylabel('C / unit area')
    plt.title('Capacitance C_T vs. Temperature T')
    plt.show()


def plot_capacitance_volume():
    """
    Plots the capacitance of the super capacitor at different volumes V
    """
    d_vec = np.linspace(1, 2 * d, N, True)
    C_V = (2 * c0) / (k * d_vec * np.sinh(k * d_vec)) * (np.cosh(k * d_vec) - 1)

    plt.plot(d_vec, C_V, '-b', label="C_V")

    plt.legend(loc="upper right")
    plt.xlabel('d / nm')
    plt.ylabel('C / volume')
    plt.title('Capacitance per Volume C_V vs. Width d')
    plt.show()


plot_electrostatic_potential()
plot_capacitance_width()
plot_capacitance_debye_length()
plot_capacitance_temp()
plot_capacitance_volume()
