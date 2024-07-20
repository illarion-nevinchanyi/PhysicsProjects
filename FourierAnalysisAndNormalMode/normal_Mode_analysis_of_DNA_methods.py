# Normal Mode analysis of DNA (methods)
# TuGraz Computational physics - Assignment 2 Exercise 3
# Authors: Christoph Kircher, Gabriele Maschera, Illarion Nevinchanyi
# Date: 2023-11-21
import numpy as np
import matplotlib.pyplot as plt

def power_method(K, n_power=10):
    n = np.shape(K)[0]
    V = np.zeros((n, n_power+1))
    for i in range(n):
        V[i, 0] = np.random.rand(1, 1)
    #print(V[:, 0])
    for i in range(n_power):
        V[:, i+1] = np.dot(K, V[:, i])
        #print(V[:, 1])
    #print(V)
    lam = np.sum(V[i, n_power] / V[i, n_power-1] for i in range(n)) / n
    vec = V[:, n_power]

    return (lam, vec)

def power_method_V2(matrix, num_iterations=100, tol=1e-16):
    n = matrix.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iterations):
        w = matrix.dot(v)
        eigenvalue = np.dot(v, w)
        v = w / np.linalg.norm(w)

        if np.linalg.norm(matrix.dot(v) - eigenvalue * v) < tol:
            break

    return eigenvalue, v

def inverse_power_method_V2(matrix, num_iterations=100, tol=1e-16):
    #IT WOOOOOOOOOOOOOOORKS!
    matrix_inv = np.linalg.inv(matrix)
    n = matrix_inv.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iterations):
        w = np.dot(matrix_inv, v)
        eigenvalue = 1 / np.dot(w, v)
        #eigenvalue = np.dot(v, w)
        v = w / np.linalg.norm(w)

        if np.linalg.norm(matrix.dot(v) - eigenvalue * v) < tol:
            break

    return eigenvalue, v

# def min_eig(matrix, num_eigen = 10):
#     eigenvalues = []; eigenvectors = []
#
#     eigenvalue, eigenvector = power_method_V2(matrix)
#     eigenvalues.append(eigenvalue)
#     eigenvectors.append(eigenvector)
#     C = matrix
#
#     for _ in range(matrix.shape[0]-1):
#         C = C - eigenvalue * np.outer(eigenvector, eigenvector)
#         eigenvalue, eigenvector = power_method_V2(C)
#         eigenvalues.append(eigenvalue)
#         eigenvectors.append(eigenvector)
#
#     # Create pairs of values and associated values
#     pairs = list(zip(eigenvalues, eigenvectors))
#     sorted_pairs = sorted(pairs, key=lambda x: abs(x[0]))           # Sort pairs based on the absolute values of the first element in each pair
#     result_pairs = sorted_pairs[:num_eigen]                         # Keep only the first 10 pairs
#     ten_min_eigenvalues = [pair[0] for pair in result_pairs]        # 10 smallest eigenvalues
#     ten_eigenvectors = [pair[1] for pair in result_pairs]           # corresponding 10 eigenvectors
#     return ten_min_eigenvalues, ten_eigenvectors

def gershgorin_bounds(matrix):
    n = matrix.shape[0]
    lower_bounds = []
    upper_bounds = []

    for i in range(n):
        R_i = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])
        lower_bounds.append(matrix[i, i] - R_i)
        upper_bounds.append(matrix[i, i] + R_i)
        #if np.abs(matrix[i, i] - R_i) >= 1:
            #print('|lambda| < 1 condition is NOT satisfied: lambda_{} =\v'.format(i), matrix[i, i] - R_i)
    print(f"Minimum eigenvalue: {lower_bounds}")
    print(f"Maximum eigenvalue: {upper_bounds}")

    return np.min(lower_bounds), np.max(upper_bounds)

def plot_eigenvector(eigenvectors_matrix, z):

    iterations = eigenvectors_matrix.shape[1]       # 10
    fig, axs = plt.subplots(iterations, figsize=(6, 3 * iterations))
    #z_sort = np.sort(z)
    half = int(eigenvectors_matrix.shape[0] / 2)
    z_h = z[:half]

    # Get the indices that would sort the x array
    sorted_indices = np.argsort(z_h.flatten())

    for i in range(iterations):
        x = eigenvectors_matrix[:half, i]

        # Use the sorted indices to obtain sorted x and y vectors
        sorted_x = x[sorted_indices].flatten().reshape(x.shape)
        sorted_z = z_h[sorted_indices].flatten().reshape(x.shape)

        # Plot for each eigenvector
        axs[i].plot(sorted_z, sorted_x, label=f'', color='darkcyan')
        axs[i].set_title(f'Eigenvector {i+1}')
        axs[i].set_xlabel('z-coordinates')
        axs[i].set_ylabel('Eigenvectors')

    for i in range(iterations):
        x = eigenvectors_matrix[half:, i]

        # Use the sorted indices to obtain sorted x and y vectors
        sorted_x = x[sorted_indices].flatten().reshape(x.shape)
        sorted_z = z_h[sorted_indices].flatten().reshape(x.shape)

        # Plot for each eigenvector
        axs[i].plot(sorted_z, sorted_x, label=f'', color='darkslateblue')
        axs[i].set_title(f'Eigenvector {i + 1}')
        axs[i].set_xlabel('z-coordinates')
        axs[i].set_ylabel('Eigenvectors')


    # Adjust layout for better spacing
    plt.tight_layout()
    # Show the combined plot
    plt.savefig('Eigenvectors.pdf')
    plt.show()

def commit_succeed():
    '''Not relevant for the main code. Just to make sure, that this is the latest version of the code'''
    return None