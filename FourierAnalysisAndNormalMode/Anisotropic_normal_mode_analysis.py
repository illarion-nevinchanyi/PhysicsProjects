# Anisotropic Normal Mode analysis of DNAs
# TuGraz Computational physics - Assignment 2 Exercise 4
# Authors: Illarion Nevinchanyi, Christoph Kircher, Gabriele Maschera
# Date: 2023-11-21

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from normal_Mode_analysis_of_DNA_methods import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')

# IMPORT DATA
df = pd.read_csv('xyzm_dna.txt', sep=',', header=None, names=['x', 'y', 'z', 'm / A'])
m = []; k = 1
R_0 = np.zeros([len(df), 1, 3])

for i in range(len(df)):
    R_0[i][0, 0] = df['x'][i]
    R_0[i][0, 1] = df['y'][i]
    R_0[i][0, 2] = df['z'][i]
    m.append(df['m / A'][i])
# Initialize Functions to Calculate ||R_0i - R_0j|| < R_cut and elements H_ij of the Hessian Matrix H
def R_0_ij_norm(i, j):
    norm = np.linalg.norm(R_0[i] - R_0[j], ord=2, axis=1)
    return norm
def H_ij(i, j):
    delta_R_ij = R_0[j] - R_0[i]
    if i == j:
        H_ij = np.zeros([3, 3])
    else:
        H_ij = -k/R_0_ij_norm(i, j) * np.dot(delta_R_ij.reshape(3, 1), delta_R_ij)
    return H_ij
# Calculate Matrix M with Atom Masses
l = np.shape(m)[0]
M = np.identity(l, dtype=np.float64)
for i in range(l):
    M[i, i] = m[i]
#Calculate new matrix M with shape (3n, 3n) to calculate the Matrix K = M^(-1/2)HM^(-1/2)
m = np.array(m)
M = np.diag(m)
M_new = np.kron(M, np.eye(3))               # Repeat each diagonal element three times using Kronecker product
M_inv_sqrt = np.identity(M_new.shape[0])
for i in range(0, M_inv_sqrt.shape[0]):
    M_inv_sqrt[i, i] = M_new[i, i] ** (-1 / 2)

H = np.zeros([len(df), len(df), 3, 3])      # Calculate matrix H
for i in range(len(df)):
    for j in range(len(df)):
        H[i, j] = H_ij(i, j)

H = np.block([[H[i, j] for j in range(len(df))] for i in range(len(df))])
K = np.dot(M_inv_sqrt, np.dot(H, M_inv_sqrt))

# Calculate 3 MAX AND 3 MIN EIGENVALUES
num_eigen = 3
max_eigenvalues = []; max_eigenvectors = []

eigenvalue, eigenvector = power_method_V2(K)
max_eigenvalues.append(eigenvalue)
max_eigenvectors.append(eigenvector)

C = K
for _ in range(num_eigen-1):
    C = C - eigenvalue * np.outer(eigenvector, eigenvector)
    eigenvalue, eigenvector = power_method_V2(C)
    max_eigenvalues.append(eigenvalue)
    max_eigenvectors.append(eigenvector)

print(f"3 max eigenvalues: {max_eigenvalues}")

min_eigenvalues = []; min_eigenvectors = []

eigenvalue, eigenvector = inverse_power_method_V2(K)
min_eigenvalues.append(eigenvalue)
min_eigenvectors.append(eigenvector)

C = K
for _ in range(num_eigen-1):
    C = C - eigenvalue * np.outer(eigenvector, eigenvector)
    eigenvalue, eigenvector = inverse_power_method_V2(C)
    min_eigenvalues.append(eigenvalue)
    min_eigenvectors.append(eigenvector)

print(f"3 min igenvalues: {min_eigenvalues}")
#%%
def animate_eigenmodes(eigenvalues, eigenvectors, equilibrium_data, t, scale):
    # because the deformation that small is, I needed to scale the deformation, in order to see it
    num_modes = min(len(eigenvectors), 3)

    fig = plt.figure(figsize=(5 * num_modes, 5))
    fig.suptitle("Eigenmodes Animation of DNA Molecule", fontsize=16)

    def update(frame):
        plt.clf()  # Clear the previous frame

        for i in range(num_modes):
            complex_exponential = np.exp(complex(0, -np.sqrt(np.abs(eigenvalues[i])) * t[frame]))
            #exponential = np.exp(-np.sqrt(np.abs(eigenvalues[i]) * t[frame]))
            mode_deformation = scale*np.real(complex_exponential) * np.array(eigenvectors[i]).reshape(-1, 3)
            deformed_data = equilibrium_data + mode_deformation

            # plot the real part
            ax = fig.add_subplot(1, num_modes, i + 1, projection='3d')
            ax.plot(equilibrium_data[:, 0], equilibrium_data[:, 1], equilibrium_data[:, 2], label="Equilibrium")
            ax.plot(deformed_data[:, 0], deformed_data[:, 1], deformed_data[:, 2], label=f"Mode {i + 1} Deformation")
            ax.set_title(f"Mode {i + 1}")
            ax.legend()

    ani = FuncAnimation(fig, update, frames=len(t), interval=100, repeat=False)
    plt.show()
    return ani  # Return the animation object

usecolumns = ['x', 'y', 'z']
data = pd.read_csv('xyzm_dna.txt', sep=',', header=None, names=['x', 'y', 'z', 'm / A'], usecols=usecolumns)

t = np.linspace(0, 2000, 100)    # Set up the time parameter

# Visualize the eigenmodes with animation
#max_animation = animate_eigenmodes(max_eigenvalues, max_eigenvectors, data[['x', 'y', 'z']].values, t, 10)
min_animation = animate_eigenmodes(min_eigenvalues, min_eigenvectors, data[['x', 'y', 'z']].values, t, 15)

# Save the animations as images (PNG format)
#max_animation.save('max_eigenmodes_animation.gif', writer='pillow', fps=30)
min_animation.save('min_eigenmodes_animation.gif', writer='pillow', fps=30)

def commit_succeed():
    '''Not relevant for the main code. Just to make sure, that this is the latest version of the code'''
    return None