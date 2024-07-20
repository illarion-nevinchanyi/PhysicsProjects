# Normal Mode analysis of DNAs
# TuGraz Computational physics - Assignment 2 Exercise 3
# Authors: Illarion Nevinchanyi, Christoph Kircher, Gabriele Maschera
# Date: 2023-11-21
from normal_Mode_analysis_of_DNA_methods import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IMPORT DATA
df = pd.read_csv('xyzm_dna.txt', sep=',', header=None, names=['x', 'y', 'z', 'm / A'])
# VARIABLES
R_cut = 5; k = 1; n_power = 50; num_eigen = 10
m = []
N = len(df)
R_0 = np.zeros([N, 1, 3])

for i in range(len(df)):
    R_0[i][0, 0] = df['x'][i]
    R_0[i][0, 1] = df['y'][i]
    R_0[i][0, 2] = df['z'][i]
    m.append(df['m / A'][i])

n = np.shape(m)[0]
M = np.identity(n, dtype=np.float64)
for i in range(n):
    M[i, i] = m[i]

# CALCULATE HESSIAN MATRIX
def R_0_ij_norm(i, j):
    norm = np.linalg.norm(R_0[i] - R_0[j], ord=2, axis=1)
    return norm

H = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    for j in range(n):
        if R_0_ij_norm(i, j) > R_cut:
            H[i, j] = 0
        else:
            if i == j:
                H[i, j] = k
            else:
                H[i, j] = -k

M_inv_sqrt = np.identity(n, dtype=np.float64)
for i in range(n):
    M_inv_sqrt[i, i] = M[i, i] ** (-1/2)

# eigenvalues, eigenvectors = np.linalg.eig(M)    # Calculate eigenvalues and eigenvectors
# S = eigenvectors            # Create a matrix S with eigenvectors as columns
# D = np.diag(eigenvalues)    # Create a diagonal matrix D
# S_inv = np.linalg.inv(S)    # Calculate the inverse of S
# M_reconstructed = np.dot(S, np.dot(D, S_inv))  # Verify the decomposition A = SDS^(-1)
#
# M_inv_sqrt_2 = np.dot(S, np.dot(np.diag(np.sqrt(eigenvalues)), np.linalg.inv(S)))
# print(M_inv_sqrt, '\n')
# print(M_inv_sqrt_2)

K = np.dot(M_inv_sqrt, np.dot(H, M_inv_sqrt))

eigenvalues = []; eigenvectors = []

eigenvalue, eigenvector = power_method_V2(K)
eigenvalues.append(eigenvalue)
eigenvectors.append(eigenvector)

C = K
for _ in range(num_eigen-1):
    C = C - eigenvalue * np.outer(eigenvector, eigenvector)
    eigenvalue, eigenvector = power_method_V2(C)
    eigenvalues.append(eigenvalue)
    eigenvectors.append(eigenvector)

print(f"Matrix K Eigenvalues: {eigenvalues}")
#%%
# c)
R_0_matrix = np.matrix(R_0)
z = R_0_matrix[:, 2]
eigenvectors_matrix = np.transpose(np.matrix(eigenvectors))
plot_eigenvector(eigenvectors_matrix, z)
#%%
# d)
print('\nMatrix K:\n')
min_eigenvalue_K, max_eigenvalue_K = gershgorin_bounds(K)
# e)
K_prime = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        K_prime[i, j] = K[i, j] / (max_eigenvalue_K*1.01)

min_eigenvalue_K_prime, max_eigenvalue_K_prime = gershgorin_bounds(K_prime)
print('\nMatrix K_prime:\n')
print('\nmin eigenvalue of K_prime:\v', min_eigenvalue_K_prime)

# #f)
# min_eigv, _ = inverse_power_method_V2(K_prime)
# max_eigv, _ = power_method_V2(np.linalg.inv(K_prime + np.identity(n)))
#
# print('\nPower Method applied on K_prime:\v', max_eigv)
# print('\nInverse Power Method applied on K_prime:\v', min_eigv)

#%%
#g)
def limit_p_to_inf(n, tol=10**(-8)):
    X = sum(np.dot(np.linalg.matrix_power(-K_prime, p), (K_prime + np.identity(K_prime.shape[0]))) for p in range(n))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if i != j and X[i, j] < tol:
                X[i, j] = 0
    return X

# result = limit_p_to_inf(lambda p: np.dot(np.linalg.matrix_power(-K_new, p), (K_new + np.identity(K_new.shape[0]))))

X = limit_p_to_inf(35)
print(X)
#%%
#h) 10 MIN EIGENVALUES OF THE K_PRIME
min_eigenvalues = []; min_eigenvectors = []

eigenvalue, eigenvector = inverse_power_method_V2(K_prime)
min_eigenvalues.append(eigenvalue)
min_eigenvectors.append(eigenvector)

C = K_prime
for _ in range(num_eigen-1):
    C = C - eigenvalue * np.outer(eigenvector, eigenvector)
    eigenvalue, eigenvector = inverse_power_method_V2(C)
    min_eigenvalues.append(eigenvalue)
    min_eigenvectors.append(eigenvector)

print(f"10 min eigenvalues: {min_eigenvalues}")
#%%
#i)
eigenvectors_matrix = np.transpose(np.matrix(min_eigenvectors))
plot_eigenvector(eigenvectors_matrix, z)

def commit_succeed():
    '''Not relevant for the main code. Just to make sure, that this is the latest version of the code'''
    return None