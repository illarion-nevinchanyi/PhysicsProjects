import numpy as np
d = 4
l = 10              # n ⊂ [0, l)
e = 1               # delta_t
ndim = 4
time_vector = e * np.arange(0, l)
A = np.ones([d, d])
b = np.ones(d)
c = np.zeros(d)

def velocity(v0, theta, t, g = 9.81):
    v = np.array([[v0 * np.cos(theta)], [v0 * np.sin(theta) - g * t]])
    return v
def F(x, t, B = 1, m = 2, g = 9.81):
    x = 1
    v = velocity(10, 60, t)
    norm_v = np.linalg.norm(v)
    f = np.array([v[0], v[1], -B * v[0] * norm_v / m, -B * v[1] * norm_v / m - g])
    return f.reshape(1, 4)
def Runge_Kutta(F, n, yn, time_vector, A, b, c):
    k = np.zeros([d, ndim])
    y = np.zeros([ndim, l + 1])
    y[:, n] = yn
    k[0] = y[:, n]
    for i in range(0, d):
        k[i] = y[:, n] + e * sum(A[i, j] * F(k[j], time_vector[n] + e * c[j]) for j in range(0, d))
    for h in range(n, l):
        y[:, h + 1] = y[:, n] + e * sum(b[j] * F(k[j], time_vector[n] + e * c[j]) for j in range(0, d))
    return k, y

k, y = Runge_Kutta(F, 0, 5, time_vector, A, b, c)