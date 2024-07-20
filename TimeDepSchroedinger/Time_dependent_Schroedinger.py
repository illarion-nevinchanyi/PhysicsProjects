import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Nx = 250; Nt = 1000; dx = 1; dt = 0.1
x = np.arange(0, Nx*dx, dx)
t = np.arange(0, Nt*dt, dt)
def calculate_psi0(x, sigma, q, x0, hbar = 1):
    prefactor = 1 / np.sqrt(sigma*np.sqrt(np.pi))
    exp_term1 = np.exp(-(x-x0)**2 / (2*sigma**2))
    exp_term2 = np.exp(1/hbar * 1j * q * x)
    psi_x = prefactor * exp_term1 * exp_term2
    return psi_x
def Crank_Nicolson_Solver(Nx, dx, Nt, dt, V, m, calculate_psi0, x, sigma, q, x0, hbar=1):
    psi = np.zeros([Nx, Nt], dtype=complex)  # [Nx, Nt]
    a = np.zeros(Nx, dtype=complex)
    Omega = np.zeros([Nx, Nt], dtype=complex)
    b = np.zeros([Nx, Nt], dtype=complex)

    psi[:, 0] = calculate_psi0(x, sigma, q, x0)  # Initial Conditions
    psi[0, :] = 0; psi[Nx - 1, :] = 0            # Boundary Conditions

    a[1] = 2 * (1 + m * dx ** 2 / hbar ** 2 * V[0] - 1j * 2 * m * dx ** 2 / (hbar * dt))

    for k in range(2, Nx - 1):
        a[k] = 2 * (1 + m * dx ** 2 / hbar ** 2 * V[k] - 1j * 2 * m * dx ** 2 / (hbar * dt)) - 1 / a[k - 1]

    for n in range(1, Nt):
        for k in range(1, Nx - 1):
            Omega[k, n] = -psi[k - 1, n - 1] + 2 * (1 + (m*dx**2 / hbar**2)*V[k] + 1j*2*m*dx**2 / (hbar * dt))*psi[k, n - 1] - psi[k + 1, n - 1]

        b[1, n] = Omega[1, n]
        for k in range(2, Nx - 1):
            b[k, n] = b[k - 1, n] / a[k - 1] + Omega[k, n]

        for k in range(Nx - 2, 0, -1):
            # psi[k, n + 1] = 1/a[k] * (psi[k + 1, n + 1] - b[k, n])
            psi[k, n] = 1 / a[k] * (psi[k + 1, n] - b[k, n])

    return psi, a, b, Omega
def calculate_potentials(V0, a, b, d):
    V = np.zeros(Nx)
    V1 = np.zeros(Nx)
    V2 = np.zeros(Nx)
    V[:] = V0
    V1[a: a + d + 1] = V0
    V2[a: a + d + 1] = V0; V2[b: b + d + 1] = V0
    return V, V1, V2
def plot_snapshot(psi, t, V0, a, b, d):
    fig, ax = plt.subplots(2, 1)
    im = ax[1].plot(x, np.abs(psi[:, t])**2, label='Probability W(x,t)', color='darkslateblue')
    im2 = ax[0].plot(x, np.real(psi[:, t]), label=r'$\psi(x,t)$' + ' mit t={}'.format(t), color='darkcyan')
    fig.suptitle('Gaussian Wave packet \n' +
                      'Parameters: V0 = {}, a = {}, b = {}, d = {}'.format(V0, a, b, d))
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    plt.show()
    plt.savefig('Gauss_Wave_(V0, a, b, d)_({}, {}, {}, {}).pdf'.format(V0, a, b, d))
def update(frame, psi_line, prob_line, time_text):
    psi_t = psi[:, frame]
    prob_t = np.abs(psi_t)**2

    psi_line.set_ydata(np.real(psi_t))
    prob_line.set_ydata(prob_t)
    time_text.set_text('Time: {:.2f}'.format(frame * dt))

    return psi_line, prob_line, time_text
def animate(psi, V0, a, b, d,  save_as):
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axs[1].set_ylim([0, 0.06])
    psi_line, = axs[0].plot(x, np.real(psi[:, 0]), label=r'$\psi(x,t)$', color='darkcyan')
    prob_line, = axs[1].plot(x, np.abs(psi[:, 0])**2, label='Probability W(x,t)', color='darkslateblue')
    time_text = axs[1].text(0.8, 0.8, 'Time: {:.2f}'.format(0), transform=axs[1].transAxes, color='red')

    axs[1].yaxis.tick_right()
    axs[0].set_xlabel('Position x',  fontsize='10')
    axs[0].set_ylabel(r'$\psi (x)$', fontsize='10')
    axs[1].set_xlabel('Position x',  fontsize='10')
    axs[1].set_ylabel('Probability ' + r'W(x, t)', fontsize='10')
    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    fig.suptitle('Gaussian Wave packet \n' +
                 'Parameters: V0 = {}, a = {}, b = {}, d = {}'.format(V0, a, b, d))

    animation = FuncAnimation(fig, update, frames=Nt, fargs=(psi_line, prob_line, time_text), interval=25, blit=True)
    animation.save('{}.gif'.format(save_as), writer='pillow')
    plt.show()
def check_normalisation(funktion, t, x, dx):
    probability = np.trapz(y=np.abs(funktion[:, t])**2, x=x, dx=dx)
    return probability
#%%
#### main
# Time Evolution b)
# Let V(x) = V0 = const. be a constant Potential for x0 =< x =< xN
V0, _, _ = calculate_potentials(1.0, 100, 200, 10)
psi, _, _, _ = Crank_Nicolson_Solver(Nx, dx, Nt, dt, V0, 1, calculate_psi0, x, 10, 2, 0)
# P = check_normalisation(psi, 500, x, dx)
# print(P) # 0.47179052082256845
# plot_snapshot(psi, 500, 1.0, 100, 200, 10)

# Scattering of the Gaussian Wave Packet d)
# POTENTIALS V(x), V1(x), V2(x)
# V0 = 1.5, 2.0, 2.5
V0_1, V1_1, V2_1 = calculate_potentials(1.5, 100, 200, 10)
psi_1, _, _, _ = Crank_Nicolson_Solver(Nx, dx, Nt, dt, V1_1, 1, calculate_psi0, x, 20, 2, 0)

V0_2, V1_2, V2_2 = calculate_potentials(2, 100, 200, 10)
psi_2, _, _, _ = Crank_Nicolson_Solver(Nx, dx, Nt, dt, V1_2, 1, calculate_psi0, x, 20, 2, 0)

V0_3, V1_3, V2_3 = calculate_potentials(2.5, 100, 200, 10)
psi_3, _, _, _ = Crank_Nicolson_Solver(Nx, dx, Nt, dt, V1_3, 1, calculate_psi0, x, 20, 2, 0)

#%%
animate(psi,   1.0, None, None, None, save_as='wave_packet_psi_animation')
animate(psi_1, 1.5, 100, 200, 10, save_as='wave_packet_psi_1_animation')
animate(psi_2, 2.0, 100, 200, 10, save_as='wave_packet_psi_2_animation')
animate(psi_3, 1.5, 100, 200, 10, save_as='wave_packet_psi_3_animation')