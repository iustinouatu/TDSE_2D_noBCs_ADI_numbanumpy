import numpy as np
from matplotlib import pyplot as plt

# psi = np.load(".npy")
psi = np.load("numbanp_psi_realtimeprop_deltaz0.05_deltaepsilon0.05_deltatime0.05_timesteps500_Nz500_Nepsilon200_E0.3.npy")

N_z_divs = 500
N_epsilon_divs = 200
delta_z = 0.1
delta_epsilon = 0.1
z_max = (N_z_divs/2) * delta_z
z_range = np.linspace(-z_max, z_max, N_z_divs+1)
epsilon_range = np.linspace(0.0, N_epsilon_divs*delta_epsilon, N_epsilon_divs+1)

zz, ee = np.meshgrid(z_range, epsilon_range)


for i in range(0, 500, 50):
    fig, ax = plt.subplots()
    c = ax.pcolor(zz, ee, np.abs(psi[:, :, i].T)**2, cmap='viridis')
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('|psi|^2', fontsize=14)
    ax.set_xlabel("z", fontsize=14)
    ax.set_ylabel("epsilon", fontsize=14)
    plt.savefig("psi_{}.png".format(i))