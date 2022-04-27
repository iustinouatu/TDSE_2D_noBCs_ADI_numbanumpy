import numpy as np
from matplotlib import pyplot as plt

ground_state_pops_numbanumpy128 = np.load("numbanpground_state_pops_over_time_deltaz0.1_deltaepsilon0.1_deltatime0.05_timesteps500_Nz2000_Nepsilon320_E0.3.npy")
# ground_state_pops_numba128 = np.load("../numba_trial/ground_state_pops_over_time_deltaz0.1_deltaepsilon0.1_deltatime0.05_timesteps6280_Nz2000_Nepsilon60_E0.3.npy")
# ground_state_pops_numbanumpy64 = np.load("numbanpground_state_pops_over_time_deltaz0.1_deltaepsilon0.1_deltatime0.05_timesteps500_Nz2000_Nepsilon320_E0.3_complex64.npy")
timesteps = np.linspace(0, 500, 501)
times = timesteps * 0.05

plt.figure()
plt.plot(times, ground_state_pops_numbanumpy128, color='red', linestyle="dotted", label='ground state population')
plt.yscale("log")
# plt.plot(times, ground_state_pops_numba128[0:501], color='blue', label='numba128')
# plt.plot(times, ground_state_pops_numbanumpy64, color='black', label='numbanumpy64')
plt.legend()
plt.xlabel("Time [atomic units]")
plt.ylabel("Fraction of total population represented by the ground state")
plt.savefig("aici.png")