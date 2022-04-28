import numpy as np
import scipy
from matplotlib import pyplot as plt
au_to_SI = 4.13 * 10**16

delta_time = np.float64(0.05)
N_timesteps = np.int32(300)

omega = np.float64(0.2) 
phi0 = np.float64(0.0)

E0 = np.float64(0.3)

def E_field(timestep):
    return E0 * np.sin(omega*timestep + phi0)

def main():
    data_filename = "numbanp_ground_state_pops_over_time_deltaz0.05_deltaepsilon0.05_deltatime0.05_timesteps300_Nz500_Nepsilon200_E0.3_envsin(___)_phi00.0.npy"
    ground_state_pops_over_time = np.load(data_filename)
    print(ground_state_pops_over_time.shape)

    timesteps = np.array(range(0, N_timesteps+1))
    Efields = E_field(timesteps)

    inst_rates = - np.gradient(np.log(ground_state_pops_over_time[:8]))
    plt.figure()
    plt.scatter(Efields[:8], inst_rates, label='Instantaneous rate')
    plt.xlabel("E field")
    plt.ylabel("inst rate")
    plt.scatter(Efields[:8], 2.4*Efields[:8]**2, label='2.4 * E^2')
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("inst_rates.pdf", bbox_inches='tight')


if __name__ == "__main__":
    main()