import numpy as np

Energy_1s_au = np.float64(-0.5) # -13.6 eV in atomic units of energy
m = np.float64(0.0) # magnetic Q number

omega = np.float64(0.2) 
E0 = np.float64(0.3)

N_timesteps_imag = np.int32(100)
N_timesteps = np.int32(500)

# N_timesteps_test = 300
# sigma_psi_squared_container = np.zeros((N_timesteps_test, ))

lambd = np.float64(1.5)

delta_time = np.float64(0.05)

N_epsilon_divs = np.int32(200)
N_z_divs = np.int32(500)
K = N_z_divs # equal to the variable N_z_divs

delta_epsilon = np.float64(0.05)
delta_z = np.float64(0.05)

z_max = (N_z_divs/2) * delta_z
epsilon_range = np.linspace(0.0, N_epsilon_divs*delta_epsilon, N_epsilon_divs+1)
z_range = np.linspace(-z_max, z_max, N_z_divs+1)
