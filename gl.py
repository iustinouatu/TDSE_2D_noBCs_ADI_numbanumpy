import numpy as np

Energy_1s_au = np.float64(-0.5) # -13.6 eV in atomic units of energy
m = np.float64(0.0) # magnetic Q number
states = np.int32(3) # calculate ITP for states from the ground-state (0) to the excited state denoted as (states-1). i.e. if states=3, then ground, 1st excited, 2nd excited will be calculated.

omega = np.float64(0.057) # light angular frequency omega = 0.057 atomic units => light optical period = 2.66 * 10^(-15) sec corresponding to lambda (color) = 800nm 
phi0 = np.float64(0.0)
E0_values = np.linspace(0.2, 0.5, 10) # for E < 0.05 ionization happens in the tunnel regime: Kostyukov 2018
enve = "gauss"  # choose from ["gauss", "sinsq"]

# for gauss
FWHM = np.float64( (2*np.pi/omega) * 6.0 )  # (2*np.pi/omega) for transforming 1 optical cycle (1 light period) to atomic units
center = np.float64( (2*np.pi/omega) * 0.0) # (2*np.pi/omega) for transforming 1 optical cycle (1 light period) to atomic units

# for sinsq
# N = np.int32(4)
# T = np.float64(N * (2*np.pi / omega))

# Z = np.int32(1)

N_timesteps_imag = np.int32(100)
N_timesteps = np.int32(773)

# N_timesteps_test = 300   # to test if acummulated errors grow exponentially in time or not (ideally shall not!)
# sigma_psi_squared_container = np.zeros( (N_timesteps_test, ) )

lambd = np.float64(1.5)

delta_time = np.float64(0.05)

N_epsilon_divs = np.int32(200)
N_z_divs = np.int32(500)
K = N_z_divs # equal to the variable N_z_divs, used in main.py to match the notation from Bauer/Mulser

delta_epsilon = np.float64(0.05)
delta_z = np.float64(0.05)

z_max = (N_z_divs/2) * delta_z
epsilon_range = np.linspace(0.0, N_epsilon_divs*delta_epsilon, N_epsilon_divs+1)
z_range = np.linspace(-z_max, z_max, N_z_divs+1)


def from_OCs_to_AUs(OCs):
    return OCs * (2*np.pi/omega)