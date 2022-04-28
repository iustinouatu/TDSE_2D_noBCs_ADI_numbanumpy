import numpy as np
from numba import jit, njit
from cmath import sin, exp, sqrt

import gl

def main():
    temp = np.zeros((gl.N_z_divs+1, gl.N_epsilon_divs+1), dtype=np.complex64)
    psi_ground = np.zeros((gl.N_z_divs+1, gl.N_epsilon_divs+1, gl.N_timesteps_imag+1), dtype=np.complex64)
    psi_ground[:, :, 0] = populate_psi_ground_at_t0_for_imagtimeprop()

# Imaginary Time Propagation (ITP)
    for timestep in range(0, gl.N_timesteps_imag):
        psi_ground, temp = computations_imag_1timestep(psi_ground, temp, timestep)
        psi_ground = renormalize_psi_ground(timestep, psi_ground)
        rel_change_ampli = calculate_rel_change_in_psiground(timestep, psi_ground)
        print(rel_change_ampli)

    np.save("numbanp_psi_ground_imagiters{}_Nz{}_Nepsilon{}_deltaz{}_deltaepsilon{}_timestep{}.npy".format(gl.N_timesteps_imag, gl.N_z_divs, 
                                                                                    gl.N_epsilon_divs, gl.delta_z, gl.delta_epsilon, gl.delta_time ), 
             psi_ground)
             
# Real Time Propagation (RTP)
    psi_start_for_realcomp = np.load("numbanp_psi_ground_imagiters{}_Nz{}_Nepsilon{}_deltaz{}_deltaepsilon{}_timestep{}.npy".format(gl.N_timesteps_imag, gl.N_z_divs, 
                                                                            gl.N_epsilon_divs, gl.delta_z, gl.delta_epsilon, gl.delta_time)
                            )[:, :, -1]

    psi = np.zeros((gl.N_z_divs+1, gl.N_epsilon_divs+1, gl.N_timesteps+1), dtype=np.complex64)
    psi[:, :, 0] = psi_start_for_realcomp
    ground_state_pops = np.zeros((gl.N_timesteps+1, ), dtype=np.float64)
    ground_state_pops[0] = 1.0

    for timestep in range(0, gl.N_timesteps):
        psi = computations_real_1timestep(psi, temp, timestep)
        ground_state_pops[timestep+1] = get_ground_state_pop(psi[:, :, timestep+1], psi[:, :, 0])

# End of computations
    np.save("numbanp_ground_state_pops_over_time_deltaz{}_deltaepsilon{}_deltatime{}_timesteps{}_Nz{}_Nepsilon{}_E{}_env{}_phi0{}.npy".format(gl.delta_z, gl.delta_epsilon, gl.delta_time, gl.N_timesteps, gl.N_z_divs, gl.N_epsilon_divs, gl.E0, gl.enve, gl.phi0), 
            ground_state_pops)
    np.save("numbanp_psi_realtimeprop_deltaz{}_deltaepsilon{}_deltatime{}_timesteps{}_Nz{}_Nepsilon{}_E{}_env{}_phi0{}.npy".format(gl.delta_z, gl.delta_epsilon, gl.delta_time, gl.N_timesteps, gl.N_z_divs, gl.N_epsilon_divs, gl.E0, gl.enve, gl.phi0), 
            psi)  
# END main()

@njit
def computations_imag_1timestep(psi_gr, tem, timeste):
        # Do one sweep across i-lines   
        # | | ... | | ... | | ... | |
    for p in range(1, gl.N_z_divs):
        # temp is the same shape as psi_ground[:, :, one_timestep_here], where one_timestep_here is any from {0, 1, ..., psi_ground.shape[2]-1}
        tem[p, :] = solve_for_1_fixed_i_line_imag_trisolver(p, timeste, psi_gr) # returned from solve_for_1_fixed_i_line() is shape (N_epsilon_divs + 1, )
        if p % 200 == 0:
            print("We have done imaginary-time first-half-step propagation Iter number " + str(p) +"/" + str(gl.N_z_divs-1) + "for timestep " + str(timeste) + "/" + str(gl.N_timesteps_imag-1))
    # Do one sweep across allj-lines 
    # ---------------> u (= j) = 3
    # ---------------> u (= j) = 2
    # ---------------> u (= j) = 1
    for u in range(1, gl.N_epsilon_divs):
        psi_gr[:, u, timeste+1] = solve_for_1_fixed_j_line_imag_trisolver(u, timeste, tem) # returned from solve_for_1_fixed_j_line() is shape (N_z_divs + 1, )
        if u % 30 == 0:
            print("We have done imaginary-time second-half-step propagation Iter number " + str(u) + "/" + str(gl.N_epsilon_divs-1) + "for timestep " + str(timeste) + "/" + str(gl.N_timesteps_imag-1))
    
    print("We have done imaginary propagation timestep number " + str(timeste) + "out of a total of " + str(gl.N_timesteps_imag-1) + "imaginary-time propagation timesteps")
    return psi_gr, tem


@njit
def computations_real_1timestep(psi, tem, timeste):
# # Propagation of psi in the laser field:

    # Do one sweep across all i-lines   
    # | | ... | | ... | |
    for p in range(1, gl.N_z_divs):
        tem[p, :] = solve_for_1_fixed_i_line_Voft_ON_trisolver(p, timeste, psi)
        if p % 200 == 0:
            print("We have done first-half-step propagation Iter number" + str(p) + "/" + str(gl.N_z_divs-1) + "for timestep " + str(timeste+1) + "/" + str(gl.N_timesteps))
    # Do one sweep across all j-lines 
    # ---------------> u = j = 3
    # ---------------> u = j = 2
    # ---------------> u = j = 1
    for u in range(1, gl.N_epsilon_divs):
        psi[:, u, timeste+1] = solve_for_1_fixed_j_line_Voft_ON_trisolver(u, timeste, tem) # returned from solve_for_1_fixed_j_line() is shape (N_z_divs+1, )
        if u % 20 == 0:
            print("We have done second-half-step propagation Iter number " + str(u) + "/" + str(gl.N_epsilon_divs-1) + "for timestep " + str(timeste+1) + "/" + str(gl.N_timesteps))
    print("We have done timestep " + str(timeste+1) + "out of a total of " + str(gl.N_timesteps) + "timesteps")

    return psi


@njit
def tridiag_solver_from3vectors(d, c, a, b):
    # Uses the Thomas Algorithm for the inversion of a tridiagonal matrix
    # A is (N, N), b is (N, ), x is the solution vector shape (N, )
    # d is (N, )
    # c is (N-1, ) 
    # a is (N-1, )
    x = np.zeros((d.shape[0], ), dtype=np.complex64)

    for i in range(1, d.shape[0]):
        const = a[i-1] / d[i-1]
        d[i] = d[i] - const * c[i-1]
        b[i] = b[i] - const * b[i-1]

    x[-1] = b[-1] / d[-1]
    for i in range(d.shape[0]-2, -1, -1):
        x[i] = (b[i] - c[i] * x[i+1] ) / d[i]   

    return x

@njit
def Voft(i, j, timestep, m):
    ii = i - gl.K/2
    term1  = -gl.Z / sqrt((j*gl.delta_epsilon)**(2*gl.lambd) + (ii*gl.delta_z)**2) #  - Z / sqrt(...)
    term2 = m**2 / ( 2*(j*gl.delta_epsilon)**(2*gl.lambd) )
    term3 = (ii*gl.delta_z) * E_field(timestep) * sin(gl.omega*timestep*gl.delta_time + gl.phi0)
    return (term1 + term2 + term3)

@njit
def Voft_OFF_imag(i, j, m):
    # this is the same as Voft_OFF() above, because there is no modification due to the imag. time propagation (delta_time --> -1j*delta_time) appearing in Voft_OFF (as there is no E-field turned on)
    ii = i - gl.K/2
    term1  = -gl.Z / sqrt(  (j*gl.delta_epsilon)**(2*gl.lambd) + (ii*gl.delta_z)**2  ) #  - Z / sqrt(...)
    term2 = m**2 / ( 2*(j*gl.delta_epsilon)**(2*gl.lambd) )
    return (term1 + term2)

@njit
def E_field(timestep): # the non-fast oscillating part of the E-field of the laser
    return gl.E0 * E_time_profile(timestep) # E_time_profile is not quickly oscillating (i.e. not oscillating at the light's frequency)

@njit
def E_time_profile(timestep): # sin^2, gauss, steplike, etc...
    return np.sin( (np.pi/gl.T) * timestep * gl.delta_time)**2


@njit
def solve_for_1_fixed_i_line_Voft_ON_trisolver(i, timestep, psi): # i represents the i-line we are solving for by calling this function: i=1 is the second vertical line
    d = np.zeros((gl.N_epsilon_divs+1, ), dtype=np.complex64) # main diag
    c = np.zeros((gl.N_epsilon_divs, ),  dtype=np.complex64) # top diag
    a = np.zeros((gl.N_epsilon_divs, ),  dtype=np.complex64) # bottom diag

    d[0] = 1.0
    d[-1] = 1.0

    for con in range(1, gl.N_epsilon_divs): # top diag
        j = con
        c[con] = (1j*gl.delta_time/2) * ( -1/(2*gl.lambd**2*(j*gl.delta_epsilon)**(2*gl.lambd)) ) *  (j**2 - (gl.lambd-1)*j) # top diag
    for con in range(1, gl.N_epsilon_divs): #
        j = con
        d[con] = 1 + (1j*gl.delta_time/2) * (  (1/(2*gl.lambd**2*(j*gl.delta_epsilon)**(2*gl.lambd))) * (2*j**2 - (gl.lambd-0.5)**2) + 0.5*Voft(i, j, np.float64(timestep+0.5), gl.m)    ) # diag 
    for con in range(0, gl.N_epsilon_divs - 1): # the last element it does work on is ab[2][gl.N_epsilon_divs-2]. ab[2][gl.N_epsilon_divs-1] is 0 from A matrix, ab[2][gl.N_epsilon_divs] is 0 from ab definition
        j = con + 1 # con = 0, first element of the bottom-diag, has its expression with j = 1
        a[con] = (1j*gl.delta_time/2) * ( -1/(2*gl.lambd**2*(j*gl.delta_epsilon)**(2*gl.lambd)) )   *   (j**2 + (gl.lambd-1)*j) # bottom diag
 
    # RHS of the matrix equation Ax = b (b is a column vector)
    b = np.zeros((gl.N_epsilon_divs+1, ), dtype=np.complex64)
    # b[0] = 0.0 # BC for epsilon = epsilon_min
    # b[-1] = 0.0 # BC for epsilon = epsilon_max
    for row in range(1, gl.N_epsilon_divs):
        j = row
        second_der_wrt_z = (psi[i+1, row, timestep] - 2*psi[i, row, timestep] + psi[i-1, row, timestep]) / (gl.delta_z**2) 
        b[row] = psi[i, row, timestep] - (1j*gl.delta_time/2) * (  (-0.5*second_der_wrt_z)  + 0.5*Voft(i, j, np.float64(timestep+0.5), gl.m)*psi[i, row, timestep] )
        
    return tridiag_solver_from3vectors(d, c, a, b) # returned is shape (gl.N_epsilon_divs, ) (same as b.shape)

@njit
def solve_for_1_fixed_j_line_Voft_ON_trisolver(j, timestep, temp):
    d = np.zeros((gl.N_z_divs+1, ), dtype=np.complex64)
    a = np.zeros((gl.N_z_divs, ),   dtype=np.complex64)
    c = np.zeros((gl.N_z_divs, ),   dtype=np.complex64)

    d[0] = 1.0
    d[-1] = 1.0

    c[1:] = (1j*gl.delta_time/2) * (-0.5/gl.delta_z**2)
    
    for con in range(1, gl.N_z_divs): # last element it does work on is ab[1][gl.N_z_divs - 1]:
        i = con
        d[con] = 1 + (1j*gl.delta_time/2) * ( (1/gl.delta_z**2) + 0.5*Voft(i, j, np.float64(timestep+0.5), gl.m) )

    a[:-1] = (1j*gl.delta_time/2) * (-0.5/gl.delta_z**2) # last 2 elements of ab[2] (ab[2] is size gl.N_z_divs+1) are 0.0 (last-but-one element is 0 from the A matrix, last element is 0 because there's no such elem in reality)
   
    # RHS of the matrix equation Ax = b (b is a column vector)
    b = np.zeros((gl.N_z_divs+1, ), dtype=np.complex64)
    # b[0] = 0.0 # BC for z = z_min
    # b[-1] = 0.0 # BC for z = z_max

    two_lambd_sq = 2*gl.lambd**2 # calculated once here, not many times inside the for-loop.
    two_lambd_minus1 = 2*(gl.lambd-1) # similarly
    lambd_minus05_squared = (gl.lambd-0.5)**2 # similarly 

    for row in range(1, gl.N_z_divs): # need not to take the first-row and last-row values as they are populated by BC's
        i = row
        second_der_wrt_epsilon = temp[i, j+1] - 2*temp[i, j] + temp[i, j-1] # no division by gl.delta_epsilon because it cancels out
        first_der_wrt_epsilon = (temp[i, j+1] - temp[i, j-1])  / 2 # no division by gl.delta_epsilon because it cancels out
        pre = -1 / (two_lambd_sq * (j*gl.delta_epsilon)**(2*gl.lambd))
        square_para1 = j**2 * second_der_wrt_epsilon - two_lambd_minus1*j*first_der_wrt_epsilon + lambd_minus05_squared * temp[i,j]
        b[row] = temp[i, j] - (1j*gl.delta_time/2) * (  pre*square_para1 + 0.5*Voft(i, j, np.float64(timestep+0.5), gl.m)*temp[i, j]  )

    return tridiag_solver_from3vectors(d, c, a, b)


@njit
def solve_for_1_fixed_i_line_imag_trisolver(i, timestep, psi_gr): # i represents the i-line we are solving for by calling this function: i=1 is the second vertical line
    d = np.zeros((gl.N_epsilon_divs+1), dtype=np.complex64) # main diag
    a = np.zeros((gl.N_epsilon_divs, ), dtype=np.complex64) # bottom diag
    c = np.zeros((gl.N_epsilon_divs, ), dtype=np.complex64) # top diagonal

    d[0] = 1.0
    d[-1] = 1.0

    for con in range(1, gl.N_epsilon_divs): # top diag
        j = con
        c[con] = (gl.delta_time/2) * ( -1/(2*gl.lambd**2*(j*gl.delta_epsilon)**(2*gl.lambd)) ) *  (j**2 - (gl.lambd-1)*j) # top diag
    for con in range(1, gl.N_epsilon_divs): # main diag
        j = con
        d[con] = 1 + (gl.delta_time/2) * (  (1/(2*gl.lambd**2*(j*gl.delta_epsilon)**(2*gl.lambd))) * (2*j**2 - (gl.lambd-0.5)**2) + 0.5*Voft_OFF_imag(i, j, gl.m)    ) # main diag 
    for con in range(0, gl.N_epsilon_divs - 1): # bottom diag
        j = con + 1 # con = 0, first element of the bottom-diag, has its expression with j = 1
        a[con] = (gl.delta_time/2) * ( -1/(2*gl.lambd**2*(j*gl.delta_epsilon)**(2*gl.lambd)) )  *  (j**2 + (gl.lambd-1)*j) # bottom diag
    
    # RHS of the matrix equation Ax = b (b is a column vector)
    b = np.zeros( (gl.N_epsilon_divs+1, ), dtype=np.complex64 )
    # b[0] = 0.0 # BC for epsilon = epsilon_min
    # b[-1] = 0.0 # BC for epsilon = epsilon_max
    for row in range(1, gl.N_epsilon_divs): # need row not to take the last-row-value as it's already populated
        j = row
        second_der_wrt_z = (psi_gr[i+1, row, timestep] - 2*psi_gr[i, row, timestep] + psi_gr[i-1, row, timestep]) / (gl.delta_z**2) 
        b[row] = psi_gr[i, row, timestep] - (gl.delta_time/2) * ( (-0.5*second_der_wrt_z) + 0.5*Voft_OFF_imag(i, j, gl.m)*psi_gr[i, row, timestep] )  

    return  tridiag_solver_from3vectors(d, c, a, b) # returned is shape (gl.N_epsilon_divs+1, ) , same as d.shape[0]

@njit
def solve_for_1_fixed_j_line_imag_trisolver(j, timestep, temp):
    d = np.zeros((gl.N_z_divs+1), dtype=np.complex64) # main diag
    a = np.zeros((gl.N_z_divs, ), dtype=np.complex64) # bottom diag
    c = np.zeros((gl.N_z_divs, ), dtype=np.complex64) # top diagonal

    d[0] = 1.0
    d[-1] = 1.0

    c[1:] = (gl.delta_time/2) * (-0.5/gl.delta_z**2) # first elem of c is 0 as from the A matrix
    
    for con in range(1, gl.N_z_divs): # d[0] and d[-1] not touched in this for-loop
        i = con
        d[con] = 1 + (gl.delta_time/2) * ( (1/gl.delta_z**2) + 0.5*Voft_OFF_imag(i, j, gl.m) )

    a[:-1] = (gl.delta_time/2) * (-0.5/gl.delta_z**2) # last elem of a is 0 from the A matrix

    # RHS of the matrix equation Ax = b (b is a column vector)
    b = np.zeros( (gl.N_z_divs+1, ), dtype=np.complex64 )
    # b[0] = 0.0 # BC for z = z_min
    # b[-1] = 0.0 # BC for z = z_max
    two_lambd_sq = 2*gl.lambd**2 # calculated once here, not many times inside the for-loop.
    two_lambd_minus1 = 2*(gl.lambd-1) # similarly
    lambd_minus05_squared = (gl.lambd-0.5)**2 # similarly 
    for row in range(1, gl.N_z_divs): # need not to take the first-row and last-row values as they are populated by BC's
        i = row
        second_der_wrt_epsilon = temp[i, j+1] - 2*temp[i, j] + temp[i, j-1] # no division by gl.delta_epsilon because it cancels out
        first_der_wrt_epsilon = (temp[i, j+1] - temp[i, j-1])  / 2 # no division by gl.delta_epsilon because it cancels out
        pre = -1 / (two_lambd_sq * (j*gl.delta_epsilon)**(2*gl.lambd))
        square_para1 = j**2 * second_der_wrt_epsilon - two_lambd_minus1*j*first_der_wrt_epsilon + lambd_minus05_squared * temp[i, j]
        b[row] = temp[i, j] - (gl.delta_time/2) * (  pre*square_para1 + 0.5*Voft_OFF_imag(i, j, gl.m)*temp[i, j]  )

    return tridiag_solver_from3vectors(d, c, a, b) # returned is shape (gl.N_z_divs+1, ), same as d.shape[0]


################################################################################################################################################################################
# Utility functions
def get_ground_state_pop(psi, psi_gr):
    # Calculates <psi_1s|psi>: int_int_{}^{} d_epsilon d_z of (psi_1s_dagger * psi)
    return np.abs(dot_product(psi_gr, psi))**2

def dot_product(psi_L, psi_R):
    # Calculates <psi_L|psi_R>: int_int_{}^{} d_epsilon d_z of (psi_L_dagger * psi_R)
    # both psi's have the same shape, i.e. (N_z_divs+1, N_epsilon_divs+1)
    tp = np.trapz( np.conj(psi_L)*psi_R,  gl.z_range, axis=0)
    res = np.trapz(tp, gl.epsilon_range, axis=0)
    return res

def renormalize_psi_ground(timestep, psi_gr):
    tp = np.trapz(np.abs(psi_gr[:, :, timestep+1])**2, gl.z_range, axis=0) # tp (temporary storage) is shape (N_epsilon_divs, )
    res = np.trapz(tp, gl.epsilon_range, axis=0)
    psi_gr[:, :, timestep+1] /= sqrt(res) 
    print("we renormalized the ground state wavefunction!") 
    return psi_gr

def calculate_rel_change_in_psiground(timestep, psi_gr):
    top = np.sum(np.sum(  np.abs(  np.abs(psi_gr[:, :, timestep+1]) - np.abs(psi_gr[:, :, timestep])  ), axis=0), axis=0) # |after - before| / before, top is |after-before|, bot is before  
    bottom = np.sum(np.sum(  np.abs(psi_gr[:, :, timestep]), axis=0), axis=0)
    print("We calculated the relative change!")
    print("Top is: {}".format(top))
    print("Bottom is: {}".format(bottom))
    return top / bottom


@njit
def populate_psi_ground_at_t0_for_imagtimeprop():
    psi_ground_at_t0 = np.zeros((gl.N_z_divs+1, gl.N_epsilon_divs+1), dtype=np.complex64)

    for c1 in range(1, psi_ground_at_t0.shape[0]-1):
        for c2 in range(1, psi_ground_at_t0.shape[1]-1):
            zed = (c1 - gl.N_z_divs/2) * gl.delta_z 
            epsi = c2 * gl.delta_epsilon
            psi_ground_at_t0[c1, c2] = sqrt(3) * epsi * exp(-sqrt(epsi**(2*gl.lambd) + zed**2))

    return psi_ground_at_t0


if __name__ == "__main__":
    main()