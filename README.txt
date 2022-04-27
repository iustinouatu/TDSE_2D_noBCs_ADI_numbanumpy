This code is much faster than the code inside ../numba_trial

It doesn't use numba on the double np.trapz usage for getting the ground state populations and renormalizing the wavefunctions.

It is also written as to be separated into functions: imaginary time propagation, saving of the ground state, 
loading the last dt of the ground state, then real time propagation.
 
