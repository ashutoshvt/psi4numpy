import time
import numpy as np
from helper_cc2 import helper_cc2energy 
from helper_cc2 import helper_cc2hbar 
from helper_cc2 import helper_cc2lambda 
from helper_cc2 import helper_cc2pert 
from helper_cc2 import helper_cclinresp
np.set_printoptions(precision=15, linewidth=200, suppress=True)
import psi4

#psi4.core.set_memory(int(2e9), False)
psi4.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
noreorient 
symmetry c1
""")

psi4.set_options({'basis': 'aug-cc-pVDZ'})
#psi4.set_options({'basis': 'sto-3g'})

print('Computing RHF reference.')
psi4.core.set_active_molecule(mol)
psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
psi4.set_module_options('SCF', {'E_CONVERGENCE':1e-13})
psi4.set_module_options('SCF', {'D_CONVERGENCE':1e-13})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
print('RHF Final Energy                          % 16.10f\n' % rhf_e)

# For numpy
compare_psi4 = True

# Compute CCSD
cc2 = helper_cc2energy(mol, rhf_e, rhf_wfn, memory=2)
cc2.compute_energy(1e-10)

CC2corr_E = cc2.ccsd_corr_e
CC2_E = cc2.ccsd_e

print('\nFinal CC2 correlation energy:          % 16.15f' % CC2corr_E)
print('Total CC2 energy:                      % 16.15f' % CC2_E)
cc2hbar = helper_cc2hbar(cc2)
cc2lambda = helper_cc2lambda(cc2,cc2hbar)
cc2lambda.compute_lambda(1e-10)

cart = {0:'X', 1: 'Y', 2: 'Z'}
Mu={}; Muij={}; Muab={}; Muia={}; Muai={};
L={}; Lij={}; Lab={}; Lia={}; Lai={};
Dij_Mu={}; Dij2_Mu={}; Dab_Mu={}; Dab2_Mu={}; Dia_Mu={}; Dai_Mu={};
Dij_L={}; Dij2_L={}; Dab_L={}; Dab2_L={}; Dia_L={}; Dai_L={};
Dij2={}; Dab2={}
ccpert = {}; optrot_AB = {}; cclinresp = {};
dipole_array = cc2.mints.ao_dipole()
    #angmom_array = ccsd.mints.ao_angular_momentum()

omega = 0.0

dipole_array[0].print_out()


#print(cc2.npC)
for p in range(0,1):
    string_Mu = "MU_" + cart[p]
    #string_L = "L_" + cart[p]
    #print(np.asarray(dipole_array[p]))
    Mu[string_Mu] = np.einsum('uj,vi,uv', cc2.npC, cc2.npC, np.asarray(dipole_array[p]))
    #L[string_L] = -0.5 * np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(angmom_array[p]))
    ccpert[string_Mu] = helper_cc2pert(string_Mu, Mu[string_Mu], cc2, cc2hbar, cc2lambda, omega)
    ccpert[string_Mu].solve('right', 1e-10)
    ccpert[string_Mu].solve('left', 1e-10)

polar_PQ = {}
for p in range(0,1):
    str_p = "MU_" + cart[p]
    for q in range(0,1):
        if p == q:
            str_q = "MU_" + cart[q]
            str_pq = "<<" + str_p + ";" + str_q + ">>"
            cclinresp[str_pq]= helper_cclinresp(cc2lambda, ccpert[str_p], ccpert[str_q])
            polar_PQ[str_pq]= cclinresp[str_pq].linresp()
            print(str_pq)
            print('\n polar1 = %20.15lf \n' % cclinresp[str_pq].polar1)
            print('\n polar2 = %20.15lf \n' % cclinresp[str_pq].polar2)
            print('\n polar_response = %20.15lf \n' % polar_PQ[str_pq])


