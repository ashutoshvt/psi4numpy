import time
import numpy as np
from numpy import linalg as LA
import sys
sys.path.append('../')

from fvno_plus_plus import guess_calculate 
from fvno_plus_plus import optrot_calculate 
np.set_printoptions(precision=15, linewidth=200, suppress=True)
import psi4

#psi4.core.set_memory(int(2e9), False)
psi4.set_memory(int(5e9), False)
psi4.core.set_output_file('output.dat', False)


mol = psi4.geometry("""
   units Angstrom
   #no_com
   #no_reorient
   0 1
   C               14.600000000000000    14.529999999999999    15.130000000000001
   O               14.600000000000000    14.529999999999999    16.530000000000001
   C               15.859999999999999    14.529999999999999    15.849999999999987
   C               14.519999999999989    15.709999999999988    14.300000000000001
   H               13.579999999999993    15.709999999999988    13.750000000000000
   H               14.580000000000000    16.600000000000001    14.919999999999989
   H               15.350000000000000    15.709999999999988    13.589999999999991
   H               14.089999999999998    13.640000000000001    14.769999999999992
   H               16.430000000000000    13.640000000000001    15.590000000000000
   H               16.430000000000000    15.419999999999995    15.590000000000000

   symmetry c1
   """)

#psi4.set_options({'basis': 'aug-cc-pVDZ'})
psi4.set_options({'basis': 'sto-3g'})
#psi4.set_num_threads(24)
psi4.set_options({'guess': 'sad'})
#psi4.set_options({'basis': 'ORP'})

print('Computing RHF reference.')
psi4.core.set_active_molecule(mol)
psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
psi4.set_module_options('SCF', {'E_CONVERGENCE':1e-9})
psi4.set_module_options('SCF', {'D_CONVERGENCE':1e-9})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)  
print('RHF Final Energy                          % 16.10f\n' % rhf_e)

Evec_ab = guess_calculate(mol, rhf_e, rhf_wfn, 'CCSD', 0, 0, 0, 40, False)
optrot_calculate(mol, rhf_e, rhf_wfn, 'CCSD', 100, 100, 100, 40, True, Evec_ab)
