import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, threshold=2000, suppress=True)
import psi4

# Set memory & output file
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Set molecule to dimer
#mol = psi4.geometry("""
#Be  0  0  0
#symmetry c1
#""")
#mol = psi4.geometry("""
# O     -0.028962160801    -0.694396279686    -0.049338350190                                                                  
# O      0.028962160801     0.694396279686    -0.049338350190                                                                  
# H      0.350498145881    -0.910645626300     0.783035421467                                                                  
# H     -0.350498145881     0.910645626300     0.783035421467                                                                  
#noreorient
#symmetry c1        
#""")

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
noreorient
symmetry c1
""")

#psi4.set_options({"scf_type": "out_of_core",
#                  "basis": "aug-cc-pVTZ",
#                  "e_convergence": 1e-8,
#                  "d_convergence": 1e-8})
#
#psi4.set_options({'basis': 'aug-cc-pVDZ'})
psi4.set_options({'basis': 'sto-3g'})
#psi4.set_options({'basis': 'ORP'})

# For numpy
compare_psi4 = True

print('Computing RHF reference.')
psi4.core.set_active_molecule(mol)
psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
psi4.set_module_options('SCF', {'E_CONVERGENCE':1e-13})
psi4.set_module_options('SCF', {'D_CONVERGENCE':1e-13})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)  
print('RHF Final Energy                          % 16.10f\n' % rhf_e)

t = time.time()
#scf_e, wfn = psi4.energy('SCF', return_wfn=True)
print('SCF took                %5.3f seconds' % ( time.time() - t))

Co = rhf_wfn.Ca_subset("AO", "OCC")
Cv = rhf_wfn.Ca_subset("AO", "VIR")
epsilon = np.asarray(rhf_wfn.epsilon_a())

nbf = rhf_wfn.nmo()
ndocc = rhf_wfn.nalpha()
nvir = nbf - ndocc
nov = ndocc * nvir
print('')
print('Ndocc: %d' % ndocc)
print('Nvir:  %d' % nvir)
print('Nrot:  %d' % nov)
print('')

eps_v = epsilon[ndocc:]
eps_o = epsilon[:ndocc]

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(rhf_wfn.basisset())
S = np.asarray(mints.ao_overlap())
I = mints.ao_eri()
v_ijab = np.asarray(mints.mo_transform(I, Co, Co, Cv, Cv))
v_iajb = np.asarray(mints.mo_transform(I, Co, Cv, Co, Cv))
Co = np.asarray(Co)
Cv = np.asarray(Cv)
print('Integral transform took %5.3f seconds\n' % ( time.time() - t))

# Grab perturbation tensors in MO basis
tmp_dipoles = mints.so_dipole()
dipoles_xyz = []
for num in range(3):
    Fso = np.asarray(tmp_dipoles[num])
    Fia = (Co.T).dot(Fso).dot(Cv)
    dipoles_xyz.append(Fia)


# Since we are time dependent we need to build the full Hessian:
# | A B |      | S  D | |  x |   |  b |
# | B A |  - w | D -S | | -x | = | -b |

# Build CIS matrices A(singlet) and B(triplet) and
# solve their linear equations separately
A11  = np.einsum('ab,ij->iajb', np.diag(eps_v), np.diag(np.ones(ndocc)))
A11 -= np.einsum('ij,ab->iajb', np.diag(eps_o), np.diag(np.ones(nvir)))
A11 += 2 * v_iajb
A11 -= v_ijab.swapaxes(1, 2)

A11.shape = (nov, nov)

B11  = np.einsum('ab,ij->iajb', np.diag(eps_v), np.diag(np.ones(ndocc)))
B11 -= np.einsum('ij,ab->iajb', np.diag(eps_o), np.diag(np.ones(nvir)))
B11 -= v_ijab.swapaxes(1,2)

B11.shape = (nov, nov)

dipoles_xyz[0].shape = nov
z1 = np.linalg.solve(A11,dipoles_xyz[0])
z2 = np.linalg.solve(B11,dipoles_xyz[0])
print('\n CIS Singlet X: \n')
print(z1)
print('\n CIS Triplet X: \n')
print(z2)

# Eigenvalues 
#A_cis_ev, A_cis_emat = np.linalg.eig(A11)
#idx =  A_cis_ev.argsort()[::1]                                                                      
#A_cis_ev = A_cis_ev[idx]  
#print(A_cis_ev)

#B_cis_ev, B_cis_emat = np.linalg.eig(B11)
#idx =  B_cis_ev.argsort()[::1]
#B_cis_ev = B_cis_ev[idx]  
#print(B_cis_ev)



# solving linear equations  of CIS A and B matrixes together

A11  = np.einsum('ab,ij->iajb', np.diag(eps_v), np.diag(np.ones(ndocc)))
A11 -= np.einsum('ij,ab->iajb', np.diag(eps_o), np.diag(np.ones(nvir)))
A11 += v_iajb
A11 -= v_ijab.swapaxes(1, 2)

A11.shape = (nov, nov)

B11 =  v_iajb.copy()
B11.shape = (nov, nov)


Hess1 = np.hstack((A11, B11))
Hess2 = np.hstack((B11, A11))
Hess = np.vstack((Hess1, Hess2))
Hess.shape = (2*nov,2*nov)
pert = np.vstack((dipoles_xyz[0], dipoles_xyz[0]))
pert.shape = 2 * nov
z = np.linalg.solve(Hess, pert)
print('\n CIS: Singlet and Triplet combined X: \n')
print(z)

AB_cis_ev, AB_cis_emat = np.linalg.eig(Hess)
idx =  AB_cis_ev.argsort()[::1]
AB_cis_ev = AB_cis_ev[idx]  
print('\n CIS: Singlet and Triplet combined Eigenvalues\n')
print(AB_cis_ev)

# Linear equations of TDHF (A and B matrices)

C11  = np.einsum('ab,ij->iajb', np.diag(eps_v), np.diag(np.ones(ndocc)))
C11 -= np.einsum('ij,ab->iajb', np.diag(eps_o), np.diag(np.ones(nvir)))
C11 += 2 * v_iajb
C11 -= v_ijab.swapaxes(1, 2)
C11 *= 2

D11  = -2 * v_iajb
D11 += v_iajb.swapaxes(0, 2)
D11 *= 2

# Reshape and jam it together
C11.shape = (nov, nov)
D11.shape = (nov, nov)
dipoles_xyz[0].shape = nov
z = np.linalg.solve(C11 - D11, 2.0 * dipoles_xyz[0])
print('\n TDHF: \n')
print(z)
