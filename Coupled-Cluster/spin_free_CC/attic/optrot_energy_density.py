import time
import numpy as np
from numpy import linalg as LA
from helper_cc import helper_ccenergy
from helper_cc import helper_cchbar
from helper_cc import helper_cclambda
from helper_cc import helper_ccpert
from helper_cc import helper_cclinresp

np.set_printoptions(precision=15, linewidth=200, suppress=True)
import psi4

#psi4.core.set_memory(int(8e9), False)
psi4.set_memory(int(8e9), False)
psi4.core.set_output_file('output.dat', False)

#numpy_memory = 2

mol = psi4.geometry("""
 O     -0.028962160801    -0.694396279686    -0.049338350190                                                                  
 O      0.028962160801     0.694396279686    -0.049338350190                                                                  
 H      0.350498145881    -0.910645626300     0.783035421467                                                                  
 H     -0.350498145881     0.910645626300     0.783035421467                                                                  
noreorient
symmetry c1        
""")

#O
#H 1 1.1
#H 1 1.1 2 104
#symmetry c1
#""")

psi4.set_options({'basis': 'aug-cc-pVTZ'})
#psi4.set_num_threads(24)
#psi4.set_options({'basis': 'sto-3g'})
#psi4.set_options({'basis': 'ORP'})

# For numpy
#compare_psi4 = True

print('Computing RHF reference.')
psi4.core.set_active_molecule(mol)
psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
psi4.set_module_options('SCF', {'E_CONVERGENCE':1e-9})
psi4.set_module_options('SCF', {'D_CONVERGENCE':1e-9})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)  
print('RHF Final Energy                          % 16.10f\n' % rhf_e)

Dij2 = {}; Dab2 = {}

def fvno_procedure(mol, rhf_e, rhf_wfn, memory):

    # Compute CCSD
    ccsd = helper_ccenergy(mol, rhf_e, rhf_wfn, memory)
    ccsd.compute_energy(r_conv=1e-7)
    CCSDcorr_E = ccsd.ccsd_corr_e
    CCSD_E = ccsd.ccsd_e
    
    print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
    print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
    print('Total CCSD energy:                      % 16.15f' % CCSD_E)
    
    cchbar = helper_cchbar(ccsd)
    
    cclambda = helper_cclambda(ccsd,cchbar)
    cclambda.compute_lambda(r_conv=1e-7)
    

    Dij2["Energy"] =  0.5 * np.einsum('ia,ja->ij', ccsd.t1, cclambda.l1)
    Dij2["Energy"] += 0.5 * np.einsum('ikab,jkab->ij', ccsd.t2, cclambda.l2)
    Dij2["Energy"] =  -1.0 * Dij2["Energy"]

    Dab2["Energy"] =   0.5 * np.einsum('ia,ib->ab', ccsd.t1, cclambda.l1)
    Dab2["Energy"] +=  0.5 * np.einsum('ijac,ijbc->ab', ccsd.t2, cclambda.l2)


    Dij2["Energy"] = 0.5 * (Dij2["Energy"] + Dij2["Energy"].T)
    Dij2["Energy"] = 0.5 * (Dij2["Energy"] + Dij2["Energy"].T)

    Evec2_ij, Emat2_ij = LA.eig(Dij2["Energy"])
    Evec2_ab, Emat2_ab = LA.eig(Dab2["Energy"])

    Emat2_ij = sort(Emat2_ij, Evec2_ij)
    Emat2_ab = sort(Emat2_ab, Evec2_ab)

    #for item in Evec2_ij:
    #    print(item)

    #for item in Evec2_ab:
    #    print(item)

    return Emat2_ij, Emat2_ab


def sort(Emat, Evec):
    tmp = Evec.copy()
    tmp = abs(tmp)
    idx =  tmp.argsort()[::-1]   
    Evec = Evec[idx]
    
    print('\n Printing sorted eigenvalues\n')
    
    for item in Evec:
        print(item)
    
    Emat = Emat[:,idx]
    return Emat
    
C = psi4.core.Matrix.to_array(rhf_wfn.Ca())
F = psi4.core.Matrix.to_array(rhf_wfn.Fa())

nmo = rhf_wfn.nmo()
occ = rhf_wfn.doccpi()[0]
vir = nmo - occ
C_occ = C[:, :occ]
C_vir = C[:, occ:]
F_mo  = np.einsum('ui,vj,uv', C, C, F)
F_mo_occ = F_mo[:occ,:occ]
F_mo_vir = F_mo[occ:, occ:]

Emat_ij, Emat_ab = fvno_procedure(mol, rhf_e, rhf_wfn, 8)

frz_vir = [15,30,35,40,45,50,55,60,65]
Emat_ab1 = np.zeros_like(Emat_ab)
for k in frz_vir:

    Emat_ab1 = Emat_ab.copy()
    Emat_view = Emat_ab1[:,vir-k:]
    Emat_view.fill(0)

    C_occ_no = np.einsum('pi,ij->pj', C_occ, Emat_ij)
    C_vir_no = np.einsum('pa,ab->pb', C_vir, Emat_ab1)

    F_no_occ  = np.einsum('ki,lj,kl', Emat_ij, Emat_ij, F_mo_occ)
    F_no_vir  = np.einsum('ca,db,cd', Emat_ab1, Emat_ab1, F_mo_vir)

    tmp_occ_ev, tmp_occ_mat = LA.eig(F_no_occ)
    tmp_vir_ev, tmp_vir_mat = LA.eig(F_no_vir)

    C_occ_sc = np.einsum('pi,ij->pj', C_occ_no, tmp_occ_mat)
    C_vir_sc = np.einsum('pa,ab->pb', C_vir_no, tmp_vir_mat)

    F_occ_sc  = np.einsum('ui,vj,uv', C_occ_sc, C_occ_sc, F)
    F_vir_sc  = np.einsum('ua,vb,uv', C_vir_sc, C_vir_sc, F)

    C_np_sc = np.concatenate((C_occ_sc, C_vir_sc), axis=1)

    C_psi4_sc = psi4.core.Matrix.from_array(C_np_sc)

    rhf_wfn.Ca().copy(C_psi4_sc)
    #for item in C:
    #    print(item)
    tmp_1, tmp_2 = fvno_procedure(mol, rhf_e, rhf_wfn, 8)
