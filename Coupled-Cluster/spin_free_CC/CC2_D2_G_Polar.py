import time
import numpy as np
from numpy import linalg as LA
import sys
sys.path.append("./helpers")
from helper_cc import ndot
from helper_cc import helper_ccenergy
from helper_cc import helper_cchbar
from helper_cc import helper_cclambda
from helper_cc import helper_ccpert
from helper_cc import helper_cclinresp

from helper_cc2 import helper_cc2energy
from helper_cc2 import helper_cc2hbar
from helper_cc2 import helper_cc2lambda
from helper_cc2 import helper_cc2pert
from helper_cc2 import helper_cc2linresp
np.set_printoptions(precision=15, linewidth=200, suppress=True, threshold=np.nan)
import psi4

#psi4.core.set_memory(int(2e9), False)
psi4.set_memory(int(8e9), False)
psi4.core.set_output_file('output.dat', False)

numpy_memory = 2

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

psi4.set_options({'basis': 'aug-cc-pVDZ'})
psi4.set_num_threads(24)

# For numpy
compare_psi4 = True

print('Computing RHF reference.')
psi4.core.set_active_molecule(mol)
psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
psi4.set_module_options('SCF', {'E_CONVERGENCE':1e-9})
psi4.set_module_options('SCF', {'D_CONVERGENCE':1e-9})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)  
print('RHF Final Energy                          % 16.10f\n' % rhf_e)

def polar_density(mol, rhf_e, rhf_wfn, corr_wfn, maxiter_cc, maxiter_lambda, maxiter_pert, inhom_pert, memory):

    # Compute CCSD
    if corr_wfn == 'CCSD':
        cc_wfn = helper_ccenergy(mol, rhf_e, rhf_wfn, memory)
        cc_wfn.compute_energy(1e-7, maxiter_cc)
        CCcorr_E = cc_wfn.ccsd_corr_e
        CC_E = cc_wfn.ccsd_e
    elif corr_wfn == 'CC2':
        cc_wfn = helper_cc2energy(mol, rhf_e, rhf_wfn, memory)
        cc_wfn.compute_energy(1e-7, maxiter_cc)
        CCcorr_E = cc_wfn.cc2_corr_e
        CC_E = cc_wfn.cc2_e
   
    print('Fock')
    print(np.diag(cc_wfn.F))
 
    if corr_wfn == 'CCSD':
        print('\nFinal CCSD correlation energy:          % 16.15f' % CCcorr_E)
        print('Total CCSD energy:                      % 16.15f' % CC_E)
        cchbar = helper_cchbar(cc_wfn)
        cclambda = helper_cclambda(cc_wfn,cchbar)
    elif corr_wfn == 'CC2':
        print('\nFinal CC2 correlation energy:          % 16.15f' % CCcorr_E)
        print('Total CC2 energy:                      % 16.15f' % CC_E)
        cchbar = helper_cc2hbar(cc_wfn)
        cclambda = helper_cc2lambda(cc_wfn,cchbar)
    
    cclambda.compute_lambda(1e-7, maxiter_lambda)
    omega = 0.07735713394560646
    #omega = 0.01
    
    cart = {0:'X', 1: 'Y', 2: 'Z'}
    Mu={}; Muij={}; Muab={}; Muia={}; Muai={};
    Dij={}; Dab={}
    ccpert = {}; cclinresp = {};

    dipole_array = cc_wfn.mints.ao_dipole()
    
    
    for p in range(0,1):
        string_Mu = "MU_" + cart[p]
        Mu[string_Mu] = np.einsum('uj,vi,uv', cc_wfn.npC, cc_wfn.npC, np.asarray(dipole_array[p]))
        if corr_wfn == 'CCSD':
            ccpert[string_Mu] = helper_ccpert(string_Mu, Mu[string_Mu], cc_wfn, cchbar, cclambda, omega)
        elif corr_wfn == 'CC2':
            ccpert[string_Mu] = helper_cc2pert(string_Mu, Mu[string_Mu], cc_wfn, cchbar, cclambda, omega, inhom_pert)
        print('\nsolving right hand perturbed amplitudes for %s\n' % string_Mu)
        ccpert[string_Mu].solve('right', 1e-7, maxiter_pert)
        print('\nsolving left hand perturbed amplitudes for %s\n'% string_Mu)
        ccpert[string_Mu].solve('left', 1e-7, maxiter_pert)
        print(ccpert[string_Mu].x1)
        print(ccpert[string_Mu].y1)
    
    
    for p in range(0,1):
    
        string_Mu = "MU_" + cart[p]

        # 2nd order density 
        Dij[string_Mu] =  0.5 * np.einsum('ia,ja->ij', ccpert[string_Mu].x1, ccpert[string_Mu].y1)
        Dij[string_Mu] += 0.5 * np.einsum('ikab,jkab->ij', ccpert[string_Mu].x2, ccpert[string_Mu].y2)
        Dij[string_Mu] =  -1.0 * Dij[string_Mu]

        Dab[string_Mu] =   0.5 * np.einsum('ia,ib->ab', ccpert[string_Mu].x1, ccpert[string_Mu].y1)
        Dab[string_Mu] +=  0.5 * np.einsum('ijac,ijbc->ab', ccpert[string_Mu].x2, ccpert[string_Mu].y2)

        """
        # New 2nd order density 
        Abar_vo =    ccpert[string_Mu].build_Avo().swapaxes(0,1)
        #Abar_vo /=   ccpert[string_Mu].Dia
        Abar_vvoo =  ccpert[string_Mu].build_Avvoo().swapaxes(0,2).swapaxes(1,3)
        Abar_vvoo += ccpert[string_Mu].build_Avvoo().swapaxes(0,3).swapaxes(1,2)
        #Abar_vvoo /=   ccpert[string_Mu].Dijab

        Dab[string_Mu] =   np.einsum('ia,ib->ab', Abar_vo, ccpert[string_Mu].y1)
        #Dab[string_Mu] +=  2.0 * np.einsum('ia,ib->ab', ccpert[string_Mu].get_pert('ov'), ccpert[string_Mu].x1)
        Dab[string_Mu] +=  0.5 * np.einsum('ijac,ijbc->ab', Abar_vvoo, ccpert[string_Mu].y2)
        """
    polar = 0 

    print('\n Calculating polarization tensor from linear response function:\n')
    
    for p in range(0,1):
        str_p = "MU_" + cart[p]
        for q in range(0,1):
            if p == q:
                str_q = "MU_" + cart[q]
                str_pq = "<<" + str_p + ";" + str_q + ">>"
                if corr_wfn == 'CCSD':
                    cclinresp[str_pq]= helper_cclinresp(cclambda, ccpert[str_p], ccpert[str_q])
                elif corr_wfn == 'CC2':
                    cclinresp[str_pq]= helper_cc2linresp(cclambda, ccpert[str_p], ccpert[str_q])
                polar = cclinresp[str_pq].linresp()

                #print(str_pq)
                #print('\n polar1 = %20.15lf \n' % cclinresp[str_pq].polar1)
                #print('\n polar2 = %20.15lf \n' % cclinresp[str_pq].polar2)
                #print('\n  Polarizability:  %20.15lf \n' % polar)

        Dij[string_Mu] = 0.5 * (Dij[string_Mu] + Dij[string_Mu].T)
        Dab[string_Mu] = 0.5 * (Dab[string_Mu] + Dab[string_Mu].T)

        Evec_ij, Emat_ij = LA.eig(Dij[string_Mu])
        Evec_ab, Emat_ab = LA.eig(Dab[string_Mu])

        Emat_ij =  sort(Emat_ij, Evec_ij)
        Emat_ab =  sort(Emat_ab, Evec_ab)

        return polar, Emat_ij, Emat_ab


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

#C = np.asarray(rhf_wfn.Ca())
#F = np.asarray(rhf_wfn.Fa())



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

 
polar1, Emat_ij, Emat_ab = polar_density(mol, rhf_e, rhf_wfn, 'CC2', 1, 0, 3, 'false', 4)
#polar1, Emat_ij, Emat_ab = optrot_density(mol, rhf_e, rhf_wfn, 'CCSD', 1, 0, 1, 4)
#frz_vir = [0, int(.05 * vir), int(.10 * vir), int(.15 * vir), int(.20 * vir), int(.25 * vir)]
frz_vir = [0]
Emat_ab1 = np.zeros_like(Emat_ab)

for k in frz_vir:

    print('\nTruncation : %d \n' % k) 

    Emat_ab1 = Emat_ab.copy()
    Emat_view = Emat_ab1[:,vir-k:] 
    Emat_view.fill(0)

    #C_occ_no = np.einsum('pi,ij->pj', C_occ, Emat_ij)
    C_vir_no = np.einsum('pa,ab->pb', C_vir, Emat_ab1)

    F_no_occ  = np.einsum('ki,lj,kl', Emat_ij, Emat_ij, F_mo_occ)    
    F_no_vir  = np.einsum('ca,db,cd', Emat_ab1, Emat_ab1, F_mo_vir)    

    tmp_occ_ev, tmp_occ_mat = LA.eig(F_no_occ)
    tmp_vir_ev, tmp_vir_mat = LA.eig(F_no_vir)

    #C_occ_sc = np.einsum('pi,ij->pj', C_occ_no, tmp_occ_mat)
    C_vir_sc = np.einsum('pa,ab->pb', C_vir_no, tmp_vir_mat)

    #F_occ_sc  = np.einsum('ui,vj,uv', C_occ_sc, C_occ_sc, F)
    F_vir_sc  = np.einsum('ua,vb,uv', C_vir_sc, C_vir_sc, F)

    #C_np_sc = np.concatenate((C_occ_sc, C_vir_sc), axis=1)
    #C_np_sc = np.concatenate((C_occ, C_vir_sc), axis=1)
    C_np_no = np.concatenate((C_occ, C_vir_no), axis=1)

    #C_psi4_sc = psi4.core.Matrix.from_array(C_np_sc)
    C_psi4_no = psi4.core.Matrix.from_array(C_np_no)

    #rhf_wfn.Ca().copy(C_psi4_sc)
    rhf_wfn.Ca().copy(C_psi4_no)

    polar, tmp_ij, tmp_ab = polar_density(mol, rhf_e, rhf_wfn, 'CCSD', 100, 100, 100, 'false', 40)
    #optrot, tmp_ij, tmp_ab = polar_density(mol, rhf_e, rhf_wfn, 'CC2', 100, 100, 100, 'true', 40)
    print('\nPolarizability CCSD (truncation: %d ) : %20.14lf\n' % (k, polar)) 

