import time
import numpy as np
from numpy import linalg as LA
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
np.set_printoptions(precision=15, linewidth=200, suppress=True)
import psi4


def helper_guess_density():

print('Computing RHF reference.')
psi4.core.set_active_molecule(mol)
psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
psi4.set_module_options('SCF', {'E_CONVERGENCE':1e-9})
psi4.set_module_options('SCF', {'D_CONVERGENCE':1e-9})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)  
print('RHF Final Energy                          % 16.10f\n' % rhf_e)

def optrot_density(mol, rhf_e, rhf_wfn, corr_wfn, maxiter_cc, maxiter_lambda, maxiter_pert, inhom_pert, memory):

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
    L={}; Lij={}; Lab={}; Lia={}; Lai={};
    Dij={}; Dab={}
    ccpert = {}; optrot_AB = {}; cclinresp = {};

    dipole_array = cc_wfn.mints.ao_dipole()
    angmom_array = cc_wfn.mints.ao_angular_momentum()
    
    
    for p in range(0,1):
        string_Mu = "MU_" + cart[p]
        string_L = "L_" + cart[p]
        Mu[string_Mu] = np.einsum('uj,vi,uv', cc_wfn.npC, cc_wfn.npC, np.asarray(dipole_array[p]))
        L[string_L] = -0.5 * np.einsum('uj,vi,uv', cc_wfn.npC, cc_wfn.npC, np.asarray(angmom_array[p]))
        if corr_wfn == 'CCSD':
            ccpert[string_Mu] = helper_ccpert(string_Mu, Mu[string_Mu], cc_wfn, cchbar, cclambda, omega)
            ccpert[string_L] = helper_ccpert(string_L, L[string_L], cc_wfn, cchbar, cclambda, omega)
        elif corr_wfn == 'CC2':
            ccpert[string_Mu] = helper_cc2pert(string_Mu, Mu[string_Mu], cc_wfn, cchbar, cclambda, omega, inhom_pert)
            ccpert[string_L] = helper_cc2pert(string_L, L[string_L], cc_wfn, cchbar, cclambda, omega, inhom_pert)
        print('\nsolving right hand perturbed amplitudes for %s\n' % string_Mu)
        ccpert[string_Mu].solve('right', 1e-7, maxiter_pert)
        print('\nsolving right hand perturbed amplitudes for %s\n' % string_L)
        ccpert[string_L].solve('right', 1e-7, maxiter_pert)
        print('\nsolving left hand perturbed amplitudes for %s\n'% string_Mu)
        ccpert[string_Mu].solve('left', 1e-7, maxiter_pert)
        print('\nsolving left hand perturbed amplitudes for %s\n'% string_L)
        ccpert[string_L].solve('left', 1e-7, maxiter_pert)
    
    
    for p in range(0,1):
    
        string_Mu = "MU_" + cart[p]
        string_L = "L_" + cart[p]

        # 2nd order density 
        Dij[string_Mu+string_L] =  0.5 * np.einsum('ia,ja->ij', ccpert[string_Mu].x1, ccpert[string_L].y1)
        Dij[string_Mu+string_L] += 0.5 * np.einsum('ikab,jkab->ij', ccpert[string_Mu].x2, ccpert[string_L].y2)
        Dij[string_Mu+string_L] =  -1.0 * Dij[string_Mu + string_L]

        Dab[string_Mu+string_L] =   0.5 * np.einsum('ia,ib->ab', ccpert[string_Mu].x1, ccpert[string_L].y1)
        Dab[string_Mu+string_L] +=  0.5 * np.einsum('ijac,ijbc->ab', ccpert[string_Mu].x2, ccpert[string_L].y2)

    
    optrot_density_Mu={};optrot_density_Mu_ij={};optrot_density_Mu_ab={};
    optrot_density_Mu_ia={};optrot_density_Mu_ai={};optrot_PQ_Mu={};

    optrot_density_L={};optrot_density_L_ij={};optrot_density_L_ab={};
    optrot_density_L_ia={};optrot_density_L_ai={};optrot_PQ_L={};
    
    for p in range(0,1):
    
        string_Mu = "MU_" + cart[p]
        string_L  = "L_"  + cart[p]

        print('\n Calculating rotation tensor from linear response function:\n')
        optrot_PQ={}
        

        for p in range(0,1):
            str_p = "MU_" + cart[p]
            for q in range(0,1):
                if p == q:
                    str_q = "L_" + cart[q]
                    str_pq = "<<" + str_p + ";" + str_q + ">>"
                    str_qp = "<<" + str_q + ";" + str_p + ">>"

                    if corr_wfn == 'CCSD':
                        cclinresp[str_pq]= helper_cclinresp(cclambda, ccpert[str_p], ccpert[str_q])
                        cclinresp[str_qp]= helper_cclinresp(cclambda, ccpert[str_q], ccpert[str_p])
                    elif corr_wfn == 'CC2':
                        cclinresp[str_pq]= helper_cc2linresp(cclambda, ccpert[str_p], ccpert[str_q])
                        cclinresp[str_qp]= helper_cc2linresp(cclambda, ccpert[str_q], ccpert[str_p])

                    optrot_PQ[str_pq]= cclinresp[str_pq].linresp()
                    optrot_PQ[str_qp]= cclinresp[str_qp].linresp()

                    #print(str_pq)
                    print('\n optrot1 = %20.15lf \n' % cclinresp[str_pq].polar1)
                    print('\n optrot2 = %20.15lf \n' % cclinresp[str_pq].polar2)
                    print('\n optrot_response muL = %20.15lf \n' % optrot_PQ[str_pq])
                    print('\n optrot1 = %20.15lf \n' % cclinresp[str_qp].polar1)
                    print('\n optrot2 = %20.15lf \n' % cclinresp[str_qp].polar2)
                    print('\n optrot_response Lmu= %20.15lf \n' %  optrot_PQ[str_qp])
                    optrot =  0.5 * optrot_PQ[str_pq] - 0.5 * optrot_PQ[str_qp]
                    print('\n optrot_response = %20.15lf %s \n' % (optrot, corr_wfn))

        Dij[string_Mu + string_L] = 0.5 * (Dij[string_Mu + string_L] + Dij[string_Mu + string_L].T)
        Dab[string_Mu + string_L] = 0.5 * (Dab[string_Mu + string_L] + Dab[string_Mu + string_L].T)

        Evec_ij, Emat_ij = LA.eig(Dij[string_Mu + string_L])
        Evec_ab, Emat_ab = LA.eig(Dab[string_Mu + string_L])

        Emat_ij =  sort(Emat_ij, Evec_ij)
        Emat_ab =  sort(Emat_ab, Evec_ab)

        return optrot, Emat_ij, Emat_ab


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

 
polar1, Emat_ij, Emat_ab = optrot_density(mol, rhf_e, rhf_wfn, 'CC2', 1, 0, 3, 'true', 4)
#polar1, Emat_ij, Emat_ab = optrot_density(mol, rhf_e, rhf_wfn, 'CCSD', 1, 0, 1, 4)
frz_vir = [15,30,35,40,45,50,55,60,65]
Emat_ab1 = np.zeros_like(Emat_ab)

for k in frz_vir:

    print('\nTruncation : %d \n' % k) 

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

    optrot, tmp_ij, tmp_ab = optrot_density(mol, rhf_e, rhf_wfn, 'CCSD', 100, 100, 100, 'false', 40)
    #optrot, tmp_ij, tmp_ab = optrot_density(mol, rhf_e, rhf_wfn, 'CC2', 100, 100, 100, 'true', 40)
    print('\nOptical rotation CCSD (truncated) : %20.14lf\n' % optrot) 
