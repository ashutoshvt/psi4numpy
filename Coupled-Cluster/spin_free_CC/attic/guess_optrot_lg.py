import time
import numpy as np
from numpy import linalg as LA
from helper_cc import ndot
from helper_cc import helper_mp2_guess
from helper_cc import helper_ccenergy
from helper_cc import helper_cchbar
from helper_cc import helper_cclambda
from helper_cc import helper_ccpert
from helper_cc import helper_cclinresp

np.set_printoptions(precision=15, linewidth=200, suppress=True)
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
#psi4.set_options({'basis': 'sto-3g'})
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

def optrot_density(mol, rhf_e, rhf_wfn, maxiter_cc, maxiter_lambda, maxiter_pert, memory):

    # Compute CCSD
    ccsd = helper_ccenergy(mol, rhf_e, rhf_wfn, memory)
    ccsd.compute_energy(1e-10, maxiter_cc)
    CCSDcorr_E = ccsd.ccsd_corr_e
    CCSD_E = ccsd.ccsd_e
    
    print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
    print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
    print('Total CCSD energy:                      % 16.15f' % CCSD_E)
    
    cchbar = helper_cchbar(ccsd)
    
    cclambda = helper_cclambda(ccsd,cchbar)
    cclambda.compute_lambda(1e-10, maxiter_lambda)
    omega = 0.07735713394560646
    #omega = 0.01
    
    cart = {0:'X', 1: 'Y', 2: 'Z'}
    Mu={}; Muij={}; Muab={}; Muia={}; Muai={};
    L={}; Lij={}; Lab={}; Lia={}; Lai={};
    Dij_Mu={}; Dij2_Mu={}; Dab_Mu={}; Dab2_Mu={}; Dia_Mu={}; Dai_Mu={}; 
    Dij_L={}; Dij2_L={}; Dab_L={}; Dab2_L={}; Dia_L={}; Dai_L={}; 
    Dij={}; Dab={}
    ccpert = {}; optrot_AB = {}; cclinresp = {};
    dipole_array = ccsd.mints.ao_dipole()
    angmom_array = ccsd.mints.ao_angular_momentum()
    
    
    for p in range(0,1):
        string_Mu = "MU_" + cart[p]
        string_L = "L_" + cart[p]
        Mu[string_Mu] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(dipole_array[p]))
        L[string_L] = -0.5 * np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(angmom_array[p]))
        ccpert[string_Mu] = helper_ccpert(string_Mu, Mu[string_Mu], ccsd, cchbar, cclambda, omega)
        ccpert[string_L] = helper_ccpert(string_L, L[string_L], ccsd, cchbar, cclambda, omega)
        print('\nsolving right hand perturbed amplitudes for %s\n' % string_Mu)
        ccpert[string_Mu].solve('right', 1e-10, maxiter_pert)
        print('\nsolving right hand perturbed amplitudes for %s\n' % string_L)
        ccpert[string_L].solve('right', 1e-10, maxiter_pert)
        print('\nsolving left hand perturbed amplitudes for %s\n'% string_Mu)
        ccpert[string_Mu].solve('left', 1e-10, maxiter_pert)
        print('\nsolving left hand perturbed amplitudes for %s\n'% string_L)
        ccpert[string_L].solve('left', 1e-10, maxiter_pert)
    
    # Now that I have solved for x and y, I would like to calculate
    # first order ccsd perturbed density. I am assuming only diagonal
    # cases below
    
    
    
    for p in range(0,1):
    
        string_Mu = "MU_" + cart[p]
        string_L = "L_" + cart[p]
        Muij[string_Mu] = Mu[string_Mu][ccsd.slice_o, ccsd.slice_o]
        Muab[string_Mu] = Mu[string_Mu][ccsd.slice_v, ccsd.slice_v]
        Muia[string_Mu] = Mu[string_Mu][ccsd.slice_o, ccsd.slice_v]
        Muai[string_Mu] = Mu[string_Mu][ccsd.slice_v, ccsd.slice_o]

        Lij[string_L] = L[string_L][ccsd.slice_o, ccsd.slice_o]
        Lab[string_L] = L[string_L][ccsd.slice_v, ccsd.slice_v]
        Lia[string_L] = L[string_L][ccsd.slice_o, ccsd.slice_v]
        Lai[string_L] = L[string_L][ccsd.slice_v, ccsd.slice_o]

   
        # Occupied - Occupied block of Density #
        
        Dij_Mu[string_Mu]  =  np.einsum('ia,ja->ij', ccpert[string_Mu].x1, cclambda.l1)
        Dij_Mu[string_Mu] +=  np.einsum('ia,ja->ij', ccsd.t1, ccpert[string_Mu].y1)
        Dij_Mu[string_Mu] +=  np.einsum('ikab,jkab->ij', ccsd.t2, ccpert[string_Mu].y2)
        Dij_Mu[string_Mu] +=  np.einsum('ikab,jkab->ij', ccpert[string_Mu].x2, cclambda.l2)
        Dij_Mu[string_Mu] = -1.0 * Dij_Mu[string_Mu] 

        # Virtual - Virtual block of Density #

        # 'ia,ib->ba': ba is absolutley essential instead of ab because angular momentum
        # integrals are anti-hermitian because of which Lab*Dab = - Lab*Dba
               
        Dab_Mu[string_Mu] =   np.einsum('ia,ib->ba', ccpert[string_Mu].x1, cclambda.l1)
        Dab_Mu[string_Mu] +=  np.einsum('ia,ib->ba', ccsd.t1, ccpert[string_Mu].y1)
        Dab_Mu[string_Mu] +=  np.einsum('ijac,ijbc->ba', ccsd.t2, ccpert[string_Mu].y2)
        Dab_Mu[string_Mu] +=  np.einsum('ijac,ijbc->ba', ccpert[string_Mu].x2, cclambda.l2)
        

        # Virtual - Occupied block of Density #
        
        Dai_Mu[string_Mu] = ccpert[string_Mu].y1.swapaxes(0,1).copy()
        
        
        # Occupied - Virtual block of Density #
        
        # 1st term
        Dia_Mu[string_Mu] = 2.0 * ccpert[string_Mu].x1.copy()  
    
        # factor of 2.0 because of Y and L - derived using unitary group formalism
        
        # 2nd term
        
        Dia_Mu[string_Mu] +=  2.0 * np.einsum('imae,me->ia', ccsd.t2, ccpert[string_Mu].y1)
        Dia_Mu[string_Mu] += -1.0 * np.einsum('imea,me->ia', ccsd.t2, ccpert[string_Mu].y1)
        
        Dia_Mu[string_Mu] +=  2.0 * np.einsum('imae,me->ia', ccpert[string_Mu].x2, cclambda.l1)
        Dia_Mu[string_Mu] += -1.0 * np.einsum('imea,me->ia', ccpert[string_Mu].x2, cclambda.l1)
    
        # 3rd term
    
        Dia_Mu[string_Mu] += -1.0 * np.einsum('ie,ma,me->ia', ccsd.t1, ccsd.t1, ccpert[string_Mu].y1)
        Dia_Mu[string_Mu] += -1.0 * np.einsum('ie,ma,me->ia', ccsd.t1, ccpert[string_Mu].x1, cclambda.l1)
        Dia_Mu[string_Mu] += -1.0 * np.einsum('ie,ma,me->ia', ccpert[string_Mu].x1, ccsd.t1, cclambda.l1)
        
        # 4_th term
        
        Dia_Mu[string_Mu] += -1.0 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccsd.t2, ccpert[string_Mu].x1)
        Dia_Mu[string_Mu] += -1.0 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccpert[string_Mu].x2, ccsd.t1)
        Dia_Mu[string_Mu] += -1.0 * np.einsum('mnef,inef,ma->ia', ccpert[string_Mu].y2, ccsd.t2, ccsd.t1)
        
        # 5th term
        
        Dia_Mu[string_Mu] += -1.0 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccsd.t2, ccpert[string_Mu].x1)
        Dia_Mu[string_Mu] += -1.0 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccpert[string_Mu].x2, ccsd.t1)
        Dia_Mu[string_Mu] += -1.0 * np.einsum('mnef,mnaf,ie->ia', ccpert[string_Mu].y2, ccsd.t2, ccsd.t1)

        ########################################################################    

        #### Angmom density now  ###############################################

        ########################################################################    

        Dij_L[string_L]  =  np.einsum('ia,ja->ij', ccpert[string_L].x1, cclambda.l1)
        Dij_L[string_L] +=  np.einsum('ia,ja->ij', ccsd.t1, ccpert[string_L].y1)
        Dij_L[string_L] +=  np.einsum('ikab,jkab->ij', ccsd.t2, ccpert[string_L].y2)
        Dij_L[string_L] +=  np.einsum('ikab,jkab->ij', ccpert[string_L].x2, cclambda.l2)
        Dij_L[string_L] = -1.0 * Dij_L[string_L]

        # Virtual - Virtual block of Density #

        # 'ia,ib->ba': ba is absolutley essential instead of ab because angular momentum
        # integrals are anti-hermitian because of which Lab*Dab = - Lab*Dba

        Dab_L[string_L] =   np.einsum('ia,ib->ba', ccpert[string_L].x1, cclambda.l1)
        Dab_L[string_L] +=  np.einsum('ia,ib->ba', ccsd.t1, ccpert[string_L].y1)
        Dab_L[string_L] +=  np.einsum('ijac,ijbc->ba', ccsd.t2, ccpert[string_L].y2)
        Dab_L[string_L] +=  np.einsum('ijac,ijbc->ba', ccpert[string_L].x2, cclambda.l2)

        # Virtual - Occupied block of Density #

        Dai_L[string_L] = ccpert[string_L].y1.swapaxes(0,1).copy()

        # Occupied - Virtual block of Density #

        # 1st term
        Dia_L[string_L] = 2.0 * ccpert[string_L].x1.copy()

        # factor of 2.0 because of Y and L - derived using unitary group formalism

        # 2nd term

        Dia_L[string_L] +=  2.0 * np.einsum('imae,me->ia', ccsd.t2, ccpert[string_L].y1)
        Dia_L[string_L] += -1.0 * np.einsum('imea,me->ia', ccsd.t2, ccpert[string_L].y1)

        Dia_L[string_L] +=  2.0 * np.einsum('imae,me->ia', ccpert[string_L].x2, cclambda.l1)
        Dia_L[string_L] += -1.0 * np.einsum('imea,me->ia', ccpert[string_L].x2, cclambda.l1)

        # 3rd term

        Dia_L[string_L] += -1.0 * np.einsum('ie,ma,me->ia', ccsd.t1, ccsd.t1, ccpert[string_L].y1)
        Dia_L[string_L] += -1.0 * np.einsum('ie,ma,me->ia', ccsd.t1, ccpert[string_L].x1, cclambda.l1)
        Dia_L[string_L] += -1.0 * np.einsum('ie,ma,me->ia', ccpert[string_L].x1, ccsd.t1, cclambda.l1)

        # 4th term

        Dia_L[string_L] += -1.0 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccsd.t2, ccpert[string_L].x1)
        Dia_L[string_L] += -1.0 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccpert[string_L].x2, ccsd.t1)
        Dia_L[string_L] += -1.0 * np.einsum('mnef,inef,ma->ia', ccpert[string_L].y2, ccsd.t2, ccsd.t1)

        # 5th term

        Dia_L[string_L] += -1.0 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccsd.t2, ccpert[string_L].x1)
        Dia_L[string_L] += -1.0 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccpert[string_L].x2, ccsd.t1)
        Dia_L[string_L] += -1.0 * np.einsum('mnef,mnaf,ie->ia', ccpert[string_L].y2, ccsd.t2, ccsd.t1)


        # 2nd order density 
        Dij[string_Mu+string_L] =  0.5 * np.einsum('ia,ja->ij', ccpert[string_Mu].x1, ccpert[string_L].y1)
        Dij[string_Mu+string_L] += 0.5 * np.einsum('ikab,jkab->ij', ccpert[string_Mu].x2, ccpert[string_L].y2)
        Dij[string_Mu+string_L] =  -1.0 * Dij[string_Mu + string_L]

        Dab[string_Mu+string_L] =   0.5 * np.einsum('ia,ib->ab', ccpert[string_Mu].x1, ccpert[string_L].y1)
        Dab[string_Mu+string_L] +=  0.5 * np.einsum('ijac,ijbc->ab', ccpert[string_Mu].x2, ccpert[string_L].y2)

        #Dij[string_Mu+string_L] =  np.einsum('ia,ja->ij', ccpert[string_Mu].x1, ccpert[string_L].y1)
        #Dij[string_Mu+string_L] += 2.0 * np.einsum('ikab,jkab->ij', ccpert[string_Mu].x2, ccpert[string_L].y2)
        #Dij[string_Mu+string_L] -= np.einsum('ikab,kjab->ij', ccpert[string_Mu].x2, ccpert[string_L].y2)
        #Dij[string_Mu+string_L] =  -1.0 * Dij[string_Mu + string_L]

        #Dab[string_Mu+string_L] =   np.einsum('ia,ib->ab', ccpert[string_Mu].x1, ccpert[string_L].y1)
        #Dab[string_Mu+string_L] +=  2.0 * np.einsum('ijac,ijbc->ab', ccpert[string_Mu].x2, ccpert[string_L].y2)
        #Dab[string_Mu+string_L] -=  np.einsum('ijac,ijcb->ab', ccpert[string_Mu].x2, ccpert[string_L].y2)



    # calculate response function <<A;B>> by pert_A * density_B
    # Right now, these are only for diagonal elements.
    
    #print('\n Calculating Polarizability tensor from first order density approach:\n')
    
    optrot_density_Mu={};optrot_density_Mu_ij={};optrot_density_Mu_ab={};
    optrot_density_Mu_ia={};optrot_density_Mu_ai={};optrot_PQ_Mu={};

    optrot_density_L={};optrot_density_L_ij={};optrot_density_L_ab={};
    optrot_density_L_ia={};optrot_density_L_ai={};optrot_PQ_L={};
    
    for p in range(0,1):
    
        string_Mu = "MU_" + cart[p]
        string_L  = "L_"  + cart[p]

        optrot_density_L[string_L] = 0
        optrot_density_L_ij[string_L] =  np.einsum('ij,ij->', Dij_L[string_L], Muij[string_Mu])
        optrot_density_L[string_L] +=  optrot_density_L_ij[string_L]
        print('\nMuij * Dij_L: %20.15lf\n' % optrot_density_L_ij[string_L])

        optrot_density_L_ab[string_L] =  np.einsum('ab,ab->', Dab_L[string_L], Muab[string_Mu])
        optrot_density_L[string_L] +=  optrot_density_L_ab[string_L]
        print('\nMuab * Dab_L: %20.15lf\n' % optrot_density_L_ab[string_L])

        optrot_density_L_ia[string_L] =  np.einsum('ia,ia->', Dia_L[string_L], Muia[string_Mu])
        optrot_density_L[string_L] +=  optrot_density_L_ia[string_L]
        print('\nMuia * Dia_L: %20.15lf\n' % optrot_density_L_ia[string_L])

        optrot_density_L_ai[string_L] =  np.einsum('ai,ai->', Dai_L[string_L], Muai[string_Mu])
        optrot_density_L[string_L] +=  optrot_density_L_ai[string_L]
        print('\nMuai * Dai_L: %20.15lf\n' % optrot_density_L_ai[string_L])

        print('\noptrot_density_Mu*D_L: %20.15lf\n' % optrot_density_L[string_L])

        optrot_density_Mu[string_Mu] = 0
        optrot_density_Mu_ij[string_Mu] =  np.einsum('ij,ij->', Dij_Mu[string_Mu], Lij[string_L])
        optrot_density_Mu[string_Mu] +=  optrot_density_Mu_ij[string_Mu]
        print('\nLij * Dij_Mu: %20.15lf\n' % optrot_density_Mu_ij[string_Mu])
   
        optrot_density_Mu_ab[string_Mu] =  np.einsum('ab,ab->', Dab_Mu[string_Mu], Lab[string_L])
        optrot_density_Mu[string_Mu] +=  optrot_density_Mu_ab[string_Mu]
        print('\nLab * Dab_Mu: %20.15lf\n' % optrot_density_Mu_ab[string_Mu])
    
        optrot_density_Mu_ia[string_Mu] =  np.einsum('ia,ia->', Dia_Mu[string_Mu], Lia[string_L])
        optrot_density_Mu[string_Mu] +=  optrot_density_Mu_ia[string_Mu]
        print('\nLia * Dia_Mu: %20.15lf\n' % optrot_density_Mu_ia[string_Mu])
    
        optrot_density_Mu_ai[string_Mu] =  np.einsum('ai,ai->', Dai_Mu[string_Mu], Lai[string_L])
        optrot_density_Mu[string_Mu] +=  optrot_density_Mu_ai[string_Mu]
        print('\nLai * Dai_Mu: %20.15lf\n' % optrot_density_Mu_ai[string_Mu])

        print('\noptrot_density_L*D_Mu: %20.15lf\n' % optrot_density_Mu[string_Mu])

        optrot = 0.50 * (optrot_density_Mu[string_Mu] - optrot_density_L[string_L])


        print('\n Calculating rotation tensor from linear response function:\n')
        optrot_PQ={}
        

        for p in range(0,1):
            str_p = "MU_" + cart[p]
            for q in range(0,1):
                if p == q:
                    str_q = "L_" + cart[q]
                    str_pq = "<<" + str_p + ";" + str_q + ">>"
                    str_qp = "<<" + str_q + ";" + str_p + ">>"
                    cclinresp[str_pq]= helper_cclinresp(cclambda, ccpert[str_p], ccpert[str_q])
                    cclinresp[str_qp]= helper_cclinresp(cclambda, ccpert[str_q], ccpert[str_p])
                    optrot_PQ[str_pq]= cclinresp[str_pq].linresp()
                    optrot_PQ[str_qp]= cclinresp[str_qp].linresp()
                    #print(str_pq)
                    print('\n optrot1 = %20.15lf \n' % cclinresp[str_pq].polar1)
                    print('\n optrot2 = %20.15lf \n' % cclinresp[str_pq].polar2)
                    print('\n optrot_response muL = %20.15lf \n' % optrot_PQ[str_pq])
                    print('\n optrot1 = %20.15lf \n' % cclinresp[str_qp].polar1)
                    print('\n optrot2 = %20.15lf \n' % cclinresp[str_qp].polar2)
                    print('\n optrot_response Lmu= %20.15lf \n' %  optrot_PQ[str_qp])
                    print('\n optrot_response = %20.15lf \n' % ( 0.5 * optrot_PQ[str_pq] - 0.5 * optrot_PQ[str_qp]))

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

C = np.asarray(rhf_wfn.Ca())
F = np.asarray(rhf_wfn.Fa())
nmo = rhf_wfn.nmo()
occ = rhf_wfn.doccpi()[0]
vir = nmo - occ
C_occ = C[:, :occ]
C_vir = C[:, occ:]
F_mo  = np.einsum('ui,vj,uv', C, C, F)
F_mo_occ = F_mo[:occ,:occ]
F_mo_vir = F_mo[occ:, occ:]

 
polar1, Emat_ij, Emat_ab = optrot_density(mol, rhf_e, rhf_wfn, 1, 0, 1, 4)

frz_vir = 15
Emat_ab1 = np.zeros_like(Emat_ab)

for k in range(frz_vir, frz_vir+1):

    print('\nTruncation : %d \n' % k) 

    Emat_ab1 = Emat_ab.copy()
    Emat_view = Emat_ab1[:,vir-k:] 
    Emat_view.fill(0)

    C_occ_no = np.einsum('pi,ij->pj', C_occ, Emat_ij)
    C_vir_no = np.einsum('pa,ab->pb', C_vir, Emat_ab1)

    F_mo_occ  = np.einsum('ki,lj,kl', Emat_ij, Emat_ij, F_mo_occ)    
    F_mo_vir  = np.einsum('ca,db,cd', Emat_ab1, Emat_ab1, F_mo_vir)    

    tmp_occ_ev, tmp_occ_mat = LA.eig(F_mo_occ)
    tmp_vir_ev, tmp_vir_mat = LA.eig(F_mo_vir)

    C_occ_sc = np.einsum('pi,ij->pj', C_occ_no, tmp_occ_mat)
    C_vir_sc = np.einsum('pa,ab->pb', C_vir_no, tmp_vir_mat)

    F_occ_sc  = np.einsum('ui,vj,uv', C_occ_sc, C_occ_sc, F)
    F_vir_sc  = np.einsum('ua,vb,uv', C_vir_sc, C_vir_sc, F)

    C_np_sc = np.concatenate((C_occ_sc, C_vir_sc), axis=1)

    C_psi4_sc = psi4.core.Matrix.from_array(C_np_sc)

    rhf_wfn.Ca().copy(C_psi4_sc)

    optrot, tmp_ij, tmp_ab = optrot_density(mol, rhf_e, rhf_wfn, 100, 100, 100, 4)
    print('\nOptical rotation (truncated) : %20.14lf\n' % optrot) 
    
