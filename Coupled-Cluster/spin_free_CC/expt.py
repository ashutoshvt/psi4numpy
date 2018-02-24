import time
import numpy as np
from numpy import linalg as LA
import sys
sys.path.append('./helpers')
from helper_cc_exp import ndot
from helper_cc_exp import helper_ccenergy
from helper_cc_exp import helper_cchbar
from helper_cc_exp import helper_cclambda
from helper_cc_exp import helper_ccpert
from helper_cc_exp import helper_cclinresp

from helper_cc2 import helper_cc2energy
from helper_cc2 import helper_cc2hbar
from helper_cc2 import helper_cc2lambda
from helper_cc2 import helper_cc2pert
from helper_cc2 import helper_cc2linresp
np.set_printoptions(precision=15, linewidth=200, suppress=True)
import psi4

#psi4.core.set_memory(int(2e9), False)
psi4.set_memory(int(5e9), False)
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

#mol = psi4.geometry("""
#   units Angstrom
#   #no_com
#   #no_reorient
#   0 1
#   C               14.600000000000000    14.529999999999999    15.130000000000001
#   O               14.600000000000000    14.529999999999999    16.530000000000001
#   C               15.859999999999999    14.529999999999999    15.849999999999987
#   C               14.519999999999989    15.709999999999988    14.300000000000001
#   H               13.579999999999993    15.709999999999988    13.750000000000000
#   H               14.580000000000000    16.600000000000001    14.919999999999989
#   H               15.350000000000000    15.709999999999988    13.589999999999991
#   H               14.089999999999998    13.640000000000001    14.769999999999992
#   H               16.430000000000000    13.640000000000001    15.590000000000000
#   H               16.430000000000000    15.419999999999995    15.590000000000000
#
#   symmetry c1
#   """)
#

#O
#H 1 1.1
#H 1 1.1 2 104
#symmetry c1
#""")

psi4.set_options({'basis': 'aug-cc-pVDZ'})
psi4.set_num_threads(24)
psi4.set_options({'guess': 'sad'})
#psi4.set_options({'basis': 'ORP'})

# For numpy
compare_psi4 = True

print('Computing RHF reference.')
psi4.core.set_active_molecule(mol)
psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
psi4.set_module_options('SCF', {'E_CONVERGENCE':1e-9})
psi4.set_module_options('SCF', {'D_CONVERGENCE':1e-9})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)  
print('RHF Final Energy                          % 16.10f\n' % rhf_e)

def guess_optrot_density(mol, rhf_e, rhf_wfn, corr_wfn, maxiter_cc, maxiter_lambda, maxiter_pert, inhom_pert, memory):

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

    pert = [0]    
    
    for p in pert:
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
        #print('\nsolving right hand perturbed amplitudes for %s\n' % string_Mu)
        #ccpert[string_Mu].solve('right', 1e-7, maxiter_pert)
        #print('\nsolving right hand perturbed amplitudes for %s\n' % string_L)
        #ccpert[string_L].solve('right', 1e-7, maxiter_pert)
        #print('\nsolving left hand perturbed amplitudes for %s\n'% string_Mu)
        #ccpert[string_Mu].solve('left', 1e-7, maxiter_pert)
        #print('\nsolving left hand perturbed amplitudes for %s\n'% string_L)
        #ccpert[string_L].solve('left', 1e-7, maxiter_pert)

        #return form_first_order_densities(ccpert, pert, Mu, L, cc_wfn, cclambda, omega)

    for p in pert:
    
        string_Mu = "MU_" + cart[p]
        string_L = "L_" + cart[p]

        # 2nd order density 
        Dij[string_Mu+string_L] =  0.5 * np.einsum('ia,ja->ij', ccpert[string_Mu].x1, ccpert[string_L].y1)
        Dij[string_Mu+string_L] += 0.5 * np.einsum('ikab,jkab->ij', ccpert[string_Mu].x2, ccpert[string_L].y2)
        Dij[string_Mu+string_L] =  -1.0 * Dij[string_Mu + string_L]

        """Dab[string_Mu+string_L]  =   0.5 * np.einsum('ia,ib->ab', ccpert[string_Mu].x1, ccpert[string_L].y1)
        Dab[string_Mu+string_L]  +=  0.5 * np.einsum('ijac,ijbc->ab', ccpert[string_Mu].x2, ccpert[string_L].y2)
        Dab[string_Mu+string_L]  +=  0.5 * np.einsum('ia,ib->ab', ccpert[string_Mu].y1, ccpert[string_L].x1)
        Dab[string_Mu+string_L]  +=  0.5 * np.einsum('ijac,ijbc->ab', ccpert[string_Mu].y2, ccpert[string_L].x2)
        """
        
        # new 2nd order density
        #print("MU_vo")
        #print(ccpert[string_Mu].build_Avo()) 
        #print("L_vo")
        #print(ccpert[string_L].build_Avo()) 

        #import matplotlib
        #import matplotlib.pyplot as plt
        #plt.plot([1,2,3,4])
        #plt.ylabel('some numbers')
        #plt.show()

        
        Mubar_vo =    ccpert[string_Mu].build_Avo().swapaxes(0,1)
        Mubar_vo /=   ccpert[string_Mu].Dia
        Mubar_vvoo =  ccpert[string_Mu].build_Avvoo().swapaxes(0,2).swapaxes(1,3)
        Mubar_vvoo += ccpert[string_Mu].build_Avvoo().swapaxes(0,3).swapaxes(1,2)
        Mubar_vvoo /=   ccpert[string_Mu].Dijab

        Lbar_vo =    ccpert[string_L].build_Avo().swapaxes(0,1)
        Lbar_vo /=   ccpert[string_L].Dia
        Lbar_vvoo =  ccpert[string_L].build_Avvoo().swapaxes(0,2).swapaxes(1,3)
        Lbar_vvoo += ccpert[string_L].build_Avvoo().swapaxes(0,3).swapaxes(1,2)
        Lbar_vvoo /=   ccpert[string_L].Dijab

        Dab[string_Mu+string_L] =     np.einsum('ia,ib->ab', Mubar_vo, Lbar_vo)
        #Dab[string_Mu+string_L] =     1.0 * np.einsum('ia,ib->ab', Mu[string_Mu][cc_wfn.slice_o,cc_wfn.slice_v], Lbar_vo)
        Dab[string_Mu+string_L] +=    2.0 * np.einsum('ijac,ijbc->ab', Mubar_vvoo, Lbar_vvoo)
        Dab[string_Mu+string_L] -=    1.0 * np.einsum('ijac,ijcb->ab', Mubar_vvoo, Lbar_vvoo)

        #Dab[string_Mu+string_L] +=   2.0 * np.einsum('ijac,ijbc->ab', ccpert[string_Mu].x2, Lbar_vvoo)
        #Dab[string_Mu+string_L] -=   1.0 * np.einsum('ijac,ijcb->ab', ccpert[string_Mu].x2, Lbar_vvoo)

        #Dab[string_Mu+string_L] +=   2.0 * np.einsum('ijac,ijbc->ab', Mubar_vvoo, ccpert[string_L].x2)
        #Dab[string_Mu+string_L] -=   1.0 * np.einsum('ijac,ijcb->ab', Mubar_vvoo, ccpert[string_L].x2)

        #Dab[string_Mu+string_L] +=   0.5 * np.einsum('ia,ib->ab', ccpert[string_Mu].y1, ccpert[string_L].x1)

        #Dab[string_Mu+string_L] +=  0.5 * np.einsum('ia,ib->ab', Lbar_vo, ccpert[string_Mu].y1)
        #Dab[string_Mu+string_L] +=  0.5 * np.einsum('ijac,ijbc->ab', Lbar_vvoo, ccpert[string_Mu].y2)"""

        Dij[string_Mu + string_L] = 0.5 * (Dij[string_Mu + string_L] + Dij[string_Mu + string_L].T)
        Dab[string_Mu + string_L] = 0.5 * (Dab[string_Mu + string_L] + Dab[string_Mu + string_L].T)

    nmo = rhf_wfn.nmo()
    occ = rhf_wfn.doccpi()[0]
    vir = nmo - occ
    #Dab = np.zeros((vir,vir))
    #Dij = np.zeros((occ,occ))
    #Dab = Dab["MU_XL_X"] +  Dab["MU_YL_Y"] +  Dab["MU_ZL_Z"]    
    Dab = Dab["MU_XL_X"] 
    #Dij = Dij["MU_XL_X"] +  Dij["MU_YL_Y"] +  Dij["MU_ZL_Z"]    
    #Dab /= 3
    #Dij /= 3

    #Evec_ij, Emat_ij = LA.eig(Dij[string_Mu + string_L])
    #Evec_ab, Emat_ab = LA.eig(Dab[string_Mu + string_L])
    #Evec_ij, Emat_ij = LA.eig(Dij)
    Evec_ab, Emat_ab = LA.eig(Dab)

    #Emat_ij =  sort(Emat_ij, Evec_ij)
    Emat_ab =  sort(Emat_ab, Evec_ab)

    return Emat_ab


def optrot_calculate(mol, rhf_e, rhf_wfn, corr_wfn, maxiter_cc, maxiter_lambda, maxiter_pert, inhom_pert, memory):

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

    pert = [0]    
    
    for p in pert:
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

        
    optrot_density_Mu={};optrot_density_Mu_ij={};optrot_density_Mu_ab={};
    optrot_density_Mu_ia={};optrot_density_Mu_ai={};optrot_PQ_Mu={};

    optrot_density_L={};optrot_density_L_ij={};optrot_density_L_ab={};
    optrot_density_L_ia={};optrot_density_L_ai={};optrot_PQ_L={};
    

    for p in pert:
        print('\n Calculating rotation tensor from linear response function: %s \n'% cart[p])
        optrot_PQ={}
        str_p = "MU_" + cart[p]
        for q in pert:
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
                print('\n optrot2_1 = %20.15lf \n' % cclinresp[str_pq].polar2_1)
                print('\n optrot2_2 = %20.15lf \n' % cclinresp[str_pq].polar2_2)
                print('\n optrot_response muL = %20.15lf \n' % optrot_PQ[str_pq])
                print('\n optrot1 = %20.15lf \n' % cclinresp[str_qp].polar1)
                print('\n optrot2_1 = %20.15lf \n' % cclinresp[str_qp].polar2_1)
                print('\n optrot2_2 = %20.15lf \n' % cclinresp[str_qp].polar2_2)
                print('\n optrot_response Lmu= %20.15lf \n' %  optrot_PQ[str_qp])
                optrot =  0.5 * optrot_PQ[str_pq] - 0.5 * optrot_PQ[str_qp]
                print('\n optrot_response = %20.15lf %s \n' % (optrot, corr_wfn))


def sort(Emat, Evec):
    tmp = Evec.copy()
    tmp = abs(tmp)
    idx =  tmp.argsort()[::-1]   
    Evec = Evec[idx]
    print("Norm")
    print(np.linalg.norm(Evec))
    
    print('\n Printing sorted eigenvalues\n')
    
    for item in Evec:
        print(item)
    
    Emat = Emat[:,idx]
    return Emat

def form_first_order_densities(ccpert, pert, Mu, L, ccsd, cclambda, omega):

    cart = {0:'X', 1: 'Y', 2: 'Z'}
    Dij={}; Dab={}; Dia={}; Dai={};
    for p in pert:
        string_Mu = "MU_" + cart[p]
        string_L = "L_" + cart[p]
   
        # Occupied - Occupied block of Density #
        
        Dij[string_Mu]  =  np.einsum('ia,ja->ij', ccpert[string_Mu].x1, cclambda.l1)
        Dij[string_Mu] +=  np.einsum('ia,ja->ij', ccsd.t1, ccpert[string_Mu].y1)
        Dij[string_Mu] +=  np.einsum('ikab,jkab->ij', ccsd.t2, ccpert[string_Mu].y2)
        Dij[string_Mu] +=  np.einsum('ikab,jkab->ij', ccpert[string_Mu].x2, cclambda.l2)
        Dij[string_Mu] = -1.0 * Dij[string_Mu] 


        # Virtual - Virtual block of Density #

        # 'ia,ib->ba': ba is absolutley essential instead of ab because angular momentum
        # integrals are anti-hermitian because of which Lab*Dab = - Lab*Dba
               
        Dab[string_Mu] =   np.einsum('ia,ib->ba', ccpert[string_Mu].x1, cclambda.l1)
        Dab[string_Mu] +=  np.einsum('ia,ib->ba', ccsd.t1, ccpert[string_Mu].y1)
        Dab[string_Mu] +=  np.einsum('ijac,ijbc->ba', ccsd.t2, ccpert[string_Mu].y2)
        Dab[string_Mu] +=  np.einsum('ijac,ijbc->ba', ccpert[string_Mu].x2, cclambda.l2)
        
        # Virtual - Occupied block of Density #
        
        Dai[string_Mu] = ccpert[string_Mu].y1.swapaxes(0,1).copy()
        
        
        # Occupied - Virtual block of Density #
        
        # 1st term
        Dia[string_Mu] = 2.0 * ccpert[string_Mu].x1.copy()  
    
        # factor of 2.0 because of Y and L - derived using unitary group formalism
        
        # 2nd term
        
        Dia[string_Mu] +=  2.0 * np.einsum('imae,me->ia', ccsd.t2, ccpert[string_Mu].y1)
        Dia[string_Mu] += -1.0 * np.einsum('imea,me->ia', ccsd.t2, ccpert[string_Mu].y1)
        
        Dia[string_Mu] +=  2.0 * np.einsum('imae,me->ia', ccpert[string_Mu].x2, cclambda.l1)
        Dia[string_Mu] += -1.0 * np.einsum('imea,me->ia', ccpert[string_Mu].x2, cclambda.l1)
    
        # 3rd term
    
        Dia[string_Mu] += -1.0 * np.einsum('ie,ma,me->ia', ccsd.t1, ccsd.t1, ccpert[string_Mu].y1)
        Dia[string_Mu] += -1.0 * np.einsum('ie,ma,me->ia', ccsd.t1, ccpert[string_Mu].x1, cclambda.l1)
        Dia[string_Mu] += -1.0 * np.einsum('ie,ma,me->ia', ccpert[string_Mu].x1, ccsd.t1, cclambda.l1)
        
        # 4_th term
        
        Dia[string_Mu] += -1.0 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccsd.t2, ccpert[string_Mu].x1)
        Dia[string_Mu] += -1.0 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccpert[string_Mu].x2, ccsd.t1)
        Dia[string_Mu] += -1.0 * np.einsum('mnef,inef,ma->ia', ccpert[string_Mu].y2, ccsd.t2, ccsd.t1)
        
        # 5th term
        
        Dia[string_Mu] += -1.0 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccsd.t2, ccpert[string_Mu].x1)
        Dia[string_Mu] += -1.0 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccpert[string_Mu].x2, ccsd.t1)
        Dia[string_Mu] += -1.0 * np.einsum('mnef,mnaf,ie->ia', ccpert[string_Mu].y2, ccsd.t2, ccsd.t1)

        ########################################################################    

        #### Angmom density now  ###############################################

        ########################################################################    

        Dij[string_L]  =  np.einsum('ia,ja->ij', ccpert[string_L].x1, cclambda.l1)
        Dij[string_L] +=  np.einsum('ia,ja->ij', ccsd.t1, ccpert[string_L].y1)
        Dij[string_L] +=  np.einsum('ikab,jkab->ij', ccsd.t2, ccpert[string_L].y2)
        Dij[string_L] +=  np.einsum('ikab,jkab->ij', ccpert[string_L].x2, cclambda.l2)
        Dij[string_L] = -1.0 * Dij[string_L]


        # Virtual - Virtual block of Density #

        # 'ia,ib->ba': ba is absolutley essential instead of ab because angular momentum
        # integrals are anti-hermitian because of which Lab*Dab = - Lab*Dba

        Dab[string_L] =   np.einsum('ia,ib->ba', ccpert[string_L].x1, cclambda.l1)
        Dab[string_L] +=  np.einsum('ia,ib->ba', ccsd.t1, ccpert[string_L].y1)
        Dab[string_L] +=  np.einsum('ijac,ijbc->ba', ccsd.t2, ccpert[string_L].y2)
        Dab[string_L] +=  np.einsum('ijac,ijbc->ba', ccpert[string_L].x2, cclambda.l2)


        # Virtual - Occupied block of Density #

        Dai[string_L] = ccpert[string_L].y1.swapaxes(0,1).copy()

        # Occupied - Virtual block of Density #

        # 1st term
        Dia[string_L] = 2.0 * ccpert[string_L].x1.copy()

        # factor of 2.0 because of Y and L - derived using unitary group formalism

        # 2nd term

        Dia[string_L] +=  2.0 * np.einsum('imae,me->ia', ccsd.t2, ccpert[string_L].y1)
        Dia[string_L] += -1.0 * np.einsum('imea,me->ia', ccsd.t2, ccpert[string_L].y1)

        Dia[string_L] +=  2.0 * np.einsum('imae,me->ia', ccpert[string_L].x2, cclambda.l1)
        Dia[string_L] += -1.0 * np.einsum('imea,me->ia', ccpert[string_L].x2, cclambda.l1)

        # 3rd term

        Dia[string_L] += -1.0 * np.einsum('ie,ma,me->ia', ccsd.t1, ccsd.t1, ccpert[string_L].y1)
        Dia[string_L] += -1.0 * np.einsum('ie,ma,me->ia', ccsd.t1, ccpert[string_L].x1, cclambda.l1)
        Dia[string_L] += -1.0 * np.einsum('ie,ma,me->ia', ccpert[string_L].x1, ccsd.t1, cclambda.l1)

        # 4th term

        Dia[string_L] += -1.0 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccsd.t2, ccpert[string_L].x1)
        Dia[string_L] += -1.0 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccpert[string_L].x2, ccsd.t1)
        Dia[string_L] += -1.0 * np.einsum('mnef,inef,ma->ia', ccpert[string_L].y2, ccsd.t2, ccsd.t1)

        # 5th term

        Dia[string_L] += -1.0 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccsd.t2, ccpert[string_L].x1)
        Dia[string_L] += -1.0 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccpert[string_L].x2, ccsd.t1)
        Dia[string_L] += -1.0 * np.einsum('mnef,mnaf,ie->ia', ccpert[string_L].y2, ccsd.t2, ccsd.t1)


        # new metric he he !!

        tmp =  np.einsum("ac,bc->ab", Mu[string_Mu][ccsd.slice_v, ccsd.slice_v], Dab[string_L])  
        #tmp -= np.einsum("ac,bc", L[string_L][ccsd.slice_v, ccsd.slice_v], Dab[string_Mu])  
        #tmp = np.einsum("ia,ib->ab", Mu[string_Mu][ccsd.slice_o, ccsd.slice_v], Dia[string_L])  
        #print(tmp)
        D2ab = 0.5 * (tmp + tmp.T)    
        Evec_ab, Emat_ab = LA.eig(D2ab)   # huge errors !!
        Emat_ab =  sort(Emat_ab, Evec_ab)
        return Emat_ab



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

 
#optrot_calculate(mol, rhf_e, rhf_wfn, 'CC2', 100, 100, 100, 'true', 40)
#Emat_ij, Emat_ab = guess_optrot_density(mol, rhf_e, rhf_wfn, 'CC2', 6, 0, 6, 'false', 4)
Emat_ab = guess_optrot_density(mol, rhf_e, rhf_wfn, 'CC2', 20, 0, 20, 'false', 4)
# 1 0 3
#frz_vir = [12, 14, 16 ,18, 20]
frz_vir = [0,2,3,4]
Emat_ab1 = np.zeros_like(Emat_ab)

for k in frz_vir:

    print('\nTruncation : %d \n' % k) 

    Emat_ab1 = Emat_ab.copy()
    Emat_view = Emat_ab1[:,vir-k:] 
    Emat_view.fill(0)

    #C_occ_no = np.einsum('pi,ij->pj', C_occ, Emat_ij)
    C_vir_no = np.einsum('pa,ab->pb', C_vir, Emat_ab1)

    #F_no_occ  = np.einsum('ki,lj,kl', Emat_ij, Emat_ij, F_mo_occ)    
    F_no_vir  = np.einsum('ca,db,cd', Emat_ab1, Emat_ab1, F_mo_vir)    

    #tmp_occ_ev, tmp_occ_mat = LA.eig(F_no_occ)
    tmp_vir_ev, tmp_vir_mat = LA.eig(F_no_vir)

    #C_occ_sc = np.einsum('pi,ij->pj', C_occ_no, tmp_occ_mat)
    C_vir_sc = np.einsum('pa,ab->pb', C_vir_no, tmp_vir_mat)

    #F_occ_sc  = np.einsum('ui,vj,uv', C_occ_sc, C_occ_sc, F)
    F_vir_sc  = np.einsum('ua,vb,uv', C_vir_sc, C_vir_sc, F)

    #C_np_sc = np.concatenate((C_occ_sc, C_vir_sc), axis=1)
    C_np_sc = np.concatenate((C_occ, C_vir_sc), axis=1)

    C_psi4_sc = psi4.core.Matrix.from_array(C_np_sc)

    rhf_wfn.Ca().copy(C_psi4_sc)

    optrot_calculate(mol, rhf_e, rhf_wfn, 'CCSD', 100, 100, 100, 'false', 40)

