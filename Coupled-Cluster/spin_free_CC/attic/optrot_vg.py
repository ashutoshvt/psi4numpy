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

#psi4.core.set_memory(int(2e9), False)
psi4.set_memory(int(12e9), False)
psi4.set_num_threads(24)
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

psi4.set_options({'basis': 'aug-cc-pVQZ'})
#psi4.set_options({'basis': 'cc-pVDZ'})
#psi4.set_options({'basis': 'sto-3g'})
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
c_square = 137 * 137 

def fvno_procedure(mol, rhf_e, rhf_wfn, memory, MVG):

    # Compute CCSD
    #for i in np.asarray(rhf_wfn.Ca()):
    #    print(i)

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
    omega = 0.07735713394560646
    omega_zero = 0
    #omega = 0.01
    pert = [0] 
    cart = {0:'X', 1: 'Y', 2: 'Z'}
    P={}; Pij={}; Pab={}; Pia={}; Pai={};
    L={}; Lij={}; Lab={}; Lia={}; Lai={};
    Dij_P={}; Dij2_P={}; Dab_P={}; Dab2_P={}; Dia_P={}; Dai_P={}; 
    Dij_L={}; Dij2_L={}; Dab_L={}; Dab2_L={}; Dia_L={}; Dai_L={}; 
    Dij={}; Dab={}
    ccpert = {}; optrot_AB = {}; cclinresp = {};
    nabla_array = ccsd.mints.ao_nabla()
    angmom_array = ccsd.mints.ao_angular_momentum()
    
    
    for p in pert:
        string_P = "P_" + cart[p]
        string_L = "L_" + cart[p]
        P[string_P] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(nabla_array[p]))
        L[string_L] = -0.5 * np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(angmom_array[p]))
        ccpert[string_P + str(omega)] = helper_ccpert(string_P, P[string_P], ccsd, cchbar, cclambda, omega)
        ccpert[string_L + str(omega)] = helper_ccpert(string_L, L[string_L], ccsd, cchbar, cclambda, omega)
        print('\nsolving right hand perturbed amplitudes for %s\n' % string_P)
        ccpert[string_P + str(omega)].solve('right', r_conv=1e-7)
        print('\nsolving right hand perturbed amplitudes for %s\n' % string_L)
        ccpert[string_L + str(omega)].solve('right', r_conv=1e-7)
        print('\nsolving left hand perturbed amplitudes for %s\n'% string_P)
        ccpert[string_P + str(omega)].solve('left', r_conv=1e-7)
        print('\nsolving left hand perturbed amplitudes for %s\n'% string_L)
        ccpert[string_L + str(omega)].solve('left', r_conv=1e-7)
    
        if (MVG):
            ccpert[string_P + str(omega_zero)] = helper_ccpert(string_P, P[string_P], ccsd, cchbar, cclambda, omega_zero)
            ccpert[string_L + str(omega_zero)] = helper_ccpert(string_L, L[string_L], ccsd, cchbar, cclambda, omega_zero)
            print('\nsolving right hand perturbed amplitudes for %s\n' % string_P)
            ccpert[string_P + str(omega_zero)].solve('right', r_conv=1e-7)
            print('\nsolving right hand perturbed amplitudes for %s\n' % string_L)
            ccpert[string_L + str(omega_zero)].solve('right', r_conv=1e-7)
            print('\nsolving left hand perturbed amplitudes for %s\n'% string_P)
            ccpert[string_P + str(omega_zero)].solve('left', r_conv=1e-7)
            print('\nsolving left hand perturbed amplitudes for %s\n'% string_L)
            ccpert[string_L + str(omega_zero)].solve('left', r_conv=1e-7)
    # Now that I have solved for x and y, I would like to calculate
    # first order ccsd perturbed density. I am assuming only diagonal
    # cases below
    
    
    for p in pert:
    
        string_P = "P_" + cart[p]
        string_L = "L_" + cart[p]
        Pij[string_P] = P[string_P][ccsd.slice_o, ccsd.slice_o]
        Pab[string_P] = P[string_P][ccsd.slice_v, ccsd.slice_v]
        Pia[string_P] = P[string_P][ccsd.slice_o, ccsd.slice_v]
        Pai[string_P] = P[string_P][ccsd.slice_v, ccsd.slice_o]

        Lij[string_L] = L[string_L][ccsd.slice_o, ccsd.slice_o]
        Lab[string_L] = L[string_L][ccsd.slice_v, ccsd.slice_v]
        Lia[string_L] = L[string_L][ccsd.slice_o, ccsd.slice_v]
        Lai[string_L] = L[string_L][ccsd.slice_v, ccsd.slice_o]

        #print("\nDifference between occupied-virtual blocks of dipole and angmom integrals\n")
        #diff1 = abs(Pia[string_P]) - abs(Lia[string_L]) 
        #print(diff1)
        #print("\nDifference between virtual-occupied  blocks of dipole and angmom integrals\n")
        #diff2 = abs(Pai[string_P]) - abs(Lai[string_L]) 
        #print(diff2)


        """
        import matplotlib.pyplot as plt 
        plt.subplot(211)
        plt.imshow(abs(Lai[string_L]), cmap=plt.cm.BuPu_r)
        plt.subplot(212)
        plt.imshow(abs(Pai[string_P]), cmap=plt.cm.BuPu_r)
        plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax)
        plt.savefig("Matrix.png")
        """
        """ 
        # Occupied - Occupied block of Density #
        
        Dij_P[string_P]  =  np.einsum('ia,ja->ij', ccpert[string_P].x1, cclambda.l1)
        Dij_P[string_P] +=  np.einsum('ia,ja->ij', ccsd.t1, ccpert[string_P].y1)
        Dij_P[string_P] +=  np.einsum('ikab,jkab->ij', ccsd.t2, ccpert[string_P].y2)
        Dij_P[string_P] +=  np.einsum('ikab,jkab->ij', ccpert[string_P].x2, cclambda.l2)
        Dij_P[string_P] = -1.0 * Dij_P[string_P] 


        #Dij2_P[string_P] =  np.einsum('ia,ja->ij', ccpert[string_P].x1, ccpert[string_P].y1)
        #Dij2_P[string_P] += np.einsum('ikab,jkab->ij', ccpert[string_P].x2, ccpert[string_P].y2)
        #Dij2_P[string_P] =  -1.0 * Dij2_P[string_P] 
      
        #Dij2_P[string_P] =  np.einsum('ia,ja->ij', ccpert[string_P].y1, ccpert[string_P].y1)
        #Dij2_P[string_P] += np.einsum('ikab,jkab->ij', ccpert[string_P].y2, ccpert[string_P].y2)
        #Dij2_P[string_P] =  -1.0 * Dij2_P[string_P] 

        Dij2_P[string_P] =  np.einsum('ia,ja->ij', ccpert[string_P].x1, ccpert[string_P].x1)
        Dij2_P[string_P] += 2.0 * np.einsum('ikab,jkab->ij', ccpert[string_P].x2, ccpert[string_P].x2)
        Dij2_P[string_P] -= np.einsum('ikab,kjab->ij', ccpert[string_P].x2, ccpert[string_P].x2)
        Dij2_P[string_P] =  -1.0 * Dij2_P[string_P] 

        # Virtual - Virtual block of Density #

        # 'ia,ib->ba': ba is absolutley essential instead of ab because angular momentum
        # integrals are anti-hermitian because of which Lab*Dab = - Lab*Dba
               
        Dab_P[string_P] =   np.einsum('ia,ib->ba', ccpert[string_P].x1, cclambda.l1)
        Dab_P[string_P] +=  np.einsum('ia,ib->ba', ccsd.t1, ccpert[string_P].y1)
        Dab_P[string_P] +=  np.einsum('ijac,ijbc->ba', ccsd.t2, ccpert[string_P].y2)
        Dab_P[string_P] +=  np.einsum('ijac,ijbc->ba', ccpert[string_P].x2, cclambda.l2)
        
        
        #Dab2_P[string_P] =   np.einsum('ia,ib->ab', ccpert[string_P].x1, ccpert[string_P].y1)
        #Dab2_P[string_P] +=  np.einsum('ijac,ijbc->ab', ccpert[string_P].x2, ccpert[string_P].y2)

        #Dab2_P[string_P] =   np.einsum('ia,ib->ab', ccpert[string_P].y1, ccpert[string_P].y1)
        #Dab2_P[string_P] +=  np.einsum('ijac,ijbc->ab', ccpert[string_P].y2, ccpert[string_P].y2)

        Dab2_P[string_P] =   np.einsum('ia,ib->ab', ccpert[string_P].x1, ccpert[string_P].x1)
        Dab2_P[string_P] +=  2.0 * np.einsum('ijac,ijbc->ab', ccpert[string_P].x2, ccpert[string_P].x2)
        Dab2_P[string_P] -=  np.einsum('ijac,ijcb->ab', ccpert[string_P].x2, ccpert[string_P].x2)


        # Virtual - Occupied block of Density #
        
        Dai_P[string_P] = ccpert[string_P].y1.swapaxes(0,1).copy()
        
        
        # Occupied - Virtual block of Density #
        
        # 1st term
        Dia_P[string_P] = 2.0 * ccpert[string_P].x1.copy()  
    
        # factor of 2.0 because of Y and L - derived using unitary group formalism
        
        # 2nd term
        
        Dia_P[string_P] +=  2.0 * np.einsum('imae,me->ia', ccsd.t2, ccpert[string_P].y1)
        Dia_P[string_P] += -1.0 * np.einsum('imea,me->ia', ccsd.t2, ccpert[string_P].y1)
        
        Dia_P[string_P] +=  2.0 * np.einsum('imae,me->ia', ccpert[string_P].x2, cclambda.l1)
        Dia_P[string_P] += -1.0 * np.einsum('imea,me->ia', ccpert[string_P].x2, cclambda.l1)
    
        # 3rd term
    
        Dia_P[string_P] += -1.0 * np.einsum('ie,ma,me->ia', ccsd.t1, ccsd.t1, ccpert[string_P].y1)
        Dia_P[string_P] += -1.0 * np.einsum('ie,ma,me->ia', ccsd.t1, ccpert[string_P].x1, cclambda.l1)
        Dia_P[string_P] += -1.0 * np.einsum('ie,ma,me->ia', ccpert[string_P].x1, ccsd.t1, cclambda.l1)
        
        # 4_th term
        
        Dia_P[string_P] += -1.0 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccsd.t2, ccpert[string_P].x1)
        Dia_P[string_P] += -1.0 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccpert[string_P].x2, ccsd.t1)
        Dia_P[string_P] += -1.0 * np.einsum('mnef,inef,ma->ia', ccpert[string_P].y2, ccsd.t2, ccsd.t1)
        
        # 5th term
        
        Dia_P[string_P] += -1.0 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccsd.t2, ccpert[string_P].x1)
        Dia_P[string_P] += -1.0 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccpert[string_P].x2, ccsd.t1)
        Dia_P[string_P] += -1.0 * np.einsum('mnef,mnaf,ie->ia', ccpert[string_P].y2, ccsd.t2, ccsd.t1)

        ########################################################################    

        #### Angmom density now  ###############################################

        ########################################################################    

        Dij_L[string_L]  =  np.einsum('ia,ja->ij', ccpert[string_L].x1, cclambda.l1)
        Dij_L[string_L] +=  np.einsum('ia,ja->ij', ccsd.t1, ccpert[string_L].y1)
        Dij_L[string_L] +=  np.einsum('ikab,jkab->ij', ccsd.t2, ccpert[string_L].y2)
        Dij_L[string_L] +=  np.einsum('ikab,jkab->ij', ccpert[string_L].x2, cclambda.l2)
        Dij_L[string_L] = -1.0 * Dij_L[string_L]


        #Dij2_L[string_L] =  np.einsum('ia,ja->ij', ccpert[string_L].x1, ccpert[string_L].y1)
        #Dij2_L[string_L] += np.einsum('ikab,jkab->ij', ccpert[string_L].x2, ccpert[string_L].y2)
        #Dij2_L[string_L] =  -1.0 * Dij2_L[string_L] 

        #Dij2_L[string_L] =  np.einsum('ia,ja->ij', ccpert[string_L].y1, ccpert[string_L].y1)
        #Dij2_L[string_L] += np.einsum('ikab,jkab->ij', ccpert[string_L].y2, ccpert[string_L].y2)
        #Dij2_L[string_L] =  -1.0 * Dij2_L[string_L] 

        Dij2_L[string_L] =  np.einsum('ia,ja->ij', ccpert[string_L].x1, ccpert[string_L].x1)
        Dij2_L[string_L] += 2.0 * np.einsum('ikab,jkab->ij', ccpert[string_L].x2, ccpert[string_L].x2)
        Dij2_L[string_L] -= np.einsum('ikab,kjab->ij', ccpert[string_L].x2, ccpert[string_L].x2)
        Dij2_L[string_L] =  -1.0 * Dij2_L[string_L] 


        #print(Dij2_L[string_L])

        # Virtual - Virtual block of Density #

        # 'ia,ib->ba': ba is absolutley essential instead of ab because angular momentum
        # integrals are anti-hermitian because of which Lab*Dab = - Lab*Dba

        Dab_L[string_L] =   np.einsum('ia,ib->ba', ccpert[string_L].x1, cclambda.l1)
        Dab_L[string_L] +=  np.einsum('ia,ib->ba', ccsd.t1, ccpert[string_L].y1)
        Dab_L[string_L] +=  np.einsum('ijac,ijbc->ba', ccsd.t2, ccpert[string_L].y2)
        Dab_L[string_L] +=  np.einsum('ijac,ijbc->ba', ccpert[string_L].x2, cclambda.l2)


        #Dab2_L[string_L] =   np.einsum('ia,ib->ab', ccpert[string_L].x1, ccpert[string_L].y1)
        #Dab2_L[string_L] +=  np.einsum('ijac,ijbc->ab', ccpert[string_L].x2, ccpert[string_L].y2)

        #Dab2_L[string_L] =   np.einsum('ia,ib->ab', ccpert[string_L].y1, ccpert[string_L].y1)
        #Dab2_L[string_L] +=  np.einsum('ijac,ijbc->ab', ccpert[string_L].y2, ccpert[string_L].y2)

        Dab2_L[string_L] =   np.einsum('ia,ib->ab', ccpert[string_L].x1, ccpert[string_L].x1)
        Dab2_L[string_L] +=  2.0 * np.einsum('ijac,ijbc->ab', ccpert[string_L].x2, ccpert[string_L].x2)
        Dab2_L[string_L] -=  np.einsum('ijac,ijcb->ab', ccpert[string_L].x2, ccpert[string_L].x2)

        #print(Dab2_L[string_L])

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

        # second order density based on MU*L, this is one of the 2 approaches 
        # the 2nd approach is the joint diagonalization of second order densities based
        # on MU*MU and L*L.  

        #Dij2[string_P+string_L] =  np.einsum('ia,ja->ij', ccpert[string_P].x1, ccpert[string_L].x1)
        #Dij2[string_P+string_L] += 2.0 * np.einsum('ikab,jkab->ij', ccpert[string_P].x2, ccpert[string_L].x2)
        #Dij2[string_P+string_L] -= np.einsum('ikab,kjab->ij', ccpert[string_P].x2, ccpert[string_L].x2)
        #Dij2[string_P+string_L] =  -1.0 * Dij2[string_P + string_L]

        #Dab2[string_P+string_L] =   np.einsum('ia,ib->ab', ccpert[string_P].x1, ccpert[string_L].x1)
        #Dab2[string_P+string_L] +=  2.0 * np.einsum('ijac,ijbc->ab', ccpert[string_P].x2, ccpert[string_L].x2)
        #Dab2[string_P+string_L] -=  np.einsum('ijac,ijcb->ab', ccpert[string_P].x2, ccpert[string_L].x2)
        """

        Dij[string_P+string_L + str(omega_zero)] =  0.5 * np.einsum('ia,ja->ij', ccpert[string_P + str(omega_zero)].x1, ccpert[string_L + str(omega_zero)].y1)
        Dij[string_P+string_L + str(omega_zero)] += 0.5 * np.einsum('ikab,jkab->ij', ccpert[string_P + str(omega_zero)].x2, ccpert[string_L + str(omega_zero)].y2)
        Dij[string_P+string_L + str(omega_zero)] =  -1.0 * Dij[string_P + string_L + str(omega_zero)]

        Dab[string_P+string_L + str(omega_zero)] =   0.5 * np.einsum('ia,ib->ab', ccpert[string_P + str(omega_zero)].x1, ccpert[string_L + str(omega_zero)].y1)
        Dab[string_P+string_L + str(omega_zero)] +=  0.5 * np.einsum('ijac,ijbc->ab', ccpert[string_P + str(omega_zero)].x2, ccpert[string_L + str(omega_zero)].y2)

        #x1 = ccpert[string_P + str(omega)].x1 + ccpert[string_P + str(omega_zero)].x1
        #x2 = ccpert[string_P + str(omega)].x2 + ccpert[string_P + str(omega_zero)].x2

        #y1 = ccpert[string_P + str(omega)].y1 - ccpert[string_P + str(omega_zero)].y1
        #y2 = ccpert[string_P + str(omega)].y2 - ccpert[string_P + str(omega_zero)].y2

        #Dij[string_P+string_L + str(omega_zero)] =  0.5 * np.einsum('ia,ja->ij', x1, y1)
        #Dij[string_P+string_L + str(omega_zero)] += 0.5 * np.einsum('ikab,jkab->ij', x2, y2)
        #Dij[string_P+string_L + str(omega_zero)] =  -1.0 * Dij[string_P + string_L + str(omega_zero)]

        #Dab[string_P+string_L + str(omega_zero)] =   0.5 * np.einsum('ia,ib->ab', x1, y1)
        #Dab[string_P+string_L + str(omega_zero)] +=  0.5 * np.einsum('ijac,ijbc->ab', x2, y2)

        #Dij[string_P+string_L + str(omega)] =  0.5 * np.einsum('ia,ja->ij', ccpert[string_P + str(omega)].x1, ccpert[string_L + str(omega)].y1)
        #Dij[string_P+string_L + str(omega)] += 0.5 * np.einsum('ikab,jkab->ij', ccpert[string_P + str(omega)].x2, ccpert[string_L + str(omega)].y2)
        #Dij[string_P+string_L + str(omega)] =  -1.0 * Dij[string_P + string_L + str(omega)]

        #Dab[string_P+string_L + str(omega)] =   0.5 * np.einsum('ia,ib->ab', ccpert[string_P + str(omega)].x1, ccpert[string_L + str(omega)].y1)
        #Dab[string_P+string_L + str(omega)] +=  0.5 * np.einsum('ijac,ijbc->ab', ccpert[string_P + str(omega)].x2, ccpert[string_L + str(omega)].y2)

    
    # calculate response function <<A;B>> by pert_A * density_B
    # Right now, these are only for diagonal elements.
    
    #print('\n Calculating Polarizability tensor from first order density approach:\n')
    
    optrot_density_P={};optrot_density_P_ij={};optrot_density_P_ab={};
    optrot_density_P_ia={};optrot_density_P_ai={};optrot_PQ_P={};

    optrot_density_L={};optrot_density_L_ij={};optrot_density_L_ab={};
    optrot_density_L_ia={};optrot_density_L_ai={};optrot_PQ_L={};
    
    for p in pert:
    
        string_P = "P_" + cart[p]
        string_L  = "L_"  + cart[p]


        if(MVG):
            Dij[string_P + string_L + str(omega_zero)] =  0.5 * (Dij[string_P + string_L + str(omega_zero)] + Dij[string_P + string_L + str(omega_zero)].T)
            Dab[string_P + string_L + str(omega_zero)] =  0.5 * (Dab[string_P + string_L + str(omega_zero)] + Dab[string_P + string_L + str(omega_zero)].T)
        
        #Dij[string_P + string_L + str(omega)] =  0.5 * (Dij[string_P + string_L + str(omega)] + Dij[string_P + string_L + str(omega)].T)
        #Dab[string_P + string_L + str(omega)] =  0.5 * (Dab[string_P + string_L + str(omega)] + Dab[string_P + string_L + str(omega)].T)

        """
        optrot_density_L[string_L] = 0
        optrot_density_L_ij[string_L] =  np.einsum('ij,ij->', Dij_L[string_L], Pij[string_P])
        optrot_density_L[string_L] +=  optrot_density_L_ij[string_L]
        print('\nPij * Dij_L: %20.15lf\n' % optrot_density_L_ij[string_L])

        optrot_density_L_ab[string_L] =  np.einsum('ab,ab->', Dab_L[string_L], Pab[string_P])
        optrot_density_L[string_L] +=  optrot_density_L_ab[string_L]
        print('\nPab * Dab_L: %20.15lf\n' % optrot_density_L_ab[string_L])

        optrot_density_L_ia[string_L] =  np.einsum('ia,ia->', Dia_L[string_L], Pia[string_P])
        optrot_density_L[string_L] +=  optrot_density_L_ia[string_L]
        print('\nPia * Dia_L: %20.15lf\n' % optrot_density_L_ia[string_L])

        optrot_density_L_ai[string_L] =  np.einsum('ai,ai->', Dai_L[string_L], Pai[string_P])
        optrot_density_L[string_L] +=  optrot_density_L_ai[string_L]
        print('\nPai * Dai_L: %20.15lf\n' % optrot_density_L_ai[string_L])

        print('\noptrot_density_P*D_L: %20.15lf\n' % optrot_density_L[string_L])

        optrot_density_P[string_P] = 0
        optrot_density_P_ij[string_P] =  np.einsum('ij,ij->', Dij_P[string_P], Lij[string_L])
        optrot_density_P[string_P] +=  optrot_density_P_ij[string_P]
        print('\nLij * Dij_P: %20.15lf\n' % optrot_density_P_ij[string_P])
   
        #print(Lab[string_L]) 
        #print(Dab_P[string_P]) 
        optrot_density_P_ab[string_P] =  np.einsum('ab,ab->', Dab_P[string_P], Lab[string_L])
        optrot_density_P[string_P] +=  optrot_density_P_ab[string_P]
        print('\nLab * Dab_P: %20.15lf\n' % optrot_density_P_ab[string_P])
    
        optrot_density_P_ia[string_P] =  np.einsum('ia,ia->', Dia_P[string_P], Lia[string_L])
        optrot_density_P[string_P] +=  optrot_density_P_ia[string_P]
        print('\nLia * Dia_P: %20.15lf\n' % optrot_density_P_ia[string_P])
    
        optrot_density_P_ai[string_P] =  np.einsum('ai,ai->', Dai_P[string_P], Lai[string_L])
        optrot_density_P[string_P] +=  optrot_density_P_ai[string_P]
        print('\nLai * Dai_P: %20.15lf\n' % optrot_density_P_ai[string_P])

        print('\noptrot_density_L*D_P: %20.15lf\n' % optrot_density_P[string_P])

        print('\noptrot_density: %20.15lf\n' % (0.50 * optrot_density_P[string_P] - 0.5 * optrot_density_L[string_L]))
        """
    
    print('\n Calculating rotation tensor from linear response function:\n')
    optrot_PQ={}
   
    for p in pert:
        str_p = "P_" + cart[p]
        for q in pert:
            if p == q:
                str_q = "L_" + cart[q]
                str_pq = "<<" + str_p + ";" + str_q + ">>"
                str_qp = "<<" + str_q + ";" + str_p + ">>"
                cclinresp[str_pq + str(omega)]= helper_cclinresp(cclambda, ccpert[str_p + str(omega)], ccpert[str_q + str(omega)])
                cclinresp[str_qp + str(omega)]= helper_cclinresp(cclambda, ccpert[str_q + str(omega)], ccpert[str_p + str(omega)])
                optrot_PQ[str_pq + str(omega)]= cclinresp[str_pq + str(omega)].linresp()
                optrot_PQ[str_qp + str(omega)]= cclinresp[str_qp + str(omega)].linresp()

                PL_omega = optrot_PQ[str_pq + str(omega)]
                LP_omega = optrot_PQ[str_qp + str(omega)]
                optrot_omega = 0.5 * (PL_omega + LP_omega)

                print('\n optrot_response P*L (%s) = %20.15lf \n' % (str(omega), PL_omega))
                print('\n optrot_response L*P (%s) = %20.15lf \n' % (str(omega), LP_omega))
                print('\n optrot_response velocity gauge (%s)= %20.15lf \n' % ( str(omega), optrot_omega))

                if(MVG):

                    cclinresp[str_pq + str(omega_zero)]= helper_cclinresp(cclambda, ccpert[str_p + str(omega)], ccpert[str_q + str(omega_zero)])
                    cclinresp[str_qp + str(omega_zero)]= helper_cclinresp(cclambda, ccpert[str_q + str(omega)], ccpert[str_p + str(omega_zero)])
                    optrot_PQ[str_pq + str(omega_zero)]= cclinresp[str_pq + str(omega_zero)].linresp()
                    optrot_PQ[str_qp + str(omega_zero)]= cclinresp[str_qp + str(omega_zero)].linresp()

                    PL_zero = optrot_PQ[str_pq + str(omega_zero)]
                    LP_zero = optrot_PQ[str_qp + str(omega_zero)]
                    optrot_zero = 0.5 * ( PL_zero + LP_zero)

                    print('\n optrot_response P*L (%s) = %20.15lf \n' % (str(omega_zero), PL_zero))
                    print('\n optrot_response L*P (%s) = %20.15lf \n' % (str(omega_zero), LP_zero))
                    print('\n optrot_response (%s)= %20.15lf \n' % ( str(omega_zero), optrot_zero))

                    print('\n optrot_response Modified velocity gauge (%s): %20.15lf\n' % (str(omega), optrot_omega - optrot_zero))

    # Just pick only the x component for now. We will extend this later.

    for p in pert:

        string_P = "P_" + cart[p]
        string_L =  "L_" + cart[p]
        
        Evec_ij, Emat_ij = LA.eig(Dij[string_P + string_L + str(omega_zero)])
        Evec_ab, Emat_ab = LA.eig(Dab[string_P + string_L + str(omega_zero)])

        #Evec_ij, Emat_ij = LA.eig(Dij[string_P + string_L + str(omega)])
        #Evec_ab, Emat_ab = LA.eig(Dab[string_P + string_L + str(omega)])

        Emat_ij =  sort(Emat_ij, Evec_ij)
        Emat_ab =  sort(Emat_ab, Evec_ab)

    return  Emat_ij, Emat_ab

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

# When I use np.asarray(), once I copy truncated C 
# into rhf_wfn.Ca(), C aslo changes as C is just a 
# view of rhf_wfn.Ca() in this case.
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

Emat_ij, Emat_ab = fvno_procedure(mol, rhf_e, rhf_wfn, 50, 'true')

#frz_vir = [i for i in range(5,9)]
#frz_vir = [15,30,45,60,65,70]
#frz_vir = [35,40,45,50,55]
frz_vir = [40,60,80,100,120]
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
    tmp_1, tmp_2 = fvno_procedure(mol, rhf_e, rhf_wfn, 50, 'true')
