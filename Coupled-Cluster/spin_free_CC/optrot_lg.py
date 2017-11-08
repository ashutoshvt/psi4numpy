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
c_square = 137 * 137 

def fvno_procedure(mol, rhf_e, rhf_wfn, memory):

    # Compute CCSD
    ccsd = helper_ccenergy(mol, rhf_e, rhf_wfn, memory)
    ccsd.compute_energy(r_conv=1e-10)
    CCSDcorr_E = ccsd.ccsd_corr_e
    CCSD_E = ccsd.ccsd_e
    
    print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
    print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
    print('Total CCSD energy:                      % 16.15f' % CCSD_E)
    
    cchbar = helper_cchbar(ccsd)
    
    cclambda = helper_cclambda(ccsd,cchbar)
    cclambda.compute_lambda(r_conv=1e-10)
    omega = 0.07735713394560646
    #omega = 0.01
    
    cart = {0:'X', 1: 'Y', 2: 'Z'}
    Mu={}; Muij={}; Muab={}; Muia={}; Muai={};
    L={}; Lij={}; Lab={}; Lia={}; Lai={};
    Dij_Mu={}; Dij2_Mu={}; Dab_Mu={}; Dab2_Mu={}; Dia_Mu={}; Dai_Mu={}; 
    Dij_L={}; Dij2_L={}; Dab_L={}; Dab2_L={}; Dia_L={}; Dai_L={}; 
    Dij2={}; Dab2={}
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
        ccpert[string_Mu].solve('right', r_conv=1e-10)
        print('\nsolving right hand perturbed amplitudes for %s\n' % string_L)
        ccpert[string_L].solve('right', r_conv=1e-10)
        print('\nsolving left hand perturbed amplitudes for %s\n'% string_Mu)
        ccpert[string_Mu].solve('left', r_conv=1e-10)
        print('\nsolving left hand perturbed amplitudes for %s\n'% string_L)
        ccpert[string_L].solve('left', r_conv=1e-10)
    
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

        #print("\nDifference between occupied-virtual blocks of dipole and angmom integrals\n")
        #diff1 = abs(Muia[string_Mu]) - abs(Lia[string_L]) 
        #print(diff1)
        #print("\nDifference between virtual-occupied  blocks of dipole and angmom integrals\n")
        #diff2 = abs(Muai[string_Mu]) - abs(Lai[string_L]) 
        #print(diff2)


        """
        import matplotlib.pyplot as plt 
        plt.subplot(211)
        plt.imshow(abs(Lai[string_L]), cmap=plt.cm.BuPu_r)
        plt.subplot(212)
        plt.imshow(abs(Muai[string_Mu]), cmap=plt.cm.BuPu_r)
        plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax)
        plt.savefig("Matrix.png")
        """

   
        # Occupied - Occupied block of Density #
        
        Dij_Mu[string_Mu]  =  np.einsum('ia,ja->ij', ccpert[string_Mu].x1, cclambda.l1)
        Dij_Mu[string_Mu] +=  np.einsum('ia,ja->ij', ccsd.t1, ccpert[string_Mu].y1)
        Dij_Mu[string_Mu] +=  np.einsum('ikab,jkab->ij', ccsd.t2, ccpert[string_Mu].y2)
        Dij_Mu[string_Mu] +=  np.einsum('ikab,jkab->ij', ccpert[string_Mu].x2, cclambda.l2)
        Dij_Mu[string_Mu] = -1.0 * Dij_Mu[string_Mu] 


        #Dij2_Mu[string_Mu] =  np.einsum('ia,ja->ij', ccpert[string_Mu].x1, ccpert[string_Mu].y1)
        #Dij2_Mu[string_Mu] += np.einsum('ikab,jkab->ij', ccpert[string_Mu].x2, ccpert[string_Mu].y2)
        #Dij2_Mu[string_Mu] =  -1.0 * Dij2_Mu[string_Mu] 
      
        #Dij2_Mu[string_Mu] =  np.einsum('ia,ja->ij', ccpert[string_Mu].y1, ccpert[string_Mu].y1)
        #Dij2_Mu[string_Mu] += np.einsum('ikab,jkab->ij', ccpert[string_Mu].y2, ccpert[string_Mu].y2)
        #Dij2_Mu[string_Mu] =  -1.0 * Dij2_Mu[string_Mu] 

        Dij2_Mu[string_Mu] =  np.einsum('ia,ja->ij', ccpert[string_Mu].x1, ccpert[string_Mu].x1)
        Dij2_Mu[string_Mu] += 2.0 * np.einsum('ikab,jkab->ij', ccpert[string_Mu].x2, ccpert[string_Mu].x2)
        Dij2_Mu[string_Mu] -= np.einsum('ikab,kjab->ij', ccpert[string_Mu].x2, ccpert[string_Mu].x2)
        Dij2_Mu[string_Mu] =  -1.0 * Dij2_Mu[string_Mu] 

        # Virtual - Virtual block of Density #

        # 'ia,ib->ba': ba is absolutley essential instead of ab because angular momentum
        # integrals are anti-hermitian because of which Lab*Dab = - Lab*Dba
               
        Dab_Mu[string_Mu] =   np.einsum('ia,ib->ba', ccpert[string_Mu].x1, cclambda.l1)
        Dab_Mu[string_Mu] +=  np.einsum('ia,ib->ba', ccsd.t1, ccpert[string_Mu].y1)
        Dab_Mu[string_Mu] +=  np.einsum('ijac,ijbc->ba', ccsd.t2, ccpert[string_Mu].y2)
        Dab_Mu[string_Mu] +=  np.einsum('ijac,ijbc->ba', ccpert[string_Mu].x2, cclambda.l2)
        
        
        #Dab2_Mu[string_Mu] =   np.einsum('ia,ib->ab', ccpert[string_Mu].x1, ccpert[string_Mu].y1)
        #Dab2_Mu[string_Mu] +=  np.einsum('ijac,ijbc->ab', ccpert[string_Mu].x2, ccpert[string_Mu].y2)

        #Dab2_Mu[string_Mu] =   np.einsum('ia,ib->ab', ccpert[string_Mu].y1, ccpert[string_Mu].y1)
        #Dab2_Mu[string_Mu] +=  np.einsum('ijac,ijbc->ab', ccpert[string_Mu].y2, ccpert[string_Mu].y2)

        Dab2_Mu[string_Mu] =   np.einsum('ia,ib->ab', ccpert[string_Mu].x1, ccpert[string_Mu].x1)
        Dab2_Mu[string_Mu] +=  2.0 * np.einsum('ijac,ijbc->ab', ccpert[string_Mu].x2, ccpert[string_Mu].x2)
        Dab2_Mu[string_Mu] -=  np.einsum('ijac,ijcb->ab', ccpert[string_Mu].x2, ccpert[string_Mu].x2)


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

        #Dij2[string_Mu+string_L] =  np.einsum('ia,ja->ij', ccpert[string_Mu].x1, ccpert[string_L].x1)
        #Dij2[string_Mu+string_L] += 2.0 * np.einsum('ikab,jkab->ij', ccpert[string_Mu].x2, ccpert[string_L].x2)
        #Dij2[string_Mu+string_L] -= np.einsum('ikab,kjab->ij', ccpert[string_Mu].x2, ccpert[string_L].x2)
        #Dij2[string_Mu+string_L] =  -1.0 * Dij2[string_Mu + string_L]

        #Dab2[string_Mu+string_L] =   np.einsum('ia,ib->ab', ccpert[string_Mu].x1, ccpert[string_L].x1)
        #Dab2[string_Mu+string_L] +=  2.0 * np.einsum('ijac,ijbc->ab', ccpert[string_Mu].x2, ccpert[string_L].x2)
        #Dab2[string_Mu+string_L] -=  np.einsum('ijac,ijcb->ab', ccpert[string_Mu].x2, ccpert[string_L].x2)

        Dij2[string_Mu+string_L] =  0.5 * np.einsum('ia,ja->ij', ccpert[string_Mu].x1, ccpert[string_L].y1)
        Dij2[string_Mu+string_L] += 0.5 * np.einsum('ikab,jkab->ij', ccpert[string_Mu].x2, ccpert[string_L].y2)
        Dij2[string_Mu+string_L] =  -1.0 * Dij2[string_Mu + string_L]

        Dab2[string_Mu+string_L] =   0.5 * np.einsum('ia,ib->ab', ccpert[string_Mu].x1, ccpert[string_L].y1)
        Dab2[string_Mu+string_L] +=  0.5 * np.einsum('ijac,ijbc->ab', ccpert[string_Mu].x2, ccpert[string_L].y2)



    
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

        #Dij_Mu[string_Mu]  =  0.5 * (Dij_Mu[string_Mu] + Dij_Mu[string_Mu].T)
        #Dab_Mu[string_Mu]  =  0.5 * (Dab_Mu[string_Mu] + Dab_Mu[string_Mu].T)
        Dij2_Mu[string_Mu] =  0.5 * (Dij2_Mu[string_Mu] + Dij2_Mu[string_Mu].T)
        Dab2_Mu[string_Mu] =  0.5 * (Dab2_Mu[string_Mu] + Dab2_Mu[string_Mu].T)

        #Dij_L[string_L]  =  0.5 * (Dij_L[string_L] + Dij_L[string_L].T)
        #Dab_L[string_L]  =  0.5 * (Dab_L[string_L] + Dab_L[string_L].T)
        Dij2_L[string_L] =  0.5 * (Dij2_L[string_L] + Dij2_L[string_L].T)
        Dab2_L[string_L] =  0.5 * (Dab2_L[string_L] + Dab2_L[string_L].T)


        #Dij2[string_Mu + string_L]  =  1.0/(1.0 + c_square) * (Dij2_Mu[string_Mu] + c_square * Dij2_L[string_L])
        #Dab2[string_Mu + string_L]  =  1.0/(1.0 + c_square) * (Dab2_Mu[string_Mu] + c_square * Dab2_L[string_L])

        #Dij2[string_Mu + string_L]  =  1.0/(1.0 + c_square) * (c_square * Dij2_Mu[string_Mu] + Dij2_L[string_L])
        #Dab2[string_Mu + string_L]  =  1.0/(1.0 + c_square) * (c_square * Dab2_Mu[string_Mu] + Dab2_L[string_L])

        #Dij2[string_Mu + string_L] =  0.5 * (Dij2_Mu[string_Mu] + Dij2_L[string_L])
        #Dab2[string_Mu + string_L] =  0.5 * (Dab2_Mu[string_Mu] + Dab2_L[string_L])

        Dij2[string_Mu + string_L] =  0.5 * (Dij2[string_Mu + string_L] + Dij2[string_Mu + string_L].T)
        Dab2[string_Mu + string_L] =  0.5 * (Dab2[string_Mu + string_L] + Dab2[string_Mu + string_L].T)


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
   
        #print(Lab[string_L]) 
        #print(Dab_Mu[string_Mu]) 
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

        print('\noptrot_density: %20.15lf\n' % (0.50 * optrot_density_Mu[string_Mu] - 0.5 * optrot_density_L[string_L]))

    
        print('\n Calculating rotation tensor from linear response function:\n')
        optrot_PQ={}
        
        for p in range(0,1):
            str_p = "MU_" + cart[p]
            for q in range(0,3):
                if p == q:
                    str_q = "L_" + cart[q]
                    str_pq = "<<" + str_p + ";" + str_q + ">>"
                    str_qp = "<<" + str_q + ";" + str_p + ">>"
                    cclinresp[str_pq]= helper_cclinresp(cclambda, ccpert[str_p], ccpert[str_q])
                    cclinresp[str_qp]= helper_cclinresp(cclambda, ccpert[str_q], ccpert[str_p])
                    optrot_PQ[str_pq]= cclinresp[str_pq].linresp()
                    optrot_PQ[str_qp]= cclinresp[str_qp].linresp()
                    #print(str_pq)
                    #print('\n optrot1 = %20.15lf \n' % cclinresp[str_pq].optrot1)
                    #print('\n optrot2 = %20.15lf \n' % cclinresp[str_pq].optrot2)
                    print('\n optrot_response muL = %20.15lf \n' % optrot_PQ[str_pq])
                    print('\n optrot_response Lmu= %20.15lf \n' %  optrot_PQ[str_qp])
                    print('\n optrot_response = %20.15lf \n' % ( 0.5 * optrot_PQ[str_pq] - 0.5 * optrot_PQ[str_qp]))
    
    #print('\nPolarizability tensor (symmetrized):\n')
    #
    #for a in range(0,1):
    #    str_a = "MU_" + cart[a]
    #    for b in range(0,1):
    #        str_b = "MU_" + cart[b]
    #        str_ab = "<<" + str_a + ";" + str_b + ">>"
    #        #str_ba = "<<" + str_b + ";" + str_a + ">>"
    #        value = optrot_AB[str_ab]
    #        #value = 0.5*(optrot_AB[str_ab] + optrot_AB[str_ba])
    #        optrot_AB[str_ab] = value
    #        #optrot_AB[str_ba] = value
    #        print(str_ab + ":" + str(value))
    
    
    
    
    # Post-Processing of the densities to obtained `perturbed` natural orbitals
    # Just pick only the x component for now. We will extend this later.
    
    for p in range(0,1):

        string_Mu = "MU_" + cart[p]
        string_L =  "L_" + cart[p]
        
        #Evecij,  Ematij =  LA.eig(Dij[string_Mu])
        #Evecab,  Ematab =  LA.eig(Dab[string_Mu])

        Evec2_Mu_ij, Emat2_Mu_ij = LA.eig(Dij2_Mu[string_Mu])
        Evec2_Mu_ab, Emat2_Mu_ab = LA.eig(Dab2_Mu[string_Mu])

        Evec2_L_ij, Emat2_L_ij = LA.eig(Dij2_L[string_L])
        Evec2_L_ab, Emat2_L_ab = LA.eig(Dab2_L[string_L])

        Evec2_ij, Emat2_ij = LA.eig(Dij2[string_Mu + string_L])
        Evec2_ab, Emat2_ab = LA.eig(Dab2[string_Mu + string_L])

        #print('\n Printing eigenvalues of occupied-occupied block of first order CCSD density\n')
        #for item in Evecij:
        #    print(item)
        #
        #print('\n Printing eigenvalues of virtual-virtual block of first order CCSD density\n')
        #for item in Evecab:
        #    print(item)
        
        #print('\n Printing eigenvalues of occupied-occupied block of leading contribution to second order CCSD density\n')
        #for item in Evec2ij:
        #    print(item)

        #print('\n Printing eigenvalues of virtual-virtual block of leading contribution to second order CCSD density\n')
        #for item in Evec2ab:
        #    print(item)

        # create mu * densities now and compare their eigenvalues
        #muDij = {}; muDab = {};
        #muDij[string_Mu] = Muij[string_Mu] * Dij[string_Mu]
        #muDab[string_Mu] = Muab[string_Mu] * Dab[string_Mu]

        #Dij_gs  =  np.einsum('ia,ja->ij', ccsd.t1, cclambda.l1)
        #Dij_gs +=  np.einsum('ikab,jkab->ij', ccsd.t2, cclambda.l2)
        #Dij_gs =   -1.0 * Dij_gs

        #print(Dij[string_Mu])   
        #print(Dij[string_Mu].diagonal())   
        #print(Dij_gs.diagonal())   
 
        #DEvecij, DEmatij = LA.eig(muDij[string_Mu])
        #DEvecab, DEmatab = LA.eig(muDab[string_Mu])

        #print('\n Printing eigenvalues of occupied-occupied block of mu * first order CCSD density\n')
        #for item in DEvecij:
        #    print(item)
        
        #print('\n Printing eigenvalues of virtual-virtual block of mu * first order CCSD density\n')
        #for item in DEvecab:
        #    print(item)

        #Ematij  =  sort(Ematij, Evecij)
        #Ematab  =  sort(Ematab, Evecab)
        #DEmatij =  sort(DEmatij, DEvecij)
        #DEmatab =  sort(DEmatab, DEvecab)

        Emat2_Mu_ij =  sort(Emat2_Mu_ij, Evec2_Mu_ij)
        Emat2_Mu_ab =  sort(Emat2_Mu_ab, Evec2_Mu_ab)

        Emat2_L_ij =  sort(Emat2_L_ij, Evec2_L_ij)
        Emat2_L_ab =  sort(Emat2_L_ab, Evec2_L_ab)

        Emat2_ij =  sort(Emat2_ij, Evec2_ij)
        Emat2_ab =  sort(Emat2_ab, Evec2_ab)

    return Emat2_Mu_ij, Emat2_Mu_ab, Emat2_L_ij, Emat2_L_ab, Emat2_ij, Emat2_ab


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
    

#Emat2ij, Emat2ab = fvno_procedure(mol, rhf_e, rhf_wfn, 4)
Emat2_Mu_ij, Emat2_Mu_ab, Emat2_L_ij, Emat2_L_ab, Emat2_ij, Emat2_ab = fvno_procedure(mol, rhf_e, rhf_wfn, 4)


C = np.asarray(rhf_wfn.Ca())
F = np.asarray(rhf_wfn.Fa())
S = np.asarray(rhf_wfn.S())
nmo = rhf_wfn.nmo()
occ = rhf_wfn.doccpi()[0]
vir = nmo - occ
C_occ = C[:, :occ] 
C_vir = C[:, occ:] 
F_mo  = np.einsum('ui,vj,uv', C, C, F)
F_mo_occ = F_mo[:occ,:occ]
F_mo_vir = F_mo[occ:, occ:]

"""
C_occ_Mu_no = np.einsum('pi,ij->pj', C_occ, Emat2_Mu_ij)
C_vir_Mu_no = np.einsum('pa,ab->pb', C_vir, Emat2_Mu_ab)

C_occ_L_no = np.einsum('pi,ij->pj', C_occ, Emat2_L_ij)
C_vir_L_no = np.einsum('pa,ab->pb', C_vir, Emat2_L_ab)

C_np_Mu = np.concatenate((C_occ_Mu_no, C_vir_Mu_no), axis=1)
C_np_L = np.concatenate((C_occ_L_no, C_vir_L_no), axis=1)

Mu = np.einsum('uj,vi,uv', C_np_Mu, C_np_Mu, S)  
L= np.einsum('uj,vi,uv', C_np_L, C_np_L, S)  

#sym_Mu_L = 0.5 * (Mu_L + L_Mu)
#Mu_L_ev, Mu_L_mat = LA.eig(sym_Mu_L)

# imaginary eigenvalues #
#Mu_ev, Mu_mat = LA.eig(Mu)
#L_ev, L_mat = LA.eig(L)

print("\nEigen-values of symmetric Mu * L overlap\n")
#for item in Mu_L_ev:
#    print(item)
#print(Mu_L_ev)
#print("\nEigen-values of L * Mu overlap\n")
print(np.diag(Mu))
print(np.diag(L))

"""

"""
C_occ_Mu_no = np.einsum('pi,ij->pj', C_occ, Emat2_Mu_ij)
C_vir_Mu_no = np.einsum('pa,ab->pb', C_vir, Emat2_Mu_ab)

C_occ_L_no = np.einsum('pi,ij->pj', C_occ, Emat2_L_ij)
C_vir_L_no = np.einsum('pa,ab->pb', C_vir, Emat2_L_ab)

mints = psi4.core.MintsHelper(rhf_wfn.basisset())

quadrupole_array = mints.ao_quadrupole()
quadrupole_array_Mu = [] 
quadrupole_array_L = [] 

C_np_Mu = np.concatenate((C_occ_Mu_no, C_vir_Mu_no), axis=1)
C_np_L = np.concatenate((C_occ_L_no, C_vir_L_no), axis=1)


for p in range(0,6):
   quadrupole_array_Mu.append(np.einsum('uj,vi,uv', C_np_Mu, C_np_Mu, np.asarray(quadrupole_array[p])))
   quadrupole_array_L.append(np.einsum('uj,vi,uv', C_np_L, C_np_L, np.asarray(quadrupole_array[p])))

spatial_extent_mat_Mu = np.zeros_like(quadrupole_array_Mu[0])     
spatial_extent_mat_Mu = quadrupole_array_Mu[0] + quadrupole_array_Mu[3] + quadrupole_array_Mu[5]

spatial_extent_mat_L = np.zeros_like(quadrupole_array_L[0])     
spatial_extent_mat_L = quadrupole_array_L[0] + quadrupole_array_L[3] + quadrupole_array_L[5]

spatial_extent_Mu =  np.zeros(nmo)
spatial_extent_L = np.zeros(nmo)

spatial_extent_Mu =  np.diag(spatial_extent_mat_Mu)
spatial_extent_L = np.diag(spatial_extent_mat_L)

#for items in spatial_extent:
#    print(items)

virtuals = [i for i in range(vir)] 
se_vir_Mu = spatial_extent_Mu[occ:]
se_vir_L = spatial_extent_L[occ:]
import matplotlib.pyplot as plt 
plt.plot(virtuals, se_vir_Mu, se_vir_L)
plt.savefig("se_Mu_L.png")
"""
frz_vir = 6
Emat2_Mu_ab1 = np.zeros_like(Emat2_Mu_ab)
Emat2_L_ab1 = np.zeros_like(Emat2_L_ab)
Emat2_ab1 = np.zeros_like(Emat2_ab)
for k in range(5, frz_vir):

    #Emat2_Mu_ab1 = Emat2_Mu_ab.copy()
    #Emat2_Mu_ab_view = Emat2_Mu_ab1[:,vir-k:] 
    #Emat2_Mu_ab_view.fill(0)

    #Emat2_L_ab1 = Emat2_L_ab.copy()
    #Emat2_L_ab_view = Emat2_L_ab1[:,vir-k:] 
    #Emat2_L_ab_view.fill(0)

    Emat2_ab1 = Emat2_ab.copy()
    Emat2_view = Emat2_ab1[:,vir-k:] 
    Emat2_view.fill(0)

    #C_occ_Mu_no = np.einsum('pi,ij->pj', C_occ, Emat2_Mu_ij)
    #C_vir_Mu_no = np.einsum('pa,ab->pb', C_vir, Emat2_Mu_ab1)

    #C_occ_L_no = np.einsum('pi,ij->pj', C_occ, Emat2_L_ij)
    #C_vir_L_no = np.einsum('pa,ab->pb', C_vir, Emat2_L_ab1)

    C_occ_no = np.einsum('pi,ij->pj', C_occ, Emat2_ij)
    C_vir_no = np.einsum('pa,ab->pb', C_vir, Emat2_ab1)

    F_mo_occ  = np.einsum('ki,lj,kl', Emat2_ij, Emat2_ij, F_mo_occ)    
    F_mo_vir  = np.einsum('ca,db,cd', Emat2_ab1, Emat2_ab1, F_mo_vir)    

    tmp_occ_ev, tmp_occ_mat = LA.eig(F_mo_occ)
    tmp_vir_ev, tmp_vir_mat = LA.eig(F_mo_vir)

    C_occ_sc = np.einsum('pi,ij->pj', C_occ_no, tmp_occ_mat)
    C_vir_sc = np.einsum('pa,ab->pb', C_vir_no, tmp_vir_mat)

    F_occ_sc  = np.einsum('ui,vj,uv', C_occ_sc, C_occ_sc, F)
    F_vir_sc  = np.einsum('ua,vb,uv', C_vir_sc, C_vir_sc, F)

    #print(F_occ_sc)
    #print(F_vir_sc)

    

    #C_np_Mu = np.concatenate((C_occ_Mu_no, C_vir_Mu_no), axis=1)
    #C_np_L = np.concatenate((C_occ_L_no, C_vir_L_no), axis=1)
    C_np_sc = np.concatenate((C_occ_sc, C_vir_sc), axis=1)
    #C_np = np.concatenate((C_occ_no, C_vir_no), axis=1)

    #C_psi4_Mu = psi4.core.Matrix.from_array(C_np_Mu)
    #C_psi4_L = psi4.core.Matrix.from_array(C_np_L)
    C_psi4_sc = psi4.core.Matrix.from_array(C_np_sc)
    #C_psi4 = psi4.core.Matrix.from_array(C_np)

    #rhf_wfn.Ca().copy(C_psi4_Mu)
    #rhf_wfn.Ca().copy(C_psi4_L)
    rhf_wfn.Ca().copy(C_psi4_sc)
    #rhf_wfn.Ca().copy(C_psi4)

    tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6 = fvno_procedure(mol, rhf_e, rhf_wfn, 4)
# Now I want to change the coefficient matrix appropriately#
