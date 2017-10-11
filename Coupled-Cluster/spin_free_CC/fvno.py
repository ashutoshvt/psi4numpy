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

psi4.set_options({'basis': 'cc-pVDZ'})
#psi4.set_options({'basis': '6-31g'})

# For numpy
compare_psi4 = True

print('Computing RHF reference.')
psi4.core.set_active_molecule(mol)
psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
psi4.set_module_options('SCF', {'E_CONVERGENCE':10e-13})
psi4.set_module_options('SCF', {'D_CONVERGENCE':10e-13})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)  
print('RHF Final Energy                          % 16.10f\n' % rhf_e)

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
    
    cart = {0:'X', 1: 'Y', 2: 'Z'}
    Mu={}; Muij={}; Muab={}; Muia={}; Muai={};
    Dij={}; Dab={}; Dia={}; Dai={}; 
    ccpert = {}; polar_AB = {}; cclinresp = {};
    dipole_array = ccsd.mints.ao_dipole()
    
    
    for p in range(0,1):
        string = "MU_" + cart[p]
        Mu[string] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(dipole_array[p]))
        ccpert[string] = helper_ccpert(string, Mu[string], ccsd, cchbar, cclambda, omega)
        print('\nsolving right hand perturbed amplitudes for %s\n' % string)
        ccpert[string].solve('right', r_conv=1e-10)
        print('\nsolving left hand perturbed amplitudes for %s\n'% string)
        ccpert[string].solve('left', r_conv=1e-10)
    
    # Now that I have solved for x and y, I would like to calculate
    # first order ccsd perturbed density. I am assuming only diagonal
    # cases below
    
    
    
    for p in range(0,1):
    
        string = "MU_" + cart[p]
        #Mu[string] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(dipole_array[p]))
        Muij[string] = Mu[string][ccsd.slice_o, ccsd.slice_o]
        Muab[string] = Mu[string][ccsd.slice_v, ccsd.slice_v]
        Muia[string] = Mu[string][ccsd.slice_o, ccsd.slice_v]
        Muai[string] = Mu[string][ccsd.slice_v, ccsd.slice_o]
    
    
        # Occupied - Occupied block of Density #
        
        Dij[string]  =  np.einsum('ia,ja->ij', ccpert[string].x1, cclambda.l1)
        #Dij[string] +=  np.einsum('ia,ja->ij', ccsd.t1, ccpert[string].y1)
        #Dij[string] +=  np.einsum('ikab,jkab->ij', ccsd.t2, ccpert[string].y2)
        #Dij[string] +=  np.einsum('ikab,jkab->ij', ccpert[string].x2, cclambda.l2)
        Dij[string] = -1.0 * Dij[string] 
      
       
        print(ccpert[string].x1)
        print(cclambda.l1)
        value  =  np.einsum('ia,ia->i', ccpert[string].x1, cclambda.l1)
        print(value)
        
                  
        
        # Virtual - Virtual block of Density #
        
        Dab[string] =   np.einsum('ia,ib->ab', ccpert[string].x1, cclambda.l1)
        Dab[string] +=  np.einsum('ia,ib->ab', ccsd.t1, ccpert[string].y1)
        Dab[string] +=  np.einsum('ijac,ijbc->ab', ccsd.t2, ccpert[string].y2)
        Dab[string] +=  np.einsum('ijac,ijbc->ab', ccpert[string].x2, cclambda.l2)
        
        
        # Virtual - Occupied block of Density #
        
        Dai[string] = ccpert[string].y1.swapaxes(0,1).copy()
        
        
        # Occupied - Virtual block of Density #
        
        # 1st term
        Dia[string] = 2.0 * ccpert[string].x1.copy()  
        Dia[string] = 2.0 * ccpert[string].x1.copy()  
    
        # factor of 2.0 because of Y and L - derived using unitary group formalism
        
        # 2nd term
        
        Dia[string] +=  2.0 * np.einsum('imae,me->ia', ccsd.t2, ccpert[string].y1)
        Dia[string] += -1.0 * np.einsum('imea,me->ia', ccsd.t2, ccpert[string].y1)
        
        Dia[string] +=  2.0 * np.einsum('imae,me->ia', ccpert[string].x2, cclambda.l1)
        Dia[string] += -1.0 * np.einsum('imea,me->ia', ccpert[string].x2, cclambda.l1)
    
        # 3rd term
    
        Dia[string] += -1.0 * np.einsum('ie,ma,me->ia', ccsd.t1, ccsd.t1, ccpert[string].y1)
        Dia[string] += -1.0 * np.einsum('ie,ma,me->ia', ccsd.t1, ccpert[string].x1, cclambda.l1)
        Dia[string] += -1.0 * np.einsum('ie,ma,me->ia', ccpert[string].x1, ccsd.t1, cclambda.l1)
        
        # 4th term
        
        Dia[string] += -1.0 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccsd.t2, ccpert[string].x1)
        Dia[string] += -1.0 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccpert[string].x2, ccsd.t1)
        Dia[string] += -1.0 * np.einsum('mnef,inef,ma->ia', ccpert[string].y2, ccsd.t2, ccsd.t1)
        
        # 5th term
        
        Dia[string] += -1.0 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccsd.t2, ccpert[string].x1)
        Dia[string] += -1.0 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccpert[string].x2, ccsd.t1)
        Dia[string] += -1.0 * np.einsum('mnef,mnaf,ie->ia', ccpert[string].y2, ccsd.t2, ccsd.t1)
    
    
    
    # calculate response function <<A;B>> by pert_A * density_B
    # Right now, these are only for diagonal elements.
    
    print('\n Calculating Polarizability tensor from first order density approach:\n')
    
    polar_density={};polar_density_ij={};polar_density_ab={};
    polar_density_ia={};polar_density_ai={};polar_PQ={};
    
    for p in range(0,1):
    
        string = "MU_" + cart[p]
        Dij[string] = 0.5 * (Dij[string] + Dij[string].T)
        Dab[string] = 0.5 * (Dab[string] + Dab[string].T)
        polar_density[string] = 0
        print(string)
        polar_density_ij[string] =  np.einsum('ij,ij->', Dij[string], Muij[string])
        polar_density[string] +=  polar_density_ij[string]
        print('\nDij: %20.15lf\n' % polar_density_ij[string])
    
        polar_density_ab[string] =  np.einsum('ab,ab->', Dab[string], Muab[string])
        polar_density[string] +=  polar_density_ab[string]
        print('\nDab: %20.15lf\n' % polar_density_ab[string])
    
        polar_density_ia[string] =  np.einsum('ia,ia->', Dia[string], Muia[string])
        polar_density[string] +=  polar_density_ia[string]
        print('\nDia: %20.15lf\n' % polar_density_ia[string])
    
        polar_density_ai[string] =  np.einsum('ai,ai->', Dai[string], Muai[string])
        polar_density[string] +=  polar_density_ai[string]
        print('\nDai: %20.15lf\n' % polar_density_ai[string])
        print('\npolar_density: %20.15lf\n' % polar_density[string])
    
    
    print('\n Calculating Polarizability tensor from linear response function:\n')
    
    for p in range(0,1):
        str_p = "MU_" + cart[p]
        for q in range(0,1):
            if p == q:
                str_q = "MU_" + cart[q]
                str_pq = "<<" + str_p + ";" + str_q + ">>"
                cclinresp[str_pq]= helper_cclinresp(cclambda, ccpert[str_p], ccpert[str_q])
                polar_PQ[str_pq]= cclinresp[str_pq].linresp()
                print(str_pq)
                print('\n polar1 = %20.15lf \n' % cclinresp[str_pq].polar1)
                print('\n polar2 = %20.15lf \n' % cclinresp[str_pq].polar2)
                print('\n polar_response = %20.15lf \n' % polar_PQ[str_pq])
    
    #print('\nPolarizability tensor (symmetrized):\n')
    #
    #for a in range(0,1):
    #    str_a = "MU_" + cart[a]
    #    for b in range(0,1):
    #        str_b = "MU_" + cart[b]
    #        str_ab = "<<" + str_a + ";" + str_b + ">>"
    #        #str_ba = "<<" + str_b + ";" + str_a + ">>"
    #        value = polar_AB[str_ab]
    #        #value = 0.5*(polar_AB[str_ab] + polar_AB[str_ba])
    #        polar_AB[str_ab] = value
    #        #polar_AB[str_ba] = value
    #        print(str_ab + ":" + str(value))
    
    
    
    
    # Post-Processing of the densities to obtained `perturbed` natural orbitals
    # Just pick only the x component for now. We will extend this later.
    
    string = "MU_" + cart[0]
    #Dij[string] = 0.5 * (Dij[string] + Dij[string].T)
    #Dab[string] = 0.5 * (Dab[string] + Dab[string].T)
    
    Evecij, Ematij = LA.eig(Dij[string])
    Evecab, Ematab = LA.eig(Dab[string])
    #
    print('\n Printing eigenvalues of occupied-occupied block of first order CCSD density\n')
    for item in Evecij:
        print(item)
    
    print('\n Printing eigenvalues of virtual-virtual block of first order CCSD density\n')
    for item in Evecab:
        print(item)
    
    # create mu * densities now and compare their eigenvalues
    muDij = {}; muDab = {};
    muDij[string] = Muij[string] * Dij[string]
    muDab[string] = Muab[string] * Dab[string]

    Dij_gs  =  np.einsum('ia,ja->ij', ccsd.t1, cclambda.l1)
    Dij_gs +=  np.einsum('ikab,jkab->ij', ccsd.t2, cclambda.l2)
    Dij_gs =   -1.0 * Dij_gs


    
    print(Dij[string])   
    print(Dij[string].diagonal())   
    print(Dij_gs.diagonal())   
 
    DEvecij, DEmatij = LA.eig(muDij[string])
    DEvecab, DEmatab = LA.eig(muDab[string])
    
    print('\n Printing eigenvalues of occupied-occupied block of mu * first order CCSD density\n')
    for item in DEvecij:
        print(item)
    
    print('\n Printing eigenvalues of virtual-virtual block of mu * first order CCSD density\n')
    for item in DEvecab:
        print(item)

    Ematij  =  sort(Ematij, Evecij)
    Ematab  =  sort(Ematab, Evecab)
    DEmatij =  sort(DEmatij, DEvecij)
    DEmatab =  sort(DEmatab, DEvecab)

    return Ematij, Ematab, DEmatij, DEmatab


def sort(Emat, Evec):
    tmp = Evec.copy()
    tmp = abs(tmp)
    idx =  tmp.argsort()[::-1]   
    Evec = Evec[idx]
    
    print('\n Printing sorted eigenvalues of occupied-occupied block of first order CCSD density\n')
    
    for item in Evec:
        print(item)
    
    Emat = Emat[:,idx]
    return Emat
    

Ematij, Ematab, DEmatij, DEmatab = fvno_procedure(mol, rhf_e, rhf_wfn, 4)

# Now I want to change the coefficient matrix appropriately#

