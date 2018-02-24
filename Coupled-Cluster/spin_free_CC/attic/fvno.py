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

#psi4.set_options({'basis': 'cc-pVDZ'})
#psi4.set_options({'basis': 'sto-3g'})
psi4.set_options({'basis': 'ORP'})

# For numpy
compare_psi4 = True

print('Computing RHF reference.')
psi4.core.set_active_molecule(mol)
psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
psi4.set_module_options('SCF', {'E_CONVERGENCE':1e-13})
psi4.set_module_options('SCF', {'D_CONVERGENCE':1e-13})
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
    #omega = 0.07735713394560646
    omega = 0.01
    
    cart = {0:'X', 1: 'Y', 2: 'Z'}
    Mu={}; Muij={}; Muab={}; Muia={}; Muai={};
    Dij={}; Dij2={}; Dab={}; Dab2={}; Dia={}; Dai={}; 
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
  

        ## l1, l2 and y1, y2 have been calculated by the bi-orthogonal(BON) formalism as
        ## shown in the pink book: so 
        ## l1(BON) = 2.0 * l1, l2(BON) = 2.0 * (2.0 * lij_ab - lij_ba)  
        ## same for y1 and y2.
        ##
        ## This is the density in regular formalism
        ##
        ## The spin-orbital expressions for the onepdm components are:
        ##
        ## D_ij = -1/2 t_im^ef L^jm_ef - t_i^e L^j_e
        ##
        ## D_ab = 1/2 L^mn_ae t_mn^be + L^m_a t_m^b
        ##
        ## D_ia = t_i^a + (t_im^ae - t_i^e t_m^a) L^m_e
        ##        - 1/2 L^mn_ef (t_in^ef t_m^a + t_i^e t_mn^af)
        ##
        ## D_ai = L^i_a
        ##
        ## [cf. Gauss and Stanton, JCP 103, 3561-3577 (1995).]

    
        # Occupied - Occupied block of Density #
        
        Dij[string]  =  0.5 * np.einsum('ia,ja->ij', ccpert[string].x1, cclambda.l1)
        Dij[string] +=  0.5 * np.einsum('ia,ja->ij', ccsd.t1, ccpert[string].y1)
        Dij[string] +=  0.5 * np.einsum('ikab,jkab->ij', ccsd.t2, ccpert[string].y2)
        Dij[string] +=  0.5 * np.einsum('ikab,jkab->ij', ccpert[string].x2, cclambda.l2)
        Dij[string] =  -1.0 * Dij[string] 


        #Dij2[string] =  np.einsum('ia,ja->ij', ccpert[string].x1, ccpert[string].y1)
        #Dij2[string] += np.einsum('ikab,jkab->ij', ccpert[string].x2, ccpert[string].y2)
        #Dij2[string] =  -1.0 * Dij2[string] 
      
        Dij2[string] =  np.einsum('ia,ja->ij', ccpert[string].y1, ccpert[string].y1)
        Dij2[string] += np.einsum('ikab,jkab->ij', ccpert[string].y2, ccpert[string].y2)
        Dij2[string] =  -1.0 * Dij2[string] 

        #Dij2[string] =  np.einsum('ia,ja->ij', ccpert[string].x1, ccpert[string].x1)
        #Dij2[string] += 2.0 * np.einsum('ikab,jkab->ij', ccpert[string].x2, ccpert[string].x2)
        #Dij2[string] -= np.einsum('ikab,kjab->ij', ccpert[string].x2, ccpert[string].x2)
        #Dij2[string] =  -1.0 * Dij2[string] 

        # Virtual - Virtual block of Density #
        
        Dab[string] =   0.5 * np.einsum('ia,ib->ba', ccpert[string].x1, cclambda.l1)
        Dab[string] +=  0.5 * np.einsum('ia,ib->ba', ccsd.t1, ccpert[string].y1)
        Dab[string] +=  0.5 * np.einsum('ijac,ijbc->ba', ccsd.t2, ccpert[string].y2)
        Dab[string] +=  0.5 * np.einsum('ijac,ijbc->ba', ccpert[string].x2, cclambda.l2)
       
        print(Dab[string]) 
        
        #Dab2[string] =   np.einsum('ia,ib->ab', ccpert[string].x1, ccpert[string].y1)
        #Dab2[string] +=  np.einsum('ijac,ijbc->ab', ccpert[string].x2, ccpert[string].y2)

        Dab2[string] =   np.einsum('ia,ib->ab', ccpert[string].y1, ccpert[string].y1)
        Dab2[string] +=  np.einsum('ijac,ijbc->ab', ccpert[string].y2, ccpert[string].y2)

        #Dab2[string] =   np.einsum('ia,ib->ab', ccpert[string].x1, ccpert[string].x1)
        #Dab2[string] +=  2.0 * np.einsum('ijac,ijbc->ab', ccpert[string].x2, ccpert[string].x2)
        #Dab2[string] -=  np.einsum('ijac,ijcb->ab', ccpert[string].x2, ccpert[string].x2)
        # Virtual - Occupied block of Density #
        
        Dai[string] = 0.5 * ccpert[string].y1.swapaxes(0,1).copy()
        
        
        # Occupied - Virtual block of Density #
        
        # 1st term
        Dia[string] = ccpert[string].x1.copy()  
    
        # factor of 2.0 because of Y and L - derived using unitary group formalism
        
        # 2nd term
        
        Dia[string] +=  np.einsum('imae,me->ia', ccsd.t2, ccpert[string].y1)
        Dia[string] +=  -0.5 * np.einsum('imea,me->ia', ccsd.t2, ccpert[string].y1)
        
        Dia[string] +=  np.einsum('imae,me->ia', ccpert[string].x2, cclambda.l1)
        Dia[string] +=  -0.5 * np.einsum('imea,me->ia', ccpert[string].x2, cclambda.l1)
    
        # 3rd term
    
        Dia[string] += -0.5 * np.einsum('ie,ma,me->ia', ccsd.t1, ccsd.t1, ccpert[string].y1)
        Dia[string] += -0.5 * np.einsum('ie,ma,me->ia', ccsd.t1, ccpert[string].x1, cclambda.l1)
        Dia[string] += -0.5 * np.einsum('ie,ma,me->ia', ccpert[string].x1, ccsd.t1, cclambda.l1)
        
        # 4th term
        
        Dia[string] += -0.5 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccsd.t2, ccpert[string].x1)
        Dia[string] += -0.5 * np.einsum('mnef,inef,ma->ia', cclambda.l2, ccpert[string].x2, ccsd.t1)
        Dia[string] += -0.5 * np.einsum('mnef,inef,ma->ia', ccpert[string].y2, ccsd.t2, ccsd.t1)
        
        # 5th term
        
        Dia[string] += -0.5 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccsd.t2, ccpert[string].x1)
        Dia[string] += -0.5 * np.einsum('mnef,mnaf,ie->ia', cclambda.l2, ccpert[string].x2, ccsd.t1)
        Dia[string] += -0.5 * np.einsum('mnef,mnaf,ie->ia', ccpert[string].y2, ccsd.t2, ccsd.t1)
    
    
    
    # calculate response function <<A;B>> by pert_A * density_B
    # Right now, these are only for diagonal elements.
    
    #print('\n Calculating Polarizability tensor from first order density approach:\n')
    
    polar_density={};polar_density_ij={};polar_density_ab={};
    polar_density_ia={};polar_density_ai={};polar_PQ={};
    
    for p in range(0,1):
    
        string = "MU_" + cart[p]
        Dij[string]  =  0.5 * (Dij[string] + Dij[string].T)
        Dab[string]  =  0.5 * (Dab[string] + Dab[string].T)
        Dij2[string] =  0.5 * (Dij2[string] + Dij2[string].T)
        Dab2[string] =  0.5 * (Dab2[string] + Dab2[string].T)

        polar_density[string] = 0
        #print(string)
        polar_density_ij[string] =  2.0 * np.einsum('ij,ij->', Dij[string], Muij[string])
        polar_density[string] +=  polar_density_ij[string]
        #print('\nDij: %20.15lf\n' % polar_density_ij[string])
    
        polar_density_ab[string] =  2.0 * np.einsum('ab,ab->', Dab[string], Muab[string])
        polar_density[string] +=  polar_density_ab[string]
        #print('\nDab: %20.15lf\n' % polar_density_ab[string])
    
        polar_density_ia[string] =  2.0 * np.einsum('ia,ia->', Dia[string], Muia[string])
        polar_density[string] +=  polar_density_ia[string]
        print('\nDia: %20.15lf\n' % polar_density_ia[string])
    
        polar_density_ai[string] =  2.0 * np.einsum('ai,ai->', Dai[string], Muai[string])
        polar_density[string] +=  polar_density_ai[string]
        print('\nDai: %20.15lf\n' % polar_density_ai[string])
        print('\npolar_density: %20.15lf\n' % (-1.0 * polar_density[string]))
    
    
    #print('\n Calculating Polarizability tensor from linear response function:\n')
    
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
    
    #Evecij,  Ematij =  LA.eig(Dij[string])
    #Evecab,  Ematab =  LA.eig(Dab[string])
    Evec2ij, Emat2ij = LA.eig(Dij2[string])
    Evec2ab, Emat2ab = LA.eig(Dab2[string])
    #
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
    #muDij[string] = Muij[string] * Dij[string]
    #muDab[string] = Muab[string] * Dab[string]

    #Dij_gs  =  np.einsum('ia,ja->ij', ccsd.t1, cclambda.l1)
    #Dij_gs +=  np.einsum('ikab,jkab->ij', ccsd.t2, cclambda.l2)
    #Dij_gs =   -1.0 * Dij_gs

    #print(Dij[string])   
    #print(Dij[string].diagonal())   
    #print(Dij_gs.diagonal())   
 
    #DEvecij, DEmatij = LA.eig(muDij[string])
    #DEvecab, DEmatab = LA.eig(muDab[string])

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

    Emat2ij =  sort(Emat2ij, Evec2ij)
    Emat2ab =  sort(Emat2ab, Evec2ab)

    #return Ematij, Ematab, DEmatij, DEmatab
    return Emat2ij, Emat2ab


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
    

Emat2ij, Emat2ab = fvno_procedure(mol, rhf_e, rhf_wfn, 4)
C = np.asarray(rhf_wfn.Ca())
nmo = rhf_wfn.nmo()
occ = rhf_wfn.doccpi()[0]
vir = nmo - occ
C_occ = C[:, :occ] 
C_vir = C[:, occ:] 

"""C_occ_no = np.einsum('pi,ij->pj', C_occ, Emat2ij)
C_vir_no = np.einsum('pa,ab->pb', C_vir, Emat2ab)

mints = psi4.core.MintsHelper(rhf_wfn.basisset())

quadrupole_array = mints.ao_quadrupole()
quadrupole_array_no = [] 
quadrupole_array_can = [] 

C_np = np.concatenate((C_occ_no, C_vir_no), axis=1)


for p in range(0,6):
   quadrupole_array_no.append(np.einsum('uj,vi,uv', C_np, C_np, np.asarray(quadrupole_array[p])))
   quadrupole_array_can.append(np.einsum('uj,vi,uv', C, C, np.asarray(quadrupole_array[p])))

spatial_extent_mat_no = np.zeros_like(quadrupole_array_no[0])     
spatial_extent_mat_no = quadrupole_array_no[0] + quadrupole_array_no[3] + quadrupole_array_no[5]

spatial_extent_mat_can = np.zeros_like(quadrupole_array_can[0])     
spatial_extent_mat_can = quadrupole_array_can[0] + quadrupole_array_can[3] + quadrupole_array_can[5]

spatial_extent_no =  np.zeros(nmo)
spatial_extent_can = np.zeros(nmo)

spatial_extent_no =  np.diag(spatial_extent_mat_no)
spatial_extent_can = np.diag(spatial_extent_mat_can)

#for items in spatial_extent:
#    print(items)

virtuals = [i for i in range(vir)] 
se_vir_no = spatial_extent_no[occ:]
se_vir_can = spatial_extent_can[occ:]
import matplotlib.pyplot as plt 
plt.plot(virtuals, se_vir_no, se_vir_can)
plt.savefig("se_can_no.png")
"""

"""
frz_vir = 21
Emat2ab1 = np.zeros_like(Emat2ab)
for k in range(20, frz_vir):
    Emat2ab1 = Emat2ab.copy()
    Emat2ab_view = Emat2ab1[:,vir-k:] 
    Emat2ab_view.fill(0)
    #print(Emat2ab1)    
    #print(Emat2ab)    
    # need to semi-canonicalize for faster convergence

    C_occ_no = np.einsum('pi,ij->pj', C_occ, Emat2ij)
    C_vir_no = np.einsum('pa,ab->pb', C_vir, Emat2ab1)
    #print('\n occupied block of Coefficients matrix\n')
    #print(C_occ)
    #print('\n virtual block of Coefficients matrix\n')
    #print(C_vir)
    C_np = np.concatenate((C_occ_no, C_vir_no), axis=1)
    C_psi4 = psi4.core.Matrix.from_array(C_np)
    rhf_wfn.Ca().copy(C_psi4)
    psi4.core.Matrix.print_out(rhf_wfn.Ca())
    tmp_1, tmp_2 = fvno_procedure(mol, rhf_e, rhf_wfn, 4)
# Now I want to change the coefficient matrix appropriately#
"""


