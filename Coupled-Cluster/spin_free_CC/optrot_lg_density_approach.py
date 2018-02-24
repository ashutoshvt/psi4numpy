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

def form_first_order_densities(ccpert, pert, Mu, L, ccsd, cclambda, omega):

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

    return Dij, Dia, Dai, Dab
   
def calculate_optical_rotation(pert, Mu, L, omega): 

    print('\n Calculating optical rotation tensor from first order density approach:\n')

    Muij={}; Muab={}; Muia={}; Muai={};
    Lij={}; Lab={}; Lia={}; Lai={};
    optrot_density={};optrot_density_ij={};optrot_density_ab={};
    optrot_density_ia={};optrot_density_ai={};optrot_PQ={};
    
    for p in pert:
    
        string_Mu = "MU_" + cart[p]
        string_L  = "L_"  + cart[p]

        Muij[string_Mu] = Mu[string_Mu][ccsd.slice_o, ccsd.slice_o]
        Muab[string_Mu] = Mu[string_Mu][ccsd.slice_v, ccsd.slice_v]
        Muia[string_Mu] = Mu[string_Mu][ccsd.slice_o, ccsd.slice_v]
        Muai[string_Mu] = Mu[string_Mu][ccsd.slice_v, ccsd.slice_o]

        Lij[string_L] = L[string_L][ccsd.slice_o, ccsd.slice_o]
        Lab[string_L] = L[string_L][ccsd.slice_v, ccsd.slice_v]
        Lia[string_L] = L[string_L][ccsd.slice_o, ccsd.slice_v]
        Lai[string_L] = L[string_L][ccsd.slice_v, ccsd.slice_o]

        print("Lai")    
        print(Lai[string_L])    

        print("Muai")    
        print(Muai[string_Mu])    

        from numpy.linalg import norm
        print("Norm of Lai")
        print(norm(Lai[string_L]))

        print("Norm of Muai")
        print(norm(Muai[string_Mu]))

        optrot_density[string_L] = 0
        optrot_density_ij[string_L] =  np.einsum('ij,ij->', Dij[string_L], Muij[string_Mu])
        optrot_density[string_L] +=  optrot_density_ij[string_L]
        print('\nMuij * Dij_L: %20.15lf\n' % optrot_density_ij[string_L])

        optrot_density_ab[string_L] =  np.einsum('ab,ab->', Dab[string_L], Muab[string_Mu])
        optrot_density[string_L] +=  optrot_density_ab[string_L]
        print('\nMuab * Dab_L: %20.15lf\n' % optrot_density_ab[string_L])

        optrot_density_ia[string_L] =  np.einsum('ia,ia->', Dia[string_L], Muia[string_Mu])
        optrot_density[string_L] +=  optrot_density_ia[string_L]
        print('\nMuia * Dia_L: %20.15lf\n' % optrot_density_ia[string_L])

        optrot_density_ai[string_L] =  np.einsum('ai,ai->', Dai[string_L], Muai[string_Mu])
        optrot_density[string_L] +=  optrot_density_ai[string_L]
        print('\nMuai * Dai_L: %20.15lf\n' % optrot_density_ai[string_L])

        print('\noptrot_density_Mu*D_L: %20.15lf\n' % optrot_density[string_L])

        optrot_density[string_Mu] = 0
        optrot_density_ij[string_Mu] =  np.einsum('ij,ij->', Dij[string_Mu], Lij[string_L])
        optrot_density[string_Mu] +=  optrot_density_ij[string_Mu]
        print('\nLij * Dij_Mu: %20.15lf\n' % optrot_density_ij[string_Mu])
   
        optrot_density_ab[string_Mu] =  np.einsum('ab,ab->', Dab[string_Mu], Lab[string_L])
        optrot_density[string_Mu] +=  optrot_density_ab[string_Mu]
        print('\nLab * Dab_Mu: %20.15lf\n' % optrot_density_ab[string_Mu])
    
        optrot_density_ia[string_Mu] =  np.einsum('ia,ia->', Dia[string_Mu], Lia[string_L])
        optrot_density[string_Mu] +=  optrot_density_ia[string_Mu]
        print('\nLia * Dia_Mu: %20.15lf\n' % optrot_density_ia[string_Mu])
    
        optrot_density_ai[string_Mu] =  np.einsum('ai,ai->', Dai[string_Mu], Lai[string_L])
        optrot_density[string_Mu] +=  optrot_density_ai[string_Mu]
        print('\nLai * Dai_Mu: %20.15lf\n' % optrot_density_ai[string_Mu])

        print('\noptrot_density_L*D_Mu: %20.15lf\n' % optrot_density[string_Mu])

        print('\noptrot_density: %20.15lf\n' % (0.50 * optrot_density[string_Mu] - 0.5 * optrot_density[string_L]))
    return optrot_density_ij, optrot_density_ia, optrot_density_ai, optrot_density_ab, optrot_density


psi4.set_memory(int(3e9), False)
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry("""
#H
#H 1 R
#H 1 DIST 2 P
#H 3 R 1 P 2 T

#R = 0.75
#DIST = 10
#P = 90.0
#T = 60.0
H        0.000000000000  -5.00  -0.324759526419
H       -0.375000000000  -5.00   0.324759526419
nocom
symmetry C1
noreorient
""")

#mol = psi4.geometry("""
# O     -0.028962160801    -0.694396279686    -0.049338350190                                                                  
# O      0.028962160801     0.694396279686    -0.049338350190                                                                  
# H      0.350498145881    -0.910645626300     0.783035421467                                                                  
# H     -0.350498145881     0.910645626300     0.783035421467                                                                  
#noreorient
#symmetry c1        
#""")

#O
#H 1 1.1
#H 1 1.1 2 104
#symmetry c1
#""")

psi4.set_options({'basis': 'sto-3g'})

print('Computing RHF reference.')
psi4.core.set_active_molecule(mol)
psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
psi4.set_module_options('SCF', {'E_CONVERGENCE':1e-13})
psi4.set_module_options('SCF', {'D_CONVERGENCE':1e-13})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)  
print('RHF Final Energy                          % 16.10f\n' % rhf_e)

"""
# Try seeing localization effects
rhf_wfn.Ca().print_out()
npC = np.asarray(rhf_wfn.Ca())
C_occ = rhf_wfn.Ca_subset("AO", "OCC")
Local = psi4.core.Localizer.build("PIPEK_MEZEY", rhf_wfn.basisset(), C_occ)
Local.localize()
C_occ_L = np.asarray(Local.L)
docc = rhf_wfn.doccpi()[0]
npC[:,:docc] = C_occ_L
C = psi4.core.Matrix.from_array(npC)
rhf_wfn.Ca() == C
rhf_wfn.Ca().print_out()
"""
# Compute CCSD

local_type = "PAO"
local_cutoff = 0.02
local_weakp = "NONE"

ccsd = helper_ccenergy(mol, rhf_e, rhf_wfn, local_type, local_cutoff, local_weakp)
ccsd.compute_energy(r_conv=1e-10)
CCSDcorr_E = ccsd.ccsd_corr_e
CCSD_E = ccsd.ccsd_e
print("T2:")
my_t2 = ccsd.t2.copy()
print(my_t2)
print("T1:")
my_t1 = ccsd.t1.copy()
print(my_t1)


print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.15f' % CCSD_E)

cchbar = helper_cchbar(ccsd)

cclambda = helper_cclambda(ccsd, cchbar, local_type, local_cutoff, local_weakp)
cclambda.compute_lambda(r_conv=1e-10)
omega = 0.07735713394560646

print("L2:")
my_l2 = cclambda.l2.copy()
print(my_l2)
print("L1:")
my_l1 = cclambda.l1.copy()
print(my_l1)


cart = {0:'X', 1: 'Y', 2: 'Z'}
Mu={}; L={};
Dij={}; Dab={}; Dia={}; Dai={}; 
ccpert = {};
dipole_array = ccsd.mints.ao_dipole()
angmom_array = ccsd.mints.ao_angular_momentum()

dipole_array[0].print_out()
angmom_array[0].print_out()
rhf_wfn.S().print_out()
ccsd.wfn.Ca().print_out()

pert = [2]
opt_den={};opt_den_ij={};opt_den_ab={};
opt_den_ia={};opt_den_ai={};


for p in pert:
    string_Mu = "MU_" + cart[p]
    string_L = "L_" + cart[p]
    Mu[string_Mu] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(dipole_array[p]))
    L[string_L] = -0.5 * np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC, np.asarray(angmom_array[p]))
    print("Mu")
    print(Mu[string_Mu])
    print("L")
    print(L[string_L])
    ccpert[string_Mu] = helper_ccpert(string_Mu, Mu[string_Mu], ccsd, cchbar, cclambda, local_type, local_cutoff, local_weakp, omega)
    ccpert[string_L] = helper_ccpert(string_L, L[string_L], ccsd, cchbar, cclambda, local_type, local_cutoff, local_weakp, omega)
    print('\nsolving right hand perturbed amplitudes for %s\n' % string_Mu)
    ccpert[string_Mu].solve('right', r_conv=1e-10)
    print('\nsolving right hand perturbed amplitudes for %s\n' % string_L)
    ccpert[string_L].solve('right', r_conv=1e-10)
    print('\nsolving left hand perturbed amplitudes for %s\n'% string_Mu)
    ccpert[string_Mu].solve('left', r_conv=1e-10)
    print('\nsolving left hand perturbed amplitudes for %s\n'% string_L)
    ccpert[string_L].solve('left', r_conv=1e-10)
    mu_x2 = ccpert[string_Mu].x2.copy()
    mu_x1 = ccpert[string_Mu].x1.copy()
    mu_y2 = ccpert[string_Mu].y2.copy()
    mu_y1 = ccpert[string_Mu].y1.copy()
    print("MU_X2:")
    print(mu_x2)
    print("MU_X1:")
    print(mu_x1)
    print("MU_Y2:")
    print(mu_y2)
    print("MU_Y1:")
    print(mu_y1)
    l_x2 = ccpert[string_L].x2.copy()
    l_x1 = ccpert[string_L].x1.copy()
    l_y2 = ccpert[string_L].y2.copy()
    l_y1 = ccpert[string_L].y1.copy()
    print("L_X2:")
    print(l_x2)
    print("L_X1:")
    print(l_x1)
    print("L_Y2:")
    print(l_y2)
    print("L_Y1:")
    print(l_y1)

Dij, Dia, Dai, Dab = form_first_order_densities(ccpert, pert, Mu, L, ccsd, cclambda, omega)
print("Dij")
print(Dij)
print("Dia")
print(Dia)
print("Dai")
print(Dai)
print("Dab")
print(Dab)
opt_den_ij, opt_den_ia, opt_den_ai, opt_den_ab, opt_den = calculate_optical_rotation(pert, Mu, L, omega)
