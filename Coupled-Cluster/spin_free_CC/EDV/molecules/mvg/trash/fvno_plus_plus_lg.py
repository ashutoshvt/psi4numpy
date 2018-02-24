import time
import numpy as np
np.set_printoptions(precision=15, linewidth=200, suppress=True)
from numpy import linalg as LA
import sys
from helper_cc_exp import helper_ccenergy
from helper_cc_exp import helper_cchbar
from helper_cc_exp import helper_cclambda
from helper_cc_exp import helper_ccpert

from helper_cc_exp import helper_cclinresp
from helper_cc2_exp import helper_cc2energy
from helper_cc2_exp import helper_cc2hbar
from helper_cc2_exp import helper_cc2lambda
from helper_cc2_exp import helper_cc2pert
from helper_cc2_exp import helper_cc2linresp

import psi4

def sort(Evec, Eval):
    tmp = Eval.copy()
    tmp = abs(tmp)
    idx =  tmp.argsort()[::-1]   
    Eval = Eval[idx]
    print("Norm")
    print(np.linalg.norm(Eval))
    
    print('\n Printing sorted eigenvalues\n')
    
    for item in Eval:
        print(item)
    
    Evec = Evec[:,idx]
    return Evec

def guess_calculate(mol, rhf_e, rhf_wfn, corr_wfn, maxiter_cc, maxiter_lambda, maxiter_pert, memory, edv, filt_singles, density_type):

    # Compute CCSD
    if corr_wfn == 'CCSD':
        cc_wfn = helper_ccenergy(mol, rhf_e, rhf_wfn, edv, memory)
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
        cclambda = helper_cclambda(cc_wfn,cchbar,edv)
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

    pert = [0,1,2]    
    
    for p in pert:
        string_Mu = "MU_" + cart[p]
        string_L = "L_" + cart[p]
        Mu[string_Mu] = np.einsum('uj,vi,uv', cc_wfn.npC, cc_wfn.npC, np.asarray(dipole_array[p]))
        L[string_L] = -0.5 * np.einsum('uj,vi,uv', cc_wfn.npC, cc_wfn.npC, np.asarray(angmom_array[p]))
        if corr_wfn == 'CCSD':
            ccpert[string_Mu] = helper_ccpert(string_Mu, Mu[string_Mu], cc_wfn, cchbar, cclambda, omega, edv, filt_singles)
            ccpert[string_L] = helper_ccpert(string_L, L[string_L], cc_wfn, cchbar, cclambda, omega, edv, filt_singles)
        elif corr_wfn == 'CC2':
            ccpert[string_Mu] = helper_cc2pert(string_Mu, Mu[string_Mu], cc_wfn, cchbar, cclambda, omega, edv)
            ccpert[string_L] = helper_cc2pert(string_L, L[string_L], cc_wfn, cchbar, cclambda, omega, edv)

        # construct the density now
        Mubar_vo =    ccpert[string_Mu].build_Avo().swapaxes(0,1)
        Mubar_vo /=   ccpert[string_Mu].Dia_
        Mubar_vvoo =  ccpert[string_Mu].build_Avvoo().swapaxes(0,2).swapaxes(1,3)
        Mubar_vvoo += ccpert[string_Mu].build_Avvoo().swapaxes(0,3).swapaxes(1,2)
        Mubar_vvoo /=   ccpert[string_Mu].Dijab_

        Lbar_vo =    ccpert[string_L].build_Avo().swapaxes(0,1)
        Lbar_vo /=   ccpert[string_L].Dia_
        Lbar_vvoo =  ccpert[string_L].build_Avvoo().swapaxes(0,2).swapaxes(1,3)
        Lbar_vvoo += ccpert[string_L].build_Avvoo().swapaxes(0,3).swapaxes(1,2)
        Lbar_vvoo /=   ccpert[string_L].Dijab_

        if (density_type == "LL_SD"):
            Dab[string_Mu+string_L]  =  np.einsum('ia,ib->ab', Lbar_vo, Lbar_vo)
            Dab[string_Mu+string_L] +=  2.0 * np.einsum('ijac,ijbc->ab', Lbar_vvoo, Lbar_vvoo)
            Dab[string_Mu+string_L] -=  1.0 * np.einsum('ijac,ijcb->ab', Lbar_vvoo, Lbar_vvoo)
        elif (density_type == "LL_D"):
            Dab[string_Mu+string_L] =    2.0 * np.einsum('ijac,ijbc->ab', Lbar_vvoo, Lbar_vvoo)
            Dab[string_Mu+string_L] -=    1.0 * np.einsum('ijac,ijcb->ab', Lbar_vvoo, Lbar_vvoo)
        elif (density_type == "MUMU_SD"):
            Dab[string_Mu+string_L]  =  np.einsum('ia,ib->ab', Mubar_vo, Mubar_vo)
            Dab[string_Mu+string_L] +=  2.0 * np.einsum('ijac,ijbc->ab', Mubar_vvoo, Mubar_vvoo)
            Dab[string_Mu+string_L] -=  1.0 * np.einsum('ijac,ijcb->ab', Mubar_vvoo, Mubar_vvoo)
        elif (density_type == "MUMU_D"):
            Dab[string_Mu+string_L] =  2.0 * np.einsum('ijac,ijbc->ab', Mubar_vvoo, Mubar_vvoo)
            Dab[string_Mu+string_L] -=  1.0 * np.einsum('ijac,ijcb->ab', Mubar_vvoo, Mubar_vvoo)
        elif (density_type == "MUL_SD"):
            Dab[string_Mu+string_L]  =  np.einsum('ia,ib->ab', Mubar_vo, Lbar_vo)
            Dab[string_Mu+string_L] +=  2.0 * np.einsum('ijac,ijbc->ab', Mubar_vvoo, Lbar_vvoo)
            Dab[string_Mu+string_L] -=  1.0 * np.einsum('ijac,ijcb->ab', Mubar_vvoo, Lbar_vvoo)
        elif (density_type == "MUL_D"):
            Dab[string_Mu+string_L] =  2.0 * np.einsum('ijac,ijbc->ab', Mubar_vvoo, Lbar_vvoo)
            Dab[string_Mu+string_L] -=  1.0 * np.einsum('ijac,ijcb->ab', Mubar_vvoo, Lbar_vvoo)
        else:
            raise Exception("Invalid option for perturbed density, avaialble options: MUMU_SD, MUMU_D, LL_SD, LL_D, MUL_SD and MUL_D")

        Dab[string_Mu + string_L] = 0.5 * (Dab[string_Mu + string_L] + Dab[string_Mu + string_L].T)
        D2ab = np.zeros_like(Dab[string_Mu + string_L])

        
    for key in Dab:
        D2ab += Dab[key] 
        D2ab /= 3    

    Eval_ab, Evec_ab = LA.eig(D2ab)
    Evec_ab = sort(Evec_ab, Eval_ab)
    return Evec_ab    



def optrot_calculate(mol, rhf_e, rhf_wfn, corr_wfn, maxiter_cc, maxiter_lambda, maxiter_pert, memory, edv, filt_singles, Evec_ab):
    ndocc = rhf_wfn.doccpi()[0]
    nmo = rhf_wfn.nmo()
    nvir = nmo - ndocc
    C = np.asarray(rhf_wfn.Ca())
    F_ao = np.asarray(rhf_wfn.Fa())
    F_mo = np.einsum('iu,uv,vj',C.T,F_ao,C)
    # 5 % increments up to 30 % 
    #truncation = [0, int(.05 * nvir), int(.1 * nvir), int(.15 * nvir), int(.20 * nvir), int(.25 * nvir), int(.30 * nvir)]
    truncation = [0]
    for k in truncation:
        # Truncation
        print('Truncation:%d'%k)
        Evec_ab1 = Evec_ab.copy()
        Evec_ab_view = Evec_ab1[:,nvir-k:] 
        Evec_ab_view.fill(0)
        # construct vir-vir block of Fock using info from rhf_efn
        F_vir_pert = np.einsum('Aa,ab,bB', Evec_ab1.T, F_mo[ndocc:,ndocc:], Evec_ab1) 
        print("F_vir_pert")
        print(F_vir_pert)
        tmp_eval, tmp_evec = LA.eig(F_vir_pert)
        tmp_evec = sort(tmp_evec, tmp_eval)
        print("tmp_eval")
        print(tmp_eval)
        SC_evec = np.einsum('aA,AB->aB', Evec_ab1, tmp_evec)

        # Compute CCSD
        if corr_wfn == 'CCSD':
            cc_wfn = helper_ccenergy(mol, rhf_e, rhf_wfn, True, memory)
            new_Fvv = np.einsum('Aa,ab,bB', SC_evec.T, F_mo[ndocc:,ndocc:], SC_evec)
            cc_wfn.after_initialize(SC_evec,new_Fvv)
            cc_wfn.guess_transform()
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
            cclambda = helper_cclambda(cc_wfn,cchbar,True)
            cclambda.after_initialize(SC_evec,new_Fvv)
        elif corr_wfn == 'CC2':
            print('\nFinal CC2 correlation energy:          % 16.15f' % CCcorr_E)
            print('Total CC2 energy:                      % 16.15f' % CC_E)
            cchbar = helper_cc2hbar(cc_wfn)
            cclambda = helper_cc2lambda(cc_wfn,cchbar)

        cclambda.compute_lambda(1e-7, maxiter_lambda)
        omega = 0.07735713394560646
        #omega = 0.01

        cart = {0:'X', 1: 'Y', 2: 'Z'}
        Mu={}; L={}; Dab={};
        ccpert = {}; cclinresp = {}; optrot_PQ={}

        dipole_array = cc_wfn.mints.ao_dipole()
        angmom_array = cc_wfn.mints.ao_angular_momentum()

        pert = [0,1,2]

        for p in pert:
            string_Mu = "MU_" + cart[p]
            string_L = "L_" + cart[p]
            Mu[string_Mu] = np.einsum('uj,vi,uv', cc_wfn.npC, cc_wfn.npC, np.asarray(dipole_array[p]))
            L[string_L] = -0.5 * np.einsum('uj,vi,uv', cc_wfn.npC, cc_wfn.npC, np.asarray(angmom_array[p]))
            if corr_wfn == 'CCSD':
                ccpert[string_Mu] = helper_ccpert(string_Mu, Mu[string_Mu], cc_wfn, cchbar, cclambda, omega, edv, filt_singles)
                ccpert[string_L] = helper_ccpert(string_L, L[string_L], cc_wfn, cchbar, cclambda, omega, edv, filt_singles)
            elif corr_wfn == 'CC2':
                ccpert[string_Mu] = helper_cc2pert(string_Mu, Mu[string_Mu], cc_wfn, cchbar, cclambda, omega, edv)
                ccpert[string_L] = helper_cc2pert(string_L, L[string_L], cc_wfn, cchbar, cclambda, omega, edv)

            new_Hvv = np.einsum('Aa,ab,bB', SC_evec.T, cchbar.Hvv, SC_evec)
            ccpert[string_Mu].after_initialize(SC_evec, new_Hvv)
            ccpert[string_Mu].guess_transform()
            ccpert[string_L].after_initialize(SC_evec, new_Hvv)
            ccpert[string_L].guess_transform()
            #print('x1')
            #print(ccpert[string_Mu].x1)
            #print('x2')    
            #print(ccpert[string_Mu].x2)
 
            print('\nsolving right hand perturbed amplitudes for %s\n' % string_Mu)
            ccpert[string_Mu].solve('right', 1e-7, maxiter_pert)
            print('\nsolving right hand perturbed amplitudes for %s\n' % string_L)
            ccpert[string_L].solve('right', 1e-7, maxiter_pert)
            value = np.einsum('ia,ia', ccpert[string_L].x1, ccpert[string_L].build_Aov())
            #print("Lia*X_ia: %20.15lf"%value)
            print('\nsolving left hand perturbed amplitudes for %s\n'% string_Mu)
            ccpert[string_Mu].solve('left', 1e-7, maxiter_pert)
            print('\nsolving left hand perturbed amplitudes for %s\n'% string_L)
            ccpert[string_L].solve('left', 1e-7, maxiter_pert)

            print('\n Calculating rotation tensor from linear response function: %s \n'% cart[p])
            str_mu = "MU_" + cart[p]
            str_L = "L_" + cart[p]
            str_muL = "<<" + str_mu + ";" + str_L + ">>"
            str_Lmu = "<<" + str_L + ";" + str_mu + ">>"
            str_mumu = "<<" + str_mu + ";" + str_mu + ">>"
            str_LL = "<<" + str_L + ";" + str_L + ">>"

            if corr_wfn == 'CCSD':
                cclinresp[str_muL]= helper_cclinresp(cclambda, ccpert[str_mu], ccpert[str_L])
                cclinresp[str_Lmu]= helper_cclinresp(cclambda, ccpert[str_L], ccpert[str_mu])
                cclinresp[str_mumu]= helper_cclinresp(cclambda, ccpert[str_mu], ccpert[str_mu])
                cclinresp[str_LL]= helper_cclinresp(cclambda, ccpert[str_L], ccpert[str_L])
            elif corr_wfn == 'CC2':
                cclinresp[str_muL]= helper_cc2linresp(cclambda, ccpert[str_mu], ccpert[str_L])
                cclinresp[str_Lmu]= helper_cc2linresp(cclambda, ccpert[str_L], ccpert[str_mu])
                cclinresp[str_mumu]= helper_cc2linresp(cclambda, ccpert[str_mu], ccpert[str_mu])
                cclinresp[str_LL]= helper_cc2linresp(cclambda, ccpert[str_L], ccpert[str_L])

            optrot_PQ[str_muL]= cclinresp[str_muL].linresp()
            optrot_PQ[str_Lmu]= cclinresp[str_Lmu].linresp()
            optrot_PQ[str_mumu]= cclinresp[str_mumu].linresp()
            optrot_PQ[str_LL]= cclinresp[str_LL].linresp()
            str_rot = "Rotation_" + cart[p]
            optrot_PQ[str_rot] =  0.5 * optrot_PQ[str_muL] - 0.5 * optrot_PQ[str_Lmu]

            print('\n optrot_muL = %20.15lf \n' % (optrot_PQ[str_muL]))
            print('\n optrot_Lmu = %20.15lf \n' % (optrot_PQ[str_Lmu]))
            print('\n optrot_mumu = %20.15lf \n' % (optrot_PQ[str_mumu]))
            print('\n optrot_LL = %20.15lf \n' % (optrot_PQ[str_LL]))
            print('\n optrot_response = %20.15lf %s \n' % (optrot_PQ[str_rot], corr_wfn))

        print("1/3 * Traces of each linear response function")
        trc3_muL = optrot_PQ["<<MU_X;L_X>>"] + optrot_PQ["<<MU_Y;L_Y>>"] + optrot_PQ["<<MU_Z;L_Z>>"]
        trc3_muL /= 3    
        trc3_Lmu = optrot_PQ["<<L_X;MU_X>>"] + optrot_PQ["<<L_Y;MU_Y>>"] + optrot_PQ["<<L_Z;MU_Z>>"]
        trc3_Lmu /= 3    
        trc3_mumu = optrot_PQ["<<MU_X;MU_X>>"] + optrot_PQ["<<MU_Y;MU_Y>>"] + optrot_PQ["<<MU_Z;MU_Z>>"]
        trc3_mumu /= 3    
        trc3_LL = optrot_PQ["<<L_X;L_X>>"] + optrot_PQ["<<L_Y;L_Y>>"] + optrot_PQ["<<L_Z;L_Z>>"]
        trc3_LL /= 3    
        trc3_optrot = optrot_PQ["Rotation_X"] + optrot_PQ["Rotation_Y"] + optrot_PQ["Rotation_Z"]
        trc3_optrot /= 3    
        print('\n trace_3_optrot_muL = %20.15lf \n' % trc3_muL)
        print('\n trace_3_optrot_Lmu = %20.15lf \n' % trc3_Lmu)
        print('\n trace_3_optrot_mumu = %20.15lf \n' % trc3_mumu)
        print('\n trace_3_optrot_LL = %20.15lf \n' % trc3_LL)
        print('\n trace_3_actual_optrot = %20.15lf \n' % trc3_optrot)



#Evec_ab = guess_calculate(mol, rhf_e, rhf_wfn, 'CCSD', 0, 0, 0, 40, False, "")
#optrot_calculate(mol, rhf_e, rhf_wfn, 'CCSD', 100, 100, 100, 40, True, Evec_ab)
