# A simple Psi4 script to compute coupled cluster lambda amplitudes
# from a RHF reference
# Scipy and numpy python modules are required
#
# Algorithms were taken directly from Daniel Crawford's programming website:
# http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming
# Special thanks to Lori Burns for integral help
#
# Created by: Ashutosh Kumar
# Date: 4/29/2017
# License: GPL v3.0
#

import time
import numpy as np
import psi4

# N dimensional dot
# Like a mini DPD library
def ndot(input_string, op1, op2, prefactor=None):
    """
    No checks, if you get weird errors its up to you to debug.

    ndot('abcd,cdef->abef', arr1, arr2)
    """
    inp, output_ind = input_string.split('->')
    input_left, input_right = inp.split(',')

    size_dict = {}
    for s, size in zip(input_left, op1.shape):
        size_dict[s] = size
    for s, size in zip(input_right, op2.shape):
        size_dict[s] = size

    set_left = set(input_left)
    set_right = set(input_right)
    set_out = set(output_ind)

    idx_removed = (set_left | set_right) - set_out
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed

    # Tensordot axes
    left_pos, right_pos = (), ()
    for s in idx_removed:
        left_pos += (input_left.find(s),)
        right_pos += (input_right.find(s),)
    tdot_axes = (left_pos, right_pos)

    # Get result ordering
    tdot_result = input_left + input_right
    for s in idx_removed:
        tdot_result = tdot_result.replace(s, '')

    rs = len(idx_removed)
    dim_left, dim_right, dim_removed = 1, 1, 1
    for key, size in size_dict.items():
        if key in keep_left:
            dim_left *= size
        if key in keep_right:
            dim_right *= size
        if key in idx_removed:
            dim_removed *= size

    shape_result = tuple(size_dict[x] for x in tdot_result)
    used_einsum = False

    # Matrix multiply
    # No transpose needed
    if input_left[-rs:] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed),
                          op2.reshape(dim_removed, dim_right))

    # Transpose both
    elif input_left[:rs] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T,
                          op2.reshape(dim_right, dim_removed).T)

    # Transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed),
                          op2.reshape(dim_right, dim_removed).T)

    # Tranpose left
    elif input_left[:rs] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T,
                          op2.reshape(dim_removed, dim_right))

    # If we have to transpose vector-matrix, einsum is faster
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        new_view = np.einsum(input_string, op1, op2)
        used_einsum = True

    else:
        new_view = np.tensordot(op1, op2, axes=tdot_axes)

    # Make sure the resulting shape is correct
    if (new_view.shape != shape_result) and not used_einsum:
        if (len(shape_result) > 0):
            new_view = new_view.reshape(shape_result)
        else:
            new_view = np.squeeze(new_view)

    # In-place mult by prefactor if requested
    if prefactor is not None:
        new_view *= prefactor

    # Do final tranpose if needed
    if used_einsum:
        return new_view
    elif tdot_result == output_ind:
        return new_view
    else:
        return np.einsum(tdot_result + '->' + output_ind, new_view)


class helper_CCRESPONSE(object):

    def __init__(self, pert, ccsd, hbar, cclambda, memory=2):

        # Integral generation from Psi4's MintsHelper
        time_init = time.time()

        self.pert = pert
        self.MO = ccsd.MO
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nfzc = 0
        self.nocc = ccsd.ndocc 
        self.nvirt = ccsd.nmo - ccsd.nocc - ccsd.nfzc

	self.mints = ccsd.mints

        self.slice_nfzc = slice(0, self.nfzc)
        self.slice_o = slice(self.nfzc, self.nocc + self.nfzc)
        self.slice_v = slice(self.nocc + self.nfzc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {'f': self.slice_nfzc, 'o' : self.slice_o, 'v' : self.slice_v,
                           'a' : self.slice_a}

       
        self.F = ccsd.F
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2


        self.ttau  =  hbar.ttau
        self.L     =  hbar.L
        self.Hov   =  hbar.Hov
        self.Hvv   =  hbar.Hvv
        self.Hoo   =  hbar.Hoo
        self.Hoooo =  hbar.Hoooo
        self.Hvvvv =  hbar.Hvvvv
        self.Hvovv =  hbar.Hvovv
        self.Hooov =  hbar.Hooov
        self.Hovvo =  hbar.Hovvo
        self.Hovov =  hbar.Hovov
        self.Hovvv =  hbar.Hovvv
        self.Hovoo =  hbar.Hovoo


	self.l1 = cclambda.l1
        self.l2 = cclambda.l2

        self.Dia = self.Hoo.reshape(-1, 1) - self.Hvv + omega
        self.Dijab = self.Hoo.reshape(-1, 1, 1, 1) + self.Hoo.reshape(-1, 1, 1) - self.Hvv.reshape(-1, 1) - self.Hvv + omega

	#self.muX = np.asarray(self.mints.ao_dipole()[0])
	#self.muY = np.asarray(self.mints.ao_dipole()[1])
	#self.muZ = np.asarray(self.mints.ao_dipole()[2])
        #self.pert_dict = {'0' : self.muX, '1' : self.muY, '2' : self.muZ}

 
        print('\n..initialed CCRESPONSE in %.3f seconds.\n' % (time.time() - time_init))
    # occ orbitals i, j, k, l, m, n
    # virt orbitals a, b, c, d, e, f
    # all oribitals p, q, r, s, t, u, v

    def get_MO(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return self.MO[self.slice_dict[string[0]], self.slice_dict[string[1]],
                       self.slice_dict[string[2]], self.slice_dict[string[3]]]

    def get_F(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]


    def get_pert(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return self.pert[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def build_Aoo(self):
        Aoo = self.get_pert('oo').copy()
        Aoo += ndot('ie,me->mi', self.t1, self.get_pert('ov'))
	return Aoo

    def build_Aov(self):
        Aov = self.get_pert('ov').copy()
        return Aov

    def build_Avo(self):
        Avo =  self.get_pert('vo').copy()
        Avo += ndot('ae,ie->ai', self.get_pert('vv'), self.t1)
        Avo -= ndot('ma,mi->ai', self.t1, self.get_pert('oo'))
        Avo += ndot('miea,me->ai', self.t2, self.get_pert('ov'), prefactor=2.0)
        Avo += ndot('imea,me->ai', self.t2, self.get_pert('ov'), prefactor=-1.0)
        tmp = ndot('ie,ma->imea', self.t1, self.t1)
        Avo -= ndot('imea,me->ai', tmp, self.get_pert('ov'))
        return Avo 

    def build_Avv(self):
        Avv =  self.get_pert('vv').copy()
        Avv -= ndot('ma,me->ae', self.t1, self.get_pert('ov'))
        return Avv
        
    def build_Aovoo(self):
        Aovoo = 0
        Aovoo += ndot('ijeb,me->mbij', self.t2, self.get_pert('ov'))
        return Aovoo

    def build_Avvvo(self):
        Avvvo = 0
        Avvvo -= ndot('miab,me->abei', self.t2, self.get_pert('ov'))
        return Avvvo

    def build_Avvoo(self):
        Avvoo = 0
        Avvoo += ndot('ijeb,ae->abij', self.t2, self.get_pert('vv'))
        Avvoo -= ndot('mjab,mi->abij', self.t2, self.get_pert('oo'))
        return Avvoo


    def build_Goo(self):
        self.Goo = 0
        self.Goo += np.dot('mjab,ijab->mi', self.t2, self.y2)
        return self.Goo

    def build_Gvv(self):
        self.Gvv = 0
        self.Gvv -= ndot('ijab,ijeb->ae', self.y2, self.t2)
        return self.Gvv

    def build_Zvv(self):
        Zvv = 0
        Zvv += ndot('amef,mf->ae', self.Hvovv, self.x1, prefactor=2.0)
        Zvv += ndot('amfe,mf->ae', self.Hvovv, self.x1, prefactor=-1.0)
        Zvv -= ndot('mnaf,mnef->ae', self.x2, self.L)
        return Zvv

    def build_Zoo(self):
        Zoo = 0
        Zoo -= ndot('mnie,ne->mi', self.Hooov, self.x1, prefactor=2.0)
        Zoo -= ndot('nmie,ne->mi', self.Hooov, self.x1, prefactor=-1.0)
        Zoo -= ndot('mnef,inef->mi', self.L, self.x2)
        return Zoo

    def update_X(self):
        r_x1  = self.Avo.copy() 
        r_x1 -= omega * self.x1
        r_x1 += ndot('ie,ae->ia', self.x1, self.Hvv)
        r_x1 -= ndot('mi,ma->ia', self.Hoo, self.x1)
        r_x1 += ndot('maei,me->ia', self.Hovvo, self.x1, prefactor=2.0)
        r_x1 += ndot('maie,me->ia', self.Hovvo, self.x1, prefactor=-1.0)
        r_x1 += ndot('miea,me->ia', self.x2, self.Hov, prefactor=2.0)
        r_x1 += ndot('imea,me->ia', self.x2, self.Hov, prefactor=-1.0)
        r_x1 += ndot('imef,amef->ia', self.x2, self.Hvovv, prefactor=2.0)
        r_x1 += ndot('imef,amfe->ia', self.x2, self.Hvovv, prefactor=-1.0)
        r_x1 -= ndot('mnie,mnae->ia', self.Hooov, self.x2, prefactor=2.0)
        r_x1 -= ndot('nmie,mnae->ia', self.Hooov, self.x2, prefactor=-1.0)

	self.x1 += r_x1/self.Dia

        r_x2 = self.build_Avvoo.copy()
        r_x2 -= 0.5 * omega * self.x2
        r_x2 += ndot('ie,abej->ijab', self.x1, self.Hvvvo)
        r_x2 -= ndot('mbij,ma->ijab', self.Hovoo, self.x1)

        r_x2 += ndot('mi,mjab->ijab', self.build_Zoo, self.t2)
        r_x2 += ndot('ijeb,ae->ijab', self.t2, self.build_Zvv)

        r_x2 += ndot('ijeb,ae->ijab', self.x2, self.Hvv)
        r_x2 -= ndot('mi,mjab->ijab', self.Hoo, self.x2)

        r_x2 += ndot('mnij,mnab->ijab', self.Hoooo, self.x2, prefactor=0.5)
        r_x2 += ndot('ijef,abef->ijab', self.x2, self.Hvvvv, prefactor=0.5)

        r_x2 -= ndot('imeb,maje->ijab', self.x2, self.Hovov)
        r_x2 -= ndot('imea,mbej->ijab', self.x2, self.Hovvo)

        r_x2 += ndot('miea,mbej->ijab', self.x2, self.Hovvo, prefactor=2.0)
        r_x2 += ndot('miea,mbje->ijab', self.x2, self.Hovov, prefactor=-1.0)
       
        self.x2 += r_x2 + r_x2.swapaxes(0,1).swapaxes(2,3)
        self.x2 += self.x2/self.Dijab

    def update_Y(self):
        r_y1  = 2.0 * self.Aov.copy()
        r_y1 += omega * self.y1
        r_y1 += ndot('ie,ea->ia', self.y1, self.Hvv)
        r_y1 -= ndot('im,ma->ia', self.Hoo, self.y1)
        r_y1 += ndot('ieam,me->ia', self.Hovvo, self.y1, prefactor=2.0)
        r_y1 += ndot('iema,me->ia', self.Hovov, self.y1, prefactor=-1.0)
        r_y1 += ndot('imef,efam->ia', self.y2, self.Hvvvo)
        r_y1 -= ndot('iemn,mnae->ia', self.Hovoo, self.y2)
        r_y1 -= ndot('eifa,ef->ia', self.Hvovv, self.build_Gvv(), prefactor=2.0)
        r_y1 -= ndot('eiaf,ef->ia', self.Hvovv, self.build_Gvv(), prefactor=-1.0)
        r_y1 -= ndot('mina,mn->ia', self.Hooov, self.build_Goo(), prefactor=2.0)
        r_y1 -= ndot('imna,mn->ia', self.Hooov, self.build_Goo(), prefactor=-1.0)

        r_y1 += ndot('imae,me->ia', self.L, self.x1, prefactor=2.0)
        r_y1 -= ndot('im,ma->ia', self.build_Aoo, self.l1)
        r_y1 += ndot('ie,ea->ia', self.l1, self.build_Avv)
        r_y1 += ndot('imfe,feam->ia', self.l2, self.build_Avvoo)

        r_y1 -= ndot('ienm,mnea->ia', self.build_Aovoo, self.l2, prefactor=0.5)
        r_y1 -= ndot('iemn,mnae->ia', self.build_Aovoo, self.l2, prefactor=0.5)

        tmp   = ndot('ma,me->ae', self.Hov, self.x1)
        r_y1 -= ndot('ie,ae->ia', self.l1, tmp)

        tmp   = ndot('ie,me->im', self.Hov, self.x1)
        r_y1 -= ndot('im,ma->ia', tmp, self.l1)

        tmp   = ndot('me,ne->mn', self.x1, self.l1)
        r_y1 -= ndot('mina,mn->ia', self.Hooov, tmp, prefactor=2.0)
        r_y1 -= ndot('imna,mn->ia', self.Hooov, tmp, prefactor=-1.0)

        tmp   = ndot('me,na->mnea', self.x1, self.l1)
        r_y1 -= ndot('imne,mnea->ia', self.Hooov, tmp, prefactor=2.0)
        r_y1 -= ndot('mine,mnea->ia', self.Hooov, tmp, prefactor=-1.0)

        tmp   = ndot('me,ne->mn', self.x1, self.l1)
        r_y1 -= ndot('mina,mn->ia', self.Hooov, tmp, prefactor=2.0)
        r_y1 -= ndot('imna,mn->ia', self.Hooov, tmp, prefactor=-1.0)

        tmp   = ndot('me,na->mnea', self.x1, self.l1)
        r_y1 -= ndot('imne,mnea->ia', self.Hooov, tmp, prefactor=2.0)
        r_y1 -= ndot('mine,mnea->ia', self.Hooov, tmp, prefactor=-1.0)



        self.y1 += r_y1/self.Dia 




    def pseudoenergy(self):
        pseudoenergy = 0
        pseudoenergy += ndot('ijab,ijab->', self.get_MO('oovv'), self.l2, prefactor=0.5)
        return pseudoenergy

 

    def compute_lambda(self, r_conv=1.e-13, maxiter=50, max_diis=8):
        ### Setup DIIS
        diis_vals_l1 = [self.l1.copy()]
        diis_vals_l2 = [self.l2.copy()]
        diis_errors = []

        ### Start Iterations
        cclambda_tstart = time.time()

        pseudoenergy_old = self.pseudoenergy()
        print("CCLAMBDA Iteration %3d: pseudoenergy = %.12f   dE = % .5E   MP2" % (0, pseudoenergy_old, -pseudoenergy_old))

        # Iterate!
        diis_size = 0
        for CCLAMBDA_iter in range(1, maxiter + 1):

            # Save new amplitudes
            oldl1 = self.l1.copy()
            oldl2 = self.l2.copy()

            self.update()

            # Compute lambda 
            pseudoenergy = self.compute_lambda()

            # Print CCLAMBDA iteration information
            print('CCLAMBDA Iteration %3d: pseudoenergy = %.12f   dE = % .5E   DIIS = %d' % (CCLAMBDA_iter, pseudoenergy, (pseudoenergy - pseudoenergy_old), diis_size))

            # Check convergence
            if (abs(pseudoenergy - pseudoenergy_old) < r_conv):
                print('\nCCLAMBDA has converged in %.3f seconds!' % (time.time() - ccsd_tstart))
                return pseudoenergy

            # Add DIIS vectors
            diis_vals_l1.append(self.l1.copy())
            diis_vals_l2.append(self.l2.copy())

            # Build new error vector
            error_l1 = (diis_vals_l1[-1] - oldl1).ravel()
            error_l2 = (diis_vals_l2[-1] - oldl2).ravel()
            diis_errors.append(np.concatenate((error_l1, error_l2)))

            # Update old energy
            pseudoenergy_old = pseudoenergy

            if CCLAMBDA_iter >= 1:
                # Limit size of DIIS vector
                if (len(diis_vals_l1) > max_diis):
                    del diis_vals_l1[0]
                    del diis_vals_l2[0]
                    del diis_errors[0]

                diis_size = len(diis_vals_l1) - 1

                # Build error matrix B
                B = np.ones((diis_size + 1, diis_size + 1)) * -1
                B[-1, -1] = 0

                for n1, e1 in enumerate(diis_errors):
                    B[n1, n1] = np.dot(e1, e1)
                    for n2, e2 in enumerate(diis_errors):
                        if n1 >= n2: continue
                        B[n1, n2] = np.dot(e1, e2)
                        B[n2, n1] = B[n1, n2]

                B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

                # Build residual vector
                resid = np.zeros(diis_size + 1)
                resid[-1] = -1

                # Solve pulay equations
                ci = np.linalg.solve(B, resid)

                # Calculate new amplitudes
                self.l1[:] = 0
                self.l2[:] = 0
                for num in range(diis_size):
                    self.l1 += ci[num] * diis_vals_l1[num + 1]
                    self.l2 += ci[num] * diis_vals_l2[num + 1]




# End CCLAMBDA class

if __name__ == "__main__":
    arr4 = np.random.rand(4, 4, 4, 4)
    arr2 = np.random.rand(4, 4)

    def test_ndot(string, op1, op2):
        ein_ret = np.einsum(string, op1, op2)
        ndot_ret = ndot(string, op1, op2)
        assert np.allclose(ein_ret, ndot_ret)

    test_ndot('abcd,cdef->abef', arr4, arr4)
    test_ndot('acbd,cdef->abef', arr4, arr4)
    test_ndot('acbd,cdef->abfe', arr4, arr4)
    test_ndot('mnab,mnij->ijab', arr4, arr4)

    test_ndot('cd,cdef->ef', arr2, arr4)
    test_ndot('ce,cdef->df', arr2, arr4)
    test_ndot('nf,naif->ia', arr2, arr4)