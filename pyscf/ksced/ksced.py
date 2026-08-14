'''KSCED subsystem embedding: frozen environment and the SCF mixin.'''

from pyscf import gto

T_NAD_DEFAULT = 'LDA_K_TF'


def _vne_of(mol):
    '''Nuclear-electron attraction matrix for mol, pseudopotential aware.

    Mirrors what pyscf.scf.hf builds for h1e, so that V_ne[B] evaluated here is
    consistent with the V_ne[A] already inside get_hcore.
    '''
    if mol._pseudo:
        vne = gto.pp_int.get_gth_pp(mol)
    else:
        vne = mol.intor_symmetric('int1e_nuc')
    if len(mol._ecpbas) > 0:
        vne = vne + mol.intor_symmetric('ECPscalar')
    return vne


class _FrozenEnv:
    '''The frozen subsystem B.

    Everything the embedded calculation needs from B is served from here and
    cached, because rho_B never changes during A's SCF. This is the only class
    that assumes A and B share an AO basis.
    '''

    def __init__(self, mf_b, dm_b=None):
        self.mf_b = mf_b
        self.mol_b = mf_b.mol
        self.dm_b = mf_b.make_rdm1() if dm_b is None else dm_b
        self.reset()

    def reset(self):
        self._vne_b = None
        self._j_b = None
        self._e_xc = None
        self._e_tnad_b = None
        return self

    def get_vne_b(self, mol):
        '''V_ne[B] in the shared AO basis.'''
        if self._vne_b is None:
            self._vne_b = _vne_of(self.mol_b)
        return self._vne_b

    def get_j_b(self, mf, mol):
        '''J[rho_B] in the shared AO basis.'''
        if self._j_b is None:
            self._j_b = mf.get_j(mol, self.dm_b, 1)
        return self._j_b

    def e_xc(self, ni, mol, grids, xc, max_memory):
        '''E_xc[rho_B].'''
        if self._e_xc is None:
            self._e_xc = ni.nr_rks(mol, grids, xc, self.dm_b,
                                   max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b(self, ni, mol, grids, t_nad, max_memory):
        '''T_s^TF[rho_B], the B term of the non-additive kinetic energy.'''
        if self._e_tnad_b is None:
            self._e_tnad_b = ni.nr_rks(mol, grids, t_nad, self.dm_b,
                                       max_memory=max_memory)[1]
        return self._e_tnad_b
