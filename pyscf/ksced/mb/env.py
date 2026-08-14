'''The frozen subsystem B, described in its own AO basis.

Presents exactly the interface of ksced._FrozenEnv, which is what lets
KSCEDMixin -- get_hcore, energy_nuc, energy_elec, reset -- serve both basis
modes without modification. The environment object, not the SCF class, is the
polymorphic seam.
'''

import numpy
from pyscf.lib import logger

from pyscf.ksced.ksced import _is_cell, _as_like, _trace_prod, _kinetic_of
from pyscf.ksced.mb.griddens import _GridDensity
from pyscf.ksced.mb.numint import _ksced_numint


def _conc(mol_a, mol_b):
    '''mol_ab with A's AOs first. Both helpers guarantee that ordering.'''
    if _is_cell(mol_a):
        from pyscf.pbc.gto.cell import conc_cell
        return conc_cell(mol_a, mol_b)
    from pyscf.gto.mole import conc_mol
    return conc_mol(mol_a, mol_b)


def _host(a):
    return a.get() if type(a).__module__.startswith('cupy') else numpy.asarray(a)


class _FrozenEnvMB:
    '''Subsystem B in B's own basis, serving A in A's basis.

    Everything crossing between the two bases is a constant of A's SCF, because
    rho_B never changes, so it is built once at the AB dimension and sliced.
    '''

    def __init__(self, mf_b, mol_a, dm_b=None, mol_ab=None):
        self.mf_b = mf_b
        self.mol_b = mf_b.mol
        self.mol_a = mol_a

        if dm_b is None:
            if (getattr(mf_b, 'mo_coeff', None) is None
                    or getattr(mf_b, 'mo_occ', None) is None):
                raise ValueError(
                    'subsystem B has no density matrix. Call mf_b.kernel() before '
                    'embed(), or pass an explicit dm_b.')
            if not getattr(mf_b, 'converged', True):
                logger.warn(mf_b, 'KSCED: subsystem B is not converged; the frozen '
                                  'density is taken from an unconverged calculation')
            dm_b = mf_b.make_rdm1()
        self.dm_b = dm_b

        self.nao_a = mol_a.nao
        self.nao_b = self.mol_b.nao

        # mol_ab means something different here than it does in the
        # supermolecular path. There it is the real whole system, supplied only
        # so E_nn[AB] can be added. Here every cross term is a *slice* of a
        # matrix built at the AB dimension, so it must be the concatenation
        # A (+) B, with nao_a + nao_b functions and A's block first.
        #
        # E_nn is unaffected by the difference: ghost centres carry no charge,
        # so conc(mol_a, mol_b).energy_nuc() equals the real system's.
        if mol_ab is None:
            mol_ab = _conc(mol_a, self.mol_b)
        elif mol_ab.nao != self.nao_a + self.nao_b:
            raise ValueError(
                'KSCED monomolecular basis: mol_ab has nao %d, but the cross '
                'terms are slices of an A(+)B build and need nao_a + nao_b = '
                '%d + %d = %d. Omit mol_ab and it will be concatenated '
                'correctly; pass one only to override the concatenation itself.'
                % (mol_ab.nao, self.nao_a, self.nao_b,
                   self.nao_a + self.nao_b))
        self.mol_ab = mol_ab
        self.reset()

    def reset(self):
        self._vne_b = None
        self._vne_a_in_b = None
        self._j_b = None
        self._e_xc = None
        self._e_tnad_b = None
        self._e_vne_a_rho_b = None
        self._griddens = None
        self._numint = None
        self._mfs = {}
        return self

    # -- scratch mean-field objects --------------------------------------

    def _mf_on(self, mol):
        '''A copy of mf_b rebound to mol.

        mf_b itself cannot be asked for hcore or J on mol_ab: its density
        fitting object is built for mol_b, so a periodic FFTDF would use the
        wrong cell. Copying and calling reset(mol) rebuilds the fitting object
        for the right cell while keeping mf_b's class -- and therefore its
        backend and its choice of method inside get_hcore, which is what
        docs/architecture.md section 4 requires. mf_b is never mutated.
        '''
        key = id(mol)
        mf = self._mfs.get(key)
        if mf is None:
            mf = self.mf_b.copy()
            mf.reset(mol)
            self._mfs[key] = mf
        return mf

    def _vne_on(self, mol, kpt=None):
        '''V_ne of mol's own nuclei, in mol's basis: hcore - T.

        Derived from get_hcore rather than computed independently, so it
        inherits whichever method the backend used. Mixing an FFTDF V_ne[B]
        with a MultiGrid V_ne[A] is the trap documented in
        docs/architecture.md section 4.
        '''
        mf = self._mf_on(mol)
        if _is_cell(mol):
            h1e = mf.get_hcore(mol, kpt)
        else:
            h1e = mf.get_hcore(mol)
        return _host(h1e) - _host(_kinetic_of(mol, kpt))

    # -- the _FrozenEnv interface ---------------------------------------

    def get_vne_b(self, mol=None, kpt=None):
        '''V_ne[B] in A's basis: the A-A block of V_ne_ab, less A's own.'''
        if self._vne_b is None:
            n = self.nao_a
            self._vne_b = (self._vne_on(self.mol_ab, kpt)[:n, :n]
                           - self._vne_on(self.mol_a, kpt))
        return self._vne_b

    def _vne_a_in_bs_basis(self, kpt=None):
        '''V_ne[A] in B's basis: the B-B block of V_ne_ab, less B's own.'''
        if self._vne_a_in_b is None:
            n = self.nao_a
            self._vne_a_in_b = (self._vne_on(self.mol_ab, kpt)[n:, n:]
                                - self._vne_on(self.mol_b, kpt))
        return self._vne_a_in_b

    def get_j_b(self, mf=None, mol=None):
        '''J[rho_B] in A's basis. mf and mol are ignored; the AB slice is used.'''
        if self._j_b is None:
            n = self.nao_a
            nao_ab = self.nao_a + self.nao_b
            dm_pad = numpy.zeros((nao_ab, nao_ab))
            dm_pad[n:, n:] = _host(self.dm_b).real
            mf_ab = self._mf_on(self.mol_ab)
            if _is_cell(self.mol_ab):
                j_ab = mf_ab.get_j(self.mol_ab, dm_pad, 1, numpy.zeros(3), None)
            else:
                j_ab = mf_ab.get_j(self.mol_ab, dm_pad, 1)
            self._j_b = _host(j_ab)[:n, :n]
        return self._j_b

    def e_vne_a_rho_b(self, vne_a=None):
        '''<V_ne[A]|rho_B>.

        vne_a is accepted and ignored: KSCEDMixin supplies it in A's basis, and
        this contraction needs V_ne[A] in B's basis. Ignoring it is what lets
        KSCEDMixin stay untouched.
        '''
        if self._e_vne_a_rho_b is None:
            self._e_vne_a_rho_b = _trace_prod(self._vne_a_in_bs_basis(),
                                              self.dm_b)
        return self._e_vne_a_rho_b

    def e_xc(self, ni, mol, grids, xc, max_memory):
        '''E_xc[rho_B], evaluated with B's own AOs on the supermolecular grid.'''
        if self._e_xc is None:
            self._e_xc = ni.nr_rks(self.mol_b, grids, xc, self.dm_b,
                                   max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b(self, ni, mol, grids, t_nad, max_memory):
        if self._e_tnad_b is None:
            self._e_tnad_b = ni.nr_rks(self.mol_b, grids, t_nad, self.dm_b,
                                       max_memory=max_memory)[1]
        return self._e_tnad_b

    def e_xc_pbc(self, ni, cell, grids, xc, hermi, kpt, max_memory):
        if self._e_xc is None:
            self._e_xc = ni.nr_rks(self.mol_b, grids, xc, self.dm_b, 0, hermi,
                                   kpt, None, max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b_pbc(self, ni, cell, grids, t_nad, hermi, kpt, max_memory):
        if self._e_tnad_b is None:
            self._e_tnad_b = ni.nr_rks(self.mol_b, grids, t_nad, self.dm_b, 0,
                                       hermi, kpt, None, max_memory=max_memory)[1]
        return self._e_tnad_b

    # -- MB-only ---------------------------------------------------------

    def _make_evaluator(self, ni, kpt=None):
        mol_b, dm_b = self.mol_b, self.dm_b
        if _is_cell(mol_b):
            kpt_ = numpy.zeros(3) if kpt is None else kpt

            def evaluator(coords):
                ao = ni.eval_ao(mol_b, coords, kpt_, deriv=1)
                return ni.eval_rho(mol_b, ao, dm_b, xctype='GGA')
        else:
            def evaluator(coords):
                ao = ni.eval_ao(mol_b, coords, deriv=1)
                return ni.eval_rho(mol_b, ao, dm_b, xctype='GGA')
        return evaluator

    def numint_for(self, ni, kpt=None):
        '''The offset twin of ni. Built once and cached.'''
        if self._numint is None:
            self._griddens = _GridDensity(self._make_evaluator(ni, kpt))
            self._numint = _ksced_numint(ni, self._griddens)
        return self._numint
