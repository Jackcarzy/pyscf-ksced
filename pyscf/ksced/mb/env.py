'''The frozen subsystem B, described in its own AO basis.

Presents exactly the interface of ksced._FrozenEnv, which is what lets
KSCEDMixin -- get_hcore, energy_nuc, energy_elec, reset -- serve both basis
modes without modification. The environment object, not the SCF class, is the
polymorphic seam.
'''

import copy

import numpy
from pyscf.lib import logger

from pyscf.ksced.ksced import (_is_cell, _kinetic_of, _spin_sum, _stack_like,
                               _trace_prod_spin)
from pyscf.ksced.mb.arrays import to_host as _host
from pyscf.ksced.mb.griddens import _GridDensity
from pyscf.ksced.mb.numint import _ksced_numint


def _conc(mol_a, mol_b):
    '''mol_ab with A's AOs first. Both helpers guarantee that ordering.'''
    if _is_cell(mol_a):
        from pyscf.pbc.gto.cell import conc_cell
        return conc_cell(mol_a, mol_b)
    from pyscf.gto.mole import conc_mol
    return conc_mol(mol_a, mol_b)




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

        # Cross terms use slices of the concatenated A+B basis, with A first.
        # Ghost centers do not affect the resulting nuclear energy.
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
        '''
        key = id(mol)
        mf = self._mfs.get(key)
        if mf is None:
            mf = self.mf_b.copy()
            with_df = getattr(mf, 'with_df', None)
            if with_df is not None:
                mf.with_df = copy.copy(with_df)
            mf.reset(mol)
            mf._eri = None
            self._mfs[key] = mf
        return mf

    def _vne_on(self, mol, kpt=None):
        '''V_ne of mol's own nuclei, in mol's basis: hcore - T.
        '''
        mf = self._mf_on(mol)
        if _is_cell(mol):
            h1e = mf.get_hcore(mol, kpt)
        else:
            h1e = mf.get_hcore(mol)
        return _host(h1e) - _host(_kinetic_of(mol, kpt))

    # -- the _FrozenEnv interface ---------------------------------------

    @property
    def polarized(self):
        return getattr(self.dm_b, 'ndim', 2) == 3

    def _nr(self, ni):
        return ni.nr_uks if self.polarized else ni.nr_rks

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
            dm_pad[n:, n:] = _host(_spin_sum(self.dm_b)).real
            mf_ab = self._mf_on(self.mol_ab)
            if _is_cell(self.mol_ab):
                j_ab = mf_ab.get_j(self.mol_ab, dm_pad, 1, numpy.zeros(3), None)
            else:
                j_ab = mf_ab.get_j(self.mol_ab, dm_pad, 1)
            self._j_b = _host(j_ab)[:n, :n]
        return self._j_b

    def e_vne_a_rho_b(self, vne_a=None):
        '''<V_ne[A]|rho_B>.
        '''
        if self._e_vne_a_rho_b is None:
            self._e_vne_a_rho_b = _trace_prod_spin(
                self._vne_a_in_bs_basis(), self.dm_b)
        return self._e_vne_a_rho_b

    def e_xc(self, ni, mol, grids, xc, max_memory):
        '''E_xc[rho_B], evaluated with B's own AOs on the supermolecular grid.'''
        if self._e_xc is None:
            self._e_xc = self._nr(ni)(self.mol_b, grids, xc, self.dm_b,
                                      max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b(self, ni, mol, grids, t_nad, max_memory):
        if self._e_tnad_b is None:
            self._e_tnad_b = self._nr(ni)(
                self.mol_b, grids, t_nad, self.dm_b,
                max_memory=max_memory)[1]
        return self._e_tnad_b

    def e_xc_pbc(self, ni, cell, grids, xc, hermi, kpt, max_memory):
        if self._e_xc is None:
            self._e_xc = self._nr(ni)(
                self.mol_b, grids, xc, self.dm_b, 0, hermi, kpt, None,
                max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b_pbc(self, ni, cell, grids, t_nad, hermi, kpt, max_memory):
        if self._e_tnad_b is None:
            self._e_tnad_b = self._nr(ni)(
                self.mol_b, grids, t_nad, self.dm_b, 0, hermi, kpt, None,
                max_memory=max_memory)[1]
        return self._e_tnad_b

    # -- MB-only ---------------------------------------------------------

    def _make_evaluator(self, ni, kpt=None):
        '''rho_B(coords) at GGA order, using B's own AOs.
        '''
        mol_b, dm_b = self.mol_b, self.dm_b

        def eval_channels(ao, dm):
            if getattr(dm, 'ndim', 2) == 2:
                return ni.eval_rho(mol_b, ao, dm, xctype='GGA')
            rhos = [ni.eval_rho(mol_b, ao, dm[s], xctype='GGA')
                    for s in range(2)]
            return _stack_like(rhos[0], rhos)

        if not _is_cell(mol_b):
            def evaluator(coords):
                ao = ni.eval_ao(mol_b, coords, deriv=1)
                return eval_channels(ao, dm_b)
            return evaluator

        k1 = numpy.zeros(3) if kpt is None else numpy.asarray(kpt).reshape(3)
        probe = numpy.zeros((1, 3))
        errors = []
        for kk, add_k_axis in ((k1, False), (k1.reshape(1, 3), True)):
            dd = dm_b[None] if add_k_axis and numpy.ndim(dm_b) == 2 else dm_b
            try:
                ao = ni.eval_ao(mol_b, probe, kk, deriv=1)
                if self.polarized:
                    for s in range(2):
                        ds = dm_b[s][None] if add_k_axis else dm_b[s]
                        ni.eval_rho(mol_b, ao, ds, xctype='GGA')
                else:
                    ni.eval_rho(mol_b, ao, dd, xctype='GGA')
            except Exception as exc:          # noqa: BLE001 - probing
                errors.append('%s: %s' % (type(exc).__name__, exc))
                continue

            def evaluator(coords, _kk=kk, _add=add_k_axis):
                ao = ni.eval_ao(mol_b, coords, _kk, deriv=1)
                if not self.polarized:
                    dd = dm_b[None] if _add else dm_b
                    return ni.eval_rho(mol_b, ao, dd, xctype='GGA')
                rhos = []
                for s in range(2):
                    dd = dm_b[s][None] if _add else dm_b[s]
                    rhos.append(ni.eval_rho(mol_b, ao, dd, xctype='GGA'))
                return _stack_like(rhos[0], rhos)
            return evaluator

        raise RuntimeError(
            'KSCED: cannot evaluate the environment density with %s. Neither '
            'gamma-point convention worked:\n  1-D kpt: %s\n  2-D kpts: %s'
            % (type(ni).__name__, errors[0], errors[1]))

    def numint_for(self, ni, kpt=None):
        '''The offset twin of ni. Built once and cached.'''
        if self._numint is None:
            self._griddens = _GridDensity(self._make_evaluator(ni, kpt))
            self._numint = _ksced_numint(ni, self._griddens)
        return self._numint
