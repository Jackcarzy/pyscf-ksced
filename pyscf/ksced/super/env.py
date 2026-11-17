'''The frozen subsystem B in the supermolecular basis (basis_mode='S').'''

from pyscf.lib import logger

from pyscf.ksced.common import (_is_kscf, _is_polarized, _kweight_trace,
                                _spin_sum, _stack_like, _trace_prod_spin,
                                _vne_from_hcore)


class _FrozenEnv:
    '''The frozen subsystem B.
    '''

    def __init__(self, mf_b, dm_b=None):
        self.mf_b = mf_b
        self.mol_b = mf_b.mol
        if dm_b is None:
            if getattr(mf_b, 'mo_coeff', None) is None or getattr(mf_b, 'mo_occ', None) is None:
                raise ValueError(
                    'subsystem B has no density matrix. Call mf_b.kernel() before '
                    'embed(), or pass an explicit dm_b.')
            if not getattr(mf_b, 'converged', True):
                logger.warn(mf_b, 'KSCED: subsystem B is not converged; the frozen '
                                  'density is taken from an unconverged calculation')
            dm_b = mf_b.make_rdm1()
        self.dm_b = dm_b
        # k-point set of B, or None at the gamma point. A k-point dm_b is
        # (nk, nao, nao); with kpts set, its leading axis is k, never spin.
        self.kpts = mf_b.kpts if _is_kscf(mf_b) else None
        self.reset()

    def reset(self):
        self._vne_b = None
        self._j_b = None
        self._e_xc = None
        self._e_tnad_b = None
        self._e_vne_a_rho_b = None
        return self

    @property
    def polarized(self):
        '''True when B was converged unrestricted, so dm_b carries a spin axis.'''
        if self.kpts is not None:
            return False        # k-point environments are restricted (KRKS) only
        return _is_polarized(self.dm_b)

    def dm_b_for(self, polarized_a):
        '''rho_B in the spin layout subsystem A needs.

        Four cases, and three of them are not the identity:

          A restricted, B restricted    dm_b unchanged
          A restricted, B unrestricted  spin summed to the total
          A unrestricted, B restricted  halved into each channel -- adding the
                                        whole of rho_B to both would double the
                                        environment, silently
          A unrestricted, B unrestricted  dm_b unchanged
        '''
        if polarized_a and not self.polarized:
            half = self.dm_b * .5
            return _stack_like(half, (half, half))
        if not polarized_a and self.polarized:
            return _spin_sum(self.dm_b)
        return self.dm_b

    def dm_b_total(self):
        '''rho_B summed over spin. A k-point dm_b is restricted already, and its
        leading axis is k, so it is returned as it is.'''
        return self.dm_b if self.kpts is not None else _spin_sum(self.dm_b)

    def _nr(self, ni, polarized):
        '''The restricted or unrestricted numint driver, as B requires.'''
        return ni.nr_uks if polarized else ni.nr_rks

    def get_vne_b(self, mol, kpt=None):
        '''V_ne[B] in the shared AO basis, from B's own get_hcore.'''
        if self._vne_b is None:
            self._vne_b = _vne_from_hcore(self.mf_b, self.mol_b, kpt)
        return self._vne_b

    def get_j_b(self, mf, mol):
        '''J[rho_B] in the shared AO basis.
        '''
        if self._j_b is None:
            if self.kpts is None:
                self._j_b = mf.get_j(mol, self.dm_b_total(), 1)
            else:
                # Pass kpts explicitly: GPU4PySCF's KSCF.get_j does not fall back
                # to mf.kpts when kpts is None, unlike PySCF's.
                self._j_b = mf.get_j(mol, self.dm_b, 1, self.kpts)
        return self._j_b

    def e_xc(self, ni, mol, grids, xc, max_memory):
        '''E_xc[rho_B].'''
        if self._e_xc is None:
            self._e_xc = self._nr(ni, self.polarized)(
                mol, grids, xc, self.dm_b, max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b(self, ni, mol, grids, t_nad, max_memory):
        '''T_s^TF[rho_B], the B term of the non-additive kinetic energy.
        '''
        if self._e_tnad_b is None:
            self._e_tnad_b = self._nr(ni, self.polarized)(
                mol, grids, t_nad, self.dm_b, max_memory=max_memory)[1]
        return self._e_tnad_b

    def e_xc_pbc(self, ni, cell, grids, xc, hermi, kpt, max_memory):
        '''E_xc[rho_B] for the periodic path.'''
        if self._e_xc is None:
            self._e_xc = self._nr(ni, self.polarized)(
                cell, grids, xc, self.dm_b, 0, hermi, kpt, None,
                max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b_pbc(self, ni, cell, grids, t_nad, hermi, kpt, max_memory):
        '''T_s^TF[rho_B] for the periodic path.'''
        if self._e_tnad_b is None:
            self._e_tnad_b = self._nr(ni, self.polarized)(
                cell, grids, t_nad, self.dm_b, 0, hermi, kpt, None,
                max_memory=max_memory)[1]
        return self._e_tnad_b

    def e_vne_a_rho_b(self, vne_a):
        '''<V_ne[A] | rho_B>, a constant of the embedded SCF.

        vne_a is supplied by the mixin, which is the only place with access to
        the unmodified get_hcore for subsystem A. It may be the matrix itself
        or a zero-argument callable returning it; the callable form is only
        invoked on a cache miss, which is once per SCF.
        '''
        if self._e_vne_a_rho_b is None:
            if callable(vne_a):
                vne_a = vne_a()
            trace = _trace_prod_spin if self.kpts is None else _kweight_trace
            self._e_vne_a_rho_b = trace(vne_a, self.dm_b)
        return self._e_vne_a_rho_b

