'''Periodic gamma-point KSCED in a monomolecular basis.

Mirrors ksced/pbcrks.py term for term. energy_elec, energy_nuc and get_hcore are
inherited from KSCEDMixin: pbc/dft/rks.py assigns energy_elec = mol_ks.energy_elec,
so one energy expression serves both domains and only get_veff is periodic.
'''

from pyscf import lib
from pyscf.ksced.ksced import _trace_prod, _tag_array
from pyscf.ksced.mb.arrays import like as _like
from pyscf.ksced.mb.common import KSCEDMBMixin


class KSCEDMBPBCRKS(KSCEDMBMixin):
    '''KSCED embedding for pyscf.pbc.dft.rks.RKS, monomolecular basis.'''

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
                 kpt=None, kpts_band=None):
        if cell is None:
            cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        if kpt is None:
            kpt = self.kpt
        if kpts_band is not None:
            raise NotImplementedError('KSCED is gamma-point only; kpts_band is '
                                      'not supported')

        ni = self._numint
        if ni.libxc.is_hybrid_xc(self.xc):
            raise NotImplementedError(
                'KSCED supports pure functionals only; %s is a hybrid' % self.xc)
        self._check_xc_types(ni)
        self.initialize_grids(cell, dm, kpt)

        env = self.with_env
        ni_t = env.numint_for(ni, kpt)
        max_memory = self.max_memory - lib.current_memory()[0]

        dm_a = dm

        # Evaluate XC at rho_A + rho_B. Validate that the backend used the
        # density hook, including when its integration call fails.
        try:
            n, exc_t, vxc = ni_t.nr_rks(cell, self.grids, self.xc, dm_a, 0,
                                        hermi, kpt, None,
                                        max_memory=max_memory)
        except Exception as exc:
            self._assert_env_density_entered(cause=exc)
            raise
        self._assert_env_density_entered()
        self._log_electron_counts(n)
        # Non-additive kinetic energy: T[rho] - T[rho_A] - T[rho_B].
        _, t_t, v_t_t = ni_t.nr_rks(cell, self.grids, self.t_nad, dm_a, 0, hermi,
                                    kpt, None, max_memory=max_memory)
        _, t_a, v_t_a = ni.nr_rks(cell, self.grids, self.t_nad, dm_a, 0, hermi,
                                  kpt, None, max_memory=max_memory)
        t_b = env.e_tnad_b_pbc(ni, cell, self.grids, self.t_nad, hermi, kpt,
                               max_memory)
        self.e_tnad = t_t - t_a - t_b

        vxc = vxc + v_t_t - v_t_a
        exc = exc_t + self.e_tnad - env.e_xc_pbc(ni, cell, self.grids, self.xc,
                                                 hermi, kpt, max_memory)

        # Build J[rho_A + rho_B] in A's basis and on A's array backend.
        vj_a = self.get_j(cell, dm_a, hermi, kpt, None)
        vj = vj_a + _like(vj_a, env.get_j_b(self, cell))
        vxc += vj

        # Half of J_AA plus half of J_AB; energy_elec adds the other half of J_AB.
        ecoul = _trace_prod(dm_a, vj) * .5

        vxc = _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)
        return vxc
