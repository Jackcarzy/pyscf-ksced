'''Periodic KSCED for restricted Kohn-Sham with k-points (KRKS), monomolecular basis.

    pyscf.ksced.mono.pbcrks : the same at the gamma point (RKS)

Mirrors mono/pbcrks.py with kpts, as pyscf.pbc.dft.krks mirrors rks. rho_B is a
real-space density, so the offset numint, the mesh density, v_J[rho_B] and the
functional energies of B are the same objects as at the gamma point; the
environment builds them from B's k-point density matrix.
energy_elec, energy_nuc and get_hcore are inherited from KSCEDMixin.
'''

from pyscf import lib
from pyscf.ksced.common import _tag_array
from pyscf.ksced.arrays import like as _like
from pyscf.ksced.mono.mixin import KSCEDMBMixin


class KSCEDMBKPBCRKS(KSCEDMBMixin):
    '''KSCED embedding for pyscf.pbc.dft.krks.KRKS, monomolecular basis.'''

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
                 kpts=None, kpts_band=None):
        '''J + V_xc for A at rho_A + rho_B, plus the non-additive kinetic potential.'''
        if cell is None:
            cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        if kpts is None:
            kpts = self.kpts
        if kpts_band is not None:
            raise NotImplementedError('KSCED does not support kpts_band')

        ni = self._numint
        if ni.libxc.is_hybrid_xc(self.xc):
            raise NotImplementedError(
                'KSCED supports pure functionals only; %s is a hybrid' % self.xc)
        self._check_xc_types(ni)
        self.initialize_grids(cell, dm, kpts)

        env = self.with_env
        ni_t = env.numint_for(ni, kpts)
        max_memory = self.max_memory - lib.current_memory()[0]

        dm_a = dm

        # Evaluate XC at rho_A + rho_B. Validate that the backend used the
        # density hook, including when its integration call fails.
        try:
            n, exc_t, vxc = ni_t.nr_rks(cell, self.grids, self.xc, dm_a, 0,
                                        hermi, kpts, None, max_memory=max_memory)
        except Exception as exc:
            self._assert_env_density_entered(cause=exc)
            raise
        self._assert_env_density_entered()
        self._log_electron_counts(n)
        # Non-additive kinetic energy: T[rho] - T[rho_A] - T[rho_B].
        _, t_t, v_t_t = ni_t.nr_rks(cell, self.grids, self.t_nad, dm_a, 0, hermi,
                                    kpts, None, max_memory=max_memory)
        _, t_a, v_t_a = ni.nr_rks(cell, self.grids, self.t_nad, dm_a, 0, hermi,
                                  kpts, None, max_memory=max_memory)
        t_b = env.e_tnad_b_pbc(ni, cell, self.grids, self.t_nad, hermi, kpts, max_memory)
        self.e_tnad = t_t - t_a - t_b

        vxc = vxc + v_t_t - v_t_a
        exc = exc_t + self.e_tnad - env.e_xc_pbc(ni, cell, self.grids, self.xc,
                                                 hermi, kpts, max_memory)

        # Build J[rho_A + rho_B] in A's basis and on A's array backend.
        vj_a = self.get_j(cell, dm_a, hermi, kpts, None)
        vj = vj_a + _like(vj_a, env.get_j_b(self, cell))
        vxc += vj

        # Half of J_AA plus half of J_AB; energy_elec adds the other half of J_AB.
        ecoul = self._trace_dm(vj, dm_a) * .5

        return _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)
