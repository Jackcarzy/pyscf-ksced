'''Periodic KSCED for restricted Kohn-Sham with k-points (KRKS), supermolecular basis.

    pyscf.ksced.super.pbcrks : the same at the gamma point (RKS)

Mirrors super/pbcrks.py with kpts, as pyscf.pbc.dft.krks mirrors rks. Every
matrix is (nk, nao, nao); KNumInt.nr_rks already averages the density over k,
and the mixin's _trace_dm averages each energy trace over k.
'''

from pyscf import lib
from pyscf.ksced.common import KSCEDMixin, _as_like, _tag_array


class KSCEDKPBCRKS(KSCEDMixin):
    '''KSCED embedding for pyscf.pbc.dft.krks.KRKS (and GPU4PySCF's KRKS).'''

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
        self.initialize_grids(cell, dm, kpts)

        env = self.with_env
        max_memory = self.max_memory - lib.current_memory()[0]

        dm_a = dm
        dm_t = dm_a + _as_like(dm_a, env.dm_b)

        # Exchange-correlation at the total density.
        n, exc_t, vxc = ni.nr_rks(cell, self.grids, self.xc, dm_t, 0, hermi,
                                  kpts, None, max_memory=max_memory)
        # Non-additive kinetic energy: T[rho] - T[rho_A] - T[rho_B].
        _, t_t, v_t_t = ni.nr_rks(cell, self.grids, self.t_nad, dm_t, 0, hermi,
                                  kpts, None, max_memory=max_memory)
        _, t_a, v_t_a = ni.nr_rks(cell, self.grids, self.t_nad, dm_a, 0, hermi,
                                  kpts, None, max_memory=max_memory)
        t_b = env.e_tnad_b_pbc(ni, cell, self.grids, self.t_nad, hermi, kpts, max_memory)
        self.e_tnad = t_t - t_a - t_b

        vxc = vxc + v_t_t - v_t_a
        exc = exc_t + self.e_tnad - env.e_xc_pbc(ni, cell, self.grids, self.xc,
                                                 hermi, kpts, max_memory)

        vj = self.get_j(cell, dm_t, hermi, kpts, None)
        vxc += vj

        # Half of J_AA plus half of J_AB; energy_elec adds the other half of J_AB.
        ecoul = self._trace_dm(vj, dm_a) * .5

        return _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)
