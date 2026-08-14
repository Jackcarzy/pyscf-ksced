'''Molecular KSCED for restricted Kohn-Sham.'''

import numpy
from pyscf import lib
from pyscf.ksced.ksced import (KSCEDMixin, _as_like, _trace_prod,
                               _tag_array)


class KSCEDRKS(KSCEDMixin):
    '''KSCED embedding for pyscf.dft.rks.RKS.

    Mirrors pyscf.dft.rks.get_veff, adding the non-additive kinetic term and
    evaluating the exchange-correlation contribution at the total density.
    Hybrid functionals are rejected: the exact-exchange split for subsystems is
    not defined here.
    '''

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()

        ni = self._numint
        if ni.libxc.is_hybrid_xc(self.xc):
            raise NotImplementedError(
                'KSCED supports pure functionals only; %s is a hybrid' % self.xc)
        if self.grids.coords is None:
            self.initialize_grids(mol, dm)

        env = self.with_env
        max_memory = self.max_memory - lib.current_memory()[0]

        dm_a = dm
        dm_t = dm_a + _as_like(dm_a, env.dm_b)

        # Exchange-correlation at the total density.
        n, exc_t, vxc = ni.nr_rks(mol, self.grids, self.xc, dm_t,
                                  max_memory=max_memory)
        # Non-additive kinetic energy: T[rho] - T[rho_A] - T[rho_B].
        _, t_t, v_t_t = ni.nr_rks(mol, self.grids, self.t_nad, dm_t,
                                  max_memory=max_memory)
        _, t_a, v_t_a = ni.nr_rks(mol, self.grids, self.t_nad, dm_a,
                                  max_memory=max_memory)
        t_b = env.e_tnad_b(ni, mol, self.grids, self.t_nad, max_memory)
        self.e_tnad = t_t - t_a - t_b

        vxc = vxc + v_t_t - v_t_a
        exc = exc_t + self.e_tnad - env.e_xc(ni, mol, self.grids, self.xc,
                                             max_memory)

        vj = self.get_j(mol, dm_t, hermi)
        vxc += vj

        # Half of J_AA plus half of J_AB; energy_elec adds the other half of J_AB.
        ecoul = _trace_prod(dm_a, vj) * .5

        vxc = _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)
        return vxc
