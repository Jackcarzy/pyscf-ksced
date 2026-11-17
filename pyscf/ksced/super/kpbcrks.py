'''Periodic k-point KSCED for restricted Kohn-Sham (KRKS), supermolecular basis.

Mirrors super/pbcrks.py term for term, with a k-point axis on every matrix:
dm, hcore, J and the exchange-correlation potential are (nk, nao, nao). The
numint drivers already integrate a k-point density matrix -- KNumInt.nr_rks sums
the density over k with weight 1/nk -- so the functionals need no change. What
does change is every trace that turns a matrix into an energy: like
pyscf.pbc.dft.krks, each is the average over k, (1/nk) sum_k tr(O_k D_k).

get_hcore, _vne_a and the k-averaged _trace_dm come from KSCEDKptMixin, shared
with the monomolecular k-point class. energy_elec and energy_nuc are inherited
from KSCEDMixin: energy_elec calls krks.energy_elec through super() and adds the
two KSCED cross terms through _trace_dm.
'''

from pyscf import lib
from pyscf.ksced.common import (KSCEDKptMixin, KSCEDMixin, _as_like,
                                _kweight_trace, _tag_array)


class KSCEDKPBCRKS(KSCEDKptMixin, KSCEDMixin):
    '''KSCED embedding for pyscf.pbc.dft.krks.KRKS (and GPU4PySCF's KRKS).'''

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
                 kpts=None, kpts_band=None):
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
        t_b = env.e_tnad_b_pbc(ni, cell, self.grids, self.t_nad, hermi, kpts,
                               max_memory)
        self.e_tnad = t_t - t_a - t_b

        vxc = vxc + v_t_t - v_t_a
        exc = exc_t + self.e_tnad - env.e_xc_pbc(ni, cell, self.grids, self.xc,
                                                 hermi, kpts, max_memory)

        vj = self.get_j(cell, dm_t, hermi, kpts, None)
        vxc += vj

        # Half of J_AA plus half of J_AB, k-averaged as in krks.get_veff;
        # energy_elec adds the other half of J_AB.
        ecoul = _kweight_trace(dm_a, vj) * .5

        vxc = _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)
        return vxc
