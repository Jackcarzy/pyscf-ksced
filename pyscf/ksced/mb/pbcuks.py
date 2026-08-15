'''Periodic unrestricted KSCED in a monomolecular basis.'''

from pyscf import lib
from pyscf.ksced.ksced import (_as_pair, _avg_spin, _spin_sum, _trace_prod,
                               _tag_array)
from pyscf.ksced.mb.arrays import like as _like
from pyscf.ksced.mb.common import KSCEDMBMixin


class KSCEDMBPBCUKS(KSCEDMBMixin):
    '''KSCED embedding for periodic UKS, monomolecular basis.'''

    a_restricted = False

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
        dm_a = _as_pair(dm)

        try:
            n, exc_t, vxc = ni_t.nr_uks(
                cell, self.grids, self.xc, dm_a, 0, hermi, kpt, None,
                max_memory=max_memory)
        except Exception as exc:
            self._assert_env_density_entered(cause=exc)
            raise
        self._assert_env_density_entered()
        self._log_electron_counts(n)
        _, t_t, v_t_t = ni_t.nr_uks(
            cell, self.grids, self.t_nad, dm_a, 0, hermi, kpt, None,
            max_memory=max_memory)
        _, t_a, v_t_a = ni.nr_uks(
            cell, self.grids, self.t_nad, dm_a, 0, hermi, kpt, None,
            max_memory=max_memory)
        t_b = env.e_tnad_b_pbc(
            ni, cell, self.grids, self.t_nad, hermi, kpt, max_memory)
        self.e_tnad = t_t - t_a - t_b

        vxc = vxc + v_t_t - v_t_a
        if self.a_restricted:
            vxc = _avg_spin(vxc)
        exc = exc_t + self.e_tnad - env.e_xc_pbc(
            ni, cell, self.grids, self.xc, hermi, kpt, max_memory)

        vj_a = self.get_j(cell, _spin_sum(dm_a), hermi, kpt, None)
        vj = vj_a + _like(vj_a, env.get_j_b(self, cell))
        vxc += vj
        ecoul = _trace_prod(_spin_sum(dm_a), vj) * .5
        return _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)


class KSCEDMBPBCRKSinU(KSCEDMBPBCUKS):
    '''Restricted periodic A in an unrestricted monomolecular environment.'''

    a_restricted = True

