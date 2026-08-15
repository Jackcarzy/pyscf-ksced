'''Molecular unrestricted KSCED in a monomolecular basis.'''

from pyscf import lib
from pyscf.ksced.ksced import (_as_pair, _avg_spin, _spin_sum, _trace_prod,
                               _tag_array)
from pyscf.ksced.mb.arrays import like as _like
from pyscf.ksced.mb.common import KSCEDMBMixin
from pyscf.ksced.mb.rks import _grid_mol
from pyscf import dft


class KSCEDMBUKS(KSCEDMBMixin):
    '''KSCED embedding for molecular UKS, monomolecular basis.'''

    a_restricted = False

    def initialize_grids(self, mol=None, dm=None):
        if self.grids.coords is None:
            self.grids = dft.gen_grid.Grids(_grid_mol(self.with_env.mol_ab))
            self.grids.build()
        return self.grids

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        ni = self._numint
        if ni.libxc.is_hybrid_xc(self.xc):
            raise NotImplementedError(
                'KSCED supports pure functionals only; %s is a hybrid' % self.xc)
        self._check_xc_types(ni)
        self.initialize_grids(mol, dm)

        env = self.with_env
        ni_t = env.numint_for(ni)
        max_memory = self.max_memory - lib.current_memory()[0]
        dm_a = _as_pair(dm)

        try:
            n, exc_t, vxc = ni_t.nr_uks(
                mol, self.grids, self.xc, dm_a, max_memory=max_memory)
        except Exception as exc:
            self._assert_env_density_entered(cause=exc)
            raise
        self._assert_env_density_entered()
        self._log_electron_counts(n)
        _, t_t, v_t_t = ni_t.nr_uks(
            mol, self.grids, self.t_nad, dm_a, max_memory=max_memory)
        _, t_a, v_t_a = ni.nr_uks(
            mol, self.grids, self.t_nad, dm_a, max_memory=max_memory)
        t_b = env.e_tnad_b(ni, mol, self.grids, self.t_nad, max_memory)
        self.e_tnad = t_t - t_a - t_b

        vxc = vxc + v_t_t - v_t_a
        if self.a_restricted:
            vxc = _avg_spin(vxc)
        exc = exc_t + self.e_tnad - env.e_xc(
            ni, mol, self.grids, self.xc, max_memory)

        vj_a = self.get_j(mol, _spin_sum(dm_a), hermi)
        vj = vj_a + _like(vj_a, env.get_j_b(self, mol))
        vxc += vj
        ecoul = _trace_prod(_spin_sum(dm_a), vj) * .5
        return _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)


class KSCEDMBRKSinU(KSCEDMBUKS):
    '''Restricted molecular A in an unrestricted monomolecular environment.'''

    a_restricted = True

