'''Periodic KSCED for restricted Kohn-Sham, supermolecular basis.

One get_veff body serves the gamma point (RKS) and k-points (KRKS). At the gamma
point every matrix is (nao, nao); with k-points it is (nk, nao, nao), and the
numint drivers already integrate a k-point density -- KNumInt.nr_rks averages
the density over k. The only other difference is the energy trace, which the
mixin's _trace_dm averages over k on a k-point object, as pyscf.pbc.dft.krks does.

energy_elec, energy_nuc and get_hcore are inherited from KSCEDMixin:
pbc/dft/rks.py assigns energy_elec = mol_ks.energy_elec, and krks.energy_elec
has the same form, so one energy expression serves every case.
'''

from pyscf import lib
from pyscf.ksced.common import KSCEDMixin, _as_like, _tag_array


def _get_veff(mf, cell, dm, hermi, k, kpts_band):
    '''J + V_xc for A at rho_A + rho_B, plus the non-additive kinetic potential.

    k is the gamma-point kpt (3,) or the k-point set (nk, 3).
    '''
    if cell is None:
        cell = mf.cell
    if dm is None:
        dm = mf.make_rdm1()
    if kpts_band is not None:
        raise NotImplementedError('KSCED does not support kpts_band')

    ni = mf._numint
    if ni.libxc.is_hybrid_xc(mf.xc):
        raise NotImplementedError(
            'KSCED supports pure functionals only; %s is a hybrid' % mf.xc)
    mf.initialize_grids(cell, dm, k)

    env = mf.with_env
    max_memory = mf.max_memory - lib.current_memory()[0]

    dm_a = dm
    dm_t = dm_a + _as_like(dm_a, env.dm_b)

    # Exchange-correlation at the total density.
    n, exc_t, vxc = ni.nr_rks(cell, mf.grids, mf.xc, dm_t, 0, hermi,
                              k, None, max_memory=max_memory)
    # Non-additive kinetic energy: T[rho] - T[rho_A] - T[rho_B].
    _, t_t, v_t_t = ni.nr_rks(cell, mf.grids, mf.t_nad, dm_t, 0, hermi,
                              k, None, max_memory=max_memory)
    _, t_a, v_t_a = ni.nr_rks(cell, mf.grids, mf.t_nad, dm_a, 0, hermi,
                              k, None, max_memory=max_memory)
    t_b = env.e_tnad_b_pbc(ni, cell, mf.grids, mf.t_nad, hermi, k, max_memory)
    mf.e_tnad = t_t - t_a - t_b

    vxc = vxc + v_t_t - v_t_a
    exc = exc_t + mf.e_tnad - env.e_xc_pbc(ni, cell, mf.grids, mf.xc,
                                           hermi, k, max_memory)

    vj = mf.get_j(cell, dm_t, hermi, k, None)
    vxc += vj

    # Half of J_AA plus half of J_AB; energy_elec adds the other half of J_AB.
    ecoul = mf._trace_dm(vj, dm_a) * .5

    return _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)


class KSCEDPBCRKS(KSCEDMixin):
    '''KSCED embedding for pyscf.pbc.dft.rks.RKS at the gamma point.'''

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
                 kpt=None, kpts_band=None):
        return _get_veff(self, cell, dm, hermi,
                         self.kpt if kpt is None else kpt, kpts_band)


class KSCEDKPBCRKS(KSCEDMixin):
    '''KSCED embedding for pyscf.pbc.dft.krks.KRKS (and GPU4PySCF's KRKS).'''

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
                 kpts=None, kpts_band=None):
        return _get_veff(self, cell, dm, hermi,
                         self.kpts if kpts is None else kpts, kpts_band)
