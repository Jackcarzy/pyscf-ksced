'''Periodic KSCED for restricted Kohn-Sham, monomolecular basis.

Mirrors super/pbcrks.py: one get_veff body serves the gamma point (RKS) and
k-points (KRKS). rho_B is a real-space density, so the offset numint, the mesh
density, v_J[rho_B] and the functional energies of B are the same objects in
both cases; the environment builds them from B's density matrix, gamma-point or
k-point. energy_elec, energy_nuc and get_hcore are inherited from KSCEDMixin.
'''

from pyscf import lib
from pyscf.ksced.common import _tag_array
from pyscf.ksced.arrays import like as _like
from pyscf.ksced.mono.mixin import KSCEDMBMixin


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
    mf._check_xc_types(ni)
    mf.initialize_grids(cell, dm, k)

    env = mf.with_env
    ni_t = env.numint_for(ni, k)
    max_memory = mf.max_memory - lib.current_memory()[0]

    dm_a = dm

    # Evaluate XC at rho_A + rho_B. Validate that the backend used the
    # density hook, including when its integration call fails.
    try:
        n, exc_t, vxc = ni_t.nr_rks(cell, mf.grids, mf.xc, dm_a, 0,
                                    hermi, k, None, max_memory=max_memory)
    except Exception as exc:
        mf._assert_env_density_entered(cause=exc)
        raise
    mf._assert_env_density_entered()
    mf._log_electron_counts(n)
    # Non-additive kinetic energy: T[rho] - T[rho_A] - T[rho_B].
    _, t_t, v_t_t = ni_t.nr_rks(cell, mf.grids, mf.t_nad, dm_a, 0, hermi,
                                k, None, max_memory=max_memory)
    _, t_a, v_t_a = ni.nr_rks(cell, mf.grids, mf.t_nad, dm_a, 0, hermi,
                              k, None, max_memory=max_memory)
    t_b = env.e_tnad_b_pbc(ni, cell, mf.grids, mf.t_nad, hermi, k, max_memory)
    mf.e_tnad = t_t - t_a - t_b

    vxc = vxc + v_t_t - v_t_a
    exc = exc_t + mf.e_tnad - env.e_xc_pbc(ni, cell, mf.grids, mf.xc,
                                           hermi, k, max_memory)

    # Build J[rho_A + rho_B] in A's basis and on A's array backend.
    vj_a = mf.get_j(cell, dm_a, hermi, k, None)
    vj = vj_a + _like(vj_a, env.get_j_b(mf, cell))
    vxc += vj

    # Half of J_AA plus half of J_AB; energy_elec adds the other half of J_AB.
    ecoul = mf._trace_dm(vj, dm_a) * .5

    return _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)


class KSCEDMBPBCRKS(KSCEDMBMixin):
    '''KSCED embedding for pyscf.pbc.dft.rks.RKS, monomolecular basis.'''

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
                 kpt=None, kpts_band=None):
        return _get_veff(self, cell, dm, hermi,
                         self.kpt if kpt is None else kpt, kpts_band)


class KSCEDMBKPBCRKS(KSCEDMBMixin):
    '''KSCED embedding for pyscf.pbc.dft.krks.KRKS, monomolecular basis.'''

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
                 kpts=None, kpts_band=None):
        return _get_veff(self, cell, dm, hermi,
                         self.kpts if kpts is None else kpts, kpts_band)
