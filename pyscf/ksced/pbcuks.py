'''Periodic gamma-point KSCED for unrestricted Kohn-Sham.

Mirrors ksced/pbcrks.py term for term, with the same three spin adjustments as
the molecular unrestricted path in ksced/uks.py. energy_elec, energy_nuc and
get_hcore are inherited from KSCEDMixin: pbc/dft/uks.py assigns
energy_elec = pyscf.dft.uks.energy_elec, exactly as the restricted module
assigns the molecular restricted one, so a single energy expression again serves
both domains and only get_veff is periodic-specific.
'''

from pyscf import lib
from pyscf.ksced.ksced import (KSCEDMixin, _as_like, _as_pair, _avg_spin,
                               _spin_sum, _trace_prod,
                               _tag_array)


class KSCEDPBCUKS(KSCEDMixin):
    '''KSCED embedding for pyscf.pbc.dft.uks.UKS at the gamma point.'''

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
        self.initialize_grids(cell, dm, kpt)

        env = self.with_env
        max_memory = self.max_memory - lib.current_memory()[0]

        dm_a = _as_pair(dm)
        # Split a restricted rho_B evenly between spin channels.
        dm_t = dm_a + _as_like(dm_a, env.dm_b_for(True))

        # Exchange-correlation at the total density.
        n, exc_t, vxc = ni.nr_uks(cell, self.grids, self.xc, dm_t, 0, hermi,
                                  kpt, None, max_memory=max_memory)
        # Non-additive kinetic energy: T[rho] - T[rho_A] - T[rho_B].
        _, t_t, v_t_t = ni.nr_uks(cell, self.grids, self.t_nad, dm_t, 0, hermi,
                                  kpt, None, max_memory=max_memory)
        _, t_a, v_t_a = ni.nr_uks(cell, self.grids, self.t_nad, dm_a, 0, hermi,
                                  kpt, None, max_memory=max_memory)
        t_b = env.e_tnad_b_pbc(ni, cell, self.grids, self.t_nad, hermi, kpt,
                               max_memory)
        self.e_tnad = t_t - t_a - t_b

        vxc = vxc + v_t_t - v_t_a
        if self.a_restricted:
            # Fold spin-dependent terms before adding the spin-free J.
            vxc = _avg_spin(vxc)
        exc = exc_t + self.e_tnad - env.e_xc_pbc(ni, cell, self.grids, self.xc,
                                                 hermi, kpt, max_memory)

        # Build the spin-free J once; broadcasting adds it to both channels.
        vj = self.get_j(cell, _spin_sum(dm_t), hermi, kpt, None)
        vxc += vj

        # Half of J_AA plus half of J_AB; energy_elec adds the other half of J_AB.
        ecoul = _trace_prod(_spin_sum(dm_a), vj) * .5

        vxc = _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)
        return vxc


class KSCEDPBCRKSinU(KSCEDPBCUKS):
    '''Restricted periodic A in an unrestricted environment.'''

    a_restricted = True
