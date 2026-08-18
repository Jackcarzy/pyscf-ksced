'''Molecular KSCED for unrestricted Kohn-Sham.

Mirrors ksced/rks.py term for term. Three things differ:

  - the numint driver is nr_uks, whose vmat is (2, nao, nao) and whose nelec is
    (2,) while excsum stays a scalar
  - the Coulomb potential is built from the spin-summed density and stays 2-D,
    broadcasting over both channels when added to vxc
  - rho_B must arrive in A's spin layout: a restricted environment is halved
    into each channel, never added whole to both

The non-additive kinetic term needs no special handling. libxc applies the
Thomas-Fermi spin-scaling relation itself, so nr_uks with LDA_K_TF returns
0.5*(T[2 rho_a] + T[2 rho_b]) and collapses to the restricted value when the
two channels agree.
'''

from pyscf import lib
from pyscf.ksced.ksced import (KSCEDMixin, _as_like, _as_pair, _avg_spin,
                               _spin_sum, _trace_prod,
                               _tag_array)


class KSCEDUKS(KSCEDMixin):
    '''KSCED embedding for pyscf.dft.uks.UKS.'''

    a_restricted = False

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

        dm_a = _as_pair(dm)
        # Split a restricted rho_B evenly between spin channels.
        dm_t = dm_a + _as_like(dm_a, env.dm_b_for(True))

        # Exchange-correlation at the total density.
        n, exc_t, vxc = ni.nr_uks(mol, self.grids, self.xc, dm_t,
                                  max_memory=max_memory)
        # Non-additive kinetic energy: T[rho] - T[rho_A] - T[rho_B].
        _, t_t, v_t_t = ni.nr_uks(mol, self.grids, self.t_nad, dm_t,
                                  max_memory=max_memory)
        _, t_a, v_t_a = ni.nr_uks(mol, self.grids, self.t_nad, dm_a,
                                  max_memory=max_memory)
        t_b = env.e_tnad_b(ni, mol, self.grids, self.t_nad, max_memory)
        self.e_tnad = t_t - t_a - t_b

        vxc = vxc + v_t_t - v_t_a
        if self.a_restricted:
            # Fold spin-dependent terms before adding the spin-free J.
            vxc = _avg_spin(vxc)
        exc = exc_t + self.e_tnad - env.e_xc(ni, mol, self.grids, self.xc,
                                             max_memory)

        # Build the spin-free J once; broadcasting adds it to both channels.
        vj = self.get_j(mol, _spin_sum(dm_t), hermi)
        vxc += vj

        # Half of J_AA plus half of J_AB; energy_elec adds the other half of J_AB.
        ecoul = _trace_prod(_spin_sum(dm_a), vj) * .5

        vxc = _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)
        return vxc


class KSCEDRKSinU(KSCEDUKS):
    '''Restricted A in an unrestricted environment.

    Both spin potentials are evaluated, then their average is supplied to the
    single restricted Fock build. Energies remain fully spin resolved.
    '''

    a_restricted = True
