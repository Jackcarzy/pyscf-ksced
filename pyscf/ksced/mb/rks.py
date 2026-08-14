'''Molecular KSCED in a monomolecular basis.

Mirrors ksced/rks.py term for term. Two things differ: the total density is
formed on the grid by the offset numint rather than by adding density matrices,
and the Coulomb potential is assembled from a nao_A build plus the cached AB
slice rather than from one build at the shared dimension.
'''

import numpy

from pyscf import dft, lib
from pyscf.ksced.ksced import _trace_prod, _tag_array
from pyscf.ksced.mb.arrays import like as _like
from pyscf.ksced.mb.common import KSCEDMBMixin


def _grid_mol(mol):
    '''mol with coincident centres collapsed, for Becke grid generation.

    mol_ab is a concatenation, so a centre carried by both subsystems appears
    twice at the same coordinates -- a real ghost in one and a real nucleus in
    the other. Becke partitioning divides by the internuclear distance, so a
    duplicated site produces NaN in every grid weight and the Fock matrix comes
    out non-finite.

    Duplicated *basis functions* are harmless, because nothing is ever inverted
    or diagonalised at the AB dimension; duplicated *grid centres* are not.
    This matters beyond the supermolecular-limit test: extended MB, where
    cell_a carries boundary atoms of B as ghosts, produces exactly this
    geometry in ordinary use.

    Returns mol unchanged when there are no duplicates, which is the common
    monomolecular case.
    '''
    coords = mol.atom_coords()
    keep = []
    for i in range(mol.natm):
        dup = next((k for k, j in enumerate(keep)
                    if numpy.allclose(coords[j], coords[i], atol=1e-8)), None)
        if dup is None:
            keep.append(i)
        elif mol.atom_charge(keep[dup]) == 0 and mol.atom_charge(i) != 0:
            # Prefer the real nucleus over a coincident ghost: the Bragg radius
            # that sizes the atomic grid comes from the element.
            keep[dup] = i

    if len(keep) == mol.natm:
        return mol

    sub = mol.copy()
    sub.atom = [(mol.atom_symbol(i), coords[i]) for i in keep]
    sub.unit = 'Bohr'
    sub.build(dump_input=False, parse_arg=False)
    return sub


class KSCEDMBRKS(KSCEDMBMixin):
    '''KSCED embedding for pyscf.dft.rks.RKS, monomolecular basis.'''

    def initialize_grids(self, mol=None, dm=None):
        '''Integrate on the supermolecular Becke grid.

        mol_a's own grid is centred on A's atoms and samples rho_B poorly where
        B's nuclei are. Grid cost scales with atom count, not nao squared, so
        using the AB grid costs little and keeps the nao_A speedup intact.
        '''
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

        dm_a = dm

        # Exchange-correlation at the total density; rho_B is added on the grid.
        # n counts N_A + N_B because the offset numint integrates rho_t; it is
        # reported separately and never returned to the stock grid warning.
        n, exc_t, vxc = ni_t.nr_rks(mol, self.grids, self.xc, dm_a,
                                    max_memory=max_memory)
        self._assert_env_density_entered()
        self._log_electron_counts(n)
        # Non-additive kinetic energy: T[rho] - T[rho_A] - T[rho_B].
        _, t_t, v_t_t = ni_t.nr_rks(mol, self.grids, self.t_nad, dm_a,
                                    max_memory=max_memory)
        _, t_a, v_t_a = ni.nr_rks(mol, self.grids, self.t_nad, dm_a,
                                  max_memory=max_memory)
        t_b = env.e_tnad_b(ni, mol, self.grids, self.t_nad, max_memory)
        self.e_tnad = t_t - t_a - t_b

        vxc = vxc + v_t_t - v_t_a
        exc = exc_t + self.e_tnad - env.e_xc(ni, mol, self.grids, self.xc,
                                             max_memory)

        # J[rho_total] in A's basis: A's own build plus the cached AB slice.
        vj_a = self.get_j(mol, dm_a, hermi)
        vj = vj_a + _like(vj_a, env.get_j_b(self, mol))
        vxc += vj

        # Half of J_AA plus half of J_AB; energy_elec adds the other half of J_AB.
        ecoul = _trace_prod(dm_a, vj) * .5

        vxc = _tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)
        return vxc
