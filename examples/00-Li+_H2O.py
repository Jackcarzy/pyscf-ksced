#!/usr/bin/env python
'''
KSCED subsystem embedding for a molecular system.

Li+ is the frozen environment B; H2O is the embedded subsystem A. Both are built
in one supermolecular basis using ghost atoms, so that dm_a and dm_b share a
dimension and the non-additive terms can be formed by adding them.

The interaction energy of a two-fragment system is a two-term subtraction,
because the plugin's e_tot already excludes B's own energy:

    E[A in B] = E_total[complex] - E[B]
    Eint      = E[A in B] - E[A isolated]
'''

from pyscf import gto, dft, ksced

COMMON = dict(basis='6-31g', verbose=4)
GEOM_A = 'ghost-Li 0 0 -1.5; O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'
GEOM_B = 'Li 0 0 -1.5; ghost-O 0 0 0; ghost-H 0 -0.757 0.587; ghost-H 0 0.757 0.587'
GEOM_AB = 'Li 0 0 -1.5; O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'

mol_a = gto.M(atom=GEOM_A, **COMMON)
mol_b = gto.M(atom=GEOM_B, charge=1, **COMMON)
mol_ab = gto.M(atom=GEOM_AB, charge=1, **COMMON)

# Converge the frozen environment first. embed() refuses an environment that has
# no density, so this step is not optional.
mf_b = dft.RKS(mol_b, xc='PBE').run()

# Embed A in B's frozen density. Passing mol_ab folds the A-B nuclear repulsion
# into e_tot, which is what makes Eint a plain subtraction below.
mf_a = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b, mol_ab=mol_ab)
mf_a.t_nad = 'LDA_K_TF'
mf_a.kernel()

# The isolated fragment, in the same supermolecular basis so the subtraction is
# counterpoise corrected.
mf_a_alone = dft.RKS(mol_a, xc='PBE').run()

eint = mf_a.e_tot - mf_a_alone.e_tot
print('embedded total energy       %.10f Ha' % mf_a.e_tot)
print('non-additive kinetic energy %.10f Ha' % mf_a.e_tnad)
print('interaction energy          %.10f Ha  = %.3f kcal/mol'
      % (eint, eint * 627.503))
