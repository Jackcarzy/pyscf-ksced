#!/usr/bin/env python
'''
KSCED subsystem embedding in a MONOMOLECULAR basis, molecular case.

Diff this against 00-Li+_H2O.py. Two things change:

  - no ghost atoms anywhere: A carries only H2O's functions, B only Li+'s
  - basis_mode='M'

and one thing follows from them: the isolated-A reference is plain mol_a rather
than the ghost-laden counterpoise cell, because A never had access to B's
functions and so cannot suffer basis set superposition error.

mol_ab is built automatically by concatenating mol_a and mol_b, so the A-B
nuclear repulsion is always included in e_tot and Eint stays a plain
subtraction. Note that mol_ab means something different here than it does in
the supermolecular path: there it is the real whole system, supplied only so
E_nn[AB] can be added; here every cross term is a slice of a matrix built at
the AB dimension, so it must be the concatenation A(+)B.
'''

from pyscf import gto, dft, ksced

COMMON = dict(basis='6-31g', verbose=4)
GEOM_A = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'
GEOM_B = 'Li 0 0 -1.5'

mol_a = gto.M(atom=GEOM_A, **COMMON)
mol_b = gto.M(atom=GEOM_B, charge=1, **COMMON)

print('nao_A = %d, nao_B = %d, nao_AB = %d'
      % (mol_a.nao, mol_b.nao, mol_a.nao + mol_b.nao))

# Converge the frozen environment first, in B's own basis.
mf_b = dft.RKS(mol_b, xc='PBE').run()

mf_a = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b, basis_mode='M')
mf_a.t_nad = 'LDA_K_TF'
mf_a.kernel()

# The isolated fragment, in the same monomolecular basis. No ghosts needed.
mf_a_alone = dft.RKS(mol_a, xc='PBE').run()

eint = mf_a.e_tot - mf_a_alone.e_tot
print('embedded total energy       %.10f Ha' % mf_a.e_tot)
print('non-additive kinetic energy %.10f Ha' % mf_a.e_tnad)
print('interaction energy          %.10f Ha  = %.3f kcal/mol'
      % (eint, eint * 627.503))
