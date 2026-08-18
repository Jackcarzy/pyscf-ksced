#!/usr/bin/env python
'''
KSCED subsystem embedding for a molecular system.

H2O is the embedded subsystem A; Li+ is the frozen environment B. Both are built
in one supermolecular basis using ghost atoms, so that dm_a and dm_b share a
dimension and the non-additive terms can be formed by adding them.

The interaction energy of a two-fragment system is a two-term subtraction.
    Eint      = E[A in B] - E[A]
'''

from pyscf import gto, dft, ksced

#0 setup
mol_a = gto.M(
    verbose = 4,
    atom = '''
        o    0    0.       0.
        h    0    -0.757   0.587
        h    0    0.757    0.587
        x-li   0    0        -2''',
    basis = '6-31g')

mol_b = gto.M(
    verbose = 4,
    atom = '''
        x-o    0    0.       0.
        x-h    0    -0.757   0.587
        x-h    0    0.757    0.587
          li   0    0        -2''',
    charge = 1,
    basis = '6-31g')

mol_ab = gto.M(
    verbose = 4,
    atom = '''
        o    0    0.       0.
        h    0    -0.757   0.587
        h    0    0.757    0.587
        li   0    0        -2''',
    charge = 1,
    basis = '6-31g')

#1 embedding B (Li+)
mf_b = dft.RKS(mol_b, xc='PBE').run()

#2 A alone (H2O)
mf_a = dft.RKS(mol_a, xc='PBE')
mf_a.kernel()

#3 embedded A (H2O in Li+)
mf_ainb = ksced.embed(mf_a, mf_b, mol_ab=mol_ab)
mf_ainb.t_nad = 'LDA_K_TF'
mf_ainb.kernel()

#4 get interaction energy (kcal/mol)
eint = mf_ainb.e_tot - mf_a.e_tot
print('embedded total energy       %.10f Ha' % mf_ainb.e_tot)
print('non-additive kinetic energy %.10f Ha' % mf_ainb.e_tnad)
print('interaction energy          %.10f Ha  = %.3f kcal/mol'
      % (eint, eint * 627.503))