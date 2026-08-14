#!/usr/bin/env python
'''
KSCED subsystem embedding for a periodic system at the gamma point.

A four-atom gold cluster is the frozen environment B; an SC fragment standing in
for a thiol adsorbate is the embedded subsystem A. Both cells carry all six
centres and differ only in which are ghosts, so dm_a and dm_b share a dimension.

Gaussian smearing on the environment mirrors the production Au120 workflow: a
metallic environment does not converge without it.

The call is identical to the molecular case; embed() dispatches on whether the
system is a Cell.
'''

import numpy
from pyscf.pbc import gto, dft, scf
from pyscf import ksced

GEOM = [
    ('Au', (0.00, 0.00, 0.00)),
    ('Au', (2.88, 0.00, 0.00)),
    ('Au', (0.00, 2.88, 0.00)),
    ('Au', (2.88, 2.88, 0.00)),
    ('S',  (1.44, 1.44, 2.40)),
    ('C',  (1.44, 1.44, 4.20)),
]
ENV = [0, 1, 2, 3]        # gold is the frozen environment
SUB = [4, 5]              # the adsorbate is the embedded subsystem

COMMON = dict(a=numpy.diag([5.76, 5.76, 18.0]),
              basis='gth-dzvp-molopt-sr', pseudo='gth-pbe',
              mesh=[20, 20, 60], verbose=4)


def build(real_indices):
    '''Every centre is present; those outside real_indices become ghosts.'''
    atoms = []
    for i, (sym, xyz) in enumerate(GEOM):
        label = sym if i in real_indices else 'ghost-' + sym
        atoms.append((label, xyz))
    return gto.M(atom=atoms, **COMMON)


cell_b = build(ENV)
cell_a = build(SUB)
cell_ab = build(ENV + SUB)

# Converge the frozen metallic environment.
mf_b = dft.RKS(cell_b, xc='PBE')
mf_b = scf.addons.smearing_(mf_b, sigma=0.003, method='gauss')
mf_b.conv_tol = 1e-6
mf_b.kernel()

# Embed the adsorbate.
mf_a = ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b, mol_ab=cell_ab)
mf_a.t_nad = 'LDA_K_TF'
mf_a.conv_tol = 1e-6
mf_a.kernel()

print('embedded total energy       %.10f Ha' % mf_a.e_tot)
print('non-additive kinetic energy %.10f Ha' % mf_a.e_tnad)
