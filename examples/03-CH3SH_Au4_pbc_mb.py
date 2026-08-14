#!/usr/bin/env python
'''
KSCED subsystem embedding in a MONOMOLECULAR basis, periodic case.

Diff this against 02-CH3SH_Au4_pbc.py. The gold cluster is the frozen
environment and carries only gold functions; the adsorbate carries only its own.
Both cells keep the same lattice and the same mesh, which embed() asserts --
the uniform grids must be the same points for rho_B to transfer.

Smearing stays on both subsystems, so a comparison against basis_mode='S'
differs only in basis and not in SCF protocol.
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


def build(indices):
    '''Only the listed centres exist. No ghosts -- that is the point.'''
    return gto.M(atom=[GEOM[i] for i in indices], **COMMON)


cell_b = build(ENV)
cell_a = build(SUB)

print('nao_A = %d, nao_B = %d, nao_AB = %d'
      % (cell_a.nao, cell_b.nao, cell_a.nao + cell_b.nao))

# Converge the frozen metallic environment. In a monomolecular basis this cell
# knows nothing about the adsorbate, so it can be reused for every geometry.
mf_b = dft.RKS(cell_b, xc='PBE')
mf_b = scf.addons.smearing_(mf_b, sigma=0.003, method='gauss')
mf_b.conv_tol = 1e-6
mf_b.kernel()

mf_a = ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b, basis_mode='M')
mf_a.t_nad = 'LDA_K_TF'
mf_a.conv_tol = 1e-6
mf_a.kernel()

print('embedded total energy       %.10f Ha' % mf_a.e_tot)
print('non-additive kinetic energy %.10f Ha' % mf_a.e_tnad)
