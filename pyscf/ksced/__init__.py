'''
Kohn-Sham Equations with Constrained Electron Density (KSCED) subsystem embedding.

Subsystem A is optimised in the frozen density of subsystem B.

Two basis modes, chosen with embed(..., basis_mode=):

  'S'  supermolecular, the default. A and B share one AO basis built with ghost
       atoms, so their density matrices have the same dimension.
  'M'  monomolecular. A carries only A's functions and B only B's; the frozen
       density reaches A through the quadrature grid. This is what makes the
       embedded SCF smaller than the whole system.
'''

__version__ = '0.2.0'

from pyscf.ksced.ksced import embed

__all__ = ['embed']
