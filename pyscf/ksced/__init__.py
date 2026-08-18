'''
Kohn-Sham Equations with Constrained Electron Density (KSCED) subsystem embedding.

Subsystem A is optimized in the frozen density of subsystem B.

Two basis modes, chosen with embed(..., basis_mode=):

  'S'  supermolecular, the default.
  'M'  monomolecular. 
'''

__version__ = '0.2.0'

from pyscf.ksced.ksced import embed

__all__ = ['embed']
