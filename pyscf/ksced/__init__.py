'''
Kohn-Sham Equations with Constrained Electron Density (KSCED) subsystem embedding.

Subsystem A is optimised in the frozen density of subsystem B. A and B must share
one supermolecular AO basis, built with ghost atoms, so that their density matrices
have the same dimension.
'''

__version__ = '0.1.0'

from pyscf.ksced.ksced import embed

__all__ = ['embed']
