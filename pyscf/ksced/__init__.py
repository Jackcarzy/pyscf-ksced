'''
Kohn-Sham Equations with Constrained Electron Density (KSCED) subsystem embedding.

Subsystem A is optimized in the frozen density of subsystem B.

Two basis modes, chosen with embed(..., basis_mode=):

  'S'  supermolecular, the default.
  'M'  monomolecular.

For a run that moves subsystem A -- a scan, an optimization, a trajectory --
build the environment once with frozen_env() and pass it to each embed(). On a
periodic mesh that keeps the frozen density and its Hartree potential across
geometries instead of rebuilding both at every point.
'''

__version__ = '0.2.0'

from pyscf.ksced.ksced import embed, frozen_env

__all__ = ['embed', 'frozen_env']
