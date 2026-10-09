'''Shared molecular quadrature and native GPU density integration.'''

import copy
import numpy as np
from pyscf.ksced.ksced import _as_pair
from pyscf.ksced.mb.arrays import to_host


def is_gpu(mf):
    return type(mf._numint).__module__.startswith('gpu4pyscf')


def initialize_grids(mf):
    from pyscf.ksced.mb.rks import _grid_mol
    if mf.grids.coords is None or mf.grids.mol is mf.mol:
        # Keep the backend and all explicit quadrature settings.
        mf.grids = copy.copy(mf.grids)
        mf.grids.reset(_grid_mol(mf.with_env.mol_ab))
        mf.grids.build()
    return mf.grids


def padded_densities(mf, dm, xp=np):
    env = mf.with_env
    polarized = dm.ndim == 3 or env.polarized
    a, b = dm, env.dm_b
    if polarized:
        a, b = _as_pair(a), _as_pair(b)
    shape = ((2,) if polarized else ()) + (env.mol_ab.nao,) * 2
    da, db = xp.zeros(shape), xp.zeros(shape)
    n = env.nao_a
    da[..., :n, :n] = xp.asarray(to_host(a)) if xp is np else xp.asarray(a)
    db[..., n:, n:] = xp.asarray(to_host(b)) if xp is np else xp.asarray(b)
    return da, db


def gpu_functionals(mf, dm):
    '''Exact AB density representation with an A-sized returned potential.'''
    import cupy as cp
    env = mf.with_env
    if getattr(env, '_gpu_mol_numint', None) is None:
        env._gpu_mol_numint = copy.copy(mf._numint)
        env._gpu_mol_numint.gdftopt = None
    ni = env._gpu_mol_numint
    da, db = padded_densities(mf, dm, cp)
    nr = ni.nr_uks if da.ndim == 3 else ni.nr_rks
    args = (env.mol_ab, mf.grids)
    n, ex, vx = nr(*args, mf.xc, da + db)
    _, tt, vt = nr(*args, mf.t_nad, da + db)
    _, ta, va = nr(*args, mf.t_nad, da)
    if env._e_xc is None:
        env._e_xc = nr(*args, mf.xc, db)[1]
    if env._e_tnad_b is None:
        env._e_tnad_b = nr(*args, mf.t_nad, db)[1]
    mf.e_tnad = tt - ta - env._e_tnad_b
    v = (vx + vt - va)[..., :env.nao_a, :env.nao_a]
    if dm.ndim == 2 and v.ndim == 3:
        v = (v[0] + v[1]) * .5
    mf._log_electron_counts(n)
    return ex + mf.e_tnad - env._e_xc, v
