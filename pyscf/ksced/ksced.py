'''KSCED subsystem embedding: the public entry points and class dispatch.'''

import numpy
from pyscf import lib

from pyscf.ksced.common import _KSCED, _is_kscf
from pyscf.ksced.super.env import _FrozenEnv


def _is_pbc(mf):
    from pyscf.pbc.gto import Cell
    return isinstance(mf.mol, Cell)


def _check_numint(mf):
    '''KSCED needs the plain grid-based NumInt; MultiGrid takes a different path.
    '''
    if not _is_pbc(mf):
        return
    try:
        from pyscf.pbc.dft import multigrid
    except ImportError:
        return

    numint_cls = getattr(multigrid, 'MultiGridNumInt', None)
    if numint_cls is not None and isinstance(mf._numint, numint_cls):
        raise NotImplementedError(
            'KSCED does not support MultiGridNumInt. Build the RKS object '
            'without multigrid acceleration.')

    df_cls = getattr(multigrid, 'MultiGridFFTDF', None)
    if df_cls is not None and isinstance(getattr(mf, 'with_df', None), df_cls):
        raise NotImplementedError(
            'KSCED does not support MultiGridFFTDF. Build the RKS object '
            'without multigrid acceleration.')


def _needs_uks_machinery(mf, env):
    '''True when the unrestricted classes are required.

    That average is not an approximation. With rho_a = rho_b = rho_A/2 held by
    A's own restrictedness,

        dE/dD_A = dE/dD^a . dD^a/dD_A + dE/dD^b . dD^b/dD_A = (v_a + v_b)/2

    so the 2-D matrix handed back is the exact gradient of the energy reported.
    '''
    return _is_unrestricted(mf) or env.polarized


def _sb_class(mf, env):
    '''The supermolecular KSCED class: domain x spin.

    Whether either subsystem is polarised selects the spin axis; the A/B
    difference within the unrestricted case is absorbed by the environment,
    which reshapes rho_B to the layout A needs.
    '''
    from pyscf.ksced.super import rks as ksced_rks
    from pyscf.ksced.super import pbcrks as ksced_pbcrks

    if _is_kscf(mf):
        return ksced_pbcrks.KSCEDKPBCRKS
    if _needs_uks_machinery(mf, env):
        if _is_pbc(mf):
            from pyscf.ksced.super import pbcuks as ksced_pbcuks
            return (ksced_pbcuks.KSCEDPBCUKS if _is_unrestricted(mf)
                    else ksced_pbcuks.KSCEDPBCRKSinU)
        from pyscf.ksced.super import uks as ksced_uks
        return (ksced_uks.KSCEDUKS if _is_unrestricted(mf)
                else ksced_uks.KSCEDRKSinU)
    return ksced_pbcrks.KSCEDPBCRKS if _is_pbc(mf) else ksced_rks.KSCEDRKS


def _is_unrestricted(mf):
    '''True when mf carries separate alpha and beta orbitals.
    '''
    return bool(mf.istype('UHF'))


def _reject_restricted_open_shell(mf, what):
    '''ROHF/ROKS is neither of the two cases the energy expression covers.
    '''
    if mf.istype('ROHF'):
        raise NotImplementedError(
            'KSCED does not support restricted open-shell: %s is ROHF/ROKS. '
            'Its density matrix is spin resolved while its Fock build is not, '
            'and the KSCED energy expression is not defined for that split. '
            'Use UKS.' % what)


def _check_kpts(mf, mf_b):
    '''k-point objects: KRKS only, and A and B on the same k-points.

    A k-point density matrix is (nk, nao, nao). The spin dispatch would read a
    leading axis of length 2 as alpha/beta, so k-point objects get their own
    classes, which exist for the restricted case only.
    '''
    ka, kb = _is_kscf(mf), _is_kscf(mf_b)
    if not (ka or kb):
        return
    if ka != kb:
        raise ValueError(
            'KSCED: subsystem A is %s and subsystem B is %s. Build both at the '
            'gamma point (RKS) or both with the same k-points (KRKS).'
            % (type(mf).__name__, type(mf_b).__name__))
    for m, what in ((mf, 'subsystem A'), (mf_b, 'subsystem B')):
        if m.istype('KUHF') or m.istype('KROHF') or m.istype('KGHF'):
            raise NotImplementedError(
                'KSCED k-point support is restricted (KRKS) only: %s is %s. '
                'Use KRKS, or UKS at the gamma point.' % (what, type(m).__name__))
    kpts_a = numpy.asarray(mf.kpts).reshape(-1, 3)
    kpts_b = numpy.asarray(mf_b.kpts).reshape(-1, 3)
    if kpts_a.shape != kpts_b.shape or not numpy.allclose(kpts_a, kpts_b):
        raise ValueError(
            'KSCED: subsystems A and B must use the same k-points; A has %d and '
            'B has %d, or they differ.' % (len(kpts_a), len(kpts_b)))


def _looks_supermolecular(mol_a, mol_b):
    '''True when A and B were built in one shared basis with ghost atoms.
    '''
    if mol_a.nao != mol_b.nao:
        return False
    if mol_a.natm != mol_b.natm:
        return False
    return numpy.allclose(mol_a.atom_coords(), mol_b.atom_coords())


def _check_mb_preconditions(mf, mf_b):
    '''Periodic MB needs A, B and B's converged run on one lattice and mesh.'''
    if not _is_pbc(mf):
        return
    cell_a, cell_b = mf.mol, mf_b.mol
    if not numpy.allclose(cell_a.lattice_vectors(), cell_b.lattice_vectors()):
        raise ValueError(
            'KSCED monomolecular basis: cell_a and cell_b must share a lattice')
    if not numpy.array_equal(cell_a.mesh, cell_b.mesh):
        raise ValueError(
            'KSCED monomolecular basis: cell_a.mesh %r and cell_b.mesh %r '
            'differ, so their uniform grids are not the same points'
            % (tuple(cell_a.mesh), tuple(cell_b.mesh)))


def frozen_env(mf_b, mol_a, dm_b=None, mol_ab=None, basis_mode='M'):
    '''A reusable frozen environment for subsystem B.

    Build one of these, then hand it to embed() at every geometry of A. The
    parts of B that do not depend on A -- rho_B on the quadrature grid, and on a
    periodic mesh the Hartree potential v_J[rho_B] -- are then built once for
    the trajectory instead of once per point.

        env = ksced.frozen_env(mf_b, cell_a)
        for coords in trajectory:
            mf_a = ksced.embed(dft.RKS(build(coords), xc='PBE'), mf_b, env=env)
            mf_a.kernel()

    Args:
        mf_b : converged RKS object for subsystem B.
        mol_a : Mole or Cell for subsystem A. Later geometries must keep the
            same atom set, basis, lattice and mesh; only the coordinates move.

    Kwargs:
        dm_b, mol_ab : as for embed().
        basis_mode : 'M' only. The supermolecular basis has nothing to reuse:
            rho_B is carried by ghost functions centred on A's atoms, so it is
            not independent of A's geometry in the first place.
    '''
    if basis_mode != 'M':
        raise ValueError(
            "frozen_env() is for basis_mode='M'. In the supermolecular basis "
            "rho_B is expanded in ghost functions centred on A's atoms, so it "
            "changes when A moves and there is nothing to carry between "
            "geometries.")
    from pyscf.ksced.mono.env import _FrozenEnvMB
    return _FrozenEnvMB(mf_b, mol_a, dm_b, mol_ab)


def _check_reusable_env(env, mf_b, dm_b, basis_mode):
    '''Vet an environment handed back to embed().

    The environment carries B's frozen density; taking a second one from dm_b or
    a mismatched mf_b would leave two disagreeing copies, so both are refused
    rather than resolved by precedence.
    '''
    from pyscf.ksced.mono.env import _FrozenEnvMB
    if not isinstance(env, _FrozenEnvMB):
        raise TypeError(
            'env must come from ksced.frozen_env(); got %r' % (type(env),))
    if basis_mode == 'S':
        raise ValueError(
            "env implies basis_mode='M'; 'S' was requested. A supermolecular "
            'environment has nothing to reuse between geometries.')
    if env.mf_b is not mf_b:
        raise ValueError(
            'env was built from a different subsystem B than the one passed to '
            'embed(). Pass the same mf_b, or build a new environment.')
    if dm_b is not None:
        raise ValueError(
            'dm_b and env both carry the frozen density. Give dm_b to '
            'frozen_env() when the environment is built, not to embed().')


def embed(mf, mf_b, dm_b=None, mol_ab=None, basis_mode=None, env=None,
          _bypass_sb_guard=False):
    '''Attach a frozen KSCED environment to a restricted KS object.

    Args:
        mf : RKS object for subsystem A.
        mf_b : converged RKS object for subsystem B.

    Kwargs:
        dm_b : density matrix for B. Defaults to mf_b.make_rdm1().
        mol_ab : Mole or Cell for the whole system. In 'S' mode this is optional
            and, when given, folds the A-B nuclear repulsion into e_tot. In 'M'
            mode it is built automatically from mf.mol and mf_b.mol and is
            always included; pass it only to override.
        basis_mode : 'S' for the supermolecular basis, where A and B share one
            AO basis built with ghost atoms. 'M' for the monomolecular basis,
            where A carries only A's functions and B only B's -- which is what
            makes the embedded SCF smaller than the whole system. Defaults to
            'S', or to 'M' when env is given.
        env : an environment from frozen_env(), rebound to this geometry of A
            rather than rebuilt. Implies basis_mode='M'.
        _bypass_sb_guard : test hook. Lets the 'M' machinery run on ghost-built
            cells so that it can be compared against the 'S' path. Never set in
            production code.

    Returns:
        A new object whose class derives from both the KSCED mixin and the class
        of mf.
    '''
    from pyscf.ksced.super import rks as ksced_rks
    from pyscf.ksced.super import pbcrks as ksced_pbcrks

    if basis_mode is None:
        # A reusable environment is monomolecular by construction; asking for
        # 'S' alongside one is a contradiction, and _check_reusable_env says so.
        basis_mode = 'S' if env is None else 'M'
    if basis_mode not in ('S', 'M'):
        raise ValueError(
            "basis_mode must be 'S' (supermolecular) or 'M' (monomolecular); "
            "got %r" % (basis_mode,))

    if env is not None:
        _check_reusable_env(env, mf_b, dm_b, basis_mode)

    _reject_restricted_open_shell(mf, 'subsystem A')
    _reject_restricted_open_shell(mf_b, 'subsystem B')
    _check_kpts(mf, mf_b)

    if basis_mode == 'S':
        env = _FrozenEnv(mf_b, dm_b)
        if isinstance(mf, _KSCED):
            mf.with_env = env
            mf.mol_ab = mol_ab
            return mf
        _check_numint(mf)
        base = _sb_class(mf, env)
        obj = base(mf, env, mol_ab)
    else:
        from pyscf.ksced.mono import rks as mb_rks
        from pyscf.ksced.mono import pbcrks as mb_pbcrks
        from pyscf.ksced.mono.env import _FrozenEnvMB

        if not _bypass_sb_guard and _looks_supermolecular(mf.mol, mf_b.mol):
            raise ValueError(
                "basis_mode='M' was given cells that share one basis: A and B "
                "have the same nao and the same geometry, which is the "
                "supermolecular (ghost-atom) construction. Build A from A's "
                "atoms only and B from B's only, or use basis_mode='S'.")
        _check_mb_preconditions(mf, mf_b)

        if env is None:
            env = _FrozenEnvMB(mf_b, mf.mol, dm_b, mol_ab)
        else:
            env.rebind(mf.mol, mol_ab)
        if isinstance(mf, _KSCED):
            mf.with_env = env
            mf.mol_ab = env.mol_ab
            return mf
        _check_numint(mf)
        if _is_kscf(mf):
            base = mb_pbcrks.KSCEDMBKPBCRKS
        elif _needs_uks_machinery(mf, env):
            if _is_pbc(mf):
                from pyscf.ksced.mono import pbcuks as mb_pbcuks
                base = (mb_pbcuks.KSCEDMBPBCUKS if _is_unrestricted(mf)
                        else mb_pbcuks.KSCEDMBPBCRKSinU)
            else:
                from pyscf.ksced.mono import uks as mb_uks
                base = (mb_uks.KSCEDMBUKS if _is_unrestricted(mf)
                        else mb_uks.KSCEDMBRKSinU)
        else:
            base = (mb_pbcrks.KSCEDMBPBCRKS if _is_pbc(mf)
                    else mb_rks.KSCEDMBRKS)
        obj = base(mf, env, env.mol_ab)

    # Use a distinct name to keep the synthesized MRO and tracebacks clear.
    name = mf.__class__.__name__ + 'WithKSCED'
    return lib.set_class(obj, (base, mf.__class__), name)
