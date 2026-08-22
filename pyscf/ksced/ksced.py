'''KSCED subsystem embedding: frozen environment and the SCF mixin.'''

import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger

T_NAD_DEFAULT = 'LDA_K_TF'


def _is_cell(mol):
    from pyscf.pbc.gto import Cell
    return isinstance(mol, Cell)


def _is_gpu_array(a):
    # GPU4PySCF tagged arrays still use CuPy's explicit host transfer.
    return not isinstance(a, numpy.ndarray) and callable(getattr(a, 'get', None))


def _as_like(ref, arr):
    '''Return arr on the same array backend as ref.
    '''
    if _is_gpu_array(ref) and not _is_gpu_array(arr):
        import cupy
        return cupy.asarray(arr)
    if not _is_gpu_array(ref) and _is_gpu_array(arr):
        return arr.get()
    return arr


def _trace_prod(a, b):
    '''<a|b> = einsum('ij,ji->', a, b).real, as a Python float, on any backend.'''
    if _is_gpu_array(a) or _is_gpu_array(b):
        import cupy
        a = cupy.asarray(a) if not _is_gpu_array(a) else a
        b = cupy.asarray(b) if not _is_gpu_array(b) else b
        return float(cupy.einsum('ij,ji->', a, b).real.get())
    return float(numpy.einsum('ij,ji->', a, b).real)


def _spin_sum(dm):
    """Total density matrix from one that may be spin resolved.

    An unrestricted density matrix is (2, nao, nao); a restricted one is
    (nao, nao). Every cross term in the KSCED energy contracts a spin-free
    one-electron operator -- V_ne[A], J[rho_B] -- against a density, and those
    contractions want the total density in both cases.
    """
    return dm[0] + dm[1] if getattr(dm, 'ndim', 2) == 3 else dm


def _trace_prod_spin(a, b):
    """<a|b> where a is a 2-D operator and b may be spin resolved."""
    return _trace_prod(a, _spin_sum(b))


def _is_polarized(dm):
    """True when dm carries an explicit spin axis."""
    return getattr(dm, 'ndim', 2) == 3


def _as_pair(dm):
    """dm as (2, nao, nao), splitting a restricted matrix evenly."""
    if _is_polarized(dm):
        return dm
    half = dm * .5
    return _stack_like(half, [half, half])


def _stack_like(ref, arrays):
    """Stack on whichever backend ref lives on."""
    if _is_gpu_array(ref):
        import cupy
        return cupy.stack([cupy.asarray(a) for a in arrays])
    return numpy.stack([numpy.asarray(a) for a in arrays])


def _avg_spin(v):
    """The spin average of a (2, nao, nao) potential, as a 2-D matrix.

    Used for a restricted subsystem A embedded in a polarised environment. It
    is not an approximation: A's own restrictedness constrains
    rho_alpha = rho_beta = rho_A / 2, so

        dE/dD_A = dE/dD_alpha . dD_alpha/dD_A + dE/dD_beta . dD_beta/dD_A
                = (v_alpha + v_beta) / 2

    and the matrix returned is the exact gradient of the energy reported.
    """
    return (v[0] + v[1]) * .5 if _is_polarized(v) else v


def _tag_array(a, **tags):
    """lib.tag_array on whichever backend a lives on.

    PySCF's tag_array calls numpy.asarray, which cupy refuses. GPU4PySCF ships an
    equivalent that produces a CPArrayWithTag.
    """
    if _is_gpu_array(a):
        from gpu4pyscf.lib.cupy_helper import tag_array as gpu_tag_array
        return gpu_tag_array(a, **tags)
    return lib.tag_array(a, **tags)


def _kinetic_of(mol, kpt=None):
    """Kinetic-energy matrix. Basis-only, so A and B share it."""
    if _is_cell(mol):
        if kpt is None:
            kpt = numpy.zeros(3)
        return mol.pbc_intor('int1e_kin', hermi=1, kpts=kpt)
    return mol.intor_symmetric('int1e_kin')


def _vne_from_hcore(mf, mol, kpt=None):
    """V_ne = hcore - T, taken from the backend's own get_hcore.
    """
    if _is_cell(mol):
        h1e = mf.get_hcore(mol, kpt)
    else:
        h1e = mf.get_hcore(mol)
    return h1e - _as_like(h1e, _kinetic_of(mol, kpt))


class _FrozenEnv:
    '''The frozen subsystem B.
    '''

    def __init__(self, mf_b, dm_b=None):
        self.mf_b = mf_b
        self.mol_b = mf_b.mol
        if dm_b is None:
            if getattr(mf_b, 'mo_coeff', None) is None or getattr(mf_b, 'mo_occ', None) is None:
                raise ValueError(
                    'subsystem B has no density matrix. Call mf_b.kernel() before '
                    'embed(), or pass an explicit dm_b.')
            if not getattr(mf_b, 'converged', True):
                logger.warn(mf_b, 'KSCED: subsystem B is not converged; the frozen '
                                  'density is taken from an unconverged calculation')
            dm_b = mf_b.make_rdm1()
        self.dm_b = dm_b
        self.reset()

    def reset(self):
        self._vne_b = None
        self._j_b = None
        self._e_xc = None
        self._e_tnad_b = None
        self._e_vne_a_rho_b = None
        return self

    @property
    def polarized(self):
        '''True when B was converged unrestricted, so dm_b carries a spin axis.'''
        return _is_polarized(self.dm_b)

    def dm_b_for(self, polarized_a):
        '''rho_B in the spin layout subsystem A needs.

        Four cases, and three of them are not the identity:

          A restricted, B restricted    dm_b unchanged
          A restricted, B unrestricted  spin summed to the total
          A unrestricted, B restricted  halved into each channel -- adding the
                                        whole of rho_B to both would double the
                                        environment, silently
          A unrestricted, B unrestricted  dm_b unchanged
        '''
        if polarized_a and not self.polarized:
            half = self.dm_b * .5
            return _stack_like(half, (half, half))
        if not polarized_a and self.polarized:
            return _spin_sum(self.dm_b)
        return self.dm_b

    def _nr(self, ni, polarized):
        '''The restricted or unrestricted numint driver, as B requires.'''
        return ni.nr_uks if polarized else ni.nr_rks

    def get_vne_b(self, mol, kpt=None):
        '''V_ne[B] in the shared AO basis, from B's own get_hcore.'''
        if self._vne_b is None:
            self._vne_b = _vne_from_hcore(self.mf_b, self.mol_b, kpt)
        return self._vne_b

    def get_j_b(self, mf, mol):
        '''J[rho_B] in the shared AO basis.
        '''
        if self._j_b is None:
            self._j_b = mf.get_j(mol, _spin_sum(self.dm_b), 1)
        return self._j_b

    def e_xc(self, ni, mol, grids, xc, max_memory):
        '''E_xc[rho_B].'''
        if self._e_xc is None:
            self._e_xc = self._nr(ni, self.polarized)(
                mol, grids, xc, self.dm_b, max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b(self, ni, mol, grids, t_nad, max_memory):
        '''T_s^TF[rho_B], the B term of the non-additive kinetic energy.
        '''
        if self._e_tnad_b is None:
            self._e_tnad_b = self._nr(ni, self.polarized)(
                mol, grids, t_nad, self.dm_b, max_memory=max_memory)[1]
        return self._e_tnad_b

    def e_xc_pbc(self, ni, cell, grids, xc, hermi, kpt, max_memory):
        '''E_xc[rho_B] for the periodic path.'''
        if self._e_xc is None:
            self._e_xc = self._nr(ni, self.polarized)(
                cell, grids, xc, self.dm_b, 0, hermi, kpt, None,
                max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b_pbc(self, ni, cell, grids, t_nad, hermi, kpt, max_memory):
        '''T_s^TF[rho_B] for the periodic path.'''
        if self._e_tnad_b is None:
            self._e_tnad_b = self._nr(ni, self.polarized)(
                cell, grids, t_nad, self.dm_b, 0, hermi, kpt, None,
                max_memory=max_memory)[1]
        return self._e_tnad_b

    def e_vne_a_rho_b(self, vne_a):
        '''<V_ne[A] | rho_B>, a constant of the embedded SCF.

        vne_a is supplied by the mixin, which is the only place with access to
        the unmodified get_hcore for subsystem A. It may be the matrix itself
        or a zero-argument callable returning it; the callable form is only
        invoked on a cache miss, which is once per SCF.
        '''
        if self._e_vne_a_rho_b is None:
            if callable(vne_a):
                vne_a = vne_a()
            self._e_vne_a_rho_b = _trace_prod_spin(vne_a, self.dm_b)
        return self._e_vne_a_rho_b


class _KSCED:
    '''Tag class labelling a KSCED-embedded SCF method.'''
    pass


class KSCEDMixin(_KSCED):
    '''Behaviour shared by the molecular and periodic KSCED methods.

    The domain-specific part is get_veff, supplied by the subclasses.
    '''

    _keys = {'with_env', 't_nad', 'mol_ab', 'e_tnad'}

    def __init__(self, mf, env, mol_ab=None):
        self.__dict__.update(mf.__dict__)
        self.with_env = env
        self.mol_ab = mol_ab
        self.t_nad = T_NAD_DEFAULT
        self.e_tnad = 0.0

    def undo_ksced(self):
        '''Return a plain SCF object without the embedding.'''
        obj = lib.view(self, lib.drop_class(self.__class__, KSCEDMixin, 'KSCED'))
        for key in ('with_env', 'mol_ab', 't_nad', 'e_tnad'):
            obj.__dict__.pop(key, None)
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'KSCED non-additive kinetic functional = %s', self.t_nad)
        logger.info(self, 'KSCED environment nao = %d', self.with_env.dm_b.shape[-1])
        if self.mol_ab is None:
            logger.info(self, 'KSCED mol_ab not supplied: E_nn[AB] excluded from e_tot')
        return self

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        h1e = super().get_hcore(mol)
        vne_b = self.with_env.get_vne_b(mol, getattr(self, 'kpt', None))
        return h1e + _as_like(h1e, vne_b)

    def energy_nuc(self):
        e = super().energy_nuc()
        if self.mol_ab is not None:
            e += (self.mol_ab.energy_nuc()
                  - self.mol.energy_nuc()
                  - self.with_env.mol_b.energy_nuc())
        return e

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        e_tot_elec, e2 = super().energy_elec(dm, h1e, vhf)

        env = self.with_env
        # Attraction between A's nuclei and B's frozen electrons. Pass the
        # callable so the environment can cache its first evaluation.
        e_vne_a_rho_b = env.e_vne_a_rho_b(self._vne_a)
        # Add the half of J_AB not included in get_veff's Coulomb energy.
        e_coul_ab_half = _trace_prod_spin(
            env.get_j_b(self, self.mol), dm) * .5

        self.scf_summary['ksced_vne_a_rho_b'] = e_vne_a_rho_b
        self.scf_summary['ksced_coul_ab_half'] = e_coul_ab_half
        return (e_tot_elec + e_vne_a_rho_b + e_coul_ab_half,
                e2 + e_coul_ab_half)

    def _vne_a(self):
        """V_ne[A] from the unmodified get_hcore, i.e. without vne_b."""
        kpt = getattr(self, 'kpt', None)
        mol = self.mol
        if _is_cell(mol):
            h1e = super().get_hcore(mol, kpt)
        else:
            h1e = super().get_hcore(mol)
        return h1e - _as_like(h1e, _kinetic_of(mol, kpt))

    def reset(self, mol=None):
        self.with_env.reset()
        return super().reset(mol)


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
    from pyscf.ksced import rks as ksced_rks
    from pyscf.ksced import pbcrks as ksced_pbcrks

    if _needs_uks_machinery(mf, env):
        if _is_pbc(mf):
            from pyscf.ksced import pbcuks as ksced_pbcuks
            return (ksced_pbcuks.KSCEDPBCUKS if _is_unrestricted(mf)
                    else ksced_pbcuks.KSCEDPBCRKSinU)
        from pyscf.ksced import uks as ksced_uks
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


def _reject_kpts(mf, what):
    '''KSCED is gamma-point only, and a k-point object must be refused here.
    '''
    if mf.istype('KSCF'):
        raise NotImplementedError(
            'KSCED is gamma-point only: %s is a k-point SCF object (%s). Its '
            'density matrix carries a k-point axis that the spin dispatch '
            'would read as alpha/beta. Build the subsystem at the gamma point '
            'with RKS/UKS instead of KRKS/KUKS.' % (what, type(mf).__name__))


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
    from pyscf.ksced.mb.env import _FrozenEnvMB
    return _FrozenEnvMB(mf_b, mol_a, dm_b, mol_ab)


def _check_reusable_env(env, mf_b, dm_b, basis_mode):
    '''Vet an environment handed back to embed().

    The environment carries B's frozen density; taking a second one from dm_b or
    a mismatched mf_b would leave two disagreeing copies, so both are refused
    rather than resolved by precedence.
    '''
    from pyscf.ksced.mb.env import _FrozenEnvMB
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
    from pyscf.ksced import rks as ksced_rks
    from pyscf.ksced import pbcrks as ksced_pbcrks

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
    _reject_kpts(mf, 'subsystem A')
    _reject_kpts(mf_b, 'subsystem B')

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
        from pyscf.ksced.mb import rks as mb_rks
        from pyscf.ksced.mb import pbcrks as mb_pbcrks
        from pyscf.ksced.mb.env import _FrozenEnvMB

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
        if _needs_uks_machinery(mf, env):
            if _is_pbc(mf):
                from pyscf.ksced.mb import pbcuks as mb_pbcuks
                base = (mb_pbcuks.KSCEDMBPBCUKS if _is_unrestricted(mf)
                        else mb_pbcuks.KSCEDMBPBCRKSinU)
            else:
                from pyscf.ksced.mb import uks as mb_uks
                base = (mb_uks.KSCEDMBUKS if _is_unrestricted(mf)
                        else mb_uks.KSCEDMBRKSinU)
        else:
            base = (mb_pbcrks.KSCEDMBPBCRKS if _is_pbc(mf)
                    else mb_rks.KSCEDMBRKS)
        obj = base(mf, env, env.mol_ab)

    # Use a distinct name to keep the synthesized MRO and tracebacks clear.
    name = mf.__class__.__name__ + 'WithKSCED'
    return lib.set_class(obj, (base, mf.__class__), name)
