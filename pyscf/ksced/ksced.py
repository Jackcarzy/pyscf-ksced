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
    return type(a).__module__.startswith('cupy')


def _as_like(ref, arr):
    '''Return arr on the same array backend as ref.

    GPU4PySCF keeps density matrices and potentials in cupy while some
    one-electron integrals stay in numpy, and cupy refuses to broadcast against
    a host array. Everything that mixes the two goes through here.
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

    Computing V_ne independently is a trap. PySCF builds it from the density
    fitting object (FFTDF), while GPU4PySCF's get_hcore switches to
    MultiGridNumInt whenever prod(cell.mesh) < 500**3 -- which is every system
    here. Deriving V_ne[B] from the same get_hcore that produced V_ne[A]
    guarantees the two are computed by the same method on whichever backend is
    in use, which is what the embedding energy expression assumes.
    """
    if _is_cell(mol):
        h1e = mf.get_hcore(mol, kpt)
    else:
        h1e = mf.get_hcore(mol)
    return h1e - _as_like(h1e, _kinetic_of(mol, kpt))


class _FrozenEnv:
    '''The frozen subsystem B.

    Everything the embedded calculation needs from B is served from here and
    cached, because rho_B never changes during A's SCF. This is the only class
    that assumes A and B share an AO basis.
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

    def get_vne_b(self, mol, kpt=None):
        '''V_ne[B] in the shared AO basis, from B's own get_hcore.'''
        if self._vne_b is None:
            self._vne_b = _vne_from_hcore(self.mf_b, self.mol_b, kpt)
        return self._vne_b

    def get_j_b(self, mf, mol):
        '''J[rho_B] in the shared AO basis.'''
        if self._j_b is None:
            self._j_b = mf.get_j(mol, self.dm_b, 1)
        return self._j_b

    def e_xc(self, ni, mol, grids, xc, max_memory):
        '''E_xc[rho_B].'''
        if self._e_xc is None:
            self._e_xc = ni.nr_rks(mol, grids, xc, self.dm_b,
                                   max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b(self, ni, mol, grids, t_nad, max_memory):
        '''T_s^TF[rho_B], the B term of the non-additive kinetic energy.'''
        if self._e_tnad_b is None:
            self._e_tnad_b = ni.nr_rks(mol, grids, t_nad, self.dm_b,
                                       max_memory=max_memory)[1]
        return self._e_tnad_b

    def e_xc_pbc(self, ni, cell, grids, xc, hermi, kpt, max_memory):
        '''E_xc[rho_B] for the periodic path.'''
        if self._e_xc is None:
            self._e_xc = ni.nr_rks(cell, grids, xc, self.dm_b, 0, hermi,
                                   kpt, None, max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b_pbc(self, ni, cell, grids, t_nad, hermi, kpt, max_memory):
        '''T_s^TF[rho_B] for the periodic path.'''
        if self._e_tnad_b is None:
            self._e_tnad_b = ni.nr_rks(cell, grids, t_nad, self.dm_b, 0, hermi,
                                       kpt, None, max_memory=max_memory)[1]
        return self._e_tnad_b

    def e_vne_a_rho_b(self, vne_a):
        '''<V_ne[A] | rho_B>, a constant of the embedded SCF.

        vne_a is supplied by the mixin, which is the only place with access to
        the unmodified get_hcore for subsystem A. It may be the matrix itself
        or a zero-argument callable returning it; the callable form is only
        invoked on a cache miss, which is once per SCF.

        That laziness matters. Building V_ne[A] means building a full hcore,
        and this value is a constant of the embedded SCF, so evaluating it
        eagerly on every energy_elec call rebuilt an hcore every cycle and
        discarded it. Measured on Au120 partition 10 on an H200: the embedded
        phase took 401 s with the eager call against 215 s for an equivalent
        implementation that precomputes the constant.
        '''
        if self._e_vne_a_rho_b is None:
            if callable(vne_a):
                vne_a = vne_a()
            self._e_vne_a_rho_b = _trace_prod(vne_a, self.dm_b)
        return self._e_vne_a_rho_b


class _KSCED:
    '''Tag class labelling a KSCED-embedded SCF method.'''
    pass


class KSCEDMixin(_KSCED):
    '''Behaviour shared by the molecular and periodic KSCED methods.

    The domain-specific part is get_veff, supplied by the subclasses. Everything
    here is common because pyscf.pbc.dft.rks assigns
    energy_elec = mol_ks.energy_elec, so one energy expression serves both.
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
        # Constant: the A nuclei attracting the frozen B electrons. For a
        # periodic system, reuse A's own density fitting object so the mesh
        # matches the one the SCF runs on.
        #
        # Passed unevaluated. _vne_a() builds a full hcore, and the environment
        # caches the contraction after the first cycle, so calling it here
        # would rebuild that hcore every cycle only to discard it.
        e_vne_a_rho_b = env.e_vne_a_rho_b(self._vne_a)
        # The second half of J_AB. get_veff already contributed the first half
        # through ecoul = 0.5 * <dm_a | J[rho_total]>.
        e_coul_ab_half = _trace_prod(env.get_j_b(self, self.mol), dm) * .5

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

    The multigrid class to look for moved between PySCF releases: 2.5 routes
    through a density fitting object, MultiGridFFTDF, while 2.14 routes through
    a numint, MultiGridNumInt. Probe for whichever exists so the plugin keeps
    working across both, which the Stage 1 and Stage 2 comparisons rely on.
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


def _looks_supermolecular(mol_a, mol_b):
    '''True when A and B were built in one shared basis with ghost atoms.

    Keys on nao equality and identical geometry. An extended-MB cell -- A's
    atoms plus a few of B's as ghosts -- never satisfies nao_a == nao_b, so this
    guard does not close the door on that.
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


def embed(mf, mf_b, dm_b=None, mol_ab=None, basis_mode='S',
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
            makes the embedded SCF smaller than the whole system.
        _bypass_sb_guard : test hook. Lets the 'M' machinery run on ghost-built
            cells so that it can be compared against the 'S' path. Never set in
            production code.

    Returns:
        A new object whose class derives from both the KSCED mixin and the class
        of mf.
    '''
    from pyscf.ksced import rks as ksced_rks
    from pyscf.ksced import pbcrks as ksced_pbcrks

    if basis_mode not in ('S', 'M'):
        raise ValueError(
            "basis_mode must be 'S' (supermolecular) or 'M' (monomolecular); "
            "got %r" % (basis_mode,))

    if basis_mode == 'S':
        env = _FrozenEnv(mf_b, dm_b)
        if isinstance(mf, _KSCED):
            mf.with_env = env
            mf.mol_ab = mol_ab
            return mf
        _check_numint(mf)
        base = ksced_pbcrks.KSCEDPBCRKS if _is_pbc(mf) else ksced_rks.KSCEDRKS
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

        env = _FrozenEnvMB(mf_b, mf.mol, dm_b, mol_ab)
        if isinstance(mf, _KSCED):
            mf.with_env = env
            mf.mol_ab = env.mol_ab
            return mf
        _check_numint(mf)
        base = mb_pbcrks.KSCEDMBPBCRKS if _is_pbc(mf) else mb_rks.KSCEDMBRKS
        obj = base(mf, env, env.mol_ab)

    # The synthesised class must not reuse the mixin's name, or the MRO reads
    # "KSCEDRKS <- KSCEDRKS" and tracebacks become ambiguous. pyscf.solvent
    # avoids this the same way, by naming from the component rather than the mixin.
    name = mf.__class__.__name__ + 'WithKSCED'
    return lib.set_class(obj, (base, mf.__class__), name)
