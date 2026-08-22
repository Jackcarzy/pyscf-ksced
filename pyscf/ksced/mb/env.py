'''The frozen subsystem B, described in its own AO basis.

Presents exactly the interface of ksced._FrozenEnv, which is what lets
KSCEDMixin -- get_hcore, energy_nuc, energy_elec, reset -- serve both basis
modes without modification. The environment object, not the SCF class, is the
polymorphic seam.
'''

import copy

import numpy
from pyscf.lib import logger

from pyscf.ksced.ksced import (_is_cell, _kinetic_of, _spin_sum, _stack_like,
                               _trace_prod_spin)
from pyscf.ksced.mb.arrays import to_host as _host
from pyscf.ksced.mb.griddens import _GridDensity
from pyscf.ksced.mb.meshdata import _MeshData, is_fft_df as _is_fft_df
from pyscf.ksced.mb.numint import _ksced_numint


def _grids_bound_to(grids, mol):
    '''The same quadrature points, screened against mol's shells.

    Functionals of rho_B are evaluated over mol_b but on subsystem A's grid
    object, because that is the grid the embedded SCF integrates on. block_loop
    masks AO evaluation with grids.non0tab, which is built from the grid's own
    molecule, and the periodic one does not check that the two agree:

        pbc/dft/numint.py:1075   if non0tab is None:            <- no guard
        dft/numint.py:2904       if non0tab is None and mol is grids.mol:

    So a periodic evaluation over B's shells indexes a mask of A's width. With
    nbas_B > nbas_A that reads past the buffer: the result is wrong, it moves
    when A's nuclei move, and two identical calls disagree.

    Only the mask is wrong -- the points and weights are shared -- so rebind
    them to a grid object that knows whose shells it is screening. Nothing is
    re-quadratured. The molecular path was already safe by the guard above; this
    keeps both domains on one route rather than relying on it.

    The defect hides whenever nbas_A == nbas_B, which is every ghost-built
    supermolecular comparison and every A-equals-B test system, and it is absent
    on GPU4PySCF, whose periodic block_loop passes no mask at all.
    '''
    from pyscf.pbc.dft.gen_grid import UniformGrids
    if isinstance(grids, UniformGrids):
        # Points come from the lattice and mesh alone, so they come out
        # identical; block_loop fills in the mask, and for a uniform grid its
        # build() touches nothing else.
        out = UniformGrids(mol)
        out.mesh = grids.mesh
        return out

    out = copy.copy(grids)
    is_pbc_grid = hasattr(out, 'cell')
    for owner in ('cell', 'mol'):
        if hasattr(out, owner):
            setattr(out, owner, mol)
    out.coords = grids.coords
    out.weights = grids.weights
    out.non0tab = None
    if is_pbc_grid:
        # The periodic block_loop rebuilds any grid whose non0tab is missing,
        # and for an atom-centred grid build() regenerates the coordinates from
        # the molecule too -- quietly re-quadraturing on B's atoms. Fill the
        # mask here so it never gets that far. The molecular loop is left alone:
        # it only builds when coords are missing, and its `mol is grids.mol`
        # guard already decides the mask correctly.
        out.non0tab = out.make_mask(mol, out.coords)
    return out


def _conc(mol_a, mol_b):
    '''mol_ab with A's AOs first. Both helpers guarantee that ordering.'''
    if _is_cell(mol_a):
        from pyscf.pbc.gto.cell import conc_cell
        return conc_cell(mol_a, mol_b)
    from pyscf.gto.mole import conc_mol
    return conc_mol(mol_a, mol_b)




class _FrozenEnvMB:
    '''Subsystem B in B's own basis, serving A in A's basis.

    Everything crossing between the two bases is a constant of A's SCF, because
    rho_B never changes, so it is built once at the AB dimension and sliced.
    '''

    def __init__(self, mf_b, mol_a, dm_b=None, mol_ab=None):
        self.mf_b = mf_b
        self.mol_b = mf_b.mol
        self.mol_a = mol_a

        if dm_b is None:
            if (getattr(mf_b, 'mo_coeff', None) is None
                    or getattr(mf_b, 'mo_occ', None) is None):
                raise ValueError(
                    'subsystem B has no density matrix. Call mf_b.kernel() before '
                    'embed(), or pass an explicit dm_b.')
            if not getattr(mf_b, 'converged', True):
                logger.warn(mf_b, 'KSCED: subsystem B is not converged; the frozen '
                                  'density is taken from an unconverged calculation')
            dm_b = mf_b.make_rdm1()
        self.dm_b = dm_b

        self.nao_b = self.mol_b.nao

        # B's grid data -- rho_B and v_J^B -- lives on the quadrature grid, so
        # whether it survives A moving is a question about that grid. A periodic
        # uniform mesh is fixed by the lattice, since get_uniform_grids reads
        # lattice_vectors() and mesh and never an atom position. A molecular
        # Becke grid is centred on the atoms and follows them. reset() keys on
        # this, and it is what makes rebind() worth having.
        self._b_side_persistent = _is_cell(self.mol_b)

        self._bind_a(mol_a, mol_ab)
        self.reset_b()

    def _bind_a(self, mol_a, mol_ab=None):
        '''Point the environment at a subsystem A and size the cross terms.'''
        self.mol_a = mol_a
        self.nao_a = mol_a.nao

        # Cross terms use slices of the concatenated A+B basis, with A first.
        # Ghost centers do not affect the resulting nuclear energy.
        if mol_ab is None:
            mol_ab = _conc(mol_a, self.mol_b)
        elif mol_ab.nao != self.nao_a + self.nao_b:
            raise ValueError(
                'KSCED monomolecular basis: mol_ab has nao %d, but the cross '
                'terms are slices of an A(+)B build and need nao_a + nao_b = '
                '%d + %d = %d. Omit mol_ab and it will be concatenated '
                'correctly; pass one only to override the concatenation itself.'
                % (mol_ab.nao, self.nao_a, self.nao_b,
                   self.nao_a + self.nao_b))
        self.mol_ab = mol_ab

    def rebind(self, mol_a, mol_ab=None):
        '''Point this environment at a displaced subsystem A.

        For driving one frozen environment along a trajectory. Only the terms
        that touch A's basis are dropped; on a periodic mesh v_J^B(r) and rho_B
        are kept, which is the entire saving -- they cost N_grid * nao_B^2 to
        build and nothing to reuse.
        '''
        self._check_displaced(mol_a)
        self._bind_a(mol_a, mol_ab)
        self.reset()
        return self

    def _check_displaced(self, mol_a):
        '''A moved A, not a different one.

        Anything that would invalidate the cached grid data, or that signals a
        different partition rather than a new geometry, is refused here rather
        than silently producing a stale potential.
        '''
        if _is_cell(mol_a) != _is_cell(self.mol_a):
            raise ValueError(
                'KSCED: rebind() cannot switch subsystem A between molecular '
                'and periodic')
        if mol_a.nao != self.nao_a:
            raise ValueError(
                'KSCED: rebind() takes a displaced subsystem A, but nao went '
                'from %d to %d. A different atom set or basis is a different '
                'partition, not a new geometry; build a new environment for it.'
                % (self.nao_a, mol_a.nao))
        if _is_cell(mol_a):
            if not numpy.allclose(mol_a.lattice_vectors(),
                                  self.mol_a.lattice_vectors()):
                raise ValueError(
                    'KSCED: rebind() requires the same lattice. The cached '
                    'v_J[rho_B] is only defined on the mesh it was built for.')
            if not numpy.array_equal(mol_a.mesh, self.mol_b.mesh):
                raise ValueError(
                    'KSCED: rebind() was given mesh %r against B\'s %r; the '
                    'two grids are not the same points'
                    % (tuple(mol_a.mesh), tuple(self.mol_b.mesh)))

    def reset(self):
        '''Clear what a change in subsystem A invalidates.

        Everything keyed to A's basis: the cross-basis one-electron terms, the
        Coulomb matrix, and the numint wrapper. B's grid data survives when the
        grid does -- see _b_side_persistent.
        '''
        self._vne_b = None
        self._vne_a_in_b = None
        self._j_b = None
        self._e_vne_a_rho_b = None
        self._numint = None
        self._mfs = {}
        if not self._b_side_persistent:
            self._reset_b_side()
        return self

    def reset_b(self):
        '''Clear everything, B's grid data included.

        For when rho_B itself changes. reset() alone would leave a stale
        v_J^B(r) behind on the periodic path.
        '''
        self._reset_b_side()
        return self.reset()

    def _reset_b_side(self):
        self._e_xc = None
        self._e_tnad_b = None
        self._griddens = None
        self._grids_b = None
        self._mesh = None

    # -- scratch mean-field objects --------------------------------------

    def _mf_on(self, mol):
        '''A copy of mf_b rebound to mol.
        '''
        key = id(mol)
        mf = self._mfs.get(key)
        if mf is None:
            mf = self.mf_b.copy()
            with_df = getattr(mf, 'with_df', None)
            if with_df is not None:
                mf.with_df = copy.copy(with_df)
            mf.reset(mol)
            mf._eri = None
            self._mfs[key] = mf
        return mf

    def _vne_on(self, mol, kpt=None):
        '''V_ne of mol's own nuclei, in mol's basis: hcore - T.
        '''
        mf = self._mf_on(mol)
        if _is_cell(mol):
            h1e = mf.get_hcore(mol, kpt)
        else:
            h1e = mf.get_hcore(mol)
        return _host(h1e) - _host(_kinetic_of(mol, kpt))

    # -- the _FrozenEnv interface ---------------------------------------

    @property
    def polarized(self):
        return getattr(self.dm_b, 'ndim', 2) == 3

    def _nr(self, ni):
        return ni.nr_uks if self.polarized else ni.nr_rks

    def get_vne_b(self, mol=None, kpt=None):
        '''V_ne[B] in A's basis: the A-A block of V_ne_ab, less A's own.'''
        if self._vne_b is None:
            n = self.nao_a
            self._vne_b = (self._vne_on(self.mol_ab, kpt)[:n, :n]
                           - self._vne_on(self.mol_a, kpt))
        return self._vne_b

    def _vne_a_in_bs_basis(self, kpt=None):
        '''V_ne[A] in B's basis: the B-B block of V_ne_ab, less B's own.'''
        if self._vne_a_in_b is None:
            n = self.nao_a
            self._vne_a_in_b = (self._vne_on(self.mol_ab, kpt)[n:, n:]
                                - self._vne_on(self.mol_b, kpt))
        return self._vne_a_in_b

    def get_j_b(self, mf=None, mol=None):
        '''J[rho_B] in A's basis. mol is ignored.

        On a plane-wave mesh the Hartree potential of rho_B is a constant of A's
        geometry, so it is solved for once and contracted against A's AOs alone:
        N_grid * nao_A^2, and nothing at the A+B dimension after the first call.
        Everything else -- molecular, or a fitting scheme that never forms a
        real-space potential -- falls back to the A+B build, sliced.

        mf is the embedded object, needed for A's own fitting object. It is
        optional only so that the AB path stays reachable without one.
        '''
        if self._j_b is None:
            mesh = self._mesh_data(getattr(mf, '_numint', None),
                                   getattr(mf, 'kpt', None))
            if mesh is not None and _is_fft_df(getattr(mf, 'with_df', None)):
                self._j_b = mesh.matrix(mf)
            else:
                self._j_b = self._j_b_from_ab()
        return self._j_b

    def _mesh_data(self, ni=None, kpt=None):
        '''B's uniform-mesh data, or None when B does not live on one.

        Needs a numint the first time, to build the density evaluator. Callers
        that have one should pass it; get_j_b takes it off the embedded object.
        '''
        if not self._b_side_persistent:
            return None
        if self._mesh is None:
            if ni is None:
                return None
            data = _MeshData(self.mf_b, self._make_evaluator(ni, kpt))
            if not data.usable():
                return None
            self._mesh = data
        return self._mesh

    def _j_b_from_ab(self):
        '''J[rho_B] as the A-A block of an A+B build with a padded density.'''
        n = self.nao_a
        nao_ab = self.nao_a + self.nao_b
        dm_pad = numpy.zeros((nao_ab, nao_ab))
        dm_pad[n:, n:] = _host(_spin_sum(self.dm_b)).real
        mf_ab = self._mf_on(self.mol_ab)
        if _is_cell(self.mol_ab):
            j_ab = mf_ab.get_j(self.mol_ab, dm_pad, 1, numpy.zeros(3), None)
        else:
            j_ab = mf_ab.get_j(self.mol_ab, dm_pad, 1)
        return _host(j_ab)[:n, :n]

    def e_vne_a_rho_b(self, vne_a=None):
        '''<V_ne[A]|rho_B>.
        '''
        if self._e_vne_a_rho_b is None:
            self._e_vne_a_rho_b = _trace_prod_spin(
                self._vne_a_in_bs_basis(), self.dm_b)
        return self._e_vne_a_rho_b

    def grids_for_b(self, grids):
        '''The caller's quadrature points, screened against B's shells.

        Every functional evaluated over mol_b has to go through this. See
        _grids_bound_to for what goes wrong otherwise.
        '''
        if self._grids_b is None:
            self._grids_b = _grids_bound_to(grids, self.mol_b)
        return self._grids_b

    def e_xc(self, ni, mol, grids, xc, max_memory):
        '''E_xc[rho_B], evaluated with B's own AOs on the supermolecular grid.'''
        if self._e_xc is None:
            self._e_xc = self._nr(ni)(self.mol_b, self.grids_for_b(grids), xc,
                                      self.dm_b, max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b(self, ni, mol, grids, t_nad, max_memory):
        if self._e_tnad_b is None:
            self._e_tnad_b = self._nr(ni)(
                self.mol_b, self.grids_for_b(grids), t_nad, self.dm_b,
                max_memory=max_memory)[1]
        return self._e_tnad_b

    def _b_energy_pbc(self, ni, grids, xc, hermi, kpt, max_memory):
        '''E[rho_B] for one functional, periodic.

        Off the cached mesh density when there is one, which is ver4's route
        and the reason that fork never met the non0tab mismatch: it reads a
        stored rho_B rather than evaluating B's AOs a second time. The fallback
        is for a periodic fit that never forms a mesh density at all, and still
        needs the grid rebound before B's shells are screened with A's mask.
        '''
        mesh = self._mesh_data(ni, kpt)
        if mesh is not None:
            return mesh.energy(ni, xc)
        return self._nr(ni)(
            self.mol_b, self.grids_for_b(grids), xc, self.dm_b, 0, hermi,
            kpt, None, max_memory=max_memory)[1]

    def e_xc_pbc(self, ni, cell, grids, xc, hermi, kpt, max_memory):
        if self._e_xc is None:
            self._e_xc = self._b_energy_pbc(ni, grids, xc, hermi, kpt,
                                            max_memory)
        return self._e_xc

    def e_tnad_b_pbc(self, ni, cell, grids, t_nad, hermi, kpt, max_memory):
        if self._e_tnad_b is None:
            self._e_tnad_b = self._b_energy_pbc(ni, grids, t_nad, hermi, kpt,
                                                max_memory)
        return self._e_tnad_b

    # -- MB-only ---------------------------------------------------------

    def _make_evaluator(self, ni, kpt=None):
        '''rho_B(coords) at GGA order, using B's own AOs.'''
        return density_evaluator(ni, self.mol_b, self.dm_b, kpt)

    def numint_for(self, ni, kpt=None):
        '''The offset twin of ni.

        Two lifetimes, deliberately separate. _griddens holds rho_B on the grid
        and costs N_grid * nao_B^2 to fill; the wrapper around it is a shallow
        copy of ni and costs nothing. A new A brings its own ni, so reset()
        drops the wrapper and keeps the density.
        '''
        if self._griddens is None:
            self._griddens = _GridDensity(self._make_evaluator(ni, kpt))
        if self._numint is None:
            self._numint = _ksced_numint(ni, self._griddens)
        return self._numint


def density_evaluator(ni, mol_b, dm_b, kpt=None):
    '''rho_B(coords) at GGA order, from B's own AOs.

    Returns (4, n) for a restricted environment and (2, 4, n) for a polarised
    one. Module level so that a validation script can collocate exactly the
    density the plugin uses rather than re-deriving it.

    The periodic branch probes both gamma-point conventions because the
    backends disagree on whether eval_ao/eval_rho want a bare kpt and a 2-D
    density matrix or a (1, 3) kpts array and a 3-D one.
    '''
    polarized = getattr(dm_b, 'ndim', 2) == 3

    def eval_channels(ao, dm):
        if getattr(dm, 'ndim', 2) == 2:
            return ni.eval_rho(mol_b, ao, dm, xctype='GGA')
        rhos = [ni.eval_rho(mol_b, ao, dm[s], xctype='GGA')
                for s in range(2)]
        return _stack_like(rhos[0], rhos)

    if not _is_cell(mol_b):
        def evaluator(coords):
            ao = ni.eval_ao(mol_b, coords, deriv=1)
            return eval_channels(ao, dm_b)
        return evaluator

    k1 = numpy.zeros(3) if kpt is None else numpy.asarray(kpt).reshape(3)
    probe = numpy.zeros((1, 3))
    errors = []
    for kk, add_k_axis in ((k1, False), (k1.reshape(1, 3), True)):
        dd = dm_b[None] if add_k_axis and numpy.ndim(dm_b) == 2 else dm_b
        try:
            ao = ni.eval_ao(mol_b, probe, kk, deriv=1)
            if polarized:
                for s in range(2):
                    ds = dm_b[s][None] if add_k_axis else dm_b[s]
                    ni.eval_rho(mol_b, ao, ds, xctype='GGA')
            else:
                ni.eval_rho(mol_b, ao, dd, xctype='GGA')
        except Exception as exc:          # noqa: BLE001 - probing
            errors.append('%s: %s' % (type(exc).__name__, exc))
            continue

        def evaluator(coords, _kk=kk, _add=add_k_axis):
            ao = ni.eval_ao(mol_b, coords, _kk, deriv=1)
            if not polarized:
                dd = dm_b[None] if _add else dm_b
                return ni.eval_rho(mol_b, ao, dd, xctype='GGA')
            rhos = []
            for s in range(2):
                dd = dm_b[s][None] if _add else dm_b[s]
                rhos.append(ni.eval_rho(mol_b, ao, dd, xctype='GGA'))
            return _stack_like(rhos[0], rhos)
        return evaluator

    raise RuntimeError(
        'KSCED: cannot evaluate the environment density with %s. Neither '
        'gamma-point convention worked:\n  1-D kpt: %s\n  2-D kpts: %s'
        % (type(ni).__name__, errors[0], errors[1]))
