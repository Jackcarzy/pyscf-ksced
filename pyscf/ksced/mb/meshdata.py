'''Subsystem B on the cell's uniform FFT mesh.

One array is collocated here, rho_B(r) at GGA order in linear mesh order, and
three things are read off it: the Hartree potential v_J[rho_B], E_xc[rho_B] and
T_TF[rho_B]. None of them depends on subsystem A.
`pbc/gto/cell.py:get_uniform_grids` reads `cell.lattice_vectors()` and `mesh`
and nothing else -- atom positions never enter -- so displacing A on a fixed
lattice leaves every grid point, and therefore all of it, untouched. That is
what lets one environment drive A across a trajectory.

This is the ver4 fork's strategy, in memory rather than through the filesystem.
That fork threaded a KSCED flag through PySCF's core and wrote the same two
quantities to disk:

    pbc/df/fft_jk.py    KSCED=1   np.save('rhoR_b', rhoR_b)  # (4, ngrids)
                                  np.save('veR_b',  vR)      # (1, ngrids)
                        KSCED=2   vR = np.load('veR_b.npy')
    pbc/dft/numint.py   KSCED=20  rho = np.load('rhoR_b.npy')[0]

`Au120_CH3SH_fast/10/` in the harness repository still holds both files from a
production run.

The Coulomb term is `pbc/df/fft_jk.py:get_j_kpts` held apart at its waist. That
function collocates a density, solves Poisson on the mesh, then contracts the
resulting potential back against AO pairs:

    J_munu = SUM_r w_r phi_mu(r) phi_nu(r) v_J(r)

Run whole at the A+B dimension, which is what slicing an `mol_ab` build amounts
to, both halves cost N_grid * nao_AB^2 and only the A-A block survives. Split,
the Poisson half depends on rho_B alone and is cached; the contraction half
depends on A alone and costs N_grid * nao_A^2.

Taking the functional energies off the same array matters for more than speed.
Evaluating them the other way -- B's AOs, on subsystem A's grid object -- is
what exposed the `non0tab` mismatch described in `_FrozenEnvMB.e_xc_pbc`. ver4
never had that bug because it never re-evaluated B's AOs at all.
'''

import numpy

from pyscf.ksced.mb.arrays import like as _like
from pyscf.ksced.mb.arrays import to_host as _host

# AO values held per chunk while the mesh density is collocated. deriv=1 gives
# four components, so a chunk costs 4 * blk * nao_B * 8 bytes.
_CHUNK_BYTES = 512 << 20


def is_fft_df(df):
    '''True for the plane-wave fitting objects whose J this module can split.

    GDF and the rest never form v_J^B(r), so callers fall back to the mol_ab
    slice for them. gpu4pyscf's FFTDF does not subclass PySCF's -- it derives
    from lib.StreamObject and copies methods across by assignment -- which is
    why the name is checked as well as the type.
    '''
    if df is None:
        return False
    from pyscf.pbc.df.fft import FFTDF
    return isinstance(df, FFTDF) or type(df).__name__ == 'FFTDF'


def _tools_for(df):
    '''pbc.tools on the backend df came from; both expose fft/ifft/get_coulG.'''
    if type(df).__module__.startswith('gpu4pyscf'):
        from gpu4pyscf.pbc import tools
    else:
        from pyscf.pbc import tools
    return tools


def _ao_blocks(df, kpt=None):
    '''(ao, p0, p1) over the uniform mesh, in linear order.

    Linear order is the whole requirement: p0:p1 indexes an array that will be,
    or has been, passed through an FFT. Neither backend's loop permutes by
    default. gpu4pyscf's `ni.get_rho` looks like the obvious thing to borrow,
    but it asks for `sort_grids=True` and returns the points permuted, so it
    cannot be used here.

    The two backends differ only in how the loop is spelled:

        PySCF      df.aoR_loop   -> (ao_k1_etc, p0, p1)
        gpu4pyscf  ni.block_loop -> (ao_ks, weight, coords)
    '''
    kpts = numpy.zeros((1, 3)) if kpt is None else numpy.reshape(kpt, (1, 3))
    if hasattr(df, 'aoR_loop'):                     # PySCF
        for ao_ks_etc, p0, p1 in df.aoR_loop(df.grids, kpts):
            yield ao_ks_etc[0][0], p0, p1
    else:                                           # gpu4pyscf
        p1 = 0
        for ao_ks, _weight, coords in df._numint.block_loop(
                df.cell, df.grids, 0, kpts):
            p0, p1 = p1, p1 + coords.shape[0]
            yield ao_ks[0], p0, p1


def _total_density(rho):
    '''The LDA row of a stored rho_B, spin summed.'''
    if rho.ndim == 3:                               # (2, 4, ngrids)
        return rho[0, 0] + rho[1, 0]
    return rho[0]                                   # (4, ngrids)


class _MeshData:
    '''rho_B on the cell's uniform mesh, and what is read off it.

    Built from B alone, so it survives any displacement of A that keeps the
    lattice and mesh. `_FrozenEnvMB.reset` is what enforces that lifetime.

    `evaluator(coords) -> (4, n) or (2, 4, n)` comes from the environment; it is
    the same callable that serves the per-block density to the offset numint,
    so the two agree on what rho_B is by construction.
    '''

    def __init__(self, mf_b, evaluator):
        self.mf_b = mf_b
        self.evaluator = evaluator
        self._rho = None
        self._vj = None

    @property
    def df(self):
        return getattr(self.mf_b, 'with_df', None)

    def usable(self):
        '''False when B was not fitted on a plane-wave mesh.'''
        return is_fft_df(self.df)

    @property
    def weight(self):
        '''Uniform quadrature weight. Constant over a uniform mesh.'''
        cell = self.df.cell
        return cell.vol / int(numpy.prod(numpy.asarray(self.df.mesh)))

    def rho(self):
        '''rho_B at GGA order, linear mesh order. ver4's rhoR_b.'''
        if self._rho is None:
            self._rho = self._collocate()
        return self._rho

    def _collocate(self):
        df = self.df
        ngrids = int(numpy.prod(numpy.asarray(df.mesh)))
        coords = df.grids.coords
        nao = df.cell.nao
        blk = max(512, int(_CHUNK_BYTES / (4 * 8 * max(nao, 1))))

        out = None
        for p0 in range(0, ngrids, blk):
            p1 = min(ngrids, p0 + blk)
            block = self.evaluator(coords[p0:p1])
            if out is None:
                shape = tuple(block.shape[:-1]) + (ngrids,)
                out = _like(block, numpy.zeros(shape))
            out[..., p0:p1] = block
        return out

    def vj(self):
        '''v_J^B(r), quadrature weight folded in. One Poisson solve, cached.'''
        if self._vj is None:
            df = self.df
            cell = df.cell
            mesh = numpy.asarray(df.mesh)
            tools = _tools_for(df)

            rho = _total_density(self.rho())
            vg = tools.get_coulG(cell, mesh=mesh) * tools.fft(rho, mesh)
            vr = tools.ifft(vg, mesh).real
            vr *= self.weight
            self._vj = vr
        return self._vj

    def energy(self, ni, xc):
        '''E[rho_B] for one functional, straight off the cached density.

        Mirrors the accumulation in pbc/dft/numint.py nr_rks and nr_uks: exc is
        per electron, so it is weighted by the density itself, once per spin
        channel when polarised.
        '''
        rho = self.rho()
        xctype = ni._xc_type(xc)
        if xctype not in ('LDA', 'GGA'):
            raise NotImplementedError(
                'KSCED stores rho_B at GGA order; %s is %s' % (xc, xctype))
        w = self.weight

        def rows(r):
            return r[0] if xctype == 'LDA' else r[:4]

        def den(r):
            return (r if xctype == 'LDA' else r[0]) * w

        if rho.ndim == 3:                           # unrestricted environment
            ra, rb = rows(rho[0]), rows(rho[1])
            exc = ni.eval_xc_eff(xc, (ra, rb), 1, xctype=xctype, spin=1)[0]
            total = den(ra).dot(exc) + den(rb).dot(exc)
        else:
            r = rows(rho)
            exc = ni.eval_xc_eff(xc, r, 1, xctype=xctype)[0]
            total = den(r).dot(exc)
        return float(_host(total))

    def matrix(self, mf_a):
        '''J[rho_B] in A's basis, by quadrature over A's AOs alone.'''
        df_a = mf_a.with_df
        self._check_mesh(df_a)

        vr = self.vj()
        vr_a = None
        j = None
        for ao, p0, p1 in _ao_blocks(df_a, getattr(mf_a, 'kpt', None)):
            if vr_a is None:
                vr_a = _like(ao, vr)
            block = ao.conj().T.dot(ao * vr_a[p0:p1, None])
            j = block if j is None else j + block
        return _host(j).real

    def _check_mesh(self, df_a):
        '''A and B must be on the same mesh for the potential to transfer.

        embed() already rejects a cell_a/cell_b mesh mismatch, but FFTDF.mesh is
        a settable attribute that UniformGrids.reset does not touch, so the two
        fitting objects can still disagree with the cells they came from.
        '''
        mesh_a = numpy.asarray(df_a.mesh)
        mesh_b = numpy.asarray(self.df.mesh)
        if not numpy.array_equal(mesh_a, mesh_b):
            raise ValueError(
                'KSCED monomolecular basis: A is fitted on mesh %r and B on %r, '
                'so v_J[rho_B] is not sampled where A needs it. Give both '
                'subsystems the same mesh.'
                % (tuple(mesh_a), tuple(mesh_b)))
