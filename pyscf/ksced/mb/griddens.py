'''rho_B on the quadrature grid: gathered from the mesh, or memoised per block.

Two classes, because the two paths want opposite things.

_MeshDensity serves the periodic path. rho_B was already collocated on the
uniform mesh for the Poisson solve, and a uniform grid point carries its own
index, so the offset is a gather -- index arithmetic on the backend that
already holds the array. There is nothing worth memoising: a cache needs a key,
and a key read off device coordinates costs a transfer and a synchronisation on
every block of every iteration, which is more than the gather it would save.

_GridDensity is the fallback, for grids that are not the mesh's own: the
molecular path, a Becke grid inside a periodic calculation, a fitting scheme
that never forms a mesh. There rho_B costs N_grid * nao_B^2 to evaluate and is
worth the key.

Why that key is the coordinates and not the call order. All four nr_rks
implementations block the grid differently:

  CPU mol   ni.block_loop via a closure in nr_rks; block size depends on deriv,
            so the xc (GGA) and t_nad (LDA) calls partition the grid differently
  CPU pbc   same differing-partition hazard
  GPU pbc   ni.block_loop(..., sort_grids=True) permutes the points
  GPU mol   one ni.block_loop per device inside a ThreadPoolExecutor, over
            disjoint ranges from gen_grid_range
'''

import threading

import numpy

from pyscf.ksced.mb.arrays import to_host as _to_host


def _block_key(coords):
    '''Order-independent identity of a grid block.
    '''
    c = _to_host(coords)
    return (c.shape[0],
            float(c[0, 0]), float(c[0, 1]), float(c[0, 2]),
            float(c[-1, 0]), float(c[-1, 1]), float(c[-1, 2]))


def _block_id(coords):
    '''Identity of a grid block, without reading it.

    A block is a view into the array block_loop is walking, so its buffer
    address and its length name it exactly for as long as that walk lasts --
    all the UKS guard needs, since begin_spin_access starts a fresh tally per
    nr_uks call. Both attributes are host-side, so unlike _block_key this costs
    no device transfer and no synchronisation.
    '''
    ptr = getattr(getattr(coords, 'data', None), 'ptr', None)      # cupy
    if ptr is None and isinstance(coords, numpy.ndarray):
        ptr = coords.ctypes.data                                   # numpy
    if ptr is None:
        return _block_key(coords)              # unknown backend: pay for it
    return (coords.shape[0], int(ptr))


class _SpinAccess:
    '''The UKS guard: exactly one alpha and one beta service per grid block.

    Subclasses name the block identity they can afford through _block_ident.
    '''

    _block_ident = staticmethod(_block_key)

    def begin_spin_access(self):
        self._spin_access = {}

    def note_spin_access(self, coords, spin):
        '''Record one alpha/beta service per block and reject duplicates.'''
        key = self._block_ident(coords)
        seen = self._spin_access.setdefault(key, set())
        if spin in seen:
            raise RuntimeError(
                'KSCED: environment density was served twice for spin %d on '
                'one grid block; refusing a double-counted UKS result' % spin)
        seen.add(spin)

    def assert_spin_access(self):
        bad = [seen for seen in self._spin_access.values()
               if seen != {0, 1}]
        if bad:
            raise RuntimeError(
                'KSCED: UKS environment density did not enter exactly once '
                'per spin channel on every grid block')


class _GridDensity(_SpinAccess):
    '''Memoized rho_B(coords), stored at GGA order.

    Restricted densities have shape (4,n); unrestricted densities have shape
    (2,4,n), with the leading axis holding alpha and beta.

    evaluator(coords) -> ndarray (4, n) is supplied by the backend adapter,
    which is the only part that knows how to evaluate B's AOs.
    '''

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self._cache = {}
        self._spin_access = {}
        return self

    @property
    def nblocks(self):
        return len(self._cache)

    def rho(self, coords):
        key = _block_key(coords)
        hit = self._cache.get(key)
        if hit is not None:
            stored_coords, rho = hit
            if not numpy.array_equal(stored_coords, _to_host(coords)):
                raise RuntimeError(
                    'KSCED: grid block key collision; refusing to serve a '
                    'density for a block it was not computed on')
            return rho

        # Keep the density on its original backend; CuPy forbids implicit conversion.
        rho = self.evaluator(coords)
        if not hasattr(rho, 'ndim'):
            rho = numpy.asarray(rho)
        if rho.ndim == 1:
            rho = rho[None, :]
        if rho.ndim not in (2, 3) or rho.shape[-2] not in (1, 4, 5):
            raise RuntimeError('KSCED: unexpected rho_B shape %r' % (rho.shape,))
        with self._lock:
            self._cache[key] = (_to_host(coords), rho)
        return rho


class _MeshDensity(_SpinAccess):
    '''rho_B(coords) gathered from the uniform mesh, with no memoisation.

    mesh is a _MeshData; mesh.rho_at returns the stored values at the points it
    is handed, in the order it is handed them, or None for points that are not
    the mesh's own. Those go to a memoised _GridDensity, which is what they
    would have used had this class not existed.
    '''

    _block_ident = staticmethod(_block_id)

    def __init__(self, mesh, evaluator):
        self.mesh = mesh
        self.fallback = _GridDensity(evaluator)
        self.reset()

    def reset(self):
        self._spin_access = {}
        self._served = 0
        self.fallback.reset()
        return self

    @property
    def nblocks(self):
        '''Blocks served, gathered and evaluated together.

        Not a cache size -- there is no cache to size -- but the number this
        class exists to keep nonzero. _assert_env_density_entered reads it as
        proof that the offset density reached the functional at all, so a
        gather has to be counted exactly like the evaluation it replaced.
        '''
        return self._served + self.fallback.nblocks

    def rho(self, coords):
        got = self.mesh.rho_at(coords)
        if got is None:
            return self.fallback.rho(coords)
        self._served += 1
        return got
