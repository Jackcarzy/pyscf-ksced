'''rho_B on the quadrature grid, memoised by block coordinates.

Why coordinates and not call order. All four nr_rks implementations block the
grid differently:

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


class _GridDensity:
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

    def begin_spin_access(self):
        self._spin_access = {}

    def note_spin_access(self, coords, spin):
        '''Record one alpha/beta service per block and reject duplicates.'''
        key = _block_key(coords)
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
