'''rho_B on the quadrature grid, memoised by block coordinates.

Why coordinates and not call order. All four nr_rks implementations block the
grid differently:

  CPU mol   ni.block_loop via a closure in nr_rks; block size depends on deriv,
            so the xc (GGA) and t_nad (LDA) calls partition the grid differently
  CPU pbc   same differing-partition hazard
  GPU pbc   ni.block_loop(..., sort_grids=True) permutes the points
  GPU mol   one ni.block_loop per device inside a ThreadPoolExecutor, over
            disjoint ranges from gen_grid_range -- concurrent, and not starting
            from zero

A table keyed by position cannot serve all four. Keying on the coordinates
themselves and filling lazily on miss is correct under every one of them: an
unseen partition costs one extra evaluation, never a wrong density.
'''

import threading

import numpy

from pyscf.ksced.mb.arrays import to_host as _to_host




def _block_key(coords):
    '''Order-independent identity of a grid block.

    First point, last point and length. Collisions are checked on lookup, so a
    false match raises rather than returning the wrong density.
    '''
    c = _to_host(coords)
    return (c.shape[0],
            float(c[0, 0]), float(c[0, 1]), float(c[0, 2]),
            float(c[-1, 0]), float(c[-1, 1]), float(c[-1, 2]))


class _GridDensity:
    '''Memoised rho_B(coords), always stored at GGA order, shape (4, n).

    evaluator(coords) -> ndarray (4, n) is supplied by the backend adapter,
    which is the only part that knows how to evaluate B's AOs.
    '''

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self._cache = {}
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

        # Keep the density on whichever backend produced it. Calling
        # numpy.asarray here would raise on cupy: "Implicit conversion to a
        # NumPy array is not allowed."
        rho = self.evaluator(coords)
        if not hasattr(rho, 'ndim'):
            rho = numpy.asarray(rho)
        if rho.ndim == 1:
            rho = rho[None, :]
        if rho.shape[0] not in (1, 4, 5):
            raise RuntimeError('KSCED: unexpected rho_B shape %r' % (rho.shape,))
        with self._lock:
            self._cache[key] = (_to_host(coords), rho)
        return rho
