'''Numint adapters that add a frozen rho_B to every density they evaluate.

Stock nr_rks still drives the loop. Only the density evaluation is intercepted,
so each backend's own blocking, screening and contraction code runs untouched
and the version-drift surface is one method signature rather than a loop body.

block_loop is wrapped to record the current block's coords in a thread-local
slot; the density evaluator reads that slot. Thread-local because GPU4PySCF's
molecular nr_rks runs one block_loop per device concurrently.
'''

import copy
import threading

import numpy


def _rho_for_xctype(rho4, xctype):
    '''Slice the stored (4, n) rho_B to what a given xctype expects.

    PySCF returns a 1-D array for LDA and a (4, n) array for GGA; MGGA wants
    five rows, of which tau is not available from a frozen density stored at
    GGA order, so it is rejected upstream in get_veff.
    '''
    if xctype == 'LDA':
        return rho4[0]
    if xctype == 'GGA':
        return rho4[:4]
    raise NotImplementedError(
        'KSCED monomolecular basis supports LDA and GGA densities only; '
        'got xctype %r' % (xctype,))


class _OffsetState(threading.local):
    def __init__(self):
        self.coords = None


def _ksced_numint(ni, griddens):
    '''Return a copy of ni whose evaluated densities include rho_B.

    ni itself is not mutated: get_veff needs both the offset twin (for rho_t)
    and the stock object (for rho_A alone).
    '''
    base = type(ni)
    state = _OffsetState()

    class _KSCEDNumInt(base):
        '''ni, plus a frozen environment density on every block.'''

        def block_loop(self, *args, **kwargs):
            for block in base.block_loop(self, *args, **kwargs):
                # coords is the last element for every backend's block_loop:
                #   CPU mol  (ao, mask, weight, coords)
                #   CPU pbc  (ao_k1, ao_k2, mask, weight, coords)
                #   GPU pbc  (ao_ks, weight, coords)
                state.coords = block[-1]
                yield block
            state.coords = None

        def _offset(self, rho_a, xctype):
            coords = state.coords
            if coords is None:
                raise RuntimeError(
                    'KSCED: density evaluated outside a wrapped block_loop; '
                    'the offset numint cannot locate the environment density')
            rho_b = _rho_for_xctype(griddens.rho(coords), xctype)
            return rho_a + rho_b

        def _gen_rho_evaluator(self, mol, dms, hermi=0, with_lapl=True,
                               grids=None):
            make_rho, ndms, nao = base._gen_rho_evaluator(
                self, mol, dms, hermi, with_lapl, grids)

            def make_rho_offset(idm, ao, mask, xctype):
                return self._offset(make_rho(idm, ao, mask, xctype), xctype)

            return make_rho_offset, ndms, nao

    obj = copy.copy(ni)
    obj.__class__ = _KSCEDNumInt
    return obj
