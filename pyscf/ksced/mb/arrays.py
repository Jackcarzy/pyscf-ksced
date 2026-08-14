'''Backend-agnostic array helpers for the monomolecular path.

The plugin's supermolecular helpers decide where an array lives by testing
``type(a).__module__.startswith('cupy')``. That is right for plain cupy arrays
but misses GPU4PySCF's tagged arrays, whose module is
``gpu4pyscf.lib.cupy_helper``: the test returns False, numpy.asarray is called
on device memory, and cupy raises

    TypeError: Implicit conversion to a NumPy array is not allowed.

several frames from the cause. These helpers duck-type on ``.get()`` instead,
so a tagged array is recognised as readily as a bare one.
'''

import numpy


def is_device(a):
    '''True when a lives in device memory, tagged or not.'''
    return not isinstance(a, numpy.ndarray) and callable(getattr(a, 'get', None))


def to_host(a):
    '''A numpy view of a, whatever backend it came from.'''
    if isinstance(a, numpy.ndarray):
        return a
    if is_device(a):
        return numpy.asarray(a.get())
    return numpy.asarray(a)


def like(ref, arr):
    '''arr on the same backend as ref, so the two can be combined.

    GPU4PySCF mixes device density matrices with host one-electron integrals,
    and cupy refuses to broadcast against a host array.
    '''
    if is_device(ref) and not is_device(arr):
        import cupy
        return cupy.asarray(arr)
    if not is_device(ref) and is_device(arr):
        return to_host(arr)
    return arr
