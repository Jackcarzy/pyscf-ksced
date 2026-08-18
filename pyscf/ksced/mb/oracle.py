'''AB-dimension reference for the monomolecular path. Test-only.

No production module may import this.
'''

import numpy


def pad_dm(dm_a, dm_b, nao_a, nao_b):
    '''Block-diagonal density matrix in the AB basis, A block first.

    conc_mol / conc_cell place argument 1's AOs first, so the A block is
    [:nao_a] and the B block is [nao_a:].
    '''
    out = numpy.zeros((nao_a + nao_b, nao_a + nao_b),
                      dtype=numpy.result_type(dm_a, dm_b))
    out[:nao_a, :nao_a] = dm_a
    out[nao_a:, nao_a:] = dm_b
    return out


def oracle_nr_rks(ni, mol_ab, grids, xc, dm_a, dm_b, nao_a, **kwargs):
    '''Stock nr_rks at the AB dimension, sliced back to A's block.

    Returns (nelec, exc, vxc[A,A]). nelec and exc are for the *total* density,
    matching what the offset numint in mb/numint.py returns.
    '''
    nao_b = dm_b.shape[-1]
    dm_t = pad_dm(dm_a, dm_b, nao_a, nao_b)
    nelec, exc, vxc = ni.nr_rks(mol_ab, grids, xc, dm_t, **kwargs)
    return nelec, exc, vxc[:nao_a, :nao_a]
