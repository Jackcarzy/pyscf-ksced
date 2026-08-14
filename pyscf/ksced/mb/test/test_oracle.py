import unittest
import numpy
from pyscf import gto, dft
from pyscf.ksced.mb import oracle


def _pair():
    """Li+ and H2O, each in its OWN basis. No ghosts."""
    common = dict(basis='sto-3g', verbose=0)
    mol_a = gto.M(atom='O 0 0 1.5; H 0 0.76 2.09; H 0 -0.76 2.09', **common)
    mol_b = gto.M(atom='Li 0 0 0', charge=1, **common)
    return mol_a, mol_b


class KnownValues(unittest.TestCase):
    def test_pad_dm_is_block_diagonal_with_a_first(self):
        dm_a = numpy.arange(4.).reshape(2, 2)
        dm_b = numpy.arange(9.).reshape(3, 3) + 100.
        out = oracle.pad_dm(dm_a, dm_b, 2, 3)
        self.assertEqual(out.shape, (5, 5))
        numpy.testing.assert_allclose(out[:2, :2], dm_a)
        numpy.testing.assert_allclose(out[2:, 2:], dm_b)
        numpy.testing.assert_allclose(out[:2, 2:], 0.0)
        numpy.testing.assert_allclose(out[2:, :2], 0.0)

    def test_oracle_reproduces_plain_nr_rks_when_b_is_empty(self):
        """With dm_b = 0 the oracle must equal a plain nr_rks on mol_a alone."""
        mol_a, mol_b = _pair()
        mol_ab = gto.conc_mol(mol_a, mol_b)

        mf_a = dft.RKS(mol_a, xc='PBE')
        mf_a.grids.build()
        dm_a = mf_a.get_init_guess()
        dm_b = numpy.zeros((mol_b.nao, mol_b.nao))

        grids = dft.gen_grid.Grids(mol_ab)
        grids.build()

        ni = mf_a._numint
        ref_n, ref_exc, ref_v = ni.nr_rks(mol_a, grids, 'PBE', dm_a)
        n, exc, v = oracle.oracle_nr_rks(ni, mol_ab, grids, 'PBE',
                                         dm_a, dm_b, mol_a.nao)

        self.assertAlmostEqual(n, ref_n, 9)
        self.assertAlmostEqual(exc, ref_exc, 9)
        numpy.testing.assert_allclose(v, ref_v, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
