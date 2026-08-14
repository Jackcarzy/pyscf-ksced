import unittest
import numpy
from pyscf import gto, dft
from pyscf.ksced.mb import oracle
from pyscf.ksced.mb.griddens import _GridDensity
from pyscf.ksced.mb.numint import _ksced_numint


def _fixture():
    common = dict(basis='sto-3g', verbose=0)
    mol_a = gto.M(atom='O 0 0 1.5; H 0 0.76 2.09; H 0 -0.76 2.09', **common)
    mol_b = gto.M(atom='Li 0 0 0', charge=1, **common)
    mol_ab = gto.conc_mol(mol_a, mol_b)

    mf_b = dft.RKS(mol_b, xc='PBE')
    mf_b.kernel()
    dm_b = mf_b.make_rdm1()

    grids = dft.gen_grid.Grids(mol_ab)
    grids.build()

    mf_a = dft.RKS(mol_a, xc='PBE')
    dm_a = mf_a.get_init_guess()
    ni = mf_a._numint

    def evaluator(coords):
        ao = ni.eval_ao(mol_b, coords, deriv=1)
        return ni.eval_rho(mol_b, ao, dm_b, xctype='GGA')

    return mol_a, mol_b, mol_ab, grids, ni, dm_a, dm_b, _GridDensity(evaluator)


class G2Oracle(unittest.TestCase):
    def test_gga_matches_the_ab_oracle(self):
        mol_a, mol_b, mol_ab, grids, ni, dm_a, dm_b, gd = _fixture()
        ni_t = _ksced_numint(ni, gd)

        n, exc, v = ni_t.nr_rks(mol_a, grids, 'PBE', dm_a)
        rn, rexc, rv = oracle.oracle_nr_rks(ni, mol_ab, grids, 'PBE',
                                            dm_a, dm_b, mol_a.nao)

        self.assertAlmostEqual(n, rn, 8)
        self.assertAlmostEqual(exc, rexc, 8)
        numpy.testing.assert_allclose(v, rv, atol=1e-8)

    def test_lda_kinetic_functional_matches_the_ab_oracle(self):
        """The t_nad call is LDA, and blocks the grid differently from GGA."""
        mol_a, mol_b, mol_ab, grids, ni, dm_a, dm_b, gd = _fixture()
        ni_t = _ksced_numint(ni, gd)

        n, exc, v = ni_t.nr_rks(mol_a, grids, 'LDA_K_TF', dm_a)
        rn, rexc, rv = oracle.oracle_nr_rks(ni, mol_ab, grids, 'LDA_K_TF',
                                            dm_a, dm_b, mol_a.nao)

        self.assertAlmostEqual(exc, rexc, 8)
        numpy.testing.assert_allclose(v, rv, atol=1e-8)

    def test_stock_numint_is_not_mutated(self):
        mol_a, mol_b, mol_ab, grids, ni, dm_a, dm_b, gd = _fixture()
        before = ni.nr_rks(mol_a, grids, 'PBE', dm_a)[1]
        _ksced_numint(ni, gd)
        after = ni.nr_rks(mol_a, grids, 'PBE', dm_a)[1]
        self.assertAlmostEqual(before, after, 12)

    def test_zero_environment_reduces_to_stock(self):
        mol_a, mol_b, mol_ab, grids, ni, dm_a, dm_b, gd = _fixture()
        zero = _GridDensity(lambda coords: numpy.zeros((4, len(coords))))
        ni_t = _ksced_numint(ni, zero)
        a = ni_t.nr_rks(mol_a, grids, 'PBE', dm_a)
        b = ni.nr_rks(mol_a, grids, 'PBE', dm_a)
        self.assertAlmostEqual(a[1], b[1], 10)
        numpy.testing.assert_allclose(a[2], b[2], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
