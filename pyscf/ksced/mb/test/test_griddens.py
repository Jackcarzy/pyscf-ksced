import unittest
import numpy
from pyscf import gto, dft
from pyscf.ksced.mb.griddens import _GridDensity, _block_key


class BlockKey(unittest.TestCase):
    def test_same_coords_same_key(self):
        c = numpy.arange(30.).reshape(10, 3)
        self.assertEqual(_block_key(c), _block_key(c.copy()))

    def test_different_coords_different_key(self):
        a = numpy.arange(30.).reshape(10, 3)
        b = a + 1.0
        self.assertNotEqual(_block_key(a), _block_key(b))

    def test_different_length_different_key(self):
        a = numpy.arange(30.).reshape(10, 3)
        self.assertNotEqual(_block_key(a), _block_key(a[:5]))


class Memoisation(unittest.TestCase):
    def test_evaluator_runs_once_per_distinct_block(self):
        calls = []

        def evaluator(coords):
            calls.append(len(coords))
            return numpy.ones((4, len(coords)))

        gd = _GridDensity(evaluator)
        c = numpy.arange(30.).reshape(10, 3)
        gd.rho(c)
        gd.rho(c.copy())
        self.assertEqual(calls, [10])
        self.assertEqual(gd.nblocks, 1)

    def test_a_different_partition_is_served_correctly(self):
        """The xc call and the t_nad call may block the grid differently."""

        def evaluator(coords):
            return numpy.tile(coords[:, 0], (4, 1))

        gd = _GridDensity(evaluator)
        c = numpy.arange(30.).reshape(10, 3)
        whole = gd.rho(c)
        halves = numpy.concatenate([gd.rho(c[:4]), gd.rho(c[4:])], axis=1)
        numpy.testing.assert_allclose(whole, halves)
        self.assertEqual(gd.nblocks, 3)

    def test_reset_drops_the_cache(self):
        calls = []

        def evaluator(coords):
            calls.append(len(coords))
            return numpy.ones((4, len(coords)))

        gd = _GridDensity(evaluator)
        c = numpy.arange(30.).reshape(10, 3)
        gd.rho(c)
        gd.reset()
        gd.rho(c)
        self.assertEqual(calls, [10, 10])


class ParticleNumber(unittest.TestCase):
    def test_integrates_to_N_B(self):
        """G3: rho_B on the AB grid must integrate to B's electron count."""
        common = dict(basis='sto-3g', verbose=0)
        mol_a = gto.M(atom='O 0 0 1.5; H 0 0.76 2.09; H 0 -0.76 2.09', **common)
        mol_b = gto.M(atom='Li 0 0 0', charge=1, **common)
        mol_ab = gto.conc_mol(mol_a, mol_b)

        mf_b = dft.RKS(mol_b, xc='PBE')
        mf_b.kernel()
        dm_b = mf_b.make_rdm1()

        grids = dft.gen_grid.Grids(mol_ab)
        grids.build()
        ni = mf_b._numint

        def evaluator(coords):
            ao = ni.eval_ao(mol_b, coords, deriv=1)
            return ni.eval_rho(mol_b, ao, dm_b, xctype='GGA')

        gd = _GridDensity(evaluator)
        n = float(gd.rho(grids.coords)[0].dot(grids.weights))
        self.assertAlmostEqual(n, mol_b.nelectron, 5)


if __name__ == '__main__':
    unittest.main()
