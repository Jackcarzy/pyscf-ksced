import unittest
import numpy
from pyscf.pbc import gto, dft
from pyscf import ksced


def _null_partition():
    """A holds every atom; B is ghost-only, so rho_B = 0."""
    common = dict(a=numpy.eye(3) * 6.0, basis='gth-szv', pseudo='gth-pbe',
                  mesh=[16, 16, 16], verbose=0)
    cell_a = gto.M(atom='He 0 0 0; He 0 0 2.0', **common)
    cell_b = gto.M(atom='ghost-He 0 0 0; ghost-He 0 0 2.0', **common)
    return cell_a, cell_b


class KnownValues(unittest.TestCase):
    def test_null_partition_matches_plain_rks(self):
        cell_a, cell_b = _null_partition()
        mf_b = dft.RKS(cell_b, xc='PBE')
        mf_b.kernel()

        mf_a = ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b)
        mf_a.kernel()

        ref = dft.RKS(cell_a, xc='PBE').run()
        self.assertAlmostEqual(mf_a.e_tot, ref.e_tot, 7)
        self.assertAlmostEqual(mf_a.e_tnad, 0.0, 7)

    def test_dispatch_picked_the_periodic_class(self):
        cell_a, cell_b = _null_partition()
        mf_b = dft.RKS(cell_b, xc='PBE').run()
        mf_a = ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b)
        from pyscf.ksced.pbcrks import KSCEDPBCRKS
        self.assertIsInstance(mf_a, KSCEDPBCRKS)


if __name__ == '__main__':
    unittest.main()
