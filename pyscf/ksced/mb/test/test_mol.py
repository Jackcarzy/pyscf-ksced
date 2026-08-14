import unittest
import numpy
from pyscf import gto, dft, ksced

COMMON = dict(basis='sto-3g', verbose=0)
GEOM_A = 'O 0 0 1.5; H 0 0.76 2.09; H 0 -0.76 2.09'
GEOM_B = 'Li 0 0 0'
GHOST_A = 'ghost-Li 0 0 0; ' + GEOM_A
GHOST_B = ('Li 0 0 0; ghost-O 0 0 1.5; ghost-H 0 0.76 2.09; '
           'ghost-H 0 -0.76 2.09')
GEOM_AB = 'Li 0 0 0; ' + GEOM_A


class Dispatch(unittest.TestCase):
    def test_default_is_supermolecular(self):
        mol_a = gto.M(atom=GHOST_A, **COMMON)
        mol_b = gto.M(atom=GHOST_B, charge=1, **COMMON)
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        mf_a = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b)
        from pyscf.ksced.rks import KSCEDRKS
        self.assertIsInstance(mf_a, KSCEDRKS)

    def test_M_selects_the_monomolecular_class(self):
        mol_a = gto.M(atom=GEOM_A, **COMMON)
        mol_b = gto.M(atom=GEOM_B, charge=1, **COMMON)
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        mf_a = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b, basis_mode='M')
        from pyscf.ksced.mb.rks import KSCEDMBRKS
        self.assertIsInstance(mf_a, KSCEDMBRKS)

    def test_ghost_cells_in_M_mode_are_rejected(self):
        mol_a = gto.M(atom=GHOST_A, **COMMON)
        mol_b = gto.M(atom=GHOST_B, charge=1, **COMMON)
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        with self.assertRaises(ValueError):
            ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b, basis_mode='M')

    def test_unknown_basis_mode_is_rejected(self):
        mol_a = gto.M(atom=GEOM_A, **COMMON)
        mol_b = gto.M(atom=GEOM_B, charge=1, **COMMON)
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        with self.assertRaises(ValueError):
            ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b, basis_mode='mono')


class G1SupermolecularLimit(unittest.TestCase):
    """The MB path fed ghost-built cells must reproduce the SB path exactly."""

    def test_e_tot_and_e_tnad_agree(self):
        mol_a = gto.M(atom=GHOST_A, **COMMON)
        mol_b = gto.M(atom=GHOST_B, charge=1, **COMMON)
        mol_ab = gto.M(atom=GEOM_AB, charge=1, **COMMON)

        mf_b = dft.RKS(mol_b, xc='PBE')
        mf_b.conv_tol = 1e-10
        mf_b.kernel()

        sb = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b, mol_ab=mol_ab)
        sb.conv_tol = 1e-10
        sb.kernel()

        # Same input, MB machinery. mol_ab is NOT passed: in M mode it means the
        # concatenated A(+)B basis that the cross terms are sliced from, not the
        # real whole system. E_nn is unaffected -- ghosts carry no charge, so
        # conc(mol_a, mol_b).energy_nuc() equals mol_ab.energy_nuc().
        # _bypass_sb_guard is the documented hook for exactly this test.
        mb = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b,
                         basis_mode='M', _bypass_sb_guard=True)
        mb.conv_tol = 1e-10
        mb.kernel()

        self.assertAlmostEqual(mb.e_tot, sb.e_tot, 10)
        self.assertAlmostEqual(mb.e_tnad, sb.e_tnad, 10)

    def test_a_supermolecular_mol_ab_is_rejected(self):
        """Passing SB's mol_ab into M mode must raise, not silently mis-slice."""
        mol_a = gto.M(atom=GHOST_A, **COMMON)
        mol_b = gto.M(atom=GHOST_B, **COMMON, charge=1)
        mol_ab = gto.M(atom=GEOM_AB, charge=1, **COMMON)
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        with self.assertRaises(ValueError):
            ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b, mol_ab=mol_ab,
                        basis_mode='M', _bypass_sb_guard=True)


class Physical(unittest.TestCase):
    def test_mb_binds_and_is_cheaper(self):
        mol_a = gto.M(atom=GEOM_A, **COMMON)
        mol_b = gto.M(atom=GEOM_B, charge=1, **COMMON)
        mol_ab = gto.M(atom=GEOM_AB, charge=1, **COMMON)
        self.assertLess(mol_a.nao, mol_ab.nao)

        mf_b = dft.RKS(mol_b, xc='PBE').run()
        mf_a = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b, basis_mode='M')
        mf_a.kernel()

        ref = dft.RKS(mol_a, xc='PBE').run()
        eint = (mf_a.e_tot - ref.e_tot) * 627.503
        self.assertLess(eint, 0.0)
        self.assertGreater(mf_a.e_tnad, 0.0)


if __name__ == '__main__':
    unittest.main()
