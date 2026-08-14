import unittest
import numpy
from pyscf import gto, dft
from pyscf.ksced.ksced import _FrozenEnv
from pyscf.ksced.mb.env import _FrozenEnvMB

COMMON = dict(basis='sto-3g', verbose=0)
GEOM_A = 'O 0 0 1.5; H 0 0.76 2.09; H 0 -0.76 2.09'
GEOM_B = 'Li 0 0 0'


def _mb_pair():
    return (gto.M(atom=GEOM_A, **COMMON),
            gto.M(atom=GEOM_B, charge=1, **COMMON))


def _sb_pair():
    """The same physical system in one shared basis, for cross-checking."""
    a = gto.M(atom='ghost-Li 0 0 0; ' + GEOM_A, **COMMON)
    b = gto.M(atom='Li 0 0 0; ghost-O 0 0 1.5; ghost-H 0 0.76 2.09; '
                   'ghost-H 0 -0.76 2.09', charge=1, **COMMON)
    return a, b


class Shapes(unittest.TestCase):
    def test_cross_terms_live_in_As_basis(self):
        mol_a, mol_b = _mb_pair()
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        env = _FrozenEnvMB(mf_b, mol_a)
        self.assertEqual(env.dm_b.shape, (mol_b.nao, mol_b.nao))
        self.assertEqual(env.get_vne_b(mol_a).shape, (mol_a.nao, mol_a.nao))
        self.assertEqual(env.get_j_b(None, mol_a).shape, (mol_a.nao, mol_a.nao))

    def test_mol_ab_is_built_with_a_first(self):
        mol_a, mol_b = _mb_pair()
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        env = _FrozenEnvMB(mf_b, mol_a)
        self.assertEqual(env.mol_ab.nao, mol_a.nao + mol_b.nao)
        self.assertEqual(env.mol_ab.atom_symbol(0), 'O')
        self.assertEqual(env.mol_ab.nelectron,
                         mol_a.nelectron + mol_b.nelectron)


class AgreesWithSharedBasis(unittest.TestCase):
    """The AB slices must reproduce what the SB path computes directly."""

    def test_vne_b_matches_the_shared_basis_value(self):
        sb_a, sb_b = _sb_pair()
        mf_b_sb = dft.RKS(sb_b, xc='PBE').run()
        sb = _FrozenEnv(mf_b_sb)
        ref = sb.get_vne_b(sb_a)

        env = _FrozenEnvMB(mf_b_sb, sb_a)
        got = env.get_vne_b(sb_a)
        numpy.testing.assert_allclose(got, ref, atol=1e-10)

    def test_j_b_matches_the_shared_basis_value(self):
        sb_a, sb_b = _sb_pair()
        mf_b_sb = dft.RKS(sb_b, xc='PBE').run()
        mf_a = dft.RKS(sb_a, xc='PBE')
        sb = _FrozenEnv(mf_b_sb)
        ref = sb.get_j_b(mf_a, sb_a)

        env = _FrozenEnvMB(mf_b_sb, sb_a)
        got = env.get_j_b(mf_a, sb_a)
        numpy.testing.assert_allclose(got, ref, atol=1e-10)

    def test_e_vne_a_rho_b_matches_the_shared_basis_value(self):
        sb_a, sb_b = _sb_pair()
        mf_b_sb = dft.RKS(sb_b, xc='PBE').run()
        sb = _FrozenEnv(mf_b_sb)

        # The SB path forms <V_ne[A]|rho_B> from V_ne[A] in the shared basis.
        vne_a = sb_a.intor_symmetric('int1e_nuc')
        ref = sb.e_vne_a_rho_b(vne_a)

        env = _FrozenEnvMB(mf_b_sb, sb_a)
        got = env.e_vne_a_rho_b(None)
        self.assertAlmostEqual(got, ref, 9)


class Caching(unittest.TestCase):
    def test_caches_are_reused_and_reset(self):
        mol_a, mol_b = _mb_pair()
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        env = _FrozenEnvMB(mf_b, mol_a)
        first = env.get_vne_b(mol_a)
        self.assertIs(first, env.get_vne_b(mol_a))
        env.reset()
        self.assertIsNot(first, env.get_vne_b(mol_a))


if __name__ == '__main__':
    unittest.main()
