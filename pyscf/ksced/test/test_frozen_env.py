import unittest
import numpy
from pyscf import gto, dft
from pyscf.ksced.ksced import _FrozenEnv


def _pair():
    """Li+ and H2O in one supermolecular basis, B = Li+, A = H2O."""
    common = dict(basis='sto-3g', verbose=0)
    mol_b = gto.M(atom='Li 0 0 0; ghost-O 0 0 1.5; ghost-H 0 0.76 2.09; ghost-H 0 -0.76 2.09',
                  charge=1, **common)
    mol_a = gto.M(atom='ghost-Li 0 0 0; O 0 0 1.5; H 0 0.76 2.09; H 0 -0.76 2.09',
                  **common)
    return mol_a, mol_b


class KnownValues(unittest.TestCase):
    def test_shapes_match_shared_basis(self):
        mol_a, mol_b = _pair()
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        env = _FrozenEnv(mf_b)
        self.assertEqual(env.dm_b.shape, (mol_a.nao, mol_a.nao))
        self.assertEqual(env.get_vne_b(mol_a).shape, (mol_a.nao, mol_a.nao))

    def test_caches_are_reused(self):
        mol_a, mol_b = _pair()
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        env = _FrozenEnv(mf_b)
        first = env.get_vne_b(mol_a)
        second = env.get_vne_b(mol_a)
        self.assertIs(first, second)

    def test_reset_drops_caches(self):
        mol_a, mol_b = _pair()
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        env = _FrozenEnv(mf_b)
        first = env.get_vne_b(mol_a)
        env.reset()
        self.assertIsNot(first, env.get_vne_b(mol_a))

    def test_e_tnad_b_is_positive(self):
        mol_a, mol_b = _pair()
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        env = _FrozenEnv(mf_b)
        mf_a = dft.RKS(mol_a, xc='PBE')
        mf_a.initialize_grids(mol_a, numpy.zeros((mol_a.nao, mol_a.nao)))
        t_b = env.e_tnad_b(mf_a._numint, mol_a, mf_a.grids, 'LDA_K_TF', 2000)
        self.assertGreater(t_b, 0.0)


if __name__ == '__main__':
    unittest.main()
