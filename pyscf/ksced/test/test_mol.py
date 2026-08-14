import unittest
import numpy
from pyscf import gto, dft, ksced


def _null_partition():
    """A holds every atom; B holds none. Ghost-only B means rho_B = 0."""
    common = dict(basis='sto-3g', verbose=0)
    atoms_a = 'O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24'
    atoms_b = 'ghost-O 0 0 0; ghost-H 0 0 0.96; ghost-H 0.93 0 -0.24'
    return gto.M(atom=atoms_a, **common), gto.M(atom=atoms_b, **common)


class KnownValues(unittest.TestCase):
    def test_null_partition_matches_plain_rks(self):
        mol_a, mol_b = _null_partition()

        mf_b = dft.RKS(mol_b, xc='PBE')
        mf_b.kernel()
        self.assertAlmostEqual(numpy.einsum('ij,ji->', mf_b.make_rdm1(),
                                            mol_b.intor_symmetric('int1e_ovlp')).real,
                               0.0, 9)

        mf_a = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b)
        mf_a.kernel()

        ref = dft.RKS(mol_a, xc='PBE').run()
        self.assertAlmostEqual(mf_a.e_tot, ref.e_tot, 9)
        self.assertAlmostEqual(mf_a.e_tnad, 0.0, 9)

    def test_embed_is_idempotent_on_class(self):
        mol_a, mol_b = _null_partition()
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        mf_a = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b)
        again = ksced.embed(mf_a, mf_b)
        self.assertIs(again.__class__, mf_a.__class__)

    def test_energy_nuc_includes_cross_term_when_mol_ab_given(self):
        common = dict(basis='sto-3g', verbose=0)
        mol_ab = gto.M(atom='Li 0 0 0; O 0 0 1.5; H 0 0.76 2.09; H 0 -0.76 2.09',
                       charge=1, **common)
        mol_b = gto.M(atom='Li 0 0 0; ghost-O 0 0 1.5; ghost-H 0 0.76 2.09; ghost-H 0 -0.76 2.09',
                      charge=1, **common)
        mol_a = gto.M(atom='ghost-Li 0 0 0; O 0 0 1.5; H 0 0.76 2.09; H 0 -0.76 2.09',
                      **common)
        mf_b = dft.RKS(mol_b, xc='PBE').run()

        without = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b)
        with_ab = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b, mol_ab=mol_ab)

        expected = mol_ab.energy_nuc() - mol_a.energy_nuc() - mol_b.energy_nuc()
        self.assertAlmostEqual(with_ab.energy_nuc() - without.energy_nuc(),
                               expected, 10)


if __name__ == '__main__':
    unittest.main()
