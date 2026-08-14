import unittest
import numpy
from pyscf.pbc import gto, dft
from pyscf import ksced

COMMON = dict(a=numpy.eye(3) * 6.0, basis='gth-szv', pseudo='gth-pbe',
              mesh=[16, 16, 16], verbose=0)
A = 'He 0 0 0'
B = 'He 0 0 2.0'


def _mb_cells():
    return gto.M(atom=A, **COMMON), gto.M(atom=B, **COMMON)


def _sb_cells():
    return (gto.M(atom='He 0 0 0; ghost-He 0 0 2.0', **COMMON),
            gto.M(atom='ghost-He 0 0 0; He 0 0 2.0', **COMMON))


class Dispatch(unittest.TestCase):
    def test_M_selects_the_periodic_monomolecular_class(self):
        cell_a, cell_b = _mb_cells()
        mf_b = dft.RKS(cell_b, xc='PBE').run()
        mf_a = ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b, basis_mode='M')
        from pyscf.ksced.mb.pbcrks import KSCEDMBPBCRKS
        self.assertIsInstance(mf_a, KSCEDMBPBCRKS)

    def test_mismatched_mesh_is_rejected(self):
        cell_a, _ = _mb_cells()
        other = dict(COMMON)
        other['mesh'] = [18, 18, 18]
        cell_b = gto.M(atom=B, **other)
        mf_b = dft.RKS(cell_b, xc='PBE').run()
        with self.assertRaises(ValueError):
            ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b, basis_mode='M')


def _integrate_rho_b(mf_a):
    grids = mf_a.grids
    rho_b = mf_a.with_env._griddens.rho(grids.coords)[0]
    return float(numpy.asarray(rho_b).dot(numpy.asarray(grids.weights)))


class G3ParticleNumber(unittest.TestCase):
    """rho_B on the uniform grid must be the density PySCF itself integrates.

    Comparing straight against N_B would test PySCF's quadrature rather than
    this plugin: on a 6 Angstrom cell at mesh 16^3, stock nr_rks reports 2.0933
    electrons for a two-electron He, converging to 2.000042 only by mesh 48^3.
    T_TF integrates rho**(5/3), which has a cusp at every nucleus, and uniform
    grids are not built for that. monkey_patch/README.md records the same
    behaviour on this system.

    So the tight check is against stock nr_rks on the same grid, and the
    convergence check uses a mesh where the quadrature is actually converged.
    """

    def test_matches_what_stock_nr_rks_integrates(self):
        cell_a, cell_b = _mb_cells()
        mf_b = dft.RKS(cell_b, xc='PBE').run()
        mf_a = ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b, basis_mode='M')
        mf_a.kernel()

        stock = mf_b._numint.nr_rks(cell_b, mf_a.grids, 'PBE',
                                    mf_b.make_rdm1(), 0, 1,
                                    numpy.zeros(3), None)[0]
        self.assertAlmostEqual(_integrate_rho_b(mf_a), float(stock), 8)

    def test_converges_to_N_B_on_an_adequate_mesh(self):
        fine = dict(COMMON)
        fine['mesh'] = [32, 32, 32]
        cell_a = gto.M(atom=A, **fine)
        cell_b = gto.M(atom=B, **fine)

        mf_b = dft.RKS(cell_b, xc='PBE').run()
        mf_a = ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b, basis_mode='M')
        mf_a.kernel()
        self.assertAlmostEqual(_integrate_rho_b(mf_a), cell_b.nelectron, 2)


class G1SupermolecularLimit(unittest.TestCase):
    def test_e_tot_and_e_tnad_agree(self):
        sb_a, sb_b = _sb_cells()
        cell_ab = gto.M(atom='He 0 0 0; He 0 0 2.0', **COMMON)

        mf_b = dft.RKS(sb_b, xc='PBE')
        mf_b.conv_tol = 1e-10
        mf_b.kernel()

        sb = ksced.embed(dft.RKS(sb_a, xc='PBE'), mf_b, mol_ab=cell_ab)
        sb.conv_tol = 1e-10
        sb.kernel()

        # mol_ab is not passed in M mode: there it means the concatenated
        # A(+)B basis the cross terms are sliced from, not the real system.
        mb = ksced.embed(dft.RKS(sb_a, xc='PBE'), mf_b,
                         basis_mode='M', _bypass_sb_guard=True)
        mb.conv_tol = 1e-10
        mb.kernel()

        self.assertAlmostEqual(mb.e_tot, sb.e_tot, 9)
        self.assertAlmostEqual(mb.e_tnad, sb.e_tnad, 9)


class Physical(unittest.TestCase):
    def test_mb_is_smaller_and_tnad_is_positive(self):
        cell_a, cell_b = _mb_cells()
        cell_ab = gto.M(atom='He 0 0 0; He 0 0 2.0', **COMMON)
        self.assertLess(cell_a.nao, cell_ab.nao)

        mf_b = dft.RKS(cell_b, xc='PBE').run()
        mf_a = ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b, basis_mode='M')
        mf_a.kernel()
        self.assertGreater(mf_a.e_tnad, 0.0)


if __name__ == '__main__':
    unittest.main()
