import unittest
import numpy

# Probe what the tests actually need, not merely that the names import. A
# gpu4pyscf shim without .pbc is present on login nodes, and cupy imports fine
# there while no CUDA device exists.
try:
    import cupy
    import gpu4pyscf.pbc.dft  # noqa: F401
    HAS_GPU = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_GPU = False

COMMON = dict(a=numpy.eye(3) * 6.0, basis='gth-szv', pseudo='gth-pbe',
              mesh=[24, 24, 24], verbose=0)
A = 'He 0 0 0'
B = 'He 0 0 2.0'


@unittest.skipUnless(HAS_GPU, 'gpu4pyscf or cupy not importable')
class GPUPeriodic(unittest.TestCase):
    def test_mb_runs_and_tnad_is_positive(self):
        from pyscf.pbc import gto
        from gpu4pyscf.pbc import dft as gdft
        from pyscf import ksced

        cell_a = gto.M(atom=A, **COMMON)
        cell_b = gto.M(atom=B, **COMMON)

        mf_b = gdft.RKS(cell_b, xc='PBE')
        mf_b.conv_tol = 1e-8
        mf_b.kernel()

        mf_a = ksced.embed(gdft.RKS(cell_a, xc='PBE'), mf_b, basis_mode='M')
        mf_a.conv_tol = 1e-8
        mf_a.kernel()

        self.assertTrue(mf_a.converged)
        self.assertGreater(mf_a.e_tnad, 0.0)

    def test_gpu_agrees_with_cpu(self):
        """Same system, both backends. The tolerance is set by GPU scatter,
        not by the CPU convergence criterion."""
        from pyscf.pbc import gto, dft as cdft
        from gpu4pyscf.pbc import dft as gdft
        from pyscf import ksced

        cell_a = gto.M(atom=A, **COMMON)
        cell_b = gto.M(atom=B, **COMMON)

        def run(dft_mod):
            mf_b = dft_mod.RKS(cell_b, xc='PBE')
            mf_b.conv_tol = 1e-8
            mf_b.kernel()
            mf_a = ksced.embed(dft_mod.RKS(cell_a, xc='PBE'), mf_b,
                               basis_mode='M')
            mf_a.conv_tol = 1e-8
            mf_a.kernel()
            return mf_a.e_tot, mf_a.e_tnad

        cpu = run(cdft)
        gpu = run(gdft)
        print('\nGPU-CPU e_tot  delta = %.3e Ha' % abs(gpu[0] - cpu[0]))
        print('GPU-CPU e_tnad delta = %.3e Ha' % abs(gpu[1] - cpu[1]))
        self.assertAlmostEqual(gpu[0], cpu[0], 5)
        self.assertAlmostEqual(gpu[1], cpu[1], 5)


@unittest.skipUnless(HAS_GPU, 'gpu4pyscf or cupy not importable')
class GPUMolecularRefused(unittest.TestCase):
    """Molecular MB on GPU4PySCF is refused, and refused loudly.

    Probed on an H200: its nr_rks distributes grid blocks across devices
    without calling ni.block_loop, and the class has no _gen_rho_evaluator, so
    neither hook fires. nr_rks then runs to completion and returns an energy
    computed from rho_A alone -- 0 blocks seen out of 33792 grid points, no
    exception. The guard turns that silent wrong answer into a named error.
    """

    def test_molecular_gpu_raises_rather_than_returning_a_wrong_energy(self):
        from pyscf import gto, ksced
        from gpu4pyscf import dft as gdft

        common = dict(basis='sto-3g', verbose=0)
        mol_a = gto.M(atom='O 0 0 1.5; H 0 0.76 2.09; H 0 -0.76 2.09', **common)
        mol_b = gto.M(atom='Li 0 0 0', charge=1, **common)

        mf_b = gdft.RKS(mol_b, xc='PBE')
        mf_b.kernel()
        mf_a = ksced.embed(gdft.RKS(mol_a, xc='PBE'), mf_b, basis_mode='M')
        with self.assertRaises(NotImplementedError) as cm:
            mf_a.kernel()
        self.assertIn('environment density was never added', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
