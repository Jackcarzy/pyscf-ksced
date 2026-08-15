"""Unrestricted KSCED, molecular, supermolecular basis.

The strongest gate here is U0: a closed-shell system run with UKS on both
subsystems must reproduce the RKS answer. A converged UKS calculation on a
closed-shell system has rho_alpha == rho_beta identically, so any deviation is
a bug in the spin bookkeeping rather than physics. Plain PySCF agrees to 1.4e-14
on this system, so the embedding has no excuse to do worse.

That one test catches all three of the silent-wrongness modes at once: a
double-counted rho_B, a mis-sliced spin axis, and a wrongly summed vj.
"""
import unittest

import numpy
from pyscf import gto, dft, ksced

COMMON = dict(basis='sto-3g', verbose=0)
GEOM_A = 'O 0 0 1.5; H 0 0.76 2.09; H 0 -0.76 2.09'
GEOM_B = 'Li 0 0 0'
GEOM_AB = 'Li 0 0 0; ' + GEOM_A

# Supermolecular construction: every fragment carries all four centres.
SB_A = 'ghost-Li 0 0 0; ' + GEOM_A
SB_B = ('Li 0 0 0; ghost-O 0 0 1.5; ghost-H 0 0.76 2.09; '
        'ghost-H 0 -0.76 2.09')

CONV = 1e-12

# e_tot and e_tnad do not converge at the same rate, so they get different
# tolerances. e_tot is variational: the density error enters at second order
# and it reaches exact agreement. e_tnad is a non-variational functional of the
# converged density, so it inherits that error linearly. Measured on this
# system, RKS against UKS:
#
#   conv_tol 1e-09   d(e_tot) 3.0e-12   d(e_tnad) 1.3e-07
#   conv_tol 1e-11   d(e_tot) 3.4e-13   d(e_tnad) 1.2e-08
#   conv_tol 1e-13   d(e_tot) 0.0       d(e_tnad) 7.0e-10
#
# Gating e_tnad at e_tot's precision would be testing the SCF convergence
# threshold rather than the spin bookkeeping. A double-counted rho_B is a
# factor-of-two error, nine orders above either tolerance.
TOL_E_TOT = 11
TOL_E_TNAD = 8


def _closed_shell_cells():
    """Li+ / H2O -- both fragments closed shell."""
    return (gto.M(atom=SB_A, **COMMON),
            gto.M(atom=SB_B, charge=1, **COMMON),
            gto.M(atom=GEOM_AB, charge=1, **COMMON))


def _open_shell_cells():
    """Neutral Li (doublet) / H2O -- same geometry, Li left uncharged."""
    return (gto.M(atom=SB_A, **COMMON),
            gto.M(atom=SB_B, spin=1, **COMMON),
            gto.M(atom=GEOM_AB, spin=1, **COMMON))


class U0ClosedShellEquivalence(unittest.TestCase):
    """UKS on a closed-shell system must reproduce RKS exactly."""

    def _run(self, cls_a, cls_b):
        mol_a, mol_b, mol_ab = _closed_shell_cells()
        mf_b = cls_b(mol_b, xc='PBE')
        mf_b.conv_tol = CONV
        mf_b.kernel()

        mf_a = ksced.embed(cls_a(mol_a, xc='PBE'), mf_b, mol_ab=mol_ab)
        mf_a.conv_tol = CONV
        mf_a.kernel()
        return mf_a.e_tot, mf_a.e_tnad

    def test_u_in_u_matches_r_in_r(self):
        ref = self._run(dft.RKS, dft.RKS)
        got = self._run(dft.UKS, dft.UKS)
        self.assertAlmostEqual(got[0], ref[0], TOL_E_TOT)
        self.assertAlmostEqual(got[1], ref[1], TOL_E_TNAD)

    def test_u_in_r_matches_r_in_r(self):
        """Unrestricted A, restricted B: rho_B must split evenly, not double."""
        ref = self._run(dft.RKS, dft.RKS)
        got = self._run(dft.UKS, dft.RKS)
        self.assertAlmostEqual(got[0], ref[0], TOL_E_TOT)
        self.assertAlmostEqual(got[1], ref[1], TOL_E_TNAD)

    def test_r_in_u_matches_r_in_r(self):
        """Restricted A in an unpolarised-but-unrestricted B.

        The spin-averaged potential is exact here, because both channels of a
        closed-shell B are identical, so the average is the thing itself.
        """
        ref = self._run(dft.RKS, dft.RKS)
        got = self._run(dft.RKS, dft.UKS)
        self.assertAlmostEqual(got[0], ref[0], TOL_E_TOT)
        self.assertAlmostEqual(got[1], ref[1], TOL_E_TNAD)


class Dispatch(unittest.TestCase):
    def test_unrestricted_a_selects_the_uks_class(self):
        mol_a, mol_b, mol_ab = _closed_shell_cells()
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        mf_a = ksced.embed(dft.UKS(mol_a, xc='PBE'), mf_b, mol_ab=mol_ab)
        from pyscf.ksced.uks import KSCEDUKS
        self.assertIsInstance(mf_a, KSCEDUKS)

    def test_restricted_a_still_selects_the_rks_class(self):
        mol_a, mol_b, mol_ab = _closed_shell_cells()
        mf_b = dft.RKS(mol_b, xc='PBE').run()
        mf_a = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b, mol_ab=mol_ab)
        from pyscf.ksced.rks import KSCEDRKS
        self.assertIsInstance(mf_a, KSCEDRKS)


class OpenShell(unittest.TestCase):
    """A genuinely polarised case: neutral Li doublet frozen around H2O."""

    def test_u_in_r_runs_and_binds(self):
        mol_a, mol_b, mol_ab = _open_shell_cells()
        mf_b = dft.UKS(mol_b, xc='PBE')
        mf_b.conv_tol = 1e-9
        mf_b.kernel()

        mf_a = ksced.embed(dft.UKS(mol_a, xc='PBE'), mf_b, mol_ab=mol_ab)
        mf_a.conv_tol = 1e-9
        mf_a.kernel()

        self.assertTrue(mf_a.converged)
        self.assertGreater(mf_a.e_tnad, 0.0)

    def test_frozen_b_is_actually_polarised(self):
        """Guard the guard: if B came out unpolarised the U-in-U test above
        would be vacuous, since it would reduce to the closed-shell case."""
        _, mol_b, _ = _open_shell_cells()
        mf_b = dft.UKS(mol_b, xc='PBE')
        mf_b.conv_tol = 1e-9
        mf_b.kernel()
        dm_b = mf_b.make_rdm1()
        self.assertEqual(dm_b.ndim, 3)
        self.assertGreater(abs(dm_b[0] - dm_b[1]).max(), 1e-3)


if __name__ == '__main__':
    unittest.main()
