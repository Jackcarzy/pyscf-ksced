"""Unrestricted KSCED, periodic, gamma point.

The periodic path is the production one, and it is the path where the spin
dispatch is easiest to get silently wrong: pyscf.pbc.scf.uhf.UHF copies the
molecular UHF methods by assignment rather than inheriting them, so
isinstance(mf, pyscf.scf.uhf.UHF) is False for a periodic UKS object. A
dispatch written that way routes periodic UKS into the restricted class and
converges to a wrong answer without raising. test_periodic_uks_reaches_the_uks_class
is the guard against that regression.

The strongest numerical gate is U0, as in the molecular file: a closed-shell
system run unrestricted must reproduce the restricted answer, which catches a
double-counted rho_B, a mis-sliced spin axis and a wrongly summed vj at once.
"""
import unittest

import numpy
from pyscf.pbc import gto, dft
from pyscf import ksced

COMMON = dict(a=numpy.eye(3) * 6.0, basis='gth-szv', pseudo='gth-pade',
              mesh=[16, 16, 16], verbose=0)
CONV = 1e-10

# Periodic SCF converges the density to conv_tol, and e_tnad inherits that
# error linearly because it is a non-variational functional of the converged
# density, while e_tot is variational and reaches ~1e-13. Same split as the
# molecular file, one digit looser: the uniform grid at mesh 16 is coarser than
# a Becke grid, so the two backends' block partitions agree less exactly. A
# double-counted rho_B is a factor of two, nine orders above either tolerance.
TOL_E_TOT = 10
TOL_E_TNAD = 9


def _closed_cells():
    """He / He, both closed shell -- the U0 reference geometry."""
    return (gto.M(atom='He 0 0 0; ghost-He 0 0 2.4', **COMMON),
            gto.M(atom='ghost-He 0 0 0; He 0 0 2.4', **COMMON),
            gto.M(atom='He 0 0 0; He 0 0 2.4', **COMMON))


def _open_cells():
    """Closed-shell He as A, an H atom doublet as the frozen environment."""
    return (gto.M(atom='He 0 0 0; ghost-H 0 0 2.4', **COMMON),
            gto.M(atom='ghost-He 0 0 0; H 0 0 2.4', spin=1, **COMMON),
            gto.M(atom='He 0 0 0; H 0 0 2.4', spin=1, **COMMON))


def _run(cls_a, cls_b, cells=_closed_cells, mode='S'):
    cell_a, cell_b, cell_ab = cells()
    mf_b = cls_b(cell_b, xc='PBE')
    mf_b.conv_tol = CONV
    mf_b.kernel()

    kw = dict(mol_ab=cell_ab) if mode == 'S' else dict(
        basis_mode='M', _bypass_sb_guard=True)
    mf_a = ksced.embed(cls_a(cell_a, xc='PBE'), mf_b, **kw)
    mf_a.conv_tol = CONV
    mf_a.kernel()
    return mf_a


class U0ClosedShellEquivalence(unittest.TestCase):
    """Periodic UKS on a closed-shell system must reproduce periodic RKS."""

    @classmethod
    def setUpClass(cls):
        ref = _run(dft.RKS, dft.RKS)
        cls.ref = (ref.e_tot, ref.e_tnad)

    def _check(self, cls_a, cls_b):
        got = _run(cls_a, cls_b)
        self.assertTrue(got.converged)
        self.assertAlmostEqual(got.e_tot, self.ref[0], TOL_E_TOT)
        self.assertAlmostEqual(got.e_tnad, self.ref[1], TOL_E_TNAD)

    def test_u_in_u_matches_r_in_r(self):
        self._check(dft.UKS, dft.UKS)

    def test_u_in_r_matches_r_in_r(self):
        """Unrestricted A, restricted B: rho_B must split evenly, not double."""
        self._check(dft.UKS, dft.RKS)

    def test_r_in_u_matches_r_in_r(self):
        """Restricted A in an unpolarised-but-unrestricted B.

        The spin-averaged potential is exact here for two reasons at once: both
        channels of a closed-shell B are identical, and the average is in any
        case the exact gradient of the reported energy under A's own
        restriction.
        """
        self._check(dft.RKS, dft.UKS)


class Dispatch(unittest.TestCase):
    def test_periodic_uks_reaches_the_uks_class(self):
        """isinstance(mf, pyscf.scf.uhf.UHF) is False here; istype is not."""
        cell_a, cell_b, cell_ab = _closed_cells()
        mf_b = dft.RKS(cell_b, xc='PBE').run()
        mf_a = ksced.embed(dft.UKS(cell_a, xc='PBE'), mf_b, mol_ab=cell_ab)
        from pyscf.ksced.pbcuks import KSCEDPBCUKS
        self.assertIsInstance(mf_a, KSCEDPBCUKS)

    def test_polarised_environment_promotes_a_restricted_a(self):
        cell_a, cell_b, cell_ab = _open_cells()
        mf_b = dft.UKS(cell_b, xc='PBE').run()
        mf_a = ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b, mol_ab=cell_ab)
        from pyscf.ksced.pbcuks import KSCEDPBCRKSinU
        self.assertIsInstance(mf_a, KSCEDPBCRKSinU)
        self.assertTrue(mf_a.a_restricted)

    def test_restricted_pair_still_reaches_the_rks_class(self):
        cell_a, cell_b, cell_ab = _closed_cells()
        mf_b = dft.RKS(cell_b, xc='PBE').run()
        mf_a = ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b, mol_ab=cell_ab)
        from pyscf.ksced.pbcrks import KSCEDPBCRKS
        from pyscf.ksced.pbcuks import KSCEDPBCUKS
        self.assertIsInstance(mf_a, KSCEDPBCRKS)
        self.assertNotIsInstance(mf_a, KSCEDPBCUKS)


class MonomolecularLimit(unittest.TestCase):
    """M on ghost-built fragments must reproduce S, per pairing.

    This is the only periodic test that drives the offset numint's spin cursor,
    which is what keeps rho_B from being added once per channel.
    """

    def _check(self, cls_a, cls_b, cells=_closed_cells):
        sb = _run(cls_a, cls_b, cells, 'S')
        mb = _run(cls_a, cls_b, cells, 'M')
        self.assertTrue(mb.converged)
        self.assertAlmostEqual(mb.e_tot, sb.e_tot, TOL_E_TOT)
        self.assertAlmostEqual(mb.e_tnad, sb.e_tnad, TOL_E_TNAD)

    def test_u_in_u(self):
        self._check(dft.UKS, dft.UKS)

    def test_u_in_r(self):
        self._check(dft.UKS, dft.RKS)

    def test_r_in_u_polarised(self):
        """A genuinely polarised environment, not the closed-shell limit."""
        self._check(dft.RKS, dft.UKS, _open_cells)


class OpenShell(unittest.TestCase):
    def test_polarised_environment_binds_and_is_actually_polarised(self):
        cell_a, cell_b, cell_ab = _open_cells()
        mf_b = dft.UKS(cell_b, xc='PBE')
        mf_b.conv_tol = CONV
        mf_b.kernel()

        dm_b = mf_b.make_rdm1()
        self.assertEqual(dm_b.ndim, 3)
        # Guard the guard: an unpolarised B would make the pairing tests
        # vacuous, since they would reduce to the closed-shell case.
        self.assertGreater(abs(dm_b[0] - dm_b[1]).max(), 1e-3)

        mf_a = ksced.embed(dft.UKS(cell_a, xc='PBE'), mf_b, mol_ab=cell_ab)
        mf_a.conv_tol = CONV
        mf_a.kernel()
        self.assertTrue(mf_a.converged)
        self.assertGreater(mf_a.e_tnad, 0.0)


class KPointsRefused(unittest.TestCase):
    """k-point objects must be refused at embed(), not misread as spin.

    KRKS's density matrix is (nkpts, nao, nao) and would be read as alpha/beta;
    KUKS's is (2, nkpts, nao, nao) and would be read as restricted, with
    istype('UHF') False for it as well. Both currently die downstream by
    accident -- on a tuple unpack and a complex cast -- which depends on nkpts
    and on the k-point density being complex.
    """

    def _kpts_mf(self, cls, cell):
        return cls(cell, kpts=cell.make_kpts([2, 1, 1]), xc='PBE')

    def test_krks_environment_is_refused(self):
        cell_a, cell_b, cell_ab = _closed_cells()
        mf_b = self._kpts_mf(dft.KRKS, cell_b)
        with self.assertRaises(NotImplementedError) as cm:
            ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b, mol_ab=cell_ab)
        self.assertIn('gamma-point only', str(cm.exception))

    def test_kuks_environment_is_refused(self):
        """KUKS reports istype('UHF') False, so only this guard catches it."""
        cell_a, cell_b, cell_ab = _closed_cells()
        mf_b = self._kpts_mf(dft.KUKS, cell_b)
        self.assertFalse(mf_b.istype('UHF'))
        with self.assertRaises(NotImplementedError):
            ksced.embed(dft.RKS(cell_a, xc='PBE'), mf_b, mol_ab=cell_ab)

    def test_kpoint_subsystem_a_is_refused(self):
        cell_a, cell_b, cell_ab = _closed_cells()
        mf_b = dft.RKS(cell_b, xc='PBE').run()
        with self.assertRaises(NotImplementedError):
            ksced.embed(self._kpts_mf(dft.KRKS, cell_a), mf_b, mol_ab=cell_ab)

    def test_gamma_point_objects_are_not_caught_by_the_guard(self):
        """The guard must not fire on the supported case."""
        cell_a, cell_b, cell_ab = _closed_cells()
        for cls in (dft.RKS, dft.UKS):
            self.assertFalse(cls(cell_a, xc='PBE').istype('KSCF'))
        mf_b = dft.RKS(cell_b, xc='PBE').run()
        ksced.embed(dft.UKS(cell_a, xc='PBE'), mf_b, mol_ab=cell_ab)


if __name__ == '__main__':
    unittest.main()
