'''UKS gates for the monomolecular molecular implementation.'''

import pytest
from pyscf import dft, gto, ksced

COMMON = dict(basis='sto-3g', verbose=0)
A = 'ghost-Li 0 0 0; O 0 0 1.5; H 0 .76 2.09; H 0 -.76 2.09'
B = ('Li 0 0 0; ghost-O 0 0 1.5; ghost-H 0 .76 2.09; '
     'ghost-H 0 -.76 2.09')
AB = 'Li 0 0 0; O 0 0 1.5; H 0 .76 2.09; H 0 -.76 2.09'


def _closed(a_cls, b_cls, mode):
    mol_a = gto.M(atom=A, **COMMON)
    mol_b = gto.M(atom=B, charge=1, **COMMON)
    mol_ab = gto.M(atom=AB, charge=1, **COMMON)
    mf_b = b_cls(mol_b, xc='PBE')
    mf_b.conv_tol = 1e-11
    mf_b.kernel()
    kw = dict(mol_ab=mol_ab) if mode == 'S' else dict(
        basis_mode='M', _bypass_sb_guard=True)
    mf_a = ksced.embed(a_cls(mol_a, xc='PBE'), mf_b, **kw)
    mf_a.conv_tol = 1e-11
    mf_a.kernel()
    assert mf_a.converged
    return mf_a.e_tot, mf_a.e_tnad


@pytest.mark.parametrize('a_cls,b_cls', [
    (dft.UKS, dft.UKS), (dft.UKS, dft.RKS), (dft.RKS, dft.UKS),
])
def test_u0_closed_shell_and_u1_supermolecular_limit(a_cls, b_cls):
    rr = _closed(dft.RKS, dft.RKS, 'S')
    sb = _closed(a_cls, b_cls, 'S')
    mb = _closed(a_cls, b_cls, 'M')
    assert sb == pytest.approx(rr, abs=2e-8)
    assert mb == pytest.approx(sb, abs=1e-9)

