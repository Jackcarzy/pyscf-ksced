"""Benchmark CH3SH on Au20 with the plugin: interaction energy

    python bench_au20.py

Molecular, supermolecular basis, GPU4PySCF. Run from this directory, which
holds structure.xyz.

The interaction energy follows a three-term subtraction:

    Eint = E[A in B] - E[c in B] - E[a]

    A   = the partition's gold plus the adsorbate
    c  = the same gold without the adsorbate, embedded in the same B
    a  = the adsorbate alone
"""
import numpy
from gpu4pyscf import dft
from gpu4pyscf.scf.smearing import smearing
from pyscf import gto, ksced

#0 setup
XC = 'PBE'
T_NAD = 'LDA_K_TF'
CONV_TOL = 1e-6
SIGMA = 0.003
COMMON = dict(basis='gth-dzvp-molopt-sr', pseudo='gth-pbe', verbose=4)

SYM = ['Au'] * 20 + ['C', 'S', 'H', 'H', 'H', 'H']
ALL = list(range(26))
ADSORBATE = [20, 21, 22, 23, 24, 25]
A_AU = [1, 3, 4, 6, 7, 8, 10, 13, 14, 17]
B_AU = [i for i in range(20) if i not in A_AU]

y = numpy.loadtxt('structure.xyz')


def build(real):
    """A Mole holding `real`; the other centres come along as ghosts, so every
    fragment shares one basis and dm_a and dm_b can simply be added."""
    atoms = [(SYM[i] if i in real else 'X-' + SYM[i], tuple(y[i]))
             for i in ALL]
    return gto.M(atom=atoms, **COMMON)


def rks(mol):
    mf = dft.RKS(mol, xc=XC).density_fit()
    mf.init_guess = 'atom'
    mf.conv_tol = CONV_TOL
    mf = smearing(mf, sigma=SIGMA, method='gauss')
    mf.diis_damp = 0.8
    return mf


mol_b = build(B_AU)
mol_A = build(A_AU + ADSORBATE)
mol_c = build(A_AU)
mol_a = build(ADSORBATE)
mol_Ab = build(ALL)
mol_cb = build(list(range(20)))

#1 embedding B (Au10)
mf_b = rks(mol_b)
mf_b.kernel()

#2 A in B (CH3SH+Au10 in Au10)
mf_Ainb = ksced.embed(rks(mol_A), mf_b, mol_ab=mol_Ab)
mf_Ainb.t_nad = T_NAD
mf_Ainb.kernel()

#3 c in B (Au10 in Au10)
mf_cinb = ksced.embed(rks(mol_c), mf_b, mol_ab=mol_cb)
mf_cinb.t_nad = T_NAD
mf_cinb.kernel()

#4 a (CH3SH)
mf_a = rks(mol_a)
mf_a.kernel()

#5 get interaction energy (kcal/mol)
eint = float(mf_Ainb.e_tot) - float(mf_cinb.e_tot) - float(mf_a.e_tot)
print('E[A in B]                   %.10f Ha' % float(mf_Ainb.e_tot))
print('E[c in B]                   %.10f Ha' % float(mf_cinb.e_tot))
print('E[a]                        %.10f Ha' % float(mf_a.e_tot))
print('non-additive kinetic energy %.10f Ha' % float(mf_Ainb.e_tnad))
print('interaction energy          %.10f Ha  = %.3f kcal/mol'
      % (eint, eint * 627.503))