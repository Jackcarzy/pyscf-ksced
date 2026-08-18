"""Benchmark NH3 on Au120 with the plugin: interaction energy, gamma point

    python 03-NH3Au10_in_Au110.py

Periodic, monomolecular basis, GPU4PySCF. 02 is the same calculation in the
supermolecular basis.

Same three-term recipe as the published ver4 driver and as 01:

    Eint = E[A in B] - E[c in B] - E[a]

    A   = the partition's gold plus NH3
    c   = the same gold without NH3
    a   = NH3 alone

Every fragment now carries only its own atoms. rho_B reaches A through the
quadrature grid instead of through a shared density matrix, so the embedded
SCF is nao_A = 278 rather than the 3028 of the supermolecular basis -- which
is the entire point of subsystem embedding. Expected dimensions:

    nao_A 278   nao_c 250   nao_a 28   nao_B 2750

"""
import os

import numpy
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.scf.smearing import smearing
from pyscf.pbc import gto, scf as pbcscf
from pyscf import ksced

#0 setup
XC = 'PBE'
T_NAD = 'LDA_K_TF'
CONV_TOL = 1e-5
SIGMA = 0.003
LATTICE = '''
14.42497833620557 0.0 0.0
0.0 14.99087722583305 0.0
0.0 0.0 34.133534589762036'''
MESH = [72, 75, 170]
COMMON = dict(a=LATTICE, basis='gth-dzvp-molopt-sr', pseudo='gth-pbe',
              mesh=MESH, verbose=4)

SYM = ['Au'] * 120 + ['N', 'H', 'H', 'H']
ALL = list(range(124))
ADSORBATE = [120, 121, 122, 123]
A_AU = [87, 88, 89, 113, 114, 115, 116, 117, 118, 119]
B_AU = [i for i in range(120) if i not in A_AU]

y = numpy.loadtxt('structure.xyz')


def build(real):
    """A Cell holding `real` and nothing else -- no ghosts.

    A and B must share the lattice and the mesh."""
    atoms = [(SYM[i], tuple(y[i])) for i in real]
    return gto.M(atom=atoms, **COMMON)


def rks(cell, chkname):
    mf = dft.RKS(cell, xc=XC)
    mf.init_guess = 'atom'
    mf.conv_tol = CONV_TOL
    mf.max_cycle = 200
    mf = smearing(mf, sigma=SIGMA, method='gauss')
    mf.diis_damp = 0.8
    mf.max_memory = 100000
    mf.with_df.max_memory = 100000
    mf.chkfile = chkname
    return mf


def dm0(mf, chkname):
    """Restart from a checkpoint when one is present, else the atomic guess.
    """
    if not os.path.exists(chkname):
        return None
    rec = pbcscf.chkfile.load(chkname, 'scf')
    return mf.make_rdm1(numpy.asarray(rec['mo_coeff']),
                        numpy.asarray(rec['mo_occ']))


cell_b = build(B_AU)
cell_A = build(A_AU + ADSORBATE)
cell_c = build(A_AU)
cell_a = build(ADSORBATE)

#1 embedding B (Au110)
mf_b = rks(cell_b, 'cellb.chk')
mf_b.kernel(dm0=dm0(mf_b, 'cellb.chk'))

#2 A in B (NH3+Au10 in Au110)
mf_Ainb = ksced.embed(rks(cell_A, 'cellA.chk'), mf_b, basis_mode='M')
mf_Ainb.t_nad = T_NAD
mf_Ainb.kernel(dm0=dm0(mf_Ainb, 'cellA.chk'))

#3 c in B (Au10 in Au110)
mf_cinb = ksced.embed(rks(cell_c, 'cellc.chk'), mf_b, basis_mode='M')
mf_cinb.t_nad = T_NAD
mf_cinb.kernel(dm0=dm0(mf_cinb, 'cellc.chk'))

#4 a (NH3)
mf_a = rks(cell_a, 'cella.chk')
mf_a.kernel(dm0=dm0(mf_a, 'cella.chk'))

#5 get interaction energy (kcal/mol)
eint = float(mf_Ainb.e_tot) - float(mf_cinb.e_tot) - float(mf_a.e_tot)
print('E[A in B]                   %.10f Ha' % float(mf_Ainb.e_tot))
print('E[c in B]                   %.10f Ha' % float(mf_cinb.e_tot))
print('E[a]                        %.10f Ha' % float(mf_a.e_tot))
print('non-additive kinetic energy %.10f Ha' % float(mf_Ainb.e_tnad))
print('interaction energy          %.10f Ha  = %.3f kcal/mol'
      % (eint, eint * 627.503))
