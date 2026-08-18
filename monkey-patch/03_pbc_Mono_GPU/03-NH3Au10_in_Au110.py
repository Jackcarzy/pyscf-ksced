"""Benchmark NH3 on Au120 with the monkey patch: interaction energy, gamma point

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
from pyscf import lib

# -------------------------------------------------------------------------
# Part 1
#
# attach_ksced_mb: a drop-in for ksced.embed(..., basis_mode='M'), by monkey
# patching only.
# -------------------------------------------------------------------------

T_NAD_DEFAULT = 'LDA_K_TF'


def _is_cell(mol):
    from pyscf.pbc.gto import Cell
    return isinstance(mol, Cell)


def _is_gpu(a):
    """True when a lives in device memory, tagged or not.

    Duck-typed rather than keyed on type(a).__module__: GPU4PySCF's make_rdm1
    returns a tagged array whose module is gpu4pyscf.lib.cupy_helper, so a
    module-name test misses it.
    """
    return not isinstance(a, numpy.ndarray) and callable(getattr(a, 'get', None))


def _as_like(ref, arr):
    if _is_gpu(ref) and not _is_gpu(arr):
        import cupy
        return cupy.asarray(arr)
    if not _is_gpu(ref) and _is_gpu(arr):
        return arr.get()
    return arr


def _host(a):
    """Bring an array to the host, whatever backend it came from.

    Duck-typed on .get(), not on type(a).__module__: GPU4PySCF's make_rdm1
    returns a tagged array whose module is gpu4pyscf.lib.cupy_helper, so a
    module-name test misses it and numpy.asarray then raises.
    """
    if isinstance(a, numpy.ndarray):
        return a
    get = getattr(a, 'get', None)
    if callable(get):
        return numpy.asarray(get())
    return numpy.asarray(a)


def _trace(a, b):
    if _is_gpu(a) or _is_gpu(b):
        import cupy
        a = a if _is_gpu(a) else cupy.asarray(a)
        b = b if _is_gpu(b) else cupy.asarray(b)
        return float(cupy.einsum('ij,ji->', a, b).real.get())
    return float(numpy.einsum('ij,ji->', a, b).real)


def _spin_sum(dm):
    return dm[0] + dm[1] if getattr(dm, 'ndim', 2) == 3 else dm


def _trace_spin(a, dm):
    return _trace(a, _spin_sum(dm))


def _as_pair(dm):
    dm = _host(dm)
    if dm.ndim == 3:
        return dm
    return numpy.stack((dm * .5, dm * .5))


def _tag(a, **kw):
    if _is_gpu(a):
        from gpu4pyscf.lib.cupy_helper import tag_array
        return tag_array(a, **kw)
    return lib.tag_array(a, **kw)


def _kinetic(mol, kpt=None):
    if _is_cell(mol):
        if kpt is None:
            kpt = numpy.zeros(3)
        return mol.pbc_intor('int1e_kin', hermi=1, kpts=kpt)
    return mol.intor_symmetric('int1e_kin')


def _conc(mol_a, mol_b):
    if _is_cell(mol_a):
        from pyscf.pbc.gto.cell import conc_cell
        return conc_cell(mol_a, mol_b)
    from pyscf.gto.mole import conc_mol
    return conc_mol(mol_a, mol_b)


def _pad(dm_a, dm_b, nao_a, nao_b):
    polarized = (getattr(dm_a, 'ndim', 2) == 3
                 or getattr(dm_b, 'ndim', 2) == 3)
    shape = ((2, nao_a + nao_b, nao_a + nao_b) if polarized
             else (nao_a + nao_b, nao_a + nao_b))
    out = numpy.zeros(shape)
    aa = _as_pair(dm_a) if polarized else _host(dm_a)
    bb = _as_pair(dm_b) if polarized else _host(dm_b)
    out[..., :nao_a, :nao_a] = aa.real
    out[..., nao_a:, nao_a:] = bb.real
    return out


def _scratch(mf, mol):
    """A copy of mf rebound to mol, for the one-time AB-dimension builds.

    Not type(mf)(mol): mf is routinely wrapped -- smearing_ produces a
    _SmearingSCF whose __init__ takes sigma, method, mu0 and fix_spin, so
    reconstructing from the class alone raises TypeError. Copying preserves
    whatever stack of wrappers is in play, and reset() rebinds the object to
    the new molecule.
    """
    import copy as _copy

    out = mf.copy()
    for attr in ('with_df', 'grids', 'nlcgrids'):
        sub = getattr(out, attr, None)
        if sub is not None:
            setattr(out, attr, _copy.copy(sub))
    out.reset(mol)
    out._eri = None
    return out


def _vne_of(mf, mol, kpt=None):
    """V_ne = hcore - T, from this object's own get_hcore."""
    h1e = mf.get_hcore(mol, kpt) if _is_cell(mol) else mf.get_hcore(mol)
    return _host(h1e) - _host(_kinetic(mol, kpt))


def attach_ksced_mb(mf, mf_b, dm_b=None, mol_ab=None, t_nad=T_NAD_DEFAULT):
    """Attach a frozen monomolecular environment by attribute assignment."""
    mol_a = mf.mol
    mol_b = mf_b.mol
    if dm_b is None:
        dm_b = mf_b.make_rdm1()
    a_unrestricted = bool(mf.istype('UHF'))
    b_unrestricted = getattr(dm_b, 'ndim', 2) == 3
    spin_resolved = a_unrestricted or b_unrestricted
    if mol_ab is None:
        mol_ab = _conc(mol_a, mol_b)

    is_pbc = _is_cell(mol_a)
    kpt = numpy.zeros(3) if is_pbc else None
    nao_a, nao_b = mol_a.nao, mol_b.nao

    # Scratch objects of the same class, so the backend and the method used
    # inside get_hcore are the same ones the subsystems used.
    mf_ab = _scratch(mf, mol_ab)
    mf_a0 = _scratch(mf, mol_a)
    mf_b0 = _scratch(mf, mol_b)

    # --- one-time AB build, sliced both ways ---------------------------
    vne_ab = _vne_of(mf_ab, mol_ab, kpt)
    vne_b_in_a = vne_ab[:nao_a, :nao_a] - _vne_of(mf_a0, mol_a, kpt)
    vne_a_in_b = vne_ab[nao_a:, nao_a:] - _vne_of(mf_b0, mol_b, kpt)
    e_vne_a_rho_b = _trace_spin(vne_a_in_b, dm_b)

    # _pad builds on the host; move it to dm_b's backend before get_j.
    dm_b_pad = _as_like(dm_b, _spin_sum(_pad(
        numpy.zeros((nao_a, nao_a)), dm_b, nao_a, nao_b)))
    if is_pbc:
        j_ab = mf_ab.get_j(mol_ab, dm_b_pad, 1, kpt, None)
    else:
        j_ab = mf_ab.get_j(mol_ab, dm_b_pad, 1)
    j_b_in_a = _host(j_ab)[:nao_a, :nao_a]

    # --- grids: supermolecular for both domains ------------------------
    # For cells this is the same set of points as A's own uniform grid, since
    # the lattice and mesh match. For molecules it genuinely differs, and the
    # supermolecular Becke grid is the one that samples rho_B properly.
    mf_ab.grids.build()
    grids = mf_ab.grids
    mf.grids = grids

    ni = mf._numint
    held = {'e_tnad': 0.0}

    def _nr_ab(xc, dm_t):
        """Stock nr_rks/nr_uks at AB dimension, sliced to A's block.

        _pad builds on the host, so the padded matrix has to be moved to
        whichever backend dm_b lives on before nr_rks sees it: GPU4PySCF's
        eval_rho does ao.dot(dm) and rejects a numpy operand outright.
        """
        dm_t = _as_like(dm_b, dm_t)
        nr = ni.nr_uks if getattr(dm_t, 'ndim', 2) == 3 else ni.nr_rks
        if is_pbc:
            n, exc, v = nr(mol_ab, grids, xc, dm_t, 0, 1, kpt, None)
        else:
            n, exc, v = nr(mol_ab, grids, xc, dm_t)
        return n, exc, _host(v)[..., :nao_a, :nao_a]

    def _nr_a(xc, dm_a):
        dm_a = _as_like(dm_b, dm_a)
        nr = ni.nr_uks if getattr(dm_a, 'ndim', 2) == 3 else ni.nr_rks
        if is_pbc:
            n, exc, v = nr(mol_a, grids, xc, dm_a, 0, 1, kpt, None)
        else:
            n, exc, v = nr(mol_a, grids, xc, dm_a)
        return n, exc, _host(v)

    orig_get_hcore = mf.get_hcore
    orig_energy_nuc = mf.energy_nuc
    orig_energy_elec = mf.energy_elec

    def get_hcore(mol=None, kpts=None):
        h1e = orig_get_hcore() if mol is None else orig_get_hcore(mol)
        return h1e + _as_like(h1e, vne_b_in_a)

    def energy_nuc():
        return (orig_energy_nuc()
                + mol_ab.energy_nuc() - mol_a.energy_nuc() - mol_b.energy_nuc())

    def get_veff(mol=None, dm=None, *args, **kwargs):
        if dm is None:
            dm = mf.make_rdm1()
        dm_a = _as_pair(dm) if spin_resolved else dm
        dm_t = _pad(dm_a, dm_b, nao_a, nao_b)
        dm_b_only = _pad(numpy.zeros((nao_a, nao_a)), dm_b, nao_a, nao_b)

        _, exc_t, vxc = _nr_ab(mf.xc, dm_t)
        _, t_t, v_t_t = _nr_ab(t_nad, dm_t)
        _, t_a, v_t_a = _nr_a(t_nad, dm_a)
        t_b = _nr_ab(t_nad, dm_b_only)[1]
        e_xc_b = _nr_ab(mf.xc, dm_b_only)[1]

        held['e_tnad'] = t_t - t_a - t_b
        mf.e_tnad = held['e_tnad']

        if is_pbc:
            vj_a = mf.get_j(mol_a, _spin_sum(dm_a), 1, kpt, None)
        else:
            vj_a = mf.get_j(mol_a, _spin_sum(dm_a), 1)

        v = _as_like(vj_a, vxc) + _as_like(vj_a, v_t_t) - _as_like(vj_a, v_t_a)
        if spin_resolved and not a_unrestricted:
            v = (v[0] + v[1]) * .5
        exc = exc_t + held['e_tnad'] - e_xc_b

        vj = vj_a + _as_like(vj_a, j_b_in_a)
        v = v + vj
        ecoul = _trace_spin(vj, dm_a) * .5
        return _tag(v, ecoul=ecoul, exc=exc, vj=vj, vk=None)

    def energy_elec(dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = mf.make_rdm1()
        e1, e2 = orig_energy_elec(dm, h1e, vhf)
        e_coul_ab_half = _trace_spin(_as_like(dm, j_b_in_a), dm) * .5
        return e1 + e_vne_a_rho_b + e_coul_ab_half, e2 + e_coul_ab_half

    mf.get_hcore = get_hcore
    mf.get_veff = get_veff
    mf.energy_elec = energy_elec
    mf.energy_nuc = energy_nuc
    mf.e_tnad = 0.0
    return mf


# -------------------------------------------------------------------------
# Part 2 -- the calculation
# -------------------------------------------------------------------------

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
mf_Ainb = attach_ksced_mb(rks(cell_A, 'cellA.chk'), mf_b, t_nad=T_NAD)
mf_Ainb.kernel(dm0=dm0(mf_Ainb, 'cellA.chk'))

#3 c in B (Au10 in Au110)
mf_cinb = attach_ksced_mb(rks(cell_c, 'cellc.chk'), mf_b, t_nad=T_NAD)
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
print('reference = -1.555 kcal/mol')
