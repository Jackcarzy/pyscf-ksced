"""
KSCED subsystem embedding for a molecular system, by monkey patching.

    python "00-H2O_in_Li+.py"

H2O is the embedded subsystem A; Li+ is the frozen environment B. Both are built
in one supermolecular basis using ghost atoms, so that dm_a and dm_b share a
dimension and the non-additive terms can be formed by adding them.

The interaction energy of a two-fragment system is a two-term subtraction.
    Eint      = E[A in B] - E[A]
"""

import numpy
from pyscf import gto, dft, lib

# ---------------------------------------------------------------------------
# Part 1
#
# attach_ksced: a drop-in replacement for ksced.embed, by monkey patching only.
# ---------------------------------------------------------------------------

T_NAD_DEFAULT = 'LDA_K_TF'


def _is_cell(mol):
    from pyscf.pbc.gto import Cell
    return isinstance(mol, Cell)


def _is_gpu(a):
    return not isinstance(a, numpy.ndarray) and callable(getattr(a, 'get', None))


def _as_like(ref, arr):
    """Put arr on the same array backend as ref."""
    if _is_gpu(ref) and not _is_gpu(arr):
        import cupy
        return cupy.asarray(arr)
    if not _is_gpu(ref) and _is_gpu(arr):
        return arr.get()
    return arr


def _trace(a, b):
    """einsum('ij,ji->', a, b).real as a float, on either backend."""
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
    if getattr(dm, 'ndim', 2) == 3:
        return dm
    half = dm * .5
    if _is_gpu(dm):
        import cupy
        return cupy.stack((half, half))
    return numpy.stack((half, half))


def _tag(a, **kw):
    if _is_gpu(a):
        from gpu4pyscf.lib.cupy_helper import tag_array
        return tag_array(a, **kw)
    return lib.tag_array(a, **kw)


def _kinetic(mol, kpt=None):
    if _is_cell(mol):
        return mol.pbc_intor('int1e_kin', hermi=1,
                             kpts=numpy.zeros(3) if kpt is None else kpt)
    return mol.intor_symmetric('int1e_kin')


def _vne(mf, mol, kpt=None):
    """V_ne = hcore - T, so it matches whatever the backend's get_hcore used.

    PySCF builds V_ne from FFTDF; GPU4PySCF's get_hcore switches to MultiGrid
    when prod(cell.mesh) < 500**3. Deriving it this way keeps V_ne[A] and
    V_ne[B] on the same footing whichever is in play.
    """
    h = mf.get_hcore(mol, kpt) if _is_cell(mol) else mf.get_hcore(mol)
    return h - _as_like(h, _kinetic(mol, kpt))


def attach_ksced(mf, mf_b, dm_b=None, mol_ab=None, t_nad=T_NAD_DEFAULT):
    """Attach a frozen KSCED environment to mf by monkey patching it."""
    mol_a = mf.mol
    mol_b = mf_b.mol
    pbc = _is_cell(mol_a)
    kpt = getattr(mf, 'kpt', None)
    if dm_b is None:
        dm_b = mf_b.make_rdm1()
    a_unrestricted = bool(mf.istype('UHF'))
    b_unrestricted = getattr(dm_b, 'ndim', 2) == 3
    spin_resolved = a_unrestricted or b_unrestricted

    # Frozen-environment quantities: rho_B never changes, so each is built once.
    vne_b = _vne(mf_b, mol_b, kpt)
    j_b = (mf.get_j(mol_a, _spin_sum(dm_b), 1, kpt, None) if pbc
           else mf.get_j(mol_a, _spin_sum(dm_b), 1))
    held = {'e_tnad': 0.0, 'e_xc_b': None, 't_b': None, 'vne_a': None}

    # Capture the originals BEFORE overwriting: without super() this is the only
    # way to keep the underlying implementation reachable.
    orig_hcore = mf.get_hcore
    orig_energy_elec = mf.energy_elec
    orig_energy_nuc = mf.energy_nuc

    def get_hcore(mol=None, kpt_=None, *a, **k):
        m = mol if mol is not None else mol_a
        h = orig_hcore(m, kpt_ if kpt_ is not None else kpt) if pbc else orig_hcore(m)
        return h + _as_like(h, vne_b)

    def _xc(ni, mol, grids, xc, dm, max_memory, hermi=1):
        nr = ni.nr_uks if getattr(dm, 'ndim', 2) == 3 else ni.nr_rks
        if pbc:
            return nr(mol, grids, xc, dm, 0, hermi, kpt, None,
                      max_memory=max_memory)
        return nr(mol, grids, xc, dm, max_memory=max_memory)

    def get_veff(mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
                 kpt_=None, kpts_band=None):
        m = mol if mol is not None else mol_a
        if dm is None:
            dm = mf.make_rdm1()
        ni = mf._numint
        if pbc:
            mf.initialize_grids(m, dm, kpt)
        elif mf.grids.coords is None:
            mf.initialize_grids(m, dm)
        max_memory = mf.max_memory - lib.current_memory()[0]

        dm_a = _as_pair(dm) if spin_resolved else dm
        if spin_resolved:
            dm_b_t = dm_b if b_unrestricted else _as_pair(dm_b)
        else:
            dm_b_t = dm_b
        dm_t = dm_a + _as_like(dm_a, dm_b_t)

        n, exc_t, vxc = _xc(ni, m, mf.grids, mf.xc, dm_t, max_memory, hermi)
        _, t_t, v_t_t = _xc(ni, m, mf.grids, t_nad, dm_t, max_memory, hermi)
        _, t_a, v_t_a = _xc(ni, m, mf.grids, t_nad, dm_a, max_memory, hermi)
        if held['t_b'] is None:
            held['t_b'] = _xc(ni, m, mf.grids, t_nad, dm_b, max_memory, hermi)[1]
            held['e_xc_b'] = _xc(ni, m, mf.grids, mf.xc, dm_b, max_memory, hermi)[1]

        held['e_tnad'] = t_t - t_a - held['t_b']
        mf.e_tnad = held['e_tnad']

        vxc = vxc + v_t_t - v_t_a
        if spin_resolved and not a_unrestricted:
            vxc = (vxc[0] + vxc[1]) * .5
        exc = exc_t + held['e_tnad'] - held['e_xc_b']

        dm_j = _spin_sum(dm_t)
        vj = (mf.get_j(m, dm_j, hermi, kpt, None) if pbc
              else mf.get_j(m, dm_j, hermi))
        vxc += vj
        ecoul = _trace_spin(vj, dm_a) * .5
        return _tag(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)

    def energy_elec(dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = mf.make_rdm1()
        e_tot_elec, e2 = orig_energy_elec(dm, h1e, vhf)
        if held['vne_a'] is None:
            # The unmodified get_hcore for A, i.e. without vne_b.
            h = orig_hcore(mol_a, kpt) if pbc else orig_hcore(mol_a)
            held['vne_a'] = h - _as_like(h, _kinetic(mol_a, kpt))
        e_vne_a_rho_b = _trace_spin(held['vne_a'], dm_b)
        e_coul_ab_half = _trace_spin(j_b, dm) * .5
        return (e_tot_elec + e_vne_a_rho_b + e_coul_ab_half, e2 + e_coul_ab_half)

    def energy_nuc():
        e = orig_energy_nuc()
        if mol_ab is not None:
            e += mol_ab.energy_nuc() - mol_a.energy_nuc() - mol_b.energy_nuc()
        return e

    mf.get_hcore = get_hcore
    mf.get_veff = get_veff
    mf.energy_elec = energy_elec
    mf.energy_nuc = energy_nuc
    mf.e_tnad = 0.0
    return mf


# ---------------------------------------------------------------------------
# Part 2 -- the calculation
# ---------------------------------------------------------------------------


#0 setup
mol_a = gto.M(
    verbose = 4,
    atom = '''
        o    0    0.       0.
        h    0    -0.757   0.587
        h    0    0.757    0.587
        x-li   0    0        -2''',
    basis = '6-31g')

mol_b = gto.M(
    verbose = 4,
    atom = '''
        x-o    0    0.       0.
        x-h    0    -0.757   0.587
        x-h    0    0.757    0.587
          li   0    0        -2''',
    charge = 1,
    basis = '6-31g')

mol_ab = gto.M(
    verbose = 4,
    atom = '''
        o    0    0.       0.
        h    0    -0.757   0.587
        h    0    0.757    0.587
        li   0    0        -2''',
    charge = 1,
    basis = '6-31g')

#1 embedding B (Li+)
mf_b = dft.RKS(mol_b, xc='PBE').run()

#2 A alone (H2O)
mf_a = dft.RKS(mol_a, xc='PBE')
mf_a.kernel()

#3 embedded A (H2O in Li+)
mf_ainb = attach_ksced(dft.RKS(mol_a, xc='PBE'), mf_b, mol_ab=mol_ab,
                       t_nad='LDA_K_TF')
mf_ainb.kernel()

#4 get interaction energy (kcal/mol)
eint = mf_ainb.e_tot - mf_a.e_tot
print('embedded total energy       %.10f Ha' % mf_ainb.e_tot)
print('non-additive kinetic energy %.10f Ha' % mf_ainb.e_tnad)
print('interaction energy          %.10f Ha  = %.3f kcal/mol'
      % (eint, eint * 627.503))
