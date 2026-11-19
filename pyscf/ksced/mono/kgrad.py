'''Analytic fixed-lattice A gradients of the monomolecular embedded energy, k-points.

    pyscf.ksced.mono.grad : the same at the gamma point

Mirrors mono/grad.py with kpts, as pyscf.pbc.grad.krks mirrors rks. Every
A-side trace is the k average (1/nk) sum_k Tr[D_k O_k]. rho_A(r), rho_B(r), the
Hartree, XC and T_nad potentials on the mesh, and the nuclear repulsion do not
depend on k; only the AO values (and so drho_A/dR) carry the phase exp(ik.r).
'''

import numpy as np

from pyscf import lib
from pyscf.ksced.arrays import to_host
from pyscf.ksced.common import _is_kscf
from pyscf.ksced.mono import grad as gamma_grad
from pyscf.ksced.mono.grad import _backend, _potential
from pyscf.ksced.mono.meshdata import is_fft_df


def _kpts_of(obj):
    return np.reshape(np.asarray(to_host(obj.kpts), dtype=float), (-1, 3))


def _validate(mf):
    '''The gamma-point checks of grad._validate, with kpts in place of kpt.'''
    from pyscf.ksced.mono.grad_pp import _validate_fft_grid
    env = mf.with_env
    cell = mf.cell
    if not _is_kscf(env.mf_b):
        raise NotImplementedError('KSCED k-point gradients need a k-point B')
    kpts = _kpts_of(mf)
    if _kpts_of(env.mf_b).shape != kpts.shape or not np.allclose(
            _kpts_of(env.mf_b), kpts, rtol=0, atol=1e-12):
        raise ValueError('KSCED k-point gradients require identical A and B k-points')
    for obj in (mf, env.mf_b):
        if getattr(obj, 'kpts_descriptor', None) is not None or 'Ksym' in type(obj).__name__:
            raise NotImplementedError('KSCED k-point gradients do not support k-point symmetry')
        if not is_fft_df(obj.with_df):
            raise NotImplementedError('KSCED gradients require FFTDF for both subsystems')
        if getattr(obj, 'rsjk', None) is not None or getattr(obj, 'j_engine', False):
            raise NotImplementedError('KSCED gradients require the FFTDF Hartree operator')
        if obj.cell.dimension != 3:
            raise NotImplementedError('KSCED gradients currently require a 3D periodic cell')
        if obj.cell._ecp:
            raise NotImplementedError('KSCED periodic ECP gradients are not implemented')
        if np.any(obj.cell.atom_charges() == 0):
            raise NotImplementedError('KSCED M gradients require separate bases without ghost atoms')
        if getattr(obj, 'with_x2c', None):
            raise NotImplementedError('KSCED X2C gradients are not implemented')
        if not np.array_equal(obj.with_df.mesh, cell.mesh):
            raise ValueError('KSCED gradients require identical cell and FFTDF meshes')
        _validate_fft_grid(obj, obj.cell)
    dm_b = to_host(env.dm_b)
    if np.ndim(dm_b) != 3 or dm_b.shape[0] != len(kpts):
        raise ValueError('KSCED k-point gradients require dm_b of shape (nk, nao_b, nao_b)')
    if getattr(mf, 'disp', None) or getattr(mf, 'nlc', None):
        raise NotImplementedError('KSCED gradients do not include dispersion or nonlocal XC')
    for code in (mf.xc, mf.t_nad):
        if mf._numint._xc_type(code) not in ('LDA', 'GGA') or mf._numint.libxc.is_hybrid_xc(code):
            raise NotImplementedError('KSCED gradients support pure LDA/GGA functionals only')
    if (type(mf.grids).__name__ != 'UniformGrids'
            or not np.array_equal(mf.grids.mesh, cell.mesh)):
        raise NotImplementedError('KSCED gradients require the fixed uniform FFT grid')
    coords = cell.get_uniform_grids()
    if (mf.grids.coords.shape != coords.shape
            or not np.allclose(to_host(mf.grids.coords), coords, rtol=0, atol=1e-12)
            or not np.allclose(to_host(mf.grids.weights), cell.vol / len(coords),
                               rtol=0, atol=1e-12)):
        raise NotImplementedError('KSCED gradients do not support modified quadrature points or weights')
    if not np.allclose(env.mol_a.atom_coords(), cell.atom_coords(), rtol=0, atol=1e-12):
        raise ValueError('The frozen environment was rebound to a different A geometry')
    from pyscf.ksced.mono.env import _conc
    expected = _conc(cell, env.mol_b)
    ab = env.mol_ab
    if (ab.natm != expected.natm or not np.array_equal(ab._bas, expected._bas)
            or not np.allclose(ab.atom_coords(), expected.atom_coords(), rtol=0, atol=1e-12)
            or not np.array_equal(ab.atom_charges(), expected.atom_charges())
            or not np.array_equal(ab.mesh, cell.mesh)
            or not np.allclose(ab.lattice_vectors(), cell.lattice_vectors(), rtol=0, atol=1e-12)):
        raise ValueError('KSCED gradients require mol_ab to be the A-then-B concatenation')
    mf.energy_potential()  # Check convergence and the finite-temperature contract.


def _ao_blocks(mf, kpts, deriv):
    '''(p0, p1, [ao_k (ncomp, g, nao) for each k]) over the uniform grid.'''
    xp, ni, _ = _backend(mf)
    cell = mf.cell
    coords = cell.get_uniform_grids()
    ncomp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
    # Complex AOs at every k-point: 16 bytes per value.
    memory = max(16, min(256, mf.max_memory - lib.current_memory()[0])) * 1e6
    block = max(64, min(8192, int(memory / (16 * (ncomp + 2) * len(kpts) * max(1, cell.nao)))))
    for p0 in range(0, len(coords), block):
        p1 = min(len(coords), p0 + block)
        ao = ni.eval_ao_kpts(cell, coords[p0:p1], kpts=kpts, deriv=deriv)
        yield p0, p1, [xp.asarray(a) for a in ao]


def _density(aos, dms, xp):
    '''(density, dx, dy, dz) of rho_A = (1/nk) sum_k sum_nu phi_k,nu conj((phi_k D_k)_nu).'''
    nk = len(dms)
    rho = xp.zeros((4, aos[0].shape[1]))
    for ao, dm in zip(aos, dms):
        c = ao[0].dot(dm).conj()
        rho[0] += xp.einsum('gi,gi->g', ao[0], c).real
        rho[1:] += 2 * xp.einsum('xgi,gi->xg', ao[1:4], c).real
    return rho / nk


def _grid_gradient(mf, dms, kpts):
    xp, _, tools = _backend(mf)
    env, cell = mf.with_env, mf.cell
    nk = len(kpts)
    dms = xp.asarray(dms)
    ngrids = int(np.prod(cell.mesh))
    rho_a = xp.empty((4, ngrids))
    for p0, p1, aos in _ao_blocks(mf, kpts, 1):
        rho_a[:, p0:p1] = _density(aos, dms, xp)
    mesh_b = env._mesh_data(mf._numint, env.kpts)
    rho_b = xp.asarray(to_host(mesh_b.rho())) if xp is np else xp.asarray(mesh_b.rho())
    weight = cell.vol / ngrids
    rho_g = tools.fft(rho_a[0], cell.mesh)
    vj = tools.ifft(rho_g * tools.get_coulG(cell, mesh=cell.mesh), cell.mesh).real * weight
    vb = mesh_b.vj()
    vj += xp.asarray(to_host(vb)) if xp is np else xp.asarray(vb)
    result = {name: xp.zeros((cell.natm, 3)) for name in ('hartree', 'xc', 'kinetic_nad')}
    hess = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
    slices = cell.aoslice_by_atom()
    for p0, p1, aos in _ao_blocks(mf, kpts, 2):
        ra = rho_a[None, :, p0:p1]
        rt = ra + rho_b[None, :, p0:p1]
        vx = _potential(mf._numint, mf.xc, rt, False, xp)[0] * weight
        vt = (_potential(mf._numint, mf.t_nad, rt, False, xp)
              - _potential(mf._numint, mf.t_nad, ra, False, xp))[0] * weight
        drho = xp.zeros((cell.natm, 3, 4, p1 - p0))
        for ao, d in zip(aos, dms):
            # conj((d^j phi_k D_k)_i) for j = value, x, y, z.
            dao = xp.einsum('xgi,ij->xgj', ao[:4], d).conj()
            for atom, (_, _, a0, a1) in enumerate(slices):
                for axis in range(3):
                    # d/dR phi_mu(r-R) = -d/dr phi_mu. Both AO factors move.
                    grad = ao[axis + 1, :, a0:a1]
                    drho[atom, axis, 0] -= 2 * xp.einsum(
                        'gi,gi->g', grad, dao[0, :, a0:a1]).real
                    for k in range(3):
                        drho[atom, axis, k + 1] -= 2 * (
                            xp.einsum('gi,gi->g', ao[hess[axis][k], :, a0:a1], dao[0, :, a0:a1])
                            + xp.einsum('gi,gi->g', grad, dao[k + 1, :, a0:a1])).real
        drho /= nk
        result['hartree'] += xp.einsum('ang,g->an', drho[:, :, 0], vj[p0:p1])
        result['xc'] += xp.einsum('anxg,xg->an', drho, vx)
        result['kinetic_nad'] += xp.einsum('anxg,xg->an', drho, vt)
    return {k: to_host(v) for k, v in result.items()}


def _contract_ip(cell, integrals, dms):
    '''(2/nk) Re sum_k sum_{i in atom, j} I_k[x, i, j] D_k[j, i] for each atom.'''
    integrals = np.asarray(integrals)
    nk = len(dms)
    return np.asarray([2 / nk * np.einsum('kxij,kji->x', integrals[:, :, p0:p1],
                                          dms[:, :, p0:p1]).real
                       for _, _, p0, p1 in cell.aoslice_by_atom()])


class Gradients(gamma_grad.Gradients):
    '''dE_potential/dR_A in Hartree/Bohr for a k-point (KRKS) embedded A.'''

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        from pyscf.ksced.mono.grad_pp import vne_gradient
        mf, cell = self.base, self.cell
        _validate(mf)
        kpts = _kpts_of(mf)
        nk = len(kpts)
        mo_energy = mf.mo_energy if mo_energy is None else mo_energy
        mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
        mo_occ = mf.mo_occ if mo_occ is None else mo_occ
        dm_backend = mf.make_rdm1(mo_coeff, mo_occ)
        dms = np.asarray(to_host(dm_backend), dtype=np.complex128)
        coeff = [to_host(c) for c in mo_coeff]
        occ = [to_host(o) for o in mo_occ]
        energy = [to_host(e) for e in mo_energy]
        weighted = np.asarray([(c * (o * e)).dot(c.conj().T)
                               for c, o, e in zip(coeff, occ, energy)])
        self.components = _grid_gradient(mf, dm_backend, kpts)
        self.components['kinetic'] = -_contract_ip(
            cell, cell.pbc_intor('int1e_ipkin', kpts=kpts), dms)
        self.components['pulay'] = _contract_ip(
            cell, cell.pbc_intor('int1e_ipovlp', kpts=kpts), weighted)
        env = mf.with_env
        ab = env.mol_ab
        padded = np.zeros((nk, ab.nao, ab.nao), dtype=np.complex128)
        padded[:, :cell.nao, :cell.nao] = dms
        padded[:, cell.nao:, cell.nao:] = to_host(env.dm_b)
        # B's backend builds both cross operators. A's own operator may use
        # a different backend; keep all three terms to preserve that energy.
        pp = vne_gradient(env._mf_on(ab), ab, padded, kpts)[:cell.natm]
        pp -= vne_gradient(env._mf_on(cell), cell, dms, kpts)
        pp += vne_gradient(mf.undo_ksced(), cell, dms, kpts)
        self.components['electron_nuclear'] = pp
        out = sum(self.components.values())
        return out if atmlst is None else out[atmlst]
