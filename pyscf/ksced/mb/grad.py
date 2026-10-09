'''Analytic fixed-lattice A gradients of the monomolecular embedded energy.

The FFT quadrature is fixed in space. Only A's AO values move in the density
terms; B's basis and density do not. Differentiating the AO values (including
their spatial derivatives for GGA) therefore includes the complete grid
contribution. The overlap derivative supplies the remaining Pulay term.
'''

import numpy as np

from pyscf import lib
from pyscf.grad import rhf as mol_grad
from pyscf.pbc.grad import rhf as pbc_grad
from pyscf.ksced.ksced import _spin_sum
from pyscf.ksced.mb.arrays import to_host
from pyscf.ksced.mb.meshdata import is_fft_df


def _backend(mf):
    if type(mf._numint).__module__.startswith('gpu4pyscf'):
        import cupy as xp
        from gpu4pyscf.pbc.dft import numint
        from gpu4pyscf.pbc import tools
    else:
        xp = np
        from pyscf.pbc.dft import numint
        from pyscf.pbc import tools
    return xp, numint, tools


def _validate(mf):
    from pyscf.ksced.mb.grad_pp import _validate_fft_grid
    env = mf.with_env
    cell = mf.cell
    for obj in (mf, env.mf_b):
        if np.any(np.abs(to_host(obj.kpt)) > 1e-12):
            raise NotImplementedError('KSCED gradients support Gamma point only')
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
    # The cross terms assume the concatenation is ordered A then B.
    from pyscf.ksced.mb.env import _conc
    expected = _conc(cell, env.mol_b)
    ab = env.mol_ab
    if (ab.natm != expected.natm or not np.array_equal(ab._bas, expected._bas)
            or not np.allclose(ab.atom_coords(), expected.atom_coords(), rtol=0, atol=1e-12)
            or not np.array_equal(ab.atom_charges(), expected.atom_charges())
            or not np.array_equal(ab.mesh, cell.mesh)
            or not np.allclose(ab.lattice_vectors(), cell.lattice_vectors(), rtol=0, atol=1e-12)):
        raise ValueError('KSCED gradients require mol_ab to be the A-then-B concatenation')
    mf.energy_potential()  # Check convergence and the finite-temperature contract.


def _ao_blocks(mf, deriv):
    xp, ni, _ = _backend(mf)
    cell = mf.cell
    coords = cell.get_uniform_grids()
    # Bound AO work arrays, including ten derivative components for GGA.
    memory = max(16, min(256, mf.max_memory - lib.current_memory()[0])) * 1e6
    block = max(64, min(8192, int(memory / (8 * 20 * max(1, cell.nao)))))
    for p0 in range(0, len(coords), block):
        p1 = min(len(coords), p0 + block)
        ao = ni.eval_ao(cell, coords[p0:p1], kpt=np.zeros(3), deriv=deriv)
        yield p0, p1, xp.asarray(ao).real


def _density(ao, dms, xp):
    '''Spin channels of (density, dx, dy, dz) on one grid block.'''
    out = []
    for dm in dms:
        c = ao[0].dot(dm)
        rho = xp.empty((4, ao.shape[1]))
        rho[0] = xp.einsum('gi,gi->g', c, ao[0])
        rho[1:] = 2 * xp.einsum('gi,xgi->xg', c, ao[1:4])
        out.append(rho)
    return xp.stack(out)


def _potential(ni, code, rho, polarized, xp):
    '''Functional derivative with respect to density and its three gradients.'''
    kind = ni._xc_type(code)
    r = rho[:, :1] if kind == 'LDA' else rho
    if not polarized:
        r = r[0]
    v = ni.eval_xc_eff(code, r, deriv=1, xctype=kind, spin=int(polarized))[1]
    v = xp.asarray(v).reshape(rho.shape[0], -1, rho.shape[-1])
    out = xp.zeros_like(rho)
    out[:, :v.shape[1]] = v
    return out


def _grid_gradient(mf, dm):
    xp, _, tools = _backend(mf)
    env, cell = mf.with_env, mf.cell
    polarized = dm.ndim == 3 or env.polarized
    dm = xp.asarray(dm)
    dms = dm if dm.ndim == 3 else (xp.stack([dm * .5] * 2) if polarized else dm[None])
    ngrids = int(np.prod(cell.mesh))
    rho_a = xp.empty((len(dms), 4, ngrids))
    for p0, p1, ao in _ao_blocks(mf, 1):
        rho_a[..., p0:p1] = _density(ao, dms, xp)
    mesh_b = env._mesh_data(mf._numint, mf.kpt)
    rho_b = xp.asarray(to_host(mesh_b.rho())) if xp is np else xp.asarray(mesh_b.rho())
    if rho_b.ndim == 2:
        rho_b = xp.stack([rho_b * .5] * 2) if polarized else rho_b[None]
    weight = cell.vol / ngrids
    rho_g = tools.fft(rho_a[:, 0].sum(axis=0), cell.mesh)
    vj = tools.ifft(rho_g * tools.get_coulG(cell, mesh=cell.mesh), cell.mesh).real * weight
    vb = mesh_b.vj()
    vj += xp.asarray(to_host(vb)) if xp is np else xp.asarray(vb)
    result = {name: xp.zeros((cell.natm, 3)) for name in ('hartree', 'xc', 'kinetic_nad')}
    hess = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
    for p0, p1, ao in _ao_blocks(mf, 2):
        ra = rho_a[..., p0:p1]
        rt = ra + rho_b[..., p0:p1]
        vx = _potential(mf._numint, mf.xc, rt, polarized, xp) * weight
        vt = (_potential(mf._numint, mf.t_nad, rt, polarized, xp)
              - _potential(mf._numint, mf.t_nad, ra, polarized, xp)) * weight
        for s, d in enumerate(dms):
            dao = xp.einsum('xgi,ij->xgj', ao[:4], d)
            for atom, (_, _, a0, a1) in enumerate(cell.aoslice_by_atom()):
                for axis in range(3):
                    # d/dR phi_mu(r-R) = -d/dr phi_mu. Both AO factors move.
                    drho = xp.empty((4, p1 - p0))
                    grad = ao[axis + 1, :, a0:a1]
                    drho[0] = -2 * xp.einsum('gi,gi->g', grad, dao[0, :, a0:a1])
                    for k in range(3):
                        drho[k + 1] = -2 * (
                            xp.einsum('gi,gi->g', ao[hess[axis][k], :, a0:a1], dao[0, :, a0:a1])
                            + xp.einsum('gi,gi->g', grad, dao[k + 1, :, a0:a1]))
                    result['hartree'][atom, axis] += drho[0].dot(vj[p0:p1])
                    result['xc'][atom, axis] += xp.einsum('xg,xg->', drho, vx[s])
                    result['kinetic_nad'][atom, axis] += xp.einsum('xg,xg->', drho, vt[s])
    return {k: to_host(v) for k, v in result.items()}


def _contract_ip(cell, integral, dm):
    return np.asarray([2 * np.einsum('xij,ji->x', integral[:, p0:p1], dm[:, p0:p1]).real
                       for _, _, p0, p1 in cell.aoslice_by_atom()])


class Gradients(pbc_grad.GradientsBase):
    '''dE_potential/dR_A in Hartree/Bohr; physical forces have the opposite sign.'''

    _keys = {'components'}

    def __init__(self, mf):
        super().__init__(mf)
        self.components = {}

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        from pyscf.ksced.mb.grad_pp import vne_gradient
        mf, cell = self.base, self.cell
        _validate(mf)
        mo_energy = mf.mo_energy if mo_energy is None else mo_energy
        mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
        mo_occ = mf.mo_occ if mo_occ is None else mo_occ
        dm_backend = mf.make_rdm1(mo_coeff, mo_occ)
        dm = to_host(dm_backend)
        if np.iscomplexobj(dm) and np.max(abs(dm.imag)) > 1e-12:
            raise NotImplementedError('KSCED Gamma gradients require real density matrices')
        dm = dm.real
        total = _spin_sum(dm)
        coeff, occ, energy = map(to_host, (mo_coeff, mo_occ, mo_energy))
        if coeff.ndim == 2:
            weighted = (coeff * (occ * energy)).dot(coeff.conj().T).real
        else:
            weighted = sum((c * (o * e)).dot(c.conj().T).real
                           for c, o, e in zip(coeff, occ, energy))
        self.components = _grid_gradient(mf, dm_backend)
        self.components['kinetic'] = -_contract_ip(cell, cell.pbc_intor('int1e_ipkin'), total)
        self.components['pulay'] = _contract_ip(cell, cell.pbc_intor('int1e_ipovlp'), weighted)
        env = mf.with_env
        ab = env.mol_ab
        padded = np.zeros((ab.nao, ab.nao))
        padded[:cell.nao, :cell.nao] = total
        padded[cell.nao:, cell.nao:] = to_host(_spin_sum(env.dm_b))
        # B's backend builds both cross operators. A's own operator may use
        # a different backend; keep all three terms to preserve that energy.
        pp = vne_gradient(env._mf_on(ab), ab, padded)[:cell.natm]
        pp -= vne_gradient(env._mf_on(cell), cell, total)
        pp += vne_gradient(mf.undo_ksced(), cell, total)
        self.components['electron_nuclear'] = pp
        out = sum(self.components.values())
        return out if atmlst is None else out[atmlst]

    def grad_nuc(self, cell=None, atmlst=None):
        out = pbc_grad.grad_nuc(self.base.mol_ab)[:self.cell.natm]
        self.components['nuclear'] = out
        return out if atmlst is None else out[atmlst]

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        if atmlst is None:
            atmlst = self.atmlst
        if atmlst is not None:
            atmlst = list(atmlst)
            if any(i < 0 or i >= self.cell.natm for i in atmlst):
                raise ValueError('atmlst must contain subsystem-A atom indices')
        self.atmlst = atmlst
        self.de = (self.grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
                   + self.grad_nuc(atmlst=atmlst))
        self._finalize()
        return self.de

    grad = lib.alias(kernel, alias_name='grad')

    def as_scanner(self):
        if isinstance(self, lib.GradScanner):
            return self
        return lib.set_class(_Scanner(self), (_Scanner, self.__class__), 'KSCEDGradScanner')


class _Scanner(mol_grad.SCF_GradScanner):
    def __call__(self, mol_or_geom, **kwargs):
        _, gradient = super().__call__(mol_or_geom, **kwargs)
        return self.base.energy_potential(), gradient
