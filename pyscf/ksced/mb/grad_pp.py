"""Derivatives of the nuclear potential used by the native core Hamiltonian.

FFTDF and GPU multigrid use different finite representations.  Differentiate
the representation selected by get_hcore, without changing the SCF energy.
The density matrix is held fixed; AO and projector centers both move.
"""

import numpy as np

from pyscf.ksced.mb.arrays import to_host
from pyscf.ksced.mb.meshdata import is_fft_df


def _validate_fft_grid(mf, cell):
    '''Reject fitting grids whose operator differs from the uniform FFT sum.'''
    df = mf.with_df
    grids = df.grids
    mesh = np.asarray(df.mesh)
    if (type(grids).__name__ != 'UniformGrids'
            or not np.array_equal(grids.mesh, mesh)
            or not np.allclose(grids.cell.lattice_vectors(), cell.lattice_vectors(),
                               rtol=0, atol=1e-12)):
        raise NotImplementedError('KSCED gradients require uniform FFTDF grids bound to the cell')
    coords = cell.get_uniform_grids(mesh)
    if (grids.coords.shape != coords.shape
            or not np.allclose(to_host(grids.coords), coords, rtol=0, atol=1e-12)
            or not np.allclose(to_host(grids.weights), cell.vol / len(coords),
                               rtol=0, atol=1e-12)):
        raise NotImplementedError('KSCED gradients do not support modified FFTDF grid points or weights')


def vne_gradient(mf, cell, dm):
    """Return d Tr[D Vne]/dR, including AO response, in Hartree/Bohr.

    ``mf`` must be the plain SCF object whose get_hcore defined this term,
    rebound to ``cell``.  Pass the spin-summed density matrix.  No kinetic,
    overlap, electron interaction, or nuclear repulsion term is included.
    """
    core = getattr(mf.get_hcore, '__func__', None)
    module = getattr(core, '__module__', '')
    if module not in ('pyscf.pbc.scf.hf', 'gpu4pyscf.pbc.scf.hf'):
        raise NotImplementedError('KSCED gradients require the native PBC hcore')
    if not is_fft_df(getattr(mf, 'with_df', None)):
        raise NotImplementedError('KSCED pseudopotential gradients require FFTDF')
    _validate_fft_grid(mf, cell)
    if cell.dimension != 3 or cell._ecp:
        raise NotImplementedError('KSCED gradients require a 3D cell without ECPs')
    if cell._pseudo and any(
            cell.atom_symbol(i) not in cell._pseudo for i in range(cell.natm)
            if cell.atom_charge(i) != 0):
        raise NotImplementedError('Mixed all-electron and pseudopotential gradients')
    if np.max(np.abs(to_host(getattr(mf, 'kpt', np.zeros(3))))) > 1e-12:
        raise NotImplementedError('KSCED gradients require Gamma point')
    dm = np.asarray(to_host(dm))
    if dm.shape != (cell.nao, cell.nao):
        raise ValueError('Pass one spin-summed density matrix for this cell')
    if np.max(np.abs(dm.imag)) > 1e-12 or not np.allclose(dm, dm.T.conj()):
        raise NotImplementedError('KSCED gradients require a real symmetric density')
    dm = np.asarray(dm.real, order='C')

    if module.startswith('gpu4pyscf'):
        from gpu4pyscf.pbc.dft import multigrid, multigrid_v2
        ni = mf._numint
        if isinstance(ni, multigrid.MultiGridNumInt):
            raise NotImplementedError('KSCED gradients do not support multigrid v1')
        if isinstance(ni, multigrid_v2.MultiGridNumInt):
            return _multigrid_gradient(ni, cell, dm)
        if np.prod(cell.mesh) < 500**3:
            # GPU HF/RKS chooses this even with an ordinary NumInt.  Its mesh
            # comes from cell, not from the FFTDF Coulomb fitting object.
            return _multigrid_gradient(multigrid_v2.MultiGridNumInt(cell), cell, dm)
    else:
        from pyscf.pbc.dft.multigrid import MultiGridNumInt
        if isinstance(getattr(mf, '_numint', None), MultiGridNumInt):
            raise NotImplementedError('KSCED gradients do not support CPU multigrid')
    return _fft_gradient(cell, dm, np.asarray(mf.with_df.mesh))


def _multigrid_gradient(ni, cell, dm):
    """Native GPU local multigrid and Gaussian nonlocal projector response."""
    import cupy as cp
    from gpu4pyscf.pbc.dft import multigrid, multigrid_v2
    from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad

    kpts = np.zeros((1, 3))
    dm_gpu = cp.asarray(dm)
    if ni.sorted_gaussian_pairs is None:
        ni.build()
    if cell._pseudo:
        vloc_g = multigrid.eval_vpplocG(cell, ni.mesh)
    else:
        vloc_g = multigrid.eval_nucG(cell, ni.mesh)
    basis = multigrid_v2.convert_xc_on_g_mesh_to_fock_gradient(
        ni, vloc_g[None, None], dm_gpu, hermi=1, kpts=kpts)
    rho_g = multigrid_v2.evaluate_density_on_g_mesh(ni, dm_gpu, kpts)[0, 0]
    if cell._pseudo:
        centers = multigrid.eval_vpplocG_SI_gradient(cell, ni.mesh, rho_g)
        nonlocal_grad = vppnl_nuc_grad(cell, dm_gpu, kpts)
    else:
        centers = multigrid.eval_nucG_SI_gradient(cell, ni.mesh, rho_g)
        nonlocal_grad = 0
    return np.asarray(to_host(basis + centers)) + np.asarray(to_host(nonlocal_grad))


def _fft_gradient(cell, dm, mesh):
    """Differentiate the finite-G FFTDF.get_pp expression, on the CPU.

    This is also a CPU fallback for the GPU FFTDF core path used for meshes
    of at least 500 cubed points.  Ordinary GPU meshes use multigrid above.
    """
    from pyscf.pbc import tools
    from pyscf.pbc.dft import numint
    from pyscf.pbc.gto import pseudo

    gv = cell.get_Gv(mesh)
    si = cell.get_SI(mesh=mesh)
    if cell._pseudo:
        vloc_g = pseudo.get_vlocG(cell, gv)
    else:
        vloc_g = cell.atom_charges()[:, None] * tools.get_coulG(
            cell, mesh=mesh, Gv=gv)
    potential = tools.ifft(-np.einsum('ig,ig->g', si, vloc_g), mesh).real
    coords = cell.get_uniform_grids(mesh)
    ngrids = len(coords)
    rho = np.empty(ngrids)
    grad = np.zeros((cell.natm, 3))
    slices = cell.aoslice_by_atom()[:, 2:]
    # AO values plus three spatial derivatives; avoid a full-grid AO tensor.
    blocksize = max(256, min(8192, (128 << 20) // (32 * max(cell.nao, 1))))
    for start in range(0, ngrids, blocksize):
        stop = min(ngrids, start + blocksize)
        ao = numint.eval_ao(cell, coords[start:stop], deriv=1)
        ao_dm = ao[0] @ dm
        rho[start:stop] = np.einsum('gm,gm->g', ao[0], ao_dm)
        for atom, (p0, p1) in enumerate(slices):
            grad[atom] -= 2 * np.einsum(
                'xgm,gm,g->x', ao[1:4, :, p0:p1], ao_dm[:, p0:p1],
                potential[start:stop])
    # FFTDF's inverse transform already includes the local potential weight.
    # Multiplying by cell.vol/ngrids here would count that weight twice.
    for atom in range(cell.natm):
        dg = 1j * gv.T * (si[atom] * vloc_g[atom])
        grad[atom] += tools.ifft(dg, mesh).real @ rho
    if cell._pseudo:
        grad += _fft_nonlocal_gradient(cell, dm, gv, si, slices)
    return grad


def _fft_nonlocal_gradient(cell, dm, gv, si, slices):
    """Derivative of the finite reciprocal projector contraction in FFTDF."""
    from pyscf import gto
    from pyscf.pbc.df import ft_ao
    from pyscf.pbc.gto import pseudo

    # Keep both normalization factors of FFTDF.get_pp: one sqrt(volume) in
    # the Fourier AOs and one volume after contracting projector products.
    ao_g = ft_ao.ft_ao(cell, gv) / np.sqrt(cell.vol)
    radius = np.linalg.norm(gv, axis=1)
    grad = np.zeros((cell.natm, 3))
    fake = gto.Mole()
    fake._atm = np.zeros((1, gto.ATM_SLOTS), dtype=np.int32)
    fake._bas = np.zeros((1, gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    fake._env = np.zeros(ptr + 10)
    fake._bas[0, gto.NPRIM_OF] = 1
    fake._bas[0, gto.NCTR_OF] = 1
    fake._bas[0, gto.PTR_EXP] = ptr + 3
    fake._bas[0, gto.PTR_COEFF] = ptr + 4
    for atom in range(cell.natm):
        symbol = cell.atom_symbol(atom)
        if symbol not in cell._pseudo:
            continue
        for angular, (rl, nl, hl) in enumerate(cell._pseudo[symbol][5:]):
            if nl == 0:
                continue
            fake._bas[0, gto.ANG_OF] = angular
            fake._env[ptr + 3] = .5 * rl**2
            fake._env[ptr + 4] = rl**(angular + 1.5) * np.pi**1.25
            ylm = fake.eval_gto('GTOval', gv).T
            projectors = np.asarray([
                ylm * pseudo.pp._qli(radius * rl, angular, radial)
                for radial in range(nl)])
            projectors = projectors * si[atom].conj()
            shape = (nl, 2 * angular + 1, cell.nao)
            flat = projectors.reshape(-1, len(gv))
            p = (flat @ ao_g).reshape(shape)
            hpd = (np.einsum('ij,jmp->imp', np.asarray(hl), p) @ dm)
            for axis in range(3):
                # dP/dR_a = delta(a,projector_center) Q - Q M_a.
                q = ((flat * (1j * gv[:, axis])) @ ao_g).reshape(shape)
                response = 2 / cell.vol * (q.conj() * hpd).real.sum(axis=(0, 1))
                grad[atom, axis] += response.sum()
                for center, (p0, p1) in enumerate(slices):
                    grad[center, axis] -= response[p0:p1].sum()
    return grad
