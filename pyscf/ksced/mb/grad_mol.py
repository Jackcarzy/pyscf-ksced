'''Frozen-B molecular gradients with full atom-centred quadrature response.

AB padded densities are used only for derivative contractions. The embedded
orbitals and Pulay term remain in A's basis. B's coefficients and centres are
fixed; its functional energy still responds to the moving AB quadrature.
'''

import copy
import numpy as np
from pyscf import scf, lib
from pyscf.grad import rhf, rks, uks
from pyscf.ksced.ksced import _spin_sum
from pyscf.ksced.mb.arrays import to_host
from pyscf.ksced.mb.molecular import is_gpu, padded_densities
from pyscf.ksced.mb.grad import _Scanner, _contract_ip


def _same_integrals(ab, expected):
    from pyscf.gto import mole
    if not all(np.array_equal(getattr(ab, name), getattr(expected, name))
               for name in ('_atm', '_bas', '_ecpbas')):
        return False

    def same_slice(pointer, size):
        return (pointer >= 0 and pointer + size <= len(ab._env)
                and pointer + size <= len(expected._env)
                and np.array_equal(ab._env[pointer:pointer + size],
                                   expected._env[pointer:pointer + size]))

    # Compare only referenced physical data. libcint also stores mutable
    # scratch state in _env, including the last inverse-distance origin atom.
    for shell in expected._bas:
        nprim, nctr = shell[mole.NPRIM_OF], shell[mole.NCTR_OF]
        if (not same_slice(shell[mole.PTR_EXP], nprim)
                or not same_slice(shell[mole.PTR_COEFF], nprim * nctr)):
            return False
    for atom in expected._atm:
        if not same_slice(atom[mole.PTR_COORD], 3):
            return False
        for field in (mole.PTR_ZETA, mole.PTR_FRAC_CHARGE, mole.PTR_RADIUS):
            pointer = atom[field]
            if pointer > 0 and not same_slice(pointer, 1):
                return False
    return same_slice(mole.PTR_RANGE_OMEGA, 1)


def _validate(mf):
    from pyscf.ksced.mb.env import _conc
    env = mf.with_env
    for obj in (mf, env.mf_b):
        mol = obj.mol
        if getattr(obj, 'with_df', None) is not None:
            raise NotImplementedError('Molecular KSCED gradients require direct Coulomb integrals')
        if mol._ecp or mol._pseudo or getattr(obj, 'with_x2c', None):
            raise NotImplementedError('Molecular KSCED gradients require all-electron nonrelativistic Hamiltonians')
        if np.any(mol.atom_charges() == 0):
            raise NotImplementedError('Molecular KSCED gradients require separate bases without ghost atoms')
        if getattr(obj, 'disp', None) or getattr(obj, 'nlc', None):
            raise NotImplementedError('Molecular KSCED gradients do not include dispersion or nonlocal XC')
    for code in (mf.xc, mf.t_nad):
        if mf._numint._xc_type(code) not in ('LDA', 'GGA') or mf._numint.libxc.is_hybrid_xc(code):
            raise NotImplementedError('Molecular KSCED gradients support pure LDA/GGA only')
    expected = _conc(mf.mol, env.mol_b)
    ab = env.mol_ab
    # Shell tables alone do not contain exponent and contraction values.
    if (not _same_integrals(ab, expected) or ab.cart != expected.cart
            or ab.cart != mf.mol.cart or ab.cart != env.mol_b.cart
            or ab._ecp or ab._pseudo):
        raise ValueError('Molecular KSCED gradients require the A-then-B concatenation '
                         'with identical basis and Hamiltonian data')
    distances = np.linalg.norm(ab.atom_coords()[:, None] - ab.atom_coords(), axis=2)
    np.fill_diagonal(distances, np.inf)
    if np.min(distances) < 1e-8:
        raise NotImplementedError('Molecular KSCED gradients do not support coincident centres')
    mf.initialize_grids()
    if mf.grids.mol is not ab:
        raise NotImplementedError('Molecular KSCED gradients require the AB atom-centred grid')
    if is_gpu(mf):
        from gpu4pyscf.dft import gen_grid, radi
        allowed_radii = (None, radi.treutler_atomic_radii_adjust)
    else:
        from pyscf.dft import gen_grid, radi
        allowed_radii = (None, radi.treutler_atomic_radii_adjust,
                         radi.becke_atomic_radii_adjust)
    if type(mf.grids) is not gen_grid.Grids:
        raise NotImplementedError('Molecular KSCED gradients require the standard backend Grids class')
    # Copying/rebuilding also copies instance monkeypatches. Native response
    # reconstructs partitions independently, so such hooks cannot be trusted
    # even when rebuilding reproduces the SCF coordinates and weights.
    grid_hooks = ('build', 'kernel', 'reset', 'gen_atomic_grids',
                  'get_partition', 'gen_partition', 'make_mask',
                  '_select_grids', '_add_padding', 'prune_by_density_')
    if any(name in mf.grids.__dict__ for name in grid_hooks):
        raise NotImplementedError('Molecular KSCED gradients do not support overridden grid methods')
    if mf.grids.radii_adjust not in allowed_radii:
        raise NotImplementedError('Molecular KSCED gradients do not support custom radius adjustments')
    # Both backends reconstruct atom grids for full response. Reject external
    # points, altered weights, and settings changed after the SCF grid build.
    fresh = copy.copy(mf.grids)
    fresh.reset(ab)
    fresh.build()
    if (fresh.coords.shape != mf.grids.coords.shape
            or not np.allclose(to_host(fresh.coords), to_host(mf.grids.coords), rtol=0, atol=1e-12)
            or not np.allclose(to_host(fresh.weights), to_host(mf.grids.weights), rtol=0, atol=1e-12)):
        raise NotImplementedError('Molecular KSCED gradients require unmodified atom-centred quadrature')
    mf.energy_potential()


def _functional(mf, code, dm):
    ab = mf.with_env.mol_ab
    if is_gpu(mf):
        from gpu4pyscf.grad import rks as gr, uks as gu
        module = gu if dm.ndim == 3 else gr
        return to_host(module.get_exc_full_response(mf._numint, ab, mf.grids, code, dm)[0])
    module = uks if dm.ndim == 3 else rks
    grid_force, vmat = module.get_vxc_full_response(
        mf._numint, ab, mf.grids, code, dm, max_memory=mf.max_memory)
    if dm.ndim == 2:
        return grid_force + _contract_ip(ab, vmat, dm)
    return grid_force + sum(_contract_ip(ab, v, d) for v, d in zip(vmat, dm))


def _hartree(mf, dm):
    ab = mf.with_env.mol_ab
    if is_gpu(mf):
        import cupy as cp
        from gpu4pyscf.scf import hf as gpu_hf
        # This is a spin-summed J contraction, independent of AB occupations.
        # The public RHF factory selects unsupported ROHF for open-shell AB.
        ref = gpu_hf.RHF(ab)
        ref.direct_scf_tol = mf.direct_scf_tol
        return to_host(ref.nuc_grad_method().jk_energy_per_atom(cp.asarray(dm), 1., 0.))
    ref = scf.hf.RHF(ab)
    ref.direct_scf_tol = mf.direct_scf_tol
    vj = ref.nuc_grad_method().get_j(ab, dm)
    return _contract_ip(ab, vj, dm)


class Gradients(rhf.GradientsBase):
    '''dE_potential/dR_A in Hartree/Bohr, with frozen B and moving AB grids.'''

    _keys = {'components'}

    def __init__(self, mf):
        super().__init__(mf)
        self.components = {}

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        mf, mol = self.base, self.mol
        _validate(mf)
        coeff = to_host(mf.mo_coeff if mo_coeff is None else mo_coeff)
        occ = to_host(mf.mo_occ if mo_occ is None else mo_occ)
        energy = to_host(mf.mo_energy if mo_energy is None else mo_energy)
        dm = to_host(mf.make_rdm1(
            mf.mo_coeff if mo_coeff is None else mo_coeff,
            mf.mo_occ if mo_occ is None else mo_occ))
        if np.iscomplexobj(dm) and np.max(abs(dm.imag)) > 1e-12:
            raise NotImplementedError('Molecular KSCED gradients require real densities')
        dm = dm.real
        da, db = padded_densities(mf, dm)
        dt = da + db
        ab, n = mf.with_env.mol_ab, mol.natm
        self.components = {
            'xc': (_functional(mf, mf.xc, dt) - _functional(mf, mf.xc, db))[:n],
            'kinetic_nad': (_functional(mf, mf.t_nad, dt)
                            - _functional(mf, mf.t_nad, da)
                            - _functional(mf, mf.t_nad, db))[:n],
            'hartree': _hartree(mf, _spin_sum(dt))[:n],
        }
        # B-only one-electron and Coulomb energies have no A derivative.
        # AB hcore contracts all nuclear-centre derivatives, including V_A rho_B.
        hgen = scf.hf.RHF(ab).nuc_grad_method().hcore_generator(ab)
        self.components['one_electron'] = np.asarray([
            np.einsum('xij,ji->x', hgen(i), _spin_sum(dt)) for i in range(n)])
        if coeff.ndim == 2:
            weighted = (coeff * (occ * energy)).dot(coeff.conj().T).real
        else:
            weighted = sum((c * (o * e)).dot(c.conj().T).real
                           for c, o, e in zip(coeff, occ, energy))
        self.components['pulay'] = _contract_ip(mol, mol.intor('int1e_ipovlp'), weighted)
        out = sum(self.components.values())
        return out if atmlst is None else out[atmlst]

    def grad_nuc(self, mol=None, atmlst=None):
        out = rhf.grad_nuc(self.base.with_env.mol_ab)[:self.mol.natm]
        self.components['nuclear'] = out
        return out if atmlst is None else out[atmlst]

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        if atmlst is None:
            atmlst = self.atmlst
        if atmlst is not None:
            atmlst = list(atmlst)
            if any(i < 0 or i >= self.mol.natm for i in atmlst):
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
        return lib.set_class(_Scanner(self), (_Scanner, self.__class__), 'KSCEDMolGradScanner')
