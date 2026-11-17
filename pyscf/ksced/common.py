'''Shared by both basis modes: array helpers and the KSCED SCF mixin.'''

import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger

T_NAD_DEFAULT = 'LDA_K_TF'


def _is_cell(mol):
    from pyscf.pbc.gto import Cell
    return isinstance(mol, Cell)


def _is_gpu_array(a):
    # GPU4PySCF tagged arrays still use CuPy's explicit host transfer.
    return not isinstance(a, numpy.ndarray) and callable(getattr(a, 'get', None))


def _as_like(ref, arr):
    '''Return arr on the same array backend as ref.
    '''
    if _is_gpu_array(ref) and not _is_gpu_array(arr):
        import cupy
        return cupy.asarray(arr)
    if not _is_gpu_array(ref) and _is_gpu_array(arr):
        return arr.get()
    return arr


def _trace_prod(a, b):
    '''<a|b> = einsum('ij,ji->', a, b).real, as a Python float, on any backend.'''
    if _is_gpu_array(a) or _is_gpu_array(b):
        import cupy
        a = cupy.asarray(a) if not _is_gpu_array(a) else a
        b = cupy.asarray(b) if not _is_gpu_array(b) else b
        return float(cupy.einsum('ij,ji->', a, b).real.get())
    return float(numpy.einsum('ij,ji->', a, b).real)


def _spin_sum(dm):
    """Total density matrix from one that may be spin resolved.

    An unrestricted density matrix is (2, nao, nao); a restricted one is
    (nao, nao). Every cross term in the KSCED energy contracts a spin-free
    one-electron operator -- V_ne[A], J[rho_B] -- against a density, and those
    contractions want the total density in both cases.
    """
    return dm[0] + dm[1] if getattr(dm, 'ndim', 2) == 3 else dm


def _trace_prod_spin(a, b):
    """<a|b> where a is a 2-D operator and b may be spin resolved."""
    return _trace_prod(a, _spin_sum(b))


def _is_kscf(mf):
    """True for a k-point SCF object (KRKS, KUKS, ...)."""
    return bool(mf.istype('KSCF'))


def _kweight_trace(a, b):
    """(1/nk) sum_k tr(a_k b_k).real for (nk, nao, nao) stacks, on any backend.

    The k-point average used by pyscf.pbc.dft.krks for every one-electron energy
    term: krks.energy_elec weights tr(h_k D_k), and krks.get_veff weights
    tr(D_k J_k), by 1/nk.
    """
    if _is_gpu_array(a) or _is_gpu_array(b):
        import cupy
        a, b = cupy.asarray(a), cupy.asarray(b)
        return float(cupy.einsum('kij,kji->', a, b).real.get()) / a.shape[0]
    a, b = numpy.asarray(a), numpy.asarray(b)
    return float(numpy.einsum('kij,kji->', a, b).real) / a.shape[0]


def _is_polarized(dm):
    """True when dm carries an explicit spin axis."""
    return getattr(dm, 'ndim', 2) == 3


def _as_pair(dm):
    """dm as (2, nao, nao), splitting a restricted matrix evenly."""
    if _is_polarized(dm):
        return dm
    half = dm * .5
    return _stack_like(half, [half, half])


def _stack_like(ref, arrays):
    """Stack on whichever backend ref lives on."""
    if _is_gpu_array(ref):
        import cupy
        return cupy.stack([cupy.asarray(a) for a in arrays])
    return numpy.stack([numpy.asarray(a) for a in arrays])


def _avg_spin(v):
    """The spin average of a (2, nao, nao) potential, as a 2-D matrix.

    Used for a restricted subsystem A embedded in a polarised environment. It
    is not an approximation: A's own restrictedness constrains
    rho_alpha = rho_beta = rho_A / 2, so

        dE/dD_A = dE/dD_alpha . dD_alpha/dD_A + dE/dD_beta . dD_beta/dD_A
                = (v_alpha + v_beta) / 2

    and the matrix returned is the exact gradient of the energy reported.
    """
    return (v[0] + v[1]) * .5 if _is_polarized(v) else v


def _tag_array(a, **tags):
    """lib.tag_array on whichever backend a lives on.

    PySCF's tag_array calls numpy.asarray, which cupy refuses. GPU4PySCF ships an
    equivalent that produces a CPArrayWithTag.
    """
    if _is_gpu_array(a):
        from gpu4pyscf.lib.cupy_helper import tag_array as gpu_tag_array
        return gpu_tag_array(a, **tags)
    return lib.tag_array(a, **tags)


def _kinetic_of(mol, kpt=None):
    """Kinetic-energy matrix. Basis-only, so A and B share it.

    kpt may be a single k-point (3,), giving (nao, nao), or a k-point set
    (nk, 3), giving the stacked (nk, nao, nao) matrices.
    """
    if _is_cell(mol):
        if kpt is None:
            kpt = numpy.zeros(3)
        if numpy.ndim(kpt) == 2:
            return numpy.asarray(mol.pbc_intor('int1e_kin', hermi=1, kpts=kpt))
        return mol.pbc_intor('int1e_kin', hermi=1, kpts=kpt)
    return mol.intor_symmetric('int1e_kin')


def _vne_from_hcore(mf, mol, kpt=None):
    """V_ne = hcore - T, taken from the backend's own get_hcore.
    """
    if _is_cell(mol):
        h1e = mf.get_hcore(mol, kpt)
    else:
        h1e = mf.get_hcore(mol)
    return h1e - _as_like(h1e, _kinetic_of(mol, kpt))


class _KSCED:
    '''Tag class labelling a KSCED-embedded SCF method.'''
    pass


class KSCEDMixin(_KSCED):
    '''Behaviour shared by the molecular and periodic KSCED methods.

    The domain-specific part is get_veff, supplied by the subclasses.
    '''

    _keys = {'with_env', 't_nad', 'mol_ab', 'e_tnad'}

    def __init__(self, mf, env, mol_ab=None):
        self.__dict__.update(mf.__dict__)
        self.with_env = env
        self.mol_ab = mol_ab
        self.t_nad = T_NAD_DEFAULT
        self.e_tnad = 0.0

    def undo_ksced(self):
        '''Return a plain SCF object without the embedding.'''
        obj = lib.view(self, lib.drop_class(self.__class__, KSCEDMixin, 'KSCED'))
        for key in ('with_env', 'mol_ab', 't_nad', 'e_tnad'):
            obj.__dict__.pop(key, None)
        return obj

    def energy_potential(self):
        '''Conservative embedded energy, in Hartree, at the converged state.

        At finite electronic temperature this is E - sigma*S, not the
        internal energy or the extrapolated zero-temperature energy. Frozen
        B's self energy is a geometry-independent constant and is omitted.
        '''
        if not self.converged:
            raise RuntimeError('Converge the embedded SCF before requesting its energy')
        energy = float(self.e_tot)
        if getattr(self, 'sigma', 0) and getattr(self, 'smearing_method', None):
            if getattr(self, 'mu0', None) is not None:
                raise NotImplementedError('KSCED forces require fixed electron number, not fixed mu0')
            if self.entropy is None:
                raise RuntimeError('Smearing entropy is unavailable; rerun the SCF')
            energy -= float(self.sigma * self.entropy)
        return energy

    def nuc_grad_method(self):
        '''Analytic frozen-B A gradients for monomolecular KSCED.'''
        from pyscf.ksced.mono.env import _FrozenEnvMB
        if not isinstance(self.with_env, _FrozenEnvMB):
            raise NotImplementedError("KSCED gradients require basis_mode='M'")
        if _is_kscf(self):
            raise NotImplementedError(
                'KSCED gradients are gamma-point only; this is a k-point object')
        if _is_cell(self.mol):
            from pyscf.ksced.mono.grad import Gradients
        else:
            from pyscf.ksced.mono.grad_mol import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'KSCED non-additive kinetic functional = %s', self.t_nad)
        logger.info(self, 'KSCED environment nao = %d', self.with_env.dm_b.shape[-1])
        if self.mol_ab is None:
            logger.info(self, 'KSCED mol_ab not supplied: E_nn[AB] excluded from e_tot')
        return self

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        h1e = super().get_hcore(mol)
        vne_b = self.with_env.get_vne_b(mol, getattr(self, 'kpt', None))
        return h1e + _as_like(h1e, vne_b)

    def energy_nuc(self):
        e = super().energy_nuc()
        if self.mol_ab is not None:
            e += (self.mol_ab.energy_nuc()
                  - self.mol.energy_nuc()
                  - self.with_env.mol_b.energy_nuc())
        return e

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        e_tot_elec, e2 = super().energy_elec(dm, h1e, vhf)

        env = self.with_env
        # Attraction between A's nuclei and B's frozen electrons. Pass the
        # callable so the environment can cache its first evaluation.
        e_vne_a_rho_b = env.e_vne_a_rho_b(self._vne_a)
        # Add the half of J_AB not included in get_veff's Coulomb energy.
        e_coul_ab_half = self._trace_dm(env.get_j_b(self, self.mol), dm) * .5

        self.scf_summary['ksced_vne_a_rho_b'] = e_vne_a_rho_b
        self.scf_summary['ksced_coul_ab_half'] = e_coul_ab_half
        return (e_tot_elec + e_vne_a_rho_b + e_coul_ab_half,
                e2 + e_coul_ab_half)

    def _trace_dm(self, op, dm):
        """<op|dm> for one energy term. The k-point classes average over k."""
        return _trace_prod_spin(op, dm)

    def _vne_a(self):
        """V_ne[A] from the unmodified get_hcore, i.e. without vne_b."""
        kpt = getattr(self, 'kpt', None)
        mol = self.mol
        if _is_cell(mol):
            h1e = super().get_hcore(mol, kpt)
        else:
            h1e = super().get_hcore(mol)
        return h1e - _as_like(h1e, _kinetic_of(mol, kpt))

    def reset(self, mol=None):
        self.with_env.reset()
        return super().reset(mol)



class KSCEDKptMixin:
    '''k-point overrides shared by both basis modes. List it before the mode's
    mixin, e.g. class X(KSCEDKptMixin, KSCEDMixin).

    A k-point object carries kpts (nk, 3) rather than kpt, its one-electron
    matrices are (nk, nao, nao), and every energy trace is the k average used by
    pyscf.pbc.dft.krks: (1/nk) sum_k tr(O_k D_k).
    '''

    def get_hcore(self, cell=None, kpts=None):
        '''hcore_k of A plus V_ne[B]_k, for every k-point.

        KSCF.get_hcore takes (cell, kpts); KSCEDMixin.get_hcore takes a single
        gamma-point kpt, so the k-point classes supply their own.
        '''
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kpts
        h1e = super(KSCEDMixin, self).get_hcore(cell, kpts)
        vne_b = self.with_env.get_vne_b(cell, kpts)
        return h1e + _as_like(h1e, vne_b)

    def _vne_a(self):
        '''V_ne[A]_k from the unmodified get_hcore, i.e. without V_ne[B].'''
        h1e = super(KSCEDMixin, self).get_hcore(self.cell, self.kpts)
        return h1e - _as_like(h1e, _kinetic_of(self.cell, self.kpts))

    def _trace_dm(self, op, dm):
        return _kweight_trace(op, dm)
