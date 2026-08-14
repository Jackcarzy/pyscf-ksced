'''KSCED subsystem embedding: frozen environment and the SCF mixin.'''

import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger

T_NAD_DEFAULT = 'LDA_K_TF'


def _vne_of(mol):
    '''Nuclear-electron attraction matrix for mol, pseudopotential aware.

    Mirrors what pyscf.scf.hf builds for h1e, so that V_ne[B] evaluated here is
    consistent with the V_ne[A] already inside get_hcore.
    '''
    if mol._pseudo:
        vne = gto.pp_int.get_gth_pp(mol)
    else:
        vne = mol.intor_symmetric('int1e_nuc')
    if len(mol._ecpbas) > 0:
        vne = vne + mol.intor_symmetric('ECPscalar')
    return vne


class _FrozenEnv:
    '''The frozen subsystem B.

    Everything the embedded calculation needs from B is served from here and
    cached, because rho_B never changes during A's SCF. This is the only class
    that assumes A and B share an AO basis.
    '''

    def __init__(self, mf_b, dm_b=None):
        self.mf_b = mf_b
        self.mol_b = mf_b.mol
        self.dm_b = mf_b.make_rdm1() if dm_b is None else dm_b
        self.reset()

    def reset(self):
        self._vne_b = None
        self._j_b = None
        self._e_xc = None
        self._e_tnad_b = None
        self._e_vne_a_rho_b = None
        return self

    def get_vne_b(self, mol):
        '''V_ne[B] in the shared AO basis.'''
        if self._vne_b is None:
            self._vne_b = _vne_of(self.mol_b)
        return self._vne_b

    def get_j_b(self, mf, mol):
        '''J[rho_B] in the shared AO basis.'''
        if self._j_b is None:
            self._j_b = mf.get_j(mol, self.dm_b, 1)
        return self._j_b

    def e_xc(self, ni, mol, grids, xc, max_memory):
        '''E_xc[rho_B].'''
        if self._e_xc is None:
            self._e_xc = ni.nr_rks(mol, grids, xc, self.dm_b,
                                   max_memory=max_memory)[1]
        return self._e_xc

    def e_tnad_b(self, ni, mol, grids, t_nad, max_memory):
        '''T_s^TF[rho_B], the B term of the non-additive kinetic energy.'''
        if self._e_tnad_b is None:
            self._e_tnad_b = ni.nr_rks(mol, grids, t_nad, self.dm_b,
                                       max_memory=max_memory)[1]
        return self._e_tnad_b

    def e_vne_a_rho_b(self, mol_a):
        '''<V_ne[A] | rho_B>, a constant of the embedded SCF.'''
        if self._e_vne_a_rho_b is None:
            self._e_vne_a_rho_b = numpy.einsum(
                'ij,ji->', _vne_of(mol_a), self.dm_b).real
        return self._e_vne_a_rho_b


class _KSCED:
    '''Tag class labelling a KSCED-embedded SCF method.'''
    pass


class KSCEDMixin(_KSCED):
    '''Behaviour shared by the molecular and periodic KSCED methods.

    The domain-specific part is get_veff, supplied by the subclasses. Everything
    here is common because pyscf.pbc.dft.rks assigns
    energy_elec = mol_ks.energy_elec, so one energy expression serves both.
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
        return super().get_hcore(mol) + self.with_env.get_vne_b(mol)

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
        # Constant: the A nuclei attracting the frozen B electrons.
        e_vne_a_rho_b = env.e_vne_a_rho_b(self.mol)
        # The second half of J_AB. get_veff already contributed the first half
        # through ecoul = 0.5 * <dm_a | J[rho_total]>.
        e_coul_ab_half = numpy.einsum(
            'ij,ji->', env.get_j_b(self, self.mol), dm).real * .5

        self.scf_summary['ksced_vne_a_rho_b'] = e_vne_a_rho_b
        self.scf_summary['ksced_coul_ab_half'] = e_coul_ab_half
        return (e_tot_elec + e_vne_a_rho_b + e_coul_ab_half,
                e2 + e_coul_ab_half)

    def reset(self, mol=None):
        self.with_env.reset()
        return super().reset(mol)


def _is_pbc(mf):
    from pyscf.pbc.gto import Cell
    return isinstance(mf.mol, Cell)


def _check_numint(mf):
    '''KSCED needs the plain grid-based NumInt; MultiGrid takes a different path.'''
    if not _is_pbc(mf):
        return
    from pyscf.pbc.dft import multigrid
    if isinstance(mf._numint, multigrid.MultiGridNumInt):
        raise NotImplementedError(
            'KSCED does not support MultiGridNumInt. Build the RKS object '
            'without multigrid acceleration.')


def embed(mf, mf_b, dm_b=None, mol_ab=None):
    '''Attach a frozen KSCED environment to a restricted KS object.

    Args:
        mf : RKS object for subsystem A, built in the supermolecular basis with
            ghost atoms on B.
        mf_b : converged RKS object for subsystem B, built in the same
            supermolecular basis with ghost atoms on A.

    Kwargs:
        dm_b : density matrix for B. Defaults to mf_b.make_rdm1().
        mol_ab : Mole or Cell for the whole system. When given, the A-B nuclear
            repulsion is included in e_tot, making e_tot the complete embedded
            energy and Eint a plain three-term subtraction.

    Returns:
        A new object whose class derives from both the KSCED mixin and the class
        of mf.
    '''
    from pyscf.ksced import rks as ksced_rks
    from pyscf.ksced import pbcrks as ksced_pbcrks

    env = _FrozenEnv(mf_b, dm_b)

    if isinstance(mf, _KSCED):
        mf.with_env = env
        mf.mol_ab = mol_ab
        return mf

    _check_numint(mf)
    base = ksced_pbcrks.KSCEDPBCRKS if _is_pbc(mf) else ksced_rks.KSCEDRKS

    obj = base(mf, env, mol_ab)
    name = 'KSCED' + mf.__class__.__name__
    return lib.set_class(obj, (base, mf.__class__), name)
