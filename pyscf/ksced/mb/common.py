'''Behaviour shared by the molecular and periodic monomolecular classes.

Derives from KSCEDMixin without modifying it: get_hcore, energy_elec,
energy_nuc and reset are already correct for MB, because everything they touch
goes through the environment object and _FrozenEnvMB presents the same
interface as _FrozenEnv.
'''

import numpy
from pyscf.lib import logger
from pyscf.ksced.ksced import KSCEDMixin


class KSCEDMBMixin(KSCEDMixin):
    '''KSCEDMixin plus the reporting the monomolecular path needs.'''

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        env = self.with_env
        logger.info(self, 'KSCED basis_mode = M (monomolecular)')
        logger.info(self, 'KSCED nao_A = %d, nao_B = %d, nao_AB = %d '
                          '(dimension reduction %.1fx)',
                    env.nao_a, env.nao_b, env.mol_ab.nao,
                    env.mol_ab.nao / max(env.nao_a, 1))
        logger.info(self, 'KSCED quadrature grid = %s',
                    type(self.grids).__name__)
        return self

    def _check_xc_types(self, ni):
        '''rho_B is stored at GGA order, so meta-GGA cannot be served.'''
        for code in (self.xc, self.t_nad):
            if ni._xc_type(code) not in ('LDA', 'GGA'):
                raise NotImplementedError(
                    'KSCED monomolecular basis supports LDA and GGA only; '
                    '%s is %s' % (code, ni._xc_type(code)))

    def _assert_env_density_entered(self, cause=None):
        '''Fail loudly if the frozen density never reached the functional.
        '''
        gd = self.with_env._griddens
        if gd is not None and gd.nblocks > 0:
            return
        raise NotImplementedError(
            "KSCED basis_mode='M' cannot reach the density evaluation of %s: "
            "Use the periodic GPU path, or run molecular MB on CPU PySCF."
            % type(self._numint).__name__) from cause

    def _log_electron_counts(self, n_total):
        '''The offset numint integrates rho_t, so its nelec counts N_A + N_B.
        '''
        get = getattr(n_total, 'get', None)
        if callable(get):
            n_total = get()
        n_total = float(numpy.asarray(n_total).sum())
        n_b = self.with_env.mol_b.nelectron
        logger.debug(self, 'KSCED grid electrons: total %.6f, B %d, A %.6f',
                     float(n_total), n_b, float(n_total) - n_b)
