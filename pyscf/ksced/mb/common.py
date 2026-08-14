'''Behaviour shared by the molecular and periodic monomolecular classes.

Derives from KSCEDMixin without modifying it: get_hcore, energy_elec,
energy_nuc and reset are already correct for MB, because everything they touch
goes through the environment object and _FrozenEnvMB presents the same
interface as _FrozenEnv.
'''

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

        The offset numint adds rho_B inside whichever density hook the live
        backend calls. GPU4PySCF's *molecular* nr_rks calls neither: it walks
        ni.block_loop but computes rho inline from mo_coeff inside
        _nr_rks_task, one thread per device, never through eval_rho or
        _gen_rho_evaluator. Probed on an H200: nr_rks ran to completion with
        zero density-hook calls over 33792 grid points and returned a
        perfectly plausible number computed from rho_A alone.

        A wrong energy that looks right is the worst failure available here,
        so check that the rho_B cache actually filled rather than trusting a
        hook fired. Behavioural rather than a backend name test, so a future
        backend with the same structure is caught too.

        cause is the exception raised by nr_rks itself, when there was one:
        the same missing seam can also surface as a shape error inside the
        wrapped loop, and that error should not mask the diagnosis.
        '''
        gd = self.with_env._griddens
        if gd is not None and gd.nblocks > 0:
            return
        raise NotImplementedError(
            "KSCED basis_mode='M' cannot reach the density evaluation of %s: "
            "the environment density was never added, so the result would be "
            "subsystem A alone. This is known for GPU4PySCF's molecular "
            "numint, whose nr_rks computes rho inline per device rather than "
            "through eval_rho or _gen_rho_evaluator. Use the periodic GPU "
            "path, or run molecular MB on CPU PySCF."
            % type(self._numint).__name__) from cause

    def _log_electron_counts(self, n_total):
        '''The offset numint integrates rho_t, so its nelec counts N_A + N_B.

        PySCF would otherwise print "electron number deviates" comparing that
        total against A's electron count alone -- 1328 against 8 on Au120_NH3.
        Report the two separately instead, and never feed n_total to the stock
        warning.
        '''
        n_b = self.with_env.mol_b.nelectron
        logger.debug(self, 'KSCED grid electrons: total %.6f, B %d, A %.6f',
                     float(n_total), n_b, float(n_total) - n_b)
