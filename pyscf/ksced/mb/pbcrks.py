'''Periodic gamma-point KSCED in a monomolecular basis. Filled in by Task 6.'''

from pyscf.ksced.mb.common import KSCEDMBMixin


class KSCEDMBPBCRKS(KSCEDMBMixin):
    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
                 kpt=None, kpts_band=None):
        raise NotImplementedError('periodic monomolecular KSCED lands in Task 6')
