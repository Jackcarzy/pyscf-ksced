'''Periodic gamma-point KSCED for restricted Kohn-Sham. Implemented in Task 6.'''

from pyscf.ksced.ksced import KSCEDMixin


class KSCEDPBCRKS(KSCEDMixin):
    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
                 kpt=None, kpts_band=None):
        raise NotImplementedError('periodic KSCED arrives in Task 6')
