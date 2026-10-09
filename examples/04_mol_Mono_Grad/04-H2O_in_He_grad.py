"""
Molecular A-in-B energy and gradient on CPU.

H2O is the embedded subsystem A; He is the frozen environment B. Each uses
its own atomic basis (monomolecular mode). The embedded energy omits He's
constant self-energy. Its analytic gradient gives derivatives with respect
to H2O's nuclear positions while He stays fixed.
"""
from pyscf import dft, gto, ksced

#1 setup (coordinates in Bohr)
mol_a = gto.M(atom='O 0 0 0; H 1.43 .12 1.12; H -1.45 -.08 1.08',
              unit='Bohr', basis='sto-3g')
mol_b = gto.M(atom='He 3.2 1.7 .8', unit='Bohr', basis='sto-3g')
mf_a = dft.RKS(mol_a, xc='PBE')
mf_b = dft.RKS(mol_b, xc='PBE')

#2 embedding B (He)
mf_b.kernel()

#3 embedded A (H2O in He)
mf_ainb = ksced.embed(mf_a, mf_b, basis_mode='M')
mf_ainb.t_nad = 'GGA_K_APBE'
mf_ainb.kernel()

#4 get embedded energy and gradient on A
energy_ainb = mf_ainb.energy_potential()
gradient_ainb_on_a = mf_ainb.nuc_grad_method().kernel()

#5 report energy and forces on A
print('Embedded energy (Hartree):', energy_ainb)
print('Forces on A (Hartree/Bohr):\n', -gradient_ainb_on_a)
