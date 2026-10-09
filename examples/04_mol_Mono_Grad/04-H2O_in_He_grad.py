"""Molecular A-in-B energy and gradient; pass --gpu for GPU4PySCF."""
import argparse
from pyscf import dft, gto, ksced

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

mol_a = gto.M(atom='O 0 0 0; H 1.43 .12 1.12; H -1.45 -.08 1.08',
              unit='Bohr', basis='sto-3g')
mol_b = gto.M(atom='He 3.2 1.7 .8', unit='Bohr', basis='sto-3g')
mf_a = dft.RKS(mol_a, xc='PBE')
mf_b = dft.RKS(mol_b, xc='PBE')
if args.gpu:
    mf_a, mf_b = mf_a.to_gpu(), mf_b.to_gpu()
mf_b.conv_tol = 1e-11
mf_b.kernel()

mf_ainb = ksced.embed(mf_a, mf_b, basis_mode='M')
mf_ainb.t_nad = 'GGA_K_APBE'
mf_ainb.conv_tol = 1e-11
mf_ainb.kernel()
energy_ainb = mf_ainb.energy_potential()
gradient_ainb_on_a = mf_ainb.nuc_grad_method().kernel()
print('Embedded energy (Hartree):', energy_ainb)
print('Forces on A (Hartree/Bohr):\n', -gradient_ainb_on_a)
