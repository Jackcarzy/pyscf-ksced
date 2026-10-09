"""
Periodic A-in-B energy and gradient on GPU using GPU4PySCF.

H2 is the embedded subsystem A; He is the frozen environment B. Each uses
its own atomic basis (monomolecular mode) in the same periodic cell at the
Gamma point. The embedded energy omits He's constant self-energy. Its
analytic gradient gives derivatives with respect to H2's nuclear positions
while He's nuclei and density, and the lattice, stay fixed.
"""
import numpy as np
from pyscf import ksced
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft

#1 setup (coordinates and lattice in Bohr)
# Both subsystems use the same fixed lattice and FFT mesh.
common = dict(a=np.eye(3) * 10, unit='Bohr', basis='gth-szv',
              pseudo='gth-pbe', mesh=[21, 21, 21], precision=1e-9)
cell_a = gto.M(atom='H 2 3 4; H 3.4 3.1 4', **common)
cell_b = gto.M(atom='He 5.6 4.5 4.8', **common)
mf_a = dft.RKS(cell_a, xc='PBE')
mf_b = dft.RKS(cell_b, xc='PBE')

#2 embedding B (He)
mf_b.kernel()
if not mf_b.converged:
    raise RuntimeError('Frozen B SCF did not converge')

#3 embedded A (H2 in He)
mf_ainb = ksced.embed(mf_a, mf_b, basis_mode='M')
mf_ainb.t_nad = 'LDA_K_TF'
mf_ainb.kernel()

#4 get embedded energy and gradient on A
energy_ainb = mf_ainb.energy_potential()
gradient_ainb_on_a = mf_ainb.nuc_grad_method().kernel()

#5 report energy, gradient, and forces on A
print('Gradient on A (Hartree/Bohr):\n', gradient_ainb_on_a)
print('Forces on A (Hartree/Bohr):\n', -gradient_ainb_on_a)
