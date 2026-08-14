# pyscf-ksced

KSCED (Kohn-Sham Equations with Constrained Electron Density) subsystem embedding
for [PySCF](https://pyscf.org), for molecules and for periodic systems at the
gamma point.

Subsystem A is optimised in the frozen density of subsystem B. A and B must share
one supermolecular AO basis, built with ghost atoms.

## Install

```bash
pip install -e .
```

Or, without installing:

```bash
export PYSCF_EXT_PATH=/path/to/pyscf-ksced
```

## Use

```python
from pyscf import gto, dft, ksced

mol_a = gto.M(atom='ghost-Li 0 0 0; O 0 0 1.5; H 0 0.76 2.09; H 0 -0.76 2.09',
              basis='6-31g')
mol_b = gto.M(atom='Li 0 0 0; ghost-O 0 0 1.5; ghost-H 0 0.76 2.09; ghost-H 0 -0.76 2.09',
              basis='6-31g', charge=1)
mol_ab = gto.M(atom='Li 0 0 0; O 0 0 1.5; H 0 0.76 2.09; H 0 -0.76 2.09',
               basis='6-31g', charge=1)

mf_b = dft.RKS(mol_b, xc='PBE').run()

mf_a = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b, mol_ab=mol_ab)
mf_a.kernel()

print(mf_a.e_tot, mf_a.e_tnad)
```

The periodic call is identical with `pyscf.pbc.dft.RKS` and a `Cell`.

## Scope

Restricted KS, closed shell, gamma point. The non-additive kinetic functional
defaults to `LDA_K_TF` and is settable through `mf.t_nad`.
