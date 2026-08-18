# pyscf-ksced

`pyscf-ksced` adds frozen-density subsystem embedding to
[PySCF](https://pyscf.org). It solves the Kohn-Sham equations for subsystem A
in the frozen electron density of subsystem B.

The package supports molecular and gamma-point periodic calculations with
restricted or unrestricted Kohn-Sham methods.

## Requirements

| | Version |
|---|---|
| Python | 3.9 or newer |
| PySCF | 2.5 or newer, developed against 2.14.0 |
| NumPy | 1.13 or newer |
| GPU4PySCF | optional, for the GPU examples |
The GPU paths additionally need CuPy and a visible CUDA device.

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

You can also load the package without installing it:

```bash
export PYSCF_EXT_PATH=/path/to/pyscf-ksced
```

## Basis modes

`ksced.embed()` supports two basis representations:

- `basis_mode='S'`: subsystems A and B share a supermolecular AO basis built
  with ghost atoms. Their density matrices have the same dimensions.
- `basis_mode='M'`: each subsystem uses only its own monomolecular basis
  functions. This reduces the embedded SCF dimension. For periodic
  calculations, A and B must use the same lattice and mesh. Only gamma-point
  calculations are supported.

## Quick start

This example embeds H2O in the frozen density of Li+ using a shared basis:

```python
from pyscf import dft, gto, ksced

mol_a = gto.M(
    verbose = 4,
    atom = '''
        o    0    0.       0.
        h    0    -0.757   0.587
        h    0    0.757    0.587
        x-li   0    0        -2''',
    basis = '6-31g')

mol_b = gto.M(
    verbose = 4,
    atom = '''
        x-o    0    0.       0.
        x-h    0    -0.757   0.587
        x-h    0    0.757    0.587
          li   0    0        -2''',
    charge = 1,
    basis = '6-31g')

mol_ab = gto.M(
    verbose = 4,
    atom = '''
        o    0    0.       0.
        h    0    -0.757   0.587
        h    0    0.757    0.587
        li   0    0        -2''',
    charge = 1,
    basis = '6-31g')

mf_b = dft.RKS(mol_b, xc='PBE').run()
mf_a = ksced.embed(dft.RKS(mol_a, xc='PBE'), mf_b, mol_ab=mol_ab)
mf_a.kernel()

print(mf_a.e_tot)  
# `mf_a.e_tot` contains the embedded total energy and 
# `mf_a.e_tnad` contains the non-additive kinetic energy.
```

## Main options and examples

- `examples/00_mol_Super_CPU`: H2O in Li+
- `examples/01_mol_Super_GPU`: CH3SH+Au10 in Au10
- `examples/02_pbc_Super_GPU`: NH3+Au10 in Au110
- `examples/03_pbc_Mono_GPU`: NH3+Au10 in Au110

## Monkey-patch

Same physics as the plugin, but nothing in PySCF is modified. Could be used right away.

## License

Apache-2.0. See [LICENSE](LICENSE).
