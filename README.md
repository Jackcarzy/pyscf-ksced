# pyscf-ksced

Frozen-density subsystem embedding for [PySCF](https://pyscf.org).

`pyscf-ksced` solves the Kohn-Sham equations for one subsystem in the frozen
electron density of another. It supports restricted and unrestricted molecular
calculations, plus gamma-point periodic calculations.

## Install

Requires Python 3.9+ and PySCF 2.5+.

```bash
pip install -e .
```

Alternatively, use the package without installing it:

```bash
export PYSCF_EXT_PATH=/path/to/pyscf-ksced
```

GPU examples also require GPU4PySCF, CuPy, and a CUDA device. PySCF multigrid
objects are not supported.

## Usage

This example embeds H2O in the frozen density of Li+ using a shared basis:

```python
from pyscf import dft, gto, ksced

mol_a = gto.M(
    atom="""
        O      0.000  0.000   0.000
        H      0.000 -0.757   0.587
        H      0.000  0.757   0.587
        ghost-Li 0.000 0.000 -2.000
    """,
    basis="6-31g",
)

mol_b = gto.M(
    atom="""
        ghost-O  0.000  0.000  0.000
        ghost-H  0.000 -0.757  0.587
        ghost-H  0.000  0.757  0.587
        Li       0.000  0.000 -2.000
    """,
    charge=1,
    basis="6-31g",
)

mol_ab = gto.M(
    atom="""
        O   0.000  0.000  0.000
        H   0.000 -0.757  0.587
        H   0.000  0.757  0.587
        Li  0.000  0.000 -2.000
    """,
    charge=1,
    basis="6-31g",
)

mf_b = dft.RKS(mol_b, xc="PBE").run()
mf_a = ksced.embed(
    dft.RKS(mol_a, xc="PBE"), mf_b, mol_ab=mol_ab, basis_mode="S"
)
mf_a.kernel()

print(mf_a.e_tot)   # embedded total energy
print(mf_a.e_tnad)  # non-additive kinetic energy
```

### Basis modes

- `basis_mode="S"` uses a shared supermolecular basis with ghost atoms.
- `basis_mode="M"` uses each subsystem's own basis, reducing the embedded SCF
  dimension.

For repeated periodic calculations where only subsystem A's coordinates move,
reuse the frozen environment:

## Examples

- `examples/00_mol_Super_CPU`: molecular, shared basis, CPU
- `examples/01_mol_Super_GPU`: molecular, shared basis, GPU
- `examples/02_pbc_Super_GPU`: periodic, shared basis, GPU
- `examples/03_pbc_Mono_GPU`: periodic, separate bases, GPU

## License

[Apache-2.0](LICENSE)
