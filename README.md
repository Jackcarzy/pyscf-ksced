# pyscf-ksced

Frozen density embedding for [PySCF](https://pyscf.org).

`pyscf-ksced` solves the Kohn-Sham equations for one subsystem in the frozen
electron density of another.

## Install

Requires Python 3.9+ and PySCF 2.5+.

```bash
pip install -e .
```

Alternatively, use the package without installing it:

```bash
export PYSCF_EXT_PATH=/path/to/pyscf-ksced
```

GPU examples also require GPU4PySCF, CuPy, and a CUDA device.

## Usage

This example embeds H2O in the frozen density of Li+ using a shared basis.
`mf_a` denotes the plain subsystem-A SCF object; `mf_ainb` denotes the KSCED
calculation of A embedded in frozen B.

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
mf_a = dft.RKS(mol_a, xc="PBE")
mf_ainb = ksced.embed(
    mf_a, mf_b, mol_ab=mol_ab, basis_mode="S"
)
mf_ainb.kernel()

print(mf_ainb.e_tot)   # energy of A embedded in B
print(mf_ainb.e_tnad)  # non-additive kinetic energy

eint = mf_ainb.e_tot - mf_a.e_tot
print('interaction energy          %.10f Ha  = %.3f kcal/mol'      % (eint, eint * 627.503))
```

### Basis modes

- `basis_mode="S"` uses a shared supermolecular basis with ghost atoms.
- `basis_mode="M"` uses each subsystem's own basis, reducing the embedded SCF
  dimension.

For repeated periodic calculations where only subsystem A's coordinates move,
reuse the frozen environment:

```python
from pyscf.pbc import dft as pbc_dft

env = ksced.frozen_env(mf_b, cell_a)
mf_a = pbc_dft.RKS(cell_a, xc="PBE")
mf_ainb = ksced.embed(mf_a, mf_b, env=env)
mf_ainb.kernel()
```

### k-points

Periodic subsystems can use k-points with restricted Kohn-Sham (`KRKS`), in
both basis modes and on CPU or GPU. A and B must use the same k-points.

```python
from pyscf.pbc import dft as pbc_dft

kpts = cell_b.make_kpts([2, 2, 1])
mf_b = pbc_dft.KRKS(cell_b, kpts=kpts, xc="PBE").run()
mf_ainb = ksced.embed(pbc_dft.KRKS(cell_a, kpts=kpts, xc="PBE"), mf_b,
                      basis_mode="M")
mf_ainb.kernel()
```

Every trace in the energy below becomes a k-point average,
(1/N<sub>k</sub>) Σ<sub>k</sub> Tr[P<sub>k</sub> O<sub>k</sub>], as in PySCF's `KRKS`.

| | Γ point (`RKS`, `UKS`) | k-points (`KRKS`) |
|---|---|---|
| `basis_mode="S"` | yes | yes |
| `basis_mode="M"` | yes | yes |
| Unrestricted (`UKS` / `KUKS`) | yes | no |
| Analytic gradients (`basis_mode="M"`) | yes | no |

### Embedded energy

`mf_ainb.e_tot` is the energy of A embedded in frozen B. In
`basis_mode="M"`, it contains:

<pre>
E<sub>ainb</sub> = Tr[P<sup>A</sup> T<sup>A</sup>] + Tr[P<sup>A</sup> (V<sup>A</sup> + V<sup>B</sup>)] + Tr[P<sup>B</sup> V<sup>A</sup>]
       + ½ J<sup>AA</sup> + J<sup>AB</sup>
       + E<sub>xc</sub>[ρ<sub>A</sub> + ρ<sub>B</sub>] − E<sub>xc</sub>[ρ<sub>B</sub>]
       + T[ρ<sub>A</sub> + ρ<sub>B</sub>] − T[ρ<sub>A</sub>] − T[ρ<sub>B</sub>]
       + E<sub>nn</sub><sup>AB</sup> − E<sub>nn</sub><sup>B</sup>
</pre>

P<sup>A</sup> and P<sup>B</sup> are the subsystem density matrices. T<sup>A</sup> is the AO kinetic
operator, V<sup>A</sup> and V<sup>B</sup> are the nuclear or pseudopotential operators, and
J<sup>AA</sup> and J<sup>AB</sup> are the Hartree self and cross interactions. Each trace uses
the basis of its density matrix. `T[ρ]` is the approximate kinetic functional
selected by `t_nad`, which defaults to `LDA_K_TF`. B's self energy is omitted
because it stays constant as A moves.

```python
energy_ainb = mf_ainb.energy_potential()  # Hartree
```

At zero electronic temperature, this returns `mf_ainb.e_tot`. With smearing,
it returns the free energy `mf_ainb.e_tot − sigma * entropy` at fixed electron
number and fixed `sigma`. This is the energy differentiated by the gradient.

### Analytic gradients

For molecular and Gamma-point periodic `basis_mode="M"` calculations, the gradient differentiates
E_ainb with respect to A's nuclear positions, including the A–B interactions.
B's atoms, basis and density stay fixed.

```python
mf_ainb.kernel()
energy_ainb = mf_ainb.energy_potential()

gradient_ainb_on_a = mf_ainb.nuc_grad_method().kernel()  # Hartree/Bohr
forces_ainb_on_a = -gradient_ainb_on_a
```

To select atoms, inspect individual contributions, or reuse the calculation
at a new A geometry:

```python
grad_ainb = mf_ainb.nuc_grad_method()
forces_selected = -grad_ainb.kernel(atmlst=[0, 2])  # A-local atom indices
components = grad_ainb.components                  # full A gradient by term

scanner_ainb = mf_ainb.nuc_grad_method().as_scanner()
energy_ainb, gradient_ainb_on_a = scanner_ainb(displaced_mol_a)  # Mole or Cell
```

## Package layout

- `pyscf/ksced/ksced.py`: `embed()`, `frozen_env()` and class dispatch
- `pyscf/ksced/common.py`, `arrays.py`: code shared by both basis modes
- `pyscf/ksced/super/`: supermolecular basis (`basis_mode="S"`)
- `pyscf/ksced/mono/`: monomolecular basis (`basis_mode="M"`), including gradients

## Examples

- `examples/00_mol_Super_CPU`: molecular, S, CPU
- `examples/01_mol_Super_GPU`: molecular, S, GPU
- `examples/02_pbc_Super_GPU`: periodic, S, GPU
- `examples/03_pbc_Mono_GPU`: periodic, M, GPU
- `examples/04_mol_Mono_Grad`: molecular gradients, M, CPU
- `examples/05_pbc_Mono_Grad`: periodic gradients, M, GPU

## License

[Apache-2.0](LICENSE)
