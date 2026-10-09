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

### Embedded energy

`mf_ainb.e_tot` is the energy of A embedded in frozen B. In
`basis_mode="M"`, it contains:

```text
E_ainb = Tr[D_A T_A] + Tr[D_A (V_A + V_B)] + Tr[D_B V_A]
       + ½ J_AA + J_AB
       + E_xc[ρ_A + ρ_B] − E_xc[ρ_B]
       + T[ρ_A + ρ_B] − T[ρ_A] − T[ρ_B]
       + E_nn[A+B] − E_nn[B]
```

`D_A` and `D_B` are the subsystem density matrices. `T_A` is the AO kinetic
operator, `V_A` and `V_B` are the nuclear or pseudopotential operators, and
`J_AA` and `J_AB` are the Hartree self and cross interactions. Each trace uses
the basis of its density matrix. `T[ρ]` is the approximate kinetic functional
selected by `t_nad`, which defaults to `LDA_K_TF`. B's self energy is omitted
because it stays constant as A moves.

```python
energy_ainb = mf_ainb.energy_potential()  # Hartree
```

At zero electronic temperature, this returns `mf_ainb.e_tot`. With smearing,
it returns the free energy `mf_ainb.e_tot − sigma * entropy` at fixed electron
number and fixed `sigma`. This is the energy differentiated by the gradient.
Fixed-chemical-potential calculations (`mu0`) are not supported by the gradient.

The energy includes PySCF's molecular or periodic A/B electrostatics. MM interactions are
not included. A future QM/MM coupling must add A–MM and frozen B–MM interactions
once, using the ionic charges from `cell.atom_charges()` for pseudopotential
nuclei. PySCF owns A/B electrostatics, the MM engine owns MM–MM terms, and the
coupling layer owns the cross terms and their forces.

### Analytic gradients

For molecular and Gamma-point periodic `basis_mode="M"` calculations, the gradient differentiates
E_ainb with respect to A's nuclear positions, including the A–B interactions.
B's atoms, basis and density stay fixed. CPU PySCF and GPU4PySCF are supported
with PySCF 2.14.

```python
mf_ainb.conv_tol = 1e-11
mf_ainb.kernel()

grad_ainb = mf_ainb.nuc_grad_method()
gradient_ainb_on_a = grad_ainb.kernel()  # Hartree/Bohr
forces_ainb_on_a = -gradient_ainb_on_a
```

For a Cartesian coordinate R_I of an atom in A, the periodic working equation is:

```text
g_I = Tr[D_A ∂_I(T_A + V_A + V_B)]
    + Tr[D_B ∂_I V_A] − Tr[W_A ∂_I S_A]
    + ∂_I E_nn[A+B]
    + Σ_g w_g { (v_J,A + v_J,B) ∂_I ρ_A
              + Σ_s (v_xc,s[ρ_A + ρ_B]
                     + v_T,s[ρ_A + ρ_B] − v_T,s[ρ_A]) · ∂_I q_A,s }_g

q_A,s = (ρ_A,s, ∂xρ_A,s, ∂yρ_A,s, ∂zρ_A,s)
W_A,μν = Σ_s,p f_s,p ε_s,p C_s,μp C*_s,νp
F_I = −g_I
```

Here `∂_I` means differentiation with respect to R_I at fixed AO density
matrices. Operator derivatives include the moving A basis and nuclear or
pseudopotential centers. `S_A` is the overlap matrix, and `W_A` is the
energy-weighted density matrix formed from the embedded orbitals, energies
and occupations. The term `−Tr[W_A ∂_I S_A]` is the Pulay contribution.
`D_A`, `D_B` and `W_A` are spin summed; `s` labels the spin channels in the
grid terms.

`g` labels uniform grid points with weights `w_g`. `v_J` is the Hartree
potential. `v_xc,s` and `v_T,s` are derivatives of the XC and kinetic energy
densities with respect to `q_s`; their dot products include the density and
its three spatial derivatives. For LDA, only the density component contributes.
The density derivatives use first and second spatial derivatives of A's AOs.
The grid points and weights stay fixed as A moves, so their nuclear response
is zero. The frozen-B self terms also have zero derivative. With smearing,
the same equation gives the free-energy gradient.

For molecules, Coulomb derivatives use direct AO integrals. The AB Becke grid
moves with the nuclei, so both its coordinates and weights contribute:

```text
g_I = Tr[D_A ∂_I(T_A + V_A + V_B)]
    + Tr[D_B ∂_I V_A] − Tr[W_A ∂_I S_A]
    + ∂_I(½ J_AA + J_AB + E_nn[A+B])
    + Σ_g { w_g ∂_I e_nad(r_g) + e_nad(r_g) ∂_I w_g }

e_nad = e_xc[q_A + q_B] − e_xc[q_B]
      + t[q_A + q_B] − t[q_A] − t[q_B]
```

`e_xc` and `t` are energy densities per unit volume. Here `∂_I e_nad(r_g)`
includes the moving A basis and the motion of the grid point `r_g`. B's density
is fixed in space, but its sampled values change when grid points move. The
derivatives of the subtracted B functional terms are therefore retained at
finite quadrature resolution. Both backends include the full grid response.

To select atoms, inspect individual contributions, or reuse the calculation
at a new A geometry:

```python
forces_selected = -grad_ainb.kernel(atmlst=[0, 2])  # A-local atom indices
components = grad_ainb.components                  # full A gradient by term

scanner_ainb = mf_ainb.nuc_grad_method().as_scanner()
energy_ainb, gradient_ainb_on_a = scanner_ainb(displaced_mol_a)  # Mole or Cell
```

Both domains require a converged SCF, separate bases without ghosts, and pure
LDA/GGA XC and kinetic functionals. Restricted, unrestricted and mixed-spin
subsystems are supported. Forces on B and lattice stress are not implemented.

Molecular gradients require all-electron, nonrelativistic Hamiltonians, direct
Coulomb integrals and an unmodified AB atom-centered grid. Density fitting,
ECPs, pseudopotentials, coincident centers, X2C and dispersion are not supported.
The GPU molecular path evaluates functionals in the combined AB basis and
returns an A-sized Fock matrix. XC, grid response and Coulomb derivatives run
on the GPU; one-electron and overlap derivatives use CPU PySCF.

Periodic gradients require a fixed 3D lattice, FFTDF, an unmodified
uniform grid, separate bases without ghosts, and pure LDA/GGA XC and kinetic
functionals. The cell, Coulomb and XC meshes must agree. GTH pseudopotential
and all-electron cells are supported, but cannot be mixed in one cell.
Other fitting methods, atom-centered grids, non-Gamma points, hybrids,
meta-GGAs, ECPs and dispersion corrections are not supported.

For periodic calculations, each backend differentiates the core Hamiltonian used in its SCF. CPU FFTDF
and GPU native core energies can differ at finite grid resolution. The GPU
path uses GPU AO, XC and grid contractions, plus GPU4PySCF's native core
potential derivatives. Ewald, kinetic and overlap derivatives use CPU PySCF.

## Examples

- `examples/00_mol_Super_CPU`: molecular, shared basis, CPU
- `examples/01_mol_Super_GPU`: molecular, shared basis, GPU
- `examples/02_pbc_Super_GPU`: periodic, shared basis, GPU
- `examples/03_pbc_Mono_GPU`: periodic, separate bases, GPU
- `examples/04_mol_Mono_Grad`: molecular energy and gradients, separate bases, CPU or GPU

## License

[Apache-2.0](LICENSE)
