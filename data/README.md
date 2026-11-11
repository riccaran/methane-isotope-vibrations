# Numerical inputs and units

The two small JSON files are the bridge between electronic structure and the
animation. Neither contains a private application, machine-specific paths,
credentials, paper PDFs or historical drafts.

## `reference_calculation.json`

This is a sanitized electronic calculation behind the approved film, not a
replacement for the from-scratch generator. It lets the mass-weighting and
subspace analysis be repeated in seconds without another DFT calculation.

| Field | Meaning |
|---|---|
| `coordinates_angstrom` | Five Cartesian positions, ordered C,H,H,H,H |
| `energy_hartree` | Gas-phase electronic energy |
| `gradient_hartree_per_bohr` | Full 15-component Cartesian gradient |
| `hessian_hartree_per_bohr2` | Unsymmetrized 15 × 15 finite-difference electronic Hessian |
| `settings` | Functional, basis, grid response, grid level and displacement |
| `engine_version` | PySCF version used to obtain the reference calculation |

The method is B3LYP/def2-TZVP, grid level 6, no density fitting or dispersion,
SCF energy tolerance 10^-12 Hartree and orbital-gradient tolerance 10^-10.
The Cartesian Hessian uses central differences of grid-response gradients with
h = 0.002 bohr. The reference geometry is stationary within 10^-6 Hartree/bohr.

The independently runnable `methane.chemistry` stage constructs an ideal
tetrahedron, optimizes its bond length, checks every Cartesian force and
electronic stability, and computes full Hessians at h and h/2. Its raw JSON is
a fuller record with settings, timing, stability flags and sensitivity data.

## `methane_modes.json`

This frozen dataset preserves the exact mode basis used in the approved video.
It is also the schema written by `python -m methane.modes` for a fresh run.

| Field | Meaning |
|---|---|
| `schema_version` | Currently 1 |
| `method` | Electronic-structure provenance and harmonic model |
| `isotopologues` | `CH4` and `CH3D`, sharing equilibrium geometry |
| `coordinates_angstrom` | A 5 × 3 equilibrium geometry within each isotope record |
| `masses_amu` | Five isotope masses in unified atomic mass units |
| `frequencies_cm1` | All nine positive harmonic vibrational wavenumbers |
| `cartesian_modes` | Nine 5 × 3 displacement vectors in ascending eigenvalue order |
| `mode_normalization` | Explicit normalization convention for the direction-only input vectors |
| `comparisons` | The selected stretch and lower E-bend comparisons |
| `cartesian_mode_unit_norm` | Unit Euclidean norm for each selected 5 × 3 vector |
| `shift_percent` | 100 × [1 - frequency(CH3D)/frequency(CH4)] |
| `selected_orientation` | Proper 3 × 3 camera rotation |
| `quality` | Minimum-curvature and subspace-tracking information |
| `display_conventions` | Atom index, mode index and degeneracy convention |

The carbon mass is 12 u; H is 1.007825032 u and D is 2.01410177812 u.
Atom **1**, using zero-based indices, changes mass. Its nuclear charge is
unchanged, so one electronic potential and Hessian serve both species.

## Mode extraction and the degenerate basis

Six mass-weighted rigid-motion directions are removed before diagonalizing the
9-dimensional vibrational subspace. Eigenfrequencies follow

`wavenumber = sqrt(eigenvalue) * sqrt(Eh / (a0² * u)) / (2πc)`.

The extraction routine produces orthonormal mass-weighted eigenvectors and
Cartesian vectors `x = M^(-1/2) q`. The frozen animation inputs additionally
normalize each Cartesian vector to unit Euclidean norm, matching the accepted
film's original convention; fresh extraction retains the mass normalization.
The per-isotope `mode_normalization` field makes that distinction explicit.
Both supply the same displacement directions: the renderer independently
rescales each mode by its largest atomic-vector norm, so a scalar change in
input normalization cancels exactly. Neither convention specifies a thermal
amplitude; display amplitudes are assigned separately.

Frequency clusters within 0.01 cm^-1 are split into C3v A1/E subspaces. A
deterministic initial basis is followed through 401 mass steps using Hungarian
subspace matching and SVD parallel transport. Fractional masses are mathematical
tracking points, not claimed physical isotopologues. The E doublet's displayed
component uses the cluster mean frequency; original numerical eigenvalues are
retained in `frequencies_cm1`.

Near-degenerate eigenvector directions and signs can change with orientation,
platform and numerical precision. A fresh end-to-end dataset reproduces the
scientific frequency comparison, while the frozen dataset preserves the exact
accepted graphical basis. These are explicitly separate reproducibility goals.

## Reference comparisons

| Selected coordinate | CH4 / cm^-1 | CH3D / cm^-1 | Fewer cycles in CH3D |
|---|---:|---:|---:|
| Symmetric-stretch branch | 3032.1384163 | 2273.2854142 | 25.0269908% |
| Lower E-bend component | 1340.9286711 | 1183.2824173 | 11.7564981% |

The film rounds these to **25.03%** and **11.76%**. They concern the two selected
coordinates, not every methane vibration, a reaction rate or an observed
astronomical line shift.
