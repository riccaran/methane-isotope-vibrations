# Computational analysis and visualization of vibrational isotope effects in CH₄ and CH₃D

**One extra neutron changes how a molecule moves.** This project makes that
isotope effect visible by comparing ordinary methane, **CH₄**, with **CH₃D**,
where one hydrogen is replaced by deuterium.

The workflow uses PySCF to optimize the methane geometry and calculate its molecular Hessian.
Harmonic analysis then determines and compares all nine vibrational modes of CH₄ and CH₃D.
Manim visualizes the resulting molecular motion alongside synchronized oscillation plots.

The astronomical motivation is the unusually deuterium-rich methane reported
in interstellar comet **3I/ATLAS** by [Roth and colleagues](https://arxiv.org/abs/2603.20445v3).
The animation is a computational illustration of the molecular isotope effect ([full movie](media/methane-isotope-effect.mp4)).

This is an isolated-molecule harmonic model. It does not fit the comet's spectrum,
assign the displayed vibrations to observed JWST lines, or infer the comet's origin.

## Results

The nine harmonic modes form four frequency levels in CH₄ and six in CH₃D.
The table follows their corresponding branches under isotope substitution;
E and T₂ denote twofold and threefold degeneracy. Degenerate-level means are
used before rounding. These are **calculated harmonic B3LYP/def2-TZVP frequencies,
not experimental values**. The film compares only the two rows marked **Yes**.

| CH₄ level | CH₄ / cm⁻¹ | CH₃D level | CH₃D / cm⁻¹ | Frequency reduction | In the film |
|---|---:|---|---:|---:|:---:|
| ν₄ (T₂) | 1340.93 | ν₃ (A₁) | 1335.39 | 0.41% | — |
| ν₄ (T₂) | 1340.93 | ν₆ (E) | 1183.28 | **11.76%** | **Yes** |
| ν₂ (E) | 1556.45 | ν₅ (E) | 1495.89 | 3.89% | — |
| ν₁ (A₁) | 3032.14 | ν₂ (A₁), C–D stretch | 2273.29 | **25.03%** | **Yes** |
| ν₃ (T₂) | 3134.71 | ν₁ (A₁) | 3061.69 | 2.33% | — |
| ν₃ (T₂) | 3134.71 | ν₄ (E) | 3134.44 | 0.01% | — |

### 1. Isotopic substitution on a common potential-energy surface

Deuterium has the same nuclear charge as hydrogen (¹H) but roughly twice its mass.
This greater inertia shifts methane’s vibrational frequencies and changes how its
atoms move together, giving CH₃D a distinct infrared fingerprint.

In the Born–Oppenheimer harmonic model used here, the two isotopologues share
the same equilibrium geometry and electronic force-constant matrix:

```math
\begin{aligned}
\mathbf H_{\mathrm{CH_4}} &= \mathbf H_{\mathrm{CH_3D}}, \\
\mathbf M_{\mathrm{CH_4}} &\neq \mathbf M_{\mathrm{CH_3D}}.
\end{aligned}
```

Here $\mathbf H$ is the Hessian of the potential energy with respect to Cartesian
nuclear coordinates, and $\mathbf M$ is the diagonal atomic-mass matrix. The
isotope effect enters through the masses on this common potential-energy surface.

![Tracked isotope and nuclear magnifiers](media/01-isotope-nuclei.gif)

### 2. Normal-mode analysis of stretching and bending vibrations

Stretching vibrations change bond lengths, while bending vibrations change bond angles.
Each mode involves a characteristic pattern of coordinated atomic motion, with some atoms
moving more than others. Replacing H with the heavier D therefore shifts each frequency
by a different amount, depending on how strongly the substituted atom participates in
that vibration.

The harmonic frequencies and atomic displacement patterns follow the generalized
eigenvalue equation

```math
\mathbf H\,\mathbf u_k = (2\pi\nu_k)^2\,\mathbf M\,\mathbf u_k,
```

where $\nu_k$ is the frequency in Hz and $\mathbf u_k$ is the Cartesian
displacement vector of mode $k$. Removing three translations and three rotations
leaves nine vibrational modes for this nonlinear five-atom molecule. The
calculation solves the equivalent eigenproblem for the mass-weighted Hessian.

![Bending and stretching guides on the same superposed motion](media/02-component-guides.gif)

### 3. Isotope effects on vibrational frequencies and periods

Over the same time interval, CH₃D completes fewer oscillations in both selected modes.
The calculated frequency reductions of about 25% for stretching and 12% for bending
correspond to oscillation periods approximately 33% and 13% longer than in CH₄.
The additional nuclear mass therefore changes the timescale of each vibration by a
different amount.

![Simultaneous stretching and bending comparison](media/03-two-mode-comparison.gif)

| Selected comparison | CH₄ / cm⁻¹ | CH₃D / cm⁻¹ | Frequency reduction |
|---|---:|---:|---:|
| Symmetric-stretch branch | 3032.14 | 2273.29 | **25.03%** |
| Lower E-bend component | 1340.93 | 1183.28 | **11.76%** |

For each pair of corresponding vibrational modes, the relative frequency reduction is

```math
\delta_{\nu} = \frac{\nu_{\mathrm{CH_4}} - \nu_{\mathrm{CH_3D}}}{\nu_{\mathrm{CH_4}}} \times 100\%
```

where $\nu_{\mathrm{CH_4}}$ and $\nu_{\mathrm{CH_3D}}$ are the calculated harmonic
frequencies of the two isotopologues.

Frequency and oscillation period are inversely related:

```math
\begin{aligned}
T_k &= \frac{1}{\nu_k} = \frac{1}{c\,\widetilde{\nu}_k}, \\
\frac{T_{k,\mathrm{CH_3D}}}{T_{k,\mathrm{CH_4}}}
&= \frac{\nu_{k,\mathrm{CH_4}}}{\nu_{k,\mathrm{CH_3D}}}.
\end{aligned}
```

Here $T_k$ is the period in seconds, $\widetilde{\nu}_k$ is the wavenumber in
cm⁻¹, and $`c = 2.99792458\times10^{10}\ \mathrm{cm\,s^{-1}}`$ is the speed of
light. The calculated periods of the two selected modes are:

| Selected comparison | CH₄ period / fs | CH₃D period / fs |
|---|---:|---:|
| Symmetric-stretch branch | **11.00** | **14.67** |
| Lower E-bend component | **24.88** | **28.19** |

### 4. Harmonic oscillations and relative phase evolution

In the harmonic approximation, each normal mode oscillates sinusoidally at a fixed
frequency. The frequency differences between CH₄ and CH₃D cause their relative phase
to evolve over time. The constant amplitudes reflect an idealized model without damping.
In real systems, collisions and coupling between modes allow vibrational energy to be
exchanged and redistributed.

For either isotopologue, the harmonic normal coordinate can be written as

```math
Q_k(t) = A_k\sin(2\pi\nu_k t + \phi_k),
```

where $A_k$ is the amplitude and $\phi_k$ the initial phase. For corresponding
modes, the phase of CH₃D relative to CH₄ evolves according to

```math
\begin{aligned}
\Delta\phi_k(t) &= \Delta\phi_k(0) \\
&\quad + 2\pi\bigl(\nu_{k,\mathrm{CH_3D}} - \nu_{k,\mathrm{CH_4}}\bigr)t.
\end{aligned}
```

In these expressions, $t$ is physical time in seconds and phase is measured in
radians. The animation maps playback time onto this physical clock. The display
amplitudes and initial phases are illustrative choices.

![Enlarged continuous waveforms](media/04-enlarged-waves.gif)

The initial GIF uses 30 fps to sample the fast displayed vibrations; the three
already slowed chapters use 12 fps. The complete MP4 retains all 60 fps.

## Reproducibility and validation

The from-scratch workflow generates its own geometry, Hessians and mode dataset.
Fresh calculations can differ in the last numerical digits and in the chosen
basis within degenerate mode spaces. Fonts and encoder versions can also change
pixels, so generated media need not be byte-identical to the included film.

### Installation

Use **Python 3.12** and install the native dependencies described in the
[Manim installation guide](https://docs.manim.community/en/stable/installation.html).
An **FFmpeg** executable must be available on `PATH`, or supplied with
`--ffmpeg /path/to/ffmpeg`. The build must support `libx264` and the `duration`
option of the `setts` bitstream filter; the runner tests these capabilities
before starting calculations or rendering. LaTeX is not used.

For Linux/macOS, after installing those prerequisites:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Also install the quantum requirements for this workflow:

```bash
python -m pip install -r requirements-qm.txt
```

The requirements pin the versions used for validation. The numerical pipeline
uses PySCF 2.14.0, NumPy 2.5.4 and SciPy 1.18.1; the renderer uses Manim 0.22.1.
No private application or account is required to run the repository.

The preferred reference fonts are Segoe UI and Consolas. On systems without
them, the renderer selects **DejaVu Sans** and **DejaVu Sans Mono**; install
these open fonts if your environment does not already provide them. Font
substitution affects typography, not the computed frequencies or phase.

### Run the entire pipeline from zero

From the repository root, in an environment containing both requirement sets:

```bash
python tools/render.py --from-scratch --previews
```

The command performs every dependent stage in order:

1. Constructs C,H,H,H,H as an ideal tetrahedron.
2. Optimizes its common C–H bond length with converged B3LYP/def2-TZVP calculations.
3. Checks all Cartesian forces and internal/external electronic stability.
4. Computes the full Cartesian Hessian from analytic gradients at ±0.002 bohr,
   then repeats it at half that displacement.
5. Changes the mass of atom 1 from H to D on the same electronic potential.
6. Projects out translations/rotations and extracts all nine physical modes.
7. Tracks matching A1/E subspaces and selects the two displayed comparisons.
8. Renders the full film, normalizes timestamps without reencoding its images,
   decodes and verifies the movie, then generates the four GIF chapters.

The pipeline renders at **60 fps**, as required by the scene's frame-aligned
endpoint.

Outputs are kept under `build/run/`:

```text
calculation.json          Raw electronic structure and Hessians
calculation.log           Latest PySCF electronic evaluation
calculation.progress.json Latest completed calculation stage
methane_modes.json        Extracted modes and selected comparisons
render_metadata.json      Model, phase and geometry at every video frame
methane-isotope-effect.mp4
verification.json         Encoded-film verification
scene_inputs.json         Fingerprints for safe reuse of a completed native scene
previews/                 Four chapter GIFs
```

Console output retains the completed-work progress history. Existing completed
calculation/video files are protected; choose a new directory for a fresh run:

```bash
python tools/render.py --from-scratch --work-dir build/another-run --previews
```

After an interrupted run, repeat the same command with `--resume`. A completed
quantum calculation is checked and re-extracted; a completed native scene is
reused only when its input, source fingerprints and output hashes match.
Exported video stays under a `.pending.mp4` name until verification and any
requested previews succeed. The final MP4 is never overwritten.

The default quantum resource settings are **four CPU threads and 2048 MB** of
PySCF memory allowance, adjustable with `--threads` and `--memory-mb`. A full
calculation and render take minutes, rather than being an instant visualization.

#### Windows with a separate WSL quantum environment

The renderer may stay native on Windows while PySCF runs under Ubuntu/WSL.
Create the rendering environment from PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In an Ubuntu/WSL shell opened in the **same repository directory**, prepare the
quantum interpreter:

```bash
python3 -m venv .venv-qm
.venv-qm/bin/python -m pip install -r requirements-qm.txt
```

Back in PowerShell, one command orchestrates both environments:

```powershell
.\.venv\Scripts\python.exe tools/render.py --from-scratch --qm-wsl --qm-python .venv-qm/bin/python --previews
```

The runner sets WSL's working directory explicitly and translates local Windows
drive paths. It passes argument arrays directly, so spaces in repository paths
do not become shell-quoting problems. `--wsl-distribution` selects a distribution
other than the default `Ubuntu`.

### Validation

Run the fast numerical and rendering checks without repeating DFT:

```bash
python -m unittest discover -s tests -v
```

The suite checks isotope frequencies and mode extraction, geometry, physical
timing, waveforms and rendering recovery. The full pipeline also verifies the
encoded video automatically.

An independent calculation reproduced the two displayed reductions to **25.03%**
and **11.76%**; halving the Hessian displacement changed frequencies by at most
about **0.003 cm⁻¹**. The [numerical audit](validation/numerical_audit.json) and
[approved-film verification](validation/approved_video/verification.json) retain
the supporting results; their scope is explained in [data/README.md](data/README.md).

## Numerical method and visual conventions

**Electronic structure.** The model uses isolated, closed-shell methane in the
gas phase: B3LYP/def2-TZVP, grid level 6, grid-response gradients, no density
fitting or added dispersion. SCF energy/orbital-gradient tolerances are
10⁻¹²/10⁻¹⁰, and every Cartesian force must be below 10⁻⁶ Hartree/bohr at the
optimized geometry. A radial optimum alone is insufficient: stability and all
nine physical curvatures are also checked.

PySCF 2.14.0's default `B3LYP` uses the Gaussian-compatible convention with
VWN-RPA (`VWN3`) local correlation, as specified in the
[versioned source](https://github.com/pyscf/pyscf/blob/v2.14.0/pyscf/dft/xcfun.py#L197-L224).

**Isotopes.** Both species have the same nuclear charges, electronic Hessian
and equilibrium geometry in this Born–Oppenheimer harmonic model. Only one
nuclear mass changes. Six rigid-motion coordinates are removed in mass-weighted
space before diagonalization, leaving `3N − 6 = 9` vibrations.

**Degenerate modes.** Nearly equal frequencies are clustered within 0.01 cm⁻¹.
The common C3v subgroup separates A1 and E components. Hungarian matching and
SVD parallel transport follow corresponding subspaces through 401 mass steps.
A selected E component uses its cluster mean frequency. Full schema, units and
normalization conventions are documented in [data/README.md](data/README.md).

**Display amplitudes.** Mode vectors are rescaled by their largest atomic-vector
norm. The field superposes all nine with illustrative 0.055 Å coefficients;
the selected pair uses 0.15 Å coefficients before splitting and 0.30 Å per
isolated branch afterward. These enlarged amplitudes and phases are not a
temperature, thermal population or quantum nuclear trajectory.

**Field motion.** Opening translations, rotations and periodic wrapping are
graphical context. They are not solvent molecular dynamics, diffusion or
collisions. The selected object is tracked through the optical zoom and locks
its rigid-body motion before the component guides appear.

**Three clocks.** Actual video seconds map smoothly to model time. Model time
integrates a shared phase coordinate `q`; physical femtoseconds follow
`Δt = q_difference / (2 c ν_reference)`. Each normal coordinate is
`sin(π × frequency_ratio × q)`. Molecular displacements, waveforms and the
timer use this shared phase. The final plots show the same eight-q-unit
physical-time window for both isotopes.

The extra comparison time uses a quintic map with unit speed and zero mapping
acceleration at its joins. It extends the completed split comparison from about
8.17 to 10.17 seconds while preserving the **650 fs** plot-zoom trigger and the
continuous final phase.

## File map

| File | Responsibility |
|---|---|
| `methane/chemistry.py` | Geometry creation, quantum optimization, stability and two Hessians |
| `methane/modes.py` | Mass weighting, rigid-motion projection, isotope/subspace tracking and JSON extraction |
| `methane/model.py` | Validated data loading, harmonic coordinates, projection, tracked geometry and clocks |
| `methane/scene.py` | Documented Manim classes, graphical updaters and per-frame metadata |
| `tools/render.py` | Complete E2E orchestration, optional Windows/WSL bridge and lossless stream-copy export |
| `tools/previews.py` | Four GIF chapters decoded from the film |
| `tools/verify_render.py` | Independent clock and encoded-wave verification |
| `tests/test_pipeline.py` | Fast numerical, geometric and documentation invariants |
| `data/` | Small reference electronic result, accepted mode dataset and schema documentation |
| `media/` | Approved full-resolution MP4 and four inline GIF excerpts |
| `requirements*.txt`, `manim.cfg` | Reproducible dependency sets and Cairo/encoder settings |

Generated work stays in ignored `build/` directories. There are no paper PDFs,
private reports, intermediate renders, vendored executables or application code.

## References

- [Roth et al., *An Enriched Methane D/H Ratio in the Interstellar Object 3I/ATLAS* — free author version](https://arxiv.org/abs/2603.20445v3)
- [Associated Nature Astronomy article](https://doi.org/10.1038/s41550-026-03002-5)
- [NASA: 3I/ATLAS facts and FAQs](https://science.nasa.gov/solar-system/comets/3i-atlas/3i-atlas-facts-and-faqs/)
- [PySCF: density-functional theory](https://pyscf.org/user/dft.html)
- [PySCF: harmonic vibrational analysis](https://pyscf.org/_modules/pyscf/hessian/thermo.html#harmonic_analysis)
- [Manim Community documentation](https://docs.manim.community/en/stable/)
