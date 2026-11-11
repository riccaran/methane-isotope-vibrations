"""Extract all nine harmonic modes and track isotope-dependent subspaces.

This stage reads only the public raw electronic calculation JSON. It projects
out translations and rotations in mass-weighted coordinates, diagonalizes the
vibrational Hessian, and follows matching C3v subspaces while the selected atom's
mass changes continuously from H to D. Intermediate masses are a mathematical
continuation device, not additional chemical species or quantum calculations.

Nearly degenerate eigenvectors are not individually unique. Symmetry separation,
a deterministic initial basis and SVD parallel transport prevent arbitrary
rotations of an E doublet from masquerading as an isotope effect. The resulting
JSON is the only scientific input required by the independent animation stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import constants
from scipy.linalg import null_space
from scipy.optimize import linear_sum_assignment

H_MASS = 1.007825032
D_MASS = 2.01410177812
FREQUENCY_FACTOR = np.sqrt(
    constants.physical_constants["Hartree energy"][0]
    / (
        constants.physical_constants["Bohr radius"][0] ** 2
        * constants.physical_constants["atomic mass constant"][0]
    )
) / (2 * np.pi * constants.c * 100)


def harmonic_modes(coordinates: np.ndarray, hessian: np.ndarray, masses: np.ndarray):
    """Return wavenumbers, weighted eigenvectors and Cartesian vectors.

    Coordinates are (5,3) angstrom, Hessian (15,15) Hartree/bohr², and masses
    (5,) unified atomic mass units. Weighted eigenvectors are orthonormal rows
    of shape (9,15). Cartesian rows have shape (9,5,3) and follow x=M^-1/2 q.
    The origin used for rotational generators is the isotope-specific COM.
    """
    xyz = np.asarray(coordinates, float)
    mass = np.asarray(masses, float)
    h = np.asarray(hessian, float)
    if xyz.shape != (5, 3) or mass.shape != (5,) or h.shape != (15, 15):
        raise ValueError("Methane needs (5,3) coordinates, five masses and a (15,15) Hessian")
    if not all(np.all(np.isfinite(a)) for a in [xyz, mass, h]) or np.any(mass <= 0):
        raise ValueError("Coordinates/Hessian must be finite and masses positive")
    relative = xyz - np.average(xyz, axis=0, weights=mass)
    root = np.repeat(np.sqrt(mass), 3)
    translations = [np.tile(axis, (5, 1)).ravel() * root for axis in np.eye(3)]
    rotations = [
        (np.cross(np.tile(axis, (5, 1)), relative) * np.sqrt(mass)[:, None]).ravel()
        for axis in np.eye(3)
    ]
    # The null space has nine columns for a nonlinear five-atom molecule.
    rigid = np.column_stack(translations + rotations)
    vibrational = null_space(rigid.T)
    if vibrational.shape != (15, 9):
        raise ValueError("Rigid-motion rank is not six: geometry may be linear or degenerate")
    inverse = 1 / root
    dynamic = inverse[:, None] * (h + h.T) / 2 * inverse[None, :]
    values, columns = np.linalg.eigh(vibrational.T @ dynamic @ vibrational)
    if np.any(values <= 0):
        raise ValueError("Non-positive physical curvature: do not animate this as a stable minimum")
    frequencies = np.sqrt(values) * FREQUENCY_FACTOR
    weighted = (vibrational @ columns).T
    cartesian = (weighted / root).reshape(9, 5, 3)
    return frequencies, weighted, cartesian


def c3_projector(coordinates: np.ndarray) -> np.ndarray:
    """Build the totally symmetric projector around the substituted C-H axis.

    A 120-degree rotation fixes C and atom 1 and cyclically permutes the other
    three H atoms. In mass-weighted displacement space P=(I+C3+C3²)/3 separates
    A1 motion from E motion. A geometric mismatch is rejected explicitly.
    """
    xyz = np.asarray(coordinates, float)
    axis = xyz[1] - xyz[0]
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = 2 * np.pi / 3
    rotation = (
        np.cos(angle) * np.eye(3)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    transformed = (xyz - xyz[0]) @ rotation.T + xyz[0]
    distances = np.linalg.norm(transformed[:, None] - xyz[None, :], axis=2)
    permutation = np.argmin(distances, axis=1)
    if sorted(permutation) != list(range(5)) or np.max(distances[np.arange(5), permutation]) > 1e-4:
        raise ValueError("Geometry does not support the expected methane C3 symmetry")
    action = np.zeros((15, 15))
    for source, target in enumerate(permutation):
        action[3 * target : 3 * target + 3, 3 * source : 3 * source + 3] = rotation
    projector = (np.eye(15) + action + action @ action) / 3
    return (projector + projector.T) / 2


def deterministic_basis(rows: np.ndarray) -> np.ndarray:
    """Choose a repeatable orthonormal basis of a degenerate row subspace.

    Project Cartesian coordinate axes into the subspace, taking the largest
    remaining norm and removing already chosen directions. This depends on the
    subspace, not on arbitrary signs/rotations returned by an eigensolver.
    """
    projector = rows.T @ rows
    basis = []
    for _ in rows:
        candidates = projector.copy()
        for vector in basis:
            candidates -= np.outer(vector, vector @ candidates)
        column = int(np.argmax(np.linalg.norm(candidates, axis=0)))
        vector = candidates[:, column]
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            raise ValueError("Degenerate subspace unexpectedly lost rank")
        basis.append(vector / norm)
    return np.array(basis)


def symmetry_groups(
    frequencies: np.ndarray, weighted: np.ndarray, projector: np.ndarray
) -> list[dict]:
    """Split frequency clusters into one-dimensional A1 and two-dimensional E groups.

    Frequencies within 0.01 cm^-1 are treated as numerically degenerate. A mean
    cluster frequency is used only for rotating such a basis; the unaveraged
    nine eigenvalues are retained separately in the exported dataset.
    """
    clusters = []
    for i, frequency in enumerate(frequencies):
        if clusters and frequency - frequencies[clusters[-1][-1]] < 0.01:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    result = []
    for indices in clusters:
        rows = weighted[indices]
        values, coefficients = np.linalg.eigh(rows @ projector @ rows.T)
        for symmetry, mask in [("A1", values > 0.5), ("E", values <= 0.5)]:
            if not np.any(mask):
                continue
            basis = deterministic_basis(coefficients[:, mask].T @ rows)
            if len(basis) != (1 if symmetry == "A1" else 2):
                raise ValueError(
                    "Unexpected symmetry multiplicity; inspect the Hessian/degeneracy tolerance"
                )
            result.append(
                {
                    "symmetry": symmetry,
                    "rows": basis,
                    "frequency": float(np.mean(frequencies[indices])),
                    "source_indices": indices,
                }
            )
    if len(result) != 6 or sum(len(group["rows"]) for group in result) != 9:
        raise ValueError("Methane must have six C3v groups containing nine coordinates")
    return result


def view_orientation(coordinates: np.ndarray) -> np.ndarray:
    """Construct the same fixed camera orientation for either mass assignment.

    Two bonds define a molecular frame, which is aligned with a canonical
    tetrahedron and given the small display tilt used in the approved film.
    It is a camera convention, independent of quantum mode extraction.
    """
    xyz = np.asarray(coordinates, float)
    normal = xyz[1] - xyz[0]
    normal /= np.linalg.norm(normal)
    tangent = xyz[2] - xyz[0]
    tangent -= np.dot(tangent, normal) * normal
    tangent /= np.linalg.norm(tangent)
    target = np.ones(3) / np.sqrt(3)
    other = np.array([-1, 1, -1]) / np.sqrt(3)
    target_tangent = other - np.dot(other, target) * target
    target_tangent /= np.linalg.norm(target_tangent)
    base = (
        np.column_stack([target, target_tangent, np.cross(target, target_tangent)])
        @ np.column_stack([normal, tangent, np.cross(normal, tangent)]).T
    )
    x, y, z = 0.11, -0.14, 0.24
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx @ base


def track_isotopes(coordinates: np.ndarray, hessian: np.ndarray, steps: int = 401):
    """Follow C3v groups from H to D by subspace overlap and parallel transport.

    Hungarian assignment matches groups with equal symmetry and multiplicity.
    SVD then aligns each new basis with the previous one. The returned endpoint
    eigenvectors are true harmonic coordinates; intermediate fractional masses
    supply only a smooth identification of corresponding motions.
    """
    if steps < 101:
        raise ValueError("Use at least 101 mass-continuation steps")
    projector = c3_projector(coordinates)
    previous = None
    minimum = 1.0
    endpoints = {}
    for step, substituted_mass in enumerate(np.linspace(H_MASS, D_MASS, steps)):
        masses = np.array([12.0, substituted_mass, H_MASS, H_MASS, H_MASS])
        frequencies, weighted, cartesian = harmonic_modes(coordinates, hessian, masses)
        current = symmetry_groups(frequencies, weighted, projector)
        if previous is None:
            for group in current:
                frequency = group["frequency"]
                family = (
                    "low_bend"
                    if frequency < 1450
                    else (
                        "mid_bend"
                        if frequency < 2000
                        else "symmetric_stretch" if frequency < 3080 else "upper_stretch"
                    )
                )
                group["id"] = family + "_" + group["symmetry"]
                if group["id"] == "symmetric_stretch_A1":
                    # Fix the stretch sign by outward radial motion, not a stored vector.
                    displacement = (group["rows"][0] / np.repeat(np.sqrt(masses), 3)).reshape(5, 3)
                    bonds = coordinates[1:] - coordinates[0]
                    if np.sum((displacement[1:] - displacement[0]) * bonds) < 0:
                        group["rows"] *= -1
        else:
            score = np.full((6, 6), -1e6)
            for i, old in enumerate(previous):
                for j, new in enumerate(current):
                    if old["symmetry"] == new["symmetry"] and len(old["rows"]) == len(new["rows"]):
                        score[i, j] = np.linalg.norm(old["rows"] @ new["rows"].T) ** 2 / len(
                            old["rows"]
                        )
            indices, assignment = linear_sum_assignment(-score)
            ordered = []
            for i, j in zip(indices, assignment):
                old, new = previous[i], current[j]
                u, singular, vh = np.linalg.svd(old["rows"] @ new["rows"].T)
                minimum = min(minimum, float(singular.min()))
                new["rows"] = u @ vh @ new["rows"]
                new["id"] = old["id"]
                ordered.append(new)
            current = ordered
        if step in [0, steps - 1]:
            name = "CH4" if step == 0 else "CH3D"
            endpoints[name] = {
                "masses_amu": masses.tolist(),
                "coordinates_angstrom": coordinates.tolist(),
                "frequencies_cm1": frequencies.tolist(),
                "cartesian_modes": cartesian.tolist(),
                "mode_normalization": "Mass-weighted eigenvectors orthonormal; Cartesian x=M^-1/2 q",
                "groups": current,
            }
        previous = current
    if minimum < 0.99:
        raise ValueError("Subspace matching is not sufficiently continuous")
    return endpoints, minimum


def extract(input_path: Path, output_path: Path, steps: int = 401) -> dict:
    """Validate a raw calculation and write the animation's complete public dataset.

    All nine modes and isotope masses are retained. Only the symmetric-stretch
    branch and one component of the lower E-bend branch are selected for the
    film. Checked-in reference data and a newly computed dataset share this
    schema, so rendering needs no change of code or manual copy/paste.
    """
    raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    coordinates = np.array(raw["coordinates_angstrom"], float)
    hessian = np.array(raw["hessian_hartree_per_bohr2"], float)
    gradient = np.array(raw["gradient_hartree_per_bohr"], float)
    if np.max(abs(gradient)) > 1e-6 or np.max(abs(hessian - hessian.T)) > 1e-5:
        raise ValueError("Raw geometry/Hessian did not pass stationarity or reciprocity checks")
    endpoints, overlap = track_isotopes(coordinates, hessian, steps)
    comparisons = []
    for title, identity in [("Stretching", "symmetric_stretch_A1"), ("Bending", "low_bend_E")]:
        cases = {}
        for name, result in endpoints.items():
            group = next(group for group in result["groups"] if group["id"] == identity)
            vector = (group["rows"][0] / np.repeat(np.sqrt(result["masses_amu"]), 3)).reshape(5, 3)
            vector /= np.linalg.norm(vector)
            cases[name] = {
                "frequency_cm1": group["frequency"],
                "cartesian_mode_unit_norm": vector.tolist(),
            }
        comparisons.append(
            {
                "title": title,
                "group_id": identity,
                "shift_percent": 100
                * (1 - cases["CH3D"]["frequency_cm1"] / cases["CH4"]["frequency_cm1"]),
                "cases": cases,
            }
        )
    half = raw.get("hessian_half_step_hartree_per_bohr2")
    sensitivity = None
    if half is not None:
        errors = []
        for name, record in endpoints.items():
            refined = harmonic_modes(coordinates, np.array(half), np.array(record["masses_amu"]))[0]
            errors.append(float(np.max(abs(refined - np.array(record["frequencies_cm1"])))))
        sensitivity = max(errors)
        if sensitivity > 0.5:
            raise ValueError("Hessian half-step sensitivity exceeds 0.5 cm^-1")
    result = {
        "schema_version": 1,
        "method": {
            "engine": "PySCF",
            "engine_version": raw.get("engine_version"),
            "functional": "B3LYP",
            "basis": "def2-tzvp",
            "model": "isolated gas-phase harmonic Born-Oppenheimer",
        },
        "isotopologues": {
            name: {k: v for k, v in record.items() if k != "groups"}
            for name, record in endpoints.items()
        },
        "comparisons": comparisons,
        "selected_orientation": view_orientation(coordinates).tolist(),
        "quality": {
            "positive_physical_modes": True,
            "mass_continuation_steps": steps,
            "minimum_subspace_overlap": overlap,
            "maximum_half_step_frequency_change_cm1": sensitivity,
        },
        "display_conventions": {
            "substituted_atom_zero_based": 1,
            "stretch_raw_index_zero_based": 5,
            "degenerate_frequency_tolerance_cm1": 0.01,
            "degenerate_basis": "C3 A1/E separation and SVD parallel transport",
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "shifts_percent": [r["shift_percent"] for r in comparisons],
                "minimum_subspace_overlap": overlap,
                "maximum_half_step_frequency_change_cm1": sensitivity,
            }
        ),
        flush=True,
    )
    return result


def main() -> None:
    """Parse raw-result and output paths for the standalone extraction stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("build/calculation.json"))
    parser.add_argument("--output", type=Path, default=Path("build/methane_modes.json"))
    parser.add_argument("--steps", type=int, default=401)
    args = parser.parse_args()
    extract(args.input, args.output, args.steps)


if __name__ == "__main__":
    main()
