"""Integration and rejection checks for public-data extraction and mode identity."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from methane import model, modes

ROOT = Path(__file__).resolve().parents[1]


class ExtractionTests(unittest.TestCase):
    """Exercise the complete inexpensive Hessian-to-dataset path."""

    @classmethod
    def setUpClass(cls):
        """Extract once from the public Hessian, with no electronic calculation."""
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            cls.extracted = modes.extract(
                ROOT / "data/reference_calculation.json", Path(directory) / "modes.json"
            )
        cls.raw = json.loads((ROOT / "data/reference_calculation.json").read_text())

    def test_all_modes_and_selected_directions_reproduce_reference(self):
        """Compare frequencies and subspaces without assuming degenerate eigenvectors."""
        for name, record in self.extracted["isotopologues"].items():
            frozen = model.DATA["isotopologues"][name]
            np.testing.assert_allclose(
                record["frequencies_cm1"], frozen["frequencies_cm1"], rtol=0, atol=1e-6
            )
            roots = np.repeat(np.sqrt(record["masses_amu"]), 3)
            vectors = []
            for source in (record, frozen):
                rows = np.array(source["cartesian_modes"]).reshape(9, 15) * roots
                vectors.append(rows / np.linalg.norm(rows, axis=1)[:, None])
            for indices in np.split(
                np.arange(9), np.flatnonzero(np.diff(record["frequencies_cm1"]) > 0.01) + 1
            ):
                a, b = [rows[indices] for rows in vectors]
                np.testing.assert_allclose(a.T @ a, b.T @ b, rtol=0, atol=1e-9)
        for new, old in zip(self.extracted["comparisons"], model.PAIRS):
            self.assertEqual(new["group_id"], old["group_id"])
            self.assertAlmostEqual(new["shift_percent"], old["shift_percent"], places=9)
            for name in ["CH4", "CH3D"]:
                a, b = [
                    np.array(pair["cases"][name]["cartesian_mode_unit_norm"]).ravel()
                    for pair in [new, old]
                ]
                self.assertAlmostEqual(abs(float(a @ b)), 1, places=9)
        np.testing.assert_allclose(
            self.extracted["selected_orientation"], model.DATA["selected_orientation"], atol=1e-12
        )

    def test_tracking_quality_and_method_are_derived(self):
        """Quality is reproducible; missing half-step evidence is not converted to zero."""
        quality = self.extracted["quality"]
        self.assertGreater(quality["minimum_subspace_overlap"], 0.99)
        self.assertAlmostEqual(
            quality["minimum_subspace_overlap"],
            model.DATA["quality"]["minimum_subspace_overlap"],
            places=10,
        )
        self.assertIsNone(quality["maximum_half_step_frequency_change_cm1"])
        self.assertEqual(self.extracted["method"]["functional"], self.raw["settings"]["method"])
        self.assertEqual(self.extracted["method"]["basis"], self.raw["settings"]["basis"])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            raw = json.loads(json.dumps(self.raw))
            raw["settings"].update(method="test-method-label", basis="test-basis-label")
            path = Path(directory) / "label-fixture.json"
            path.write_text(json.dumps(raw))
            result = modes.extract(path, Path(directory) / "modes.json")
        self.assertEqual(result["method"]["functional"], "test-method-label")
        self.assertEqual(result["method"]["basis"], "test-basis-label")

    def test_model_rejects_wrong_identity_mass_and_index(self):
        """Plausible arrays with incompatible scientific identities fail at entry."""
        changes = [
            lambda d: d["comparisons"].reverse(),
            lambda d: d["display_conventions"].update(stretch_raw_index_zero_based=0),
            lambda d: d["isotopologues"]["CH3D"]["masses_amu"].__setitem__(2, modes.D_MASS),
            lambda d: d["isotopologues"]["CH4"]["cartesian_modes"].__setitem__(
                0, np.zeros((5, 3)).tolist()
            ),
            lambda d: d["comparisons"][1]["cases"]["CH3D"].update(frequency_cm1=-1183.28),
        ]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                data = json.loads(json.dumps(model.DATA))
                change(data)
                path = Path(directory) / "invalid.json"
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    model.load_dataset(path)

    def test_degenerate_basis_ignores_arbitrary_eigensolver_rotations(self):
        """Equal coordinate projections use a stable tie-break within one subspace."""
        rows = np.eye(15)[:2]
        expected = modes.deterministic_basis(rows)
        for angle in [0.3, 1.7, 2.6]:
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            np.testing.assert_allclose(
                modes.deterministic_basis(rotation @ rows), expected, rtol=0, atol=1e-12
            )

    def test_extraction_rejects_failed_checks_and_protects_outputs(self):
        """Reject failed electronic checks, unknown mode families, and overwrites."""
        for key, value in [
            ("gradient_hartree_per_bohr", [float("nan")] * 15),
            ("internal_stable", False),
            (
                "hessian_hartree_per_bohr2",
                (4 * np.array(self.raw["hessian_hartree_per_bohr2"])).tolist(),
            ),
        ]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                raw = dict(self.raw, **{key: value})
                path, output = Path(directory) / "raw.json", Path(directory) / "modes.json"
                path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    modes.extract(path, output)
                self.assertFalse(output.exists())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("preserve me")
            with self.assertRaises(FileExistsError):
                modes.extract(ROOT / "data/reference_calculation.json", output)
            self.assertEqual(output.read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
