"""Scientific invariants and public-code documentation checks.

These checks use a frozen electronic Hessian, not mocked frequency values.
The expensive from-scratch PySCF/Manim integration run is separate and described
in README; this fast suite can run before every change without recomputing DFT.
"""

import ast
import json
from pathlib import Path
import tempfile
import unittest
import re

import numpy as np
from PIL import Image

from methane import model, modes
from methane.chemistry import tetrahedron

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    """Check independent physical, numerical and distribution-level invariants."""

    @classmethod
    def setUpClass(cls):
        """Load the small reference electronic calculation exactly once."""
        cls.raw = json.loads((ROOT / "data/reference_calculation.json").read_text(encoding="utf-8"))

    def test_constructed_geometry_is_tetrahedral(self):
        """From-code construction has four equal bonds and tetrahedral angles."""
        xyz = tetrahedron(1.09)
        directions = (xyz[1:] - xyz[0]) / 1.09
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-14)
        gram = directions @ directions.T
        np.testing.assert_allclose(gram[np.triu_indices(4, 1)], -1 / 3, atol=1e-14)

    def test_hessian_reproduces_all_nine_reference_frequencies(self):
        """Independent mass weighting recovers the nine supplied values for H and D."""
        for name, record in model.DATA["isotopologues"].items():
            frequency, weighted, cartesian = modes.harmonic_modes(
                np.array(self.raw["coordinates_angstrom"]),
                np.array(self.raw["hessian_hartree_per_bohr2"]),
                np.array(record["masses_amu"]),
            )
            np.testing.assert_allclose(frequency, record["frequencies_cm1"], rtol=0, atol=1e-6)
            np.testing.assert_allclose(weighted @ weighted.T, np.eye(9), atol=1e-12)
            # Translation-free physical coordinates conserve the molecular COM.
            centre_motion = np.einsum("a,kai->ki", record["masses_amu"], cartesian)
            np.testing.assert_allclose(centre_motion, 0, atol=1e-12)

    def test_isotope_substitution_changes_mass_not_geometry_or_charge(self):
        """The Born-Oppenheimer comparison changes exactly one isotope mass."""
        h, d = [model.DATA["isotopologues"][name] for name in ["CH4", "CH3D"]]
        np.testing.assert_array_equal(h["coordinates_angstrom"], d["coordinates_angstrom"])
        differences = np.flatnonzero(np.array(h["masses_amu"]) != np.array(d["masses_amu"]))
        np.testing.assert_array_equal(differences, [1])
        self.assertAlmostEqual(d["masses_amu"][1], modes.D_MASS)

    def test_selected_caption_percentages_follow_frequency_ratios(self):
        """Both public percentages correspond to fewer cycles over the same time."""
        for pair, expected in zip(model.PAIRS, [25.03, 11.76]):
            shift = 100 * (
                1 - pair["cases"]["CH3D"]["frequency_cm1"] / pair["cases"]["CH4"]["frequency_cm1"]
            )
            self.assertAlmostEqual(shift, pair["shift_percent"], places=11)
            self.assertEqual(round(shift, 2), expected)

    def test_cartesian_motion_repeats_after_a_physical_period(self):
        """Actual selected atomic displacement repeats after 1/(c*wavenumber)."""
        for name in ["CH4", "CH3D"]:
            for k, branch in [(0, 1), (1, -1)]:
                ratio = model.PAIRS[k]["cases"][name]["frequency_cm1"] / model.FREF
                initial = model.local_atoms(name, 35.0, branch=branch)
                repeated = model.local_atoms(name, 35.0 + 2 / ratio, branch=branch)
                np.testing.assert_allclose(initial, repeated, atol=1e-12)

    def test_phase_clock_has_the_declared_continuous_derivative(self):
        """An independent centred difference checks rate/phase synchronization."""
        step = 1e-5
        for time in [0.0, 15.0, 16.0, 18.0, 20.0, 23.0, 24.0, 30.0]:
            derivative = (model.phase_clock(time + step) - model.phase_clock(time - step)) / (
                2 * step
            )
            self.assertAlmostEqual(float(derivative), float(model.phase_rate(time)), places=7)

    def test_extra_comparison_time_has_smooth_monotonic_joins(self):
        """The added two seconds have no hold, backward time or endpoint speed jump."""
        start = model.PLOTS_START
        end = model.GRAPH_FOCUS_START + 2
        times = np.linspace(start, end, 1001)
        mapped = np.array([model.comparison_model_time(t) for t in times])
        self.assertTrue(np.all(np.diff(mapped) > 0))
        self.assertAlmostEqual(mapped[0], start)
        self.assertAlmostEqual(mapped[-1], model.GRAPH_FOCUS_START)
        self.assertEqual(model.comparison_playback_rate(start), 1)
        self.assertEqual(model.comparison_playback_rate(end), 1)
        self.assertAlmostEqual(float(model.time_fs(mapped[-1])), 650, places=9)

    def test_lens_is_attached_to_the_live_molecule(self):
        """The optical lens and tracked H/D source use current atom positions."""
        for name in ["CH4", "CH3D"]:
            for time in [3.4, 5.4, 8.0, 11.2, 12.4]:
                _, points, centre, _, _ = model.lens_state(name, time)
                np.testing.assert_array_equal(centre, points[0])
                source, _, _, _ = model.nucleus_state(name, time)
                np.testing.assert_array_equal(source, points[1])

    def test_invalid_dataset_fails_before_rendering(self):
        """A malformed mode array is rejected rather than producing a plausible film."""
        dataset = json.loads(json.dumps(model.DATA))
        dataset["isotopologues"]["CH4"]["cartesian_modes"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")
            with self.assertRaises(ValueError):
                model.load_dataset(path)

    def test_every_public_module_class_and_function_is_documented(self):
        """Enforce the requested documentation on all shipped Python source files."""
        for folder in ["methane", "tools", "tests"]:
            for path in (ROOT / folder).glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                self.assertTrue(ast.get_docstring(tree), str(path))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertTrue(ast.get_docstring(node), f"{path.name}:{node.name}")

    def test_readme_links_resolve_to_shipped_files(self):
        """All local README links and four inline GIFs resolve in a clean clone."""
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        targets = re.findall(r"\]\(([^)]+)\)", text)
        local = [
            target for target in targets if not target.startswith(("https://", "http://", "#"))
        ]
        for target in local:
            self.assertTrue((ROOT / target).is_file(), target)
        self.assertEqual(sum(target.endswith(".gif") for target in local), 4)

    def test_preview_gifs_animate_and_preserve_mode_colours(self):
        """GIF chapters retain motion, fast-sampling headroom and cyan/pink cues."""
        paths = sorted((ROOT / "media").glob("*.gif"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            with Image.open(path) as image:
                self.assertGreater(image.n_frames, 1)
                image.seek(0)
                first = np.asarray(image.convert("RGB"), dtype=float)
                image.seek(1)
                second = np.asarray(image.convert("RGB"), dtype=float)
                self.assertGreater(float(np.mean(abs(second - first))), 0)
                if path.name.startswith("01-"):
                    frames = image.n_frames
                    duration = 0
                    for i in range(frames):
                        image.seek(i)
                        duration += image.info.get("duration", 0)
                    average_fps = frames / (duration / 1000)
                    largest_graphical_hz = (
                        12 * max(model.DATA["isotopologues"]["CH4"]["frequencies_cm1"]) / model.FREF
                    )
                    self.assertGreater(average_fps, 2 * largest_graphical_hz)
                if path.name.startswith("03-"):
                    cyan = (
                        (first[:, :, 1] > 120)
                        & (first[:, :, 2] > 140)
                        & (first[:, :, 1] - first[:, :, 0] > 45)
                    )
                    pink = (
                        (first[:, :, 0] > 150)
                        & (first[:, :, 2] > 135)
                        & (first[:, :, 0] - first[:, :, 1] > 30)
                        & (first[:, :, 2] - first[:, :, 1] > 8)
                    )
                    self.assertGreater(int(cyan.sum()), 300)
                    self.assertGreater(int(pink.sum()), 300)


if __name__ == "__main__":
    unittest.main()
