"""Small encoded-frame regression checks; no Manim or quantum job is run."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class PipelineRecoveryTests(unittest.TestCase):
    """Check stage orchestration with cheap stand-ins for expensive subprocesses."""

    def test_preflight_failure_stops_before_any_expensive_stage(self):
        """An unsupported encoder must fail before launching PySCF or Manim."""
        from tools import render

        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", ["render.py", "--from-scratch", "--work-dir", directory]
        ), patch.object(render, "ffmpeg_executable", return_value="unused"), patch.object(
            render, "check_ffmpeg", side_effect=RuntimeError("unsupported")
        ), patch.object(
            render, "run"
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                render.main()
            run.assert_not_called()
            self.assertFalse((Path(directory) / "calculation.json").exists())

    def test_late_failure_resume_and_changed_input(self):
        """Publish only verified output; reuse matching scenes and rebuild changed input."""
        from tools import render
        from methane import model

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            data = work / "input.json"
            data.write_text(json.dumps(model.DATA))
            state = {"fail": True, "scene_runs": 0}

            def stage(command, environment=None):
                """Represent completed native rendering or an injected verifier failure."""
                if "manim" in command:
                    state["scene_runs"] += 1
                    native = work / "manim/videos/test/methane_scene.mp4"
                    native.parent.mkdir(parents=True, exist_ok=True)
                    native.write_bytes(b"native-stage-fixture")
                    Path(environment["METHANE_METADATA"]).write_text("{}")
                elif (
                    any(str(part).endswith("verify_render.py") for part in command)
                    and state["fail"]
                ):
                    raise subprocess.CalledProcessError(97, command)

            def export(binary, source, target):
                """Represent a completed export without introducing a real encode."""
                target.write_bytes(source.read_bytes())

            args = ["render.py", "--work-dir", directory, "--data", str(data)]
            with patch.object(render, "ffmpeg_executable", return_value="unused"), patch.object(
                render, "check_ffmpeg"
            ), patch.object(render, "run", side_effect=stage), patch.object(
                render, "export_video", side_effect=export
            ):
                with patch.object(sys, "argv", args), self.assertRaises(
                    subprocess.CalledProcessError
                ):
                    render.main()
                final = work / "methane-isotope-effect.mp4"
                self.assertFalse(final.exists())
                self.assertTrue((work / "methane-isotope-effect.pending.mp4").exists())
                state["fail"] = False
                with patch.object(sys, "argv", args + ["--resume"]):
                    render.main()
                self.assertTrue(final.exists())
                self.assertEqual(state["scene_runs"], 1)
                final.unlink()  # Only this temporary test fixture is removed.
                changed = json.loads(data.read_text())
                changed["provenance"] = "different input hash"
                data.write_text(json.dumps(changed))
                with patch.object(sys, "argv", args + ["--resume"]):
                    render.main()
                self.assertEqual(state["scene_runs"], 2)


@unittest.skipUnless(importlib.util.find_spec("av"), "Encoded-frame tests need PyAV")
class EncodedFrameTests(unittest.TestCase):
    """Read real frames and reject deliberately damaged timer/curve pixels."""

    @classmethod
    def setUpClass(cls):
        """Decode two small reference samples, not the entire film."""
        import av
        from tools import verify_render
        from methane import model

        cls.verifier, cls.pairs = verify_render, model.PAIRS
        cls.reference = cls.pairs[0]["cases"]["CH4"]["frequency_cm1"]
        cls.samples = {}
        for second in [20, 50]:
            with av.open(str(ROOT / "media/methane-isotope-effect.mp4")) as movie:
                movie.seek(second * av.time_base)
                frame = next(
                    frame for frame in movie.decode(video=0) if float(frame.time) >= second
                )
                cls.samples[second] = frame.to_ndarray(format="rgb24")
        atlas = np.asarray(Image.open(ROOT / "validation/timer_digits.png")) > 0
        cls.templates = atlas.reshape(2, 32, 10, 24).transpose(0, 2, 1, 3)

    def test_timer_recognizes_pixels_and_detects_changed_digit(self):
        """Read the actual timer; copying a different digit changes its interpretation."""
        pixels = self.samples[20].copy()
        _, elapsed, _ = self.verifier.physical_state(20, self.reference)
        expected = f"{elapsed:06.2f}"
        self.assertEqual(self.verifier.read_timer(pixels, self.templates, 3), expected)
        pixels[1005:1055, 942:969] = pixels[1005:1055, 995:1022]
        self.assertNotEqual(self.verifier.read_timer(pixels, self.templates, 3), expected)

    def test_wave_check_rejects_missing_colour_and_wrong_phase(self):
        """A correct frame passes; erasing gold or assigning another phase fails."""
        pixels = self.samples[50]
        error, checked = self.verifier.check_waveforms(pixels, 50, self.pairs, self.reference)
        self.assertLess(error, 3)
        self.assertGreater(checked, 0)
        changed = pixels.copy()
        r, g, b = np.moveaxis(changed.astype(float), -1, 0)
        changed[(r > 130) & (g > 95) & (g - b > 25) & (r - b > 35)] = [237, 240, 243]
        with self.assertRaisesRegex(AssertionError, "colour"):
            self.verifier.check_waveforms(changed, 50, self.pairs, self.reference)
        with self.assertRaises(AssertionError):
            self.verifier.check_waveforms(pixels, 50.5, self.pairs, self.reference)


if __name__ == "__main__":
    unittest.main()
