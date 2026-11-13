"""Verify a rendered film against an independently evaluated physical clock.

This verifier reads the public dataset and the generated numerical metadata,
then decodes *every* video frame with PyAV. It checks timestamps, dimensions,
the 650 fs transition, absence of a frozen ending, and all enlarged waveforms
against phases derived directly from the frequency data. It does not import
the animation model's clock, avoiding a self-comparison of the implementation.
"""

from pathlib import Path
import argparse
import hashlib
import gzip
import json

import av
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def read_timer(pixels: np.ndarray, templates: np.ndarray, integer_digits: int) -> str:
    """Read only the five fixed timer digits, using real glyphs in two fonts.

    Bounding-box normalization tolerates subpixel placement and H.264 edges.
    The atlas contains Consolas and DejaVu Sans Mono at the scene's font size.
    A mismatch over 12% of the normalized binary glyph is rejected.
    """
    result = ""
    for position, x in enumerate([902.4, 928.8, 955.2, 1008.0, 1034.4]):
        if position < 3 - integer_digits:
            result += " "  # The moving 't =' label occupies unused integer slots.
            continue
        mask = pixels[1005:1055, round(x) - 13 : round(x) + 14].min(axis=2) > 110
        yy, xx = np.nonzero(mask)
        if len(xx) < 10:
            result += " "
            continue
        crop = Image.fromarray(mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1])
        normalized = np.asarray(crop.resize((24, 32), Image.Resampling.NEAREST))
        errors = np.mean(templates != normalized, axis=(2, 3)).min(axis=0)
        digit = int(np.argmin(errors))
        if errors[digit] > 0.12:
            raise AssertionError(("Unrecognized timer digit", result, float(errors[digit])))
        result += str(digit)
    return result[:3] + "." + result[3:]


def check_waveforms(pixels: np.ndarray, video_time: float, pairs: list, reference: float):
    """Check phase and isotope colour, excluding pixels hidden at crossings."""
    q, _, zoom = physical_state(video_time, reference)
    u = float(np.clip((video_time - zoom) / 2, 0, 1))
    u = u**3 * (10 - 15 * u + 6 * u * u)
    sx, amplitude = 1 + 0.05 * u, 0.2 * (1 + 2 * u)
    maximum, checked = 0.0, 0
    for k, original in enumerate([-2.8, -3.6]):
        centre = (1 - u) * original + u * (1.05 if k == 0 else -1.05)
        ratios = [pairs[k]["cases"][name]["frequency_cm1"] / reference for name in ["CH4", "CH3D"]]
        for position in np.linspace(-6.8, 6.8, 12):
            phase = q - 8 + 8 * (position + 6.9) / 13.8
            ys = [
                (4.5 - centre - amplitude * np.sin(np.pi * ratio * phase)) * 120 for ratio in ratios
            ]
            if abs(ys[0] - ys[1]) < 8:
                continue  # The gold curve can occlude the white curve here.
            for isotope, expected_y in enumerate(ys):
                expected_x = (position * sx + 8) * 120
                x, y = round(expected_x), round(expected_y)
                patch = pixels[y - 4 : y + 5, x - 4 : x + 5].astype(float)
                r, g, b = np.moveaxis(patch, -1, 0)
                mask = (
                    (np.minimum(np.minimum(r, g), b) > 125)
                    & (patch.max(axis=2) - patch.min(axis=2) < 35)
                    if isotope == 0
                    else (r > 130) & (g > 95) & (g - b > 25) & (r - b > 35)
                )
                yy, xx = np.nonzero(mask)
                if not len(xx):
                    raise AssertionError(
                        ("Missing wave or wrong isotope colour", video_time, k, isotope)
                    )
                error = float(np.min(np.hypot(x - 4 + xx - expected_x, y - 4 + yy - expected_y)))
                if error >= 3:
                    raise AssertionError(("Wave phase mismatch", video_time, k, isotope, error))
                maximum, checked = max(maximum, error), checked + 1
    return maximum, checked


def physical_state(video_time: float, reference_cm1: float):
    """Independently return phase coordinate, elapsed fs and graph-zoom progress.

    The original cosine ramp is integrated analytically. An independent
    expanded quintic maps the two extra comparison seconds into that same
    physical timeline, retaining unit speed at the two presentation joins.
    """
    fs_per_q = 1e15 / (2 * reference_cm1 * 2.99792458e10)
    original_zoom = 24 + 650 / fs_per_q - 100
    start = original_zoom + 2
    s = float(np.clip((video_time - 34) / (start - 34), 0, 1))
    model_time = video_time - 2 * (6 * s**5 - 15 * s**4 + 10 * s**3)
    if model_time <= 16:
        q = 24 * model_time
    elif model_time < 24:
        x = model_time - 16
        q = 384 + 12.5 * x + 92 / np.pi * np.sin(np.pi * x / 8)
    else:
        q = 484 + model_time - 24
    elapsed = 0.0 if model_time < 16 else fs_per_q * (q - 384)
    return q, elapsed, start


def verify(video: Path, metadata: Path, data: Path, output: Path) -> dict:
    """Decode and verify the complete film, writing a compact machine-readable audit.

    The timer/headers are checked in encoded pixels; wave phase is tested at
    twelve horizontal points in each of four curves for every enlarged frame.
    Adjacent waveform crops must keep changing, including the final pair.
    These are checks of the actual encoded output, not just scene intention.
    """
    dataset = json.loads(data.read_text(encoding="utf-8"))
    records = json.loads(
        gzip.decompress(metadata.read_bytes())
        if metadata.suffix == ".gz"
        else metadata.read_bytes()
    )
    pairs = dataset["comparisons"]
    reference = pairs[0]["cases"]["CH4"]["frequency_cm1"]
    _, _, zoom = physical_state(0, reference)
    zoom_end = zoom + 2
    end = float(np.ceil((zoom_end + 0.4 + 9) * 60) / 60)
    max_clock_error = 0.0
    for row in records["frames"]:
        if not np.all(np.isfinite([row["v"], row["q"], row["time_fs"]])):
            raise AssertionError("Non-finite frame metadata")
        q, t, _ = physical_state(row["v"], reference)
        max_clock_error = max(max_clock_error, abs(row["q"] - q), abs(row["time_fs"] - t))
    if max_clock_error > 1e-9:
        raise AssertionError(("clock mismatch", max_clock_error))
    if abs(records["graph_focus_interval"][0] - zoom) > 1e-9:
        raise AssertionError("Recorded graph transition disagrees with independent clock")
    expected_frames = round(end * 60)
    rows = records["frames"]
    terminal = len(rows) == expected_frames + 1 and abs(rows[-1]["v"] - end) < 1e-9
    if len(rows) != expected_frames + int(terminal):
        raise AssertionError(
            "Metadata must contain one sample per frame, plus at most one terminal state"
        )
    if max(abs(row["v"] - i / 60) for i, row in enumerate(rows)) > 1e-9:
        raise AssertionError("Metadata is not on the encoded frame grid")
    if records["source_data_sha256"] != hashlib.sha256(data.read_bytes()).hexdigest():
        raise AssertionError("Renderer used a different dataset")
    count = 0
    pts_error = 0.0
    enlarged = 0
    phase_checks = 0
    maximum_curve_error = 0.0
    previous = None
    motion = []
    timer_frames = timer_present = 0
    observed_hide = None
    atlas = np.asarray(Image.open(ROOT / "validation/timer_digits.png")) > 0
    templates = atlas.reshape(2, 32, 10, 24).transpose(0, 2, 1, 3)
    with av.open(str(video)) as movie:
        stream = movie.streams.video[0]
        if (stream.width, stream.height, float(stream.average_rate), len(movie.streams.audio)) != (
            1920,
            1080,
            60.0,
            0,
        ):
            raise AssertionError("Expected silent 1920x1080 video at exact 60 fps")
        for frame in movie.decode(stream):
            v = float(frame.time)
            pts_error = max(pts_error, abs(v - count / 60))
            count += 1
            pixels = frame.to_ndarray(format="rgb24")
            row = rows[count - 1]
            if abs(row["v"] - v) > 1e-9:
                raise AssertionError("Encoded frame and metadata timestamps disagree")
            if 16 <= v < zoom:
                _, elapsed, _ = physical_state(v, reference)
                integer, fraction = f"{elapsed:.2f}".split(".")
                expected_timer = integer.rjust(3) + "." + fraction
                actual_timer = read_timer(pixels, templates, len(integer))
                if actual_timer != expected_timer:
                    raise AssertionError(("Timer digits mismatch", v, actual_timer, expected_timer))
                timer_present += 1
            if v < zoom:
                if (pixels[15:105, 360:600].min(axis=2) > 130).sum() < 200:
                    raise AssertionError(("CH4 header missing", v))
                region = pixels[15:105, 1320:1560].astype(float)
                gold = (
                    (region[:, :, 0] > 140)
                    & (region[:, :, 1] > 110)
                    & (region[:, :, 1] - region[:, :, 2] > 20)
                )
                if gold.sum() < 200:
                    raise AssertionError(("CH3D header missing", v))
            else:
                timer_frames += 1
                if (pixels[1020:1070, 750:1180].max(axis=2) > 100).sum() >= 20:
                    raise AssertionError(("Timer remained visible after graph zoom", v))
                if observed_hide is None:
                    observed_hide = v
            if v >= 34:
                error, checks = check_waveforms(pixels, v, pairs, reference)
                maximum_curve_error = max(maximum_curve_error, error)
                phase_checks += checks
            if v < zoom_end:
                continue
            enlarged += 1
            roi = np.concatenate((pixels[338:490, 70:1850:2], pixels[590:742, 70:1850:2]), axis=0)
            if previous is not None:
                motion.append(float(np.mean(abs(roi.astype(np.int16) - previous.astype(np.int16)))))
            previous = roi.copy()
    if count != round(end * 60) or pts_error > 1e-9:
        raise AssertionError(("Frame-grid/duration mismatch", count, end, pts_error))
    if not motion or min(motion) <= 0.2:
        raise AssertionError("Frozen or discontinuous finale")
    if observed_hide is None or not 0 <= observed_hide - zoom < 1 / 60:
        raise AssertionError("Observed timer transition is not at the 650 fs frame boundary")
    result = {
        "passed": True,
        "frames": count,
        "duration_seconds": count / 60,
        "all_frames_decoded": True,
        "pts_max_error_seconds": pts_error,
        "clock_max_error": max_clock_error,
        "graph_zoom_physical_fs": 650.0,
        "graph_zoom_video_seconds": zoom,
        "timer_absence_frames_checked": timer_frames,
        "timer_digits_frames_checked": timer_present,
        "observed_timer_hide_seconds": observed_hide,
        "metadata_terminal_sample": terminal,
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "timer_atlas_sha256": hashlib.sha256(
            (ROOT / "validation/timer_digits.png").read_bytes()
        ).hexdigest(),
        "enlarged_frames_checked": enlarged,
        "wave_phase_pixel_checks": phase_checks,
        "maximum_wave_position_error_px": maximum_curve_error,
        "minimum_adjacent_wave_rgb_change": min(motion),
        "last_frame_phase_checked": True,
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    """Parse the movie, generated metadata, dataset and audit-output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/verification.json"))
    args = parser.parse_args()
    verify(args.video, args.metadata, args.data, args.output)


if __name__ == "__main__":
    main()
