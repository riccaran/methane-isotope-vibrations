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
import json

import av
import numpy as np


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
    records = json.loads(metadata.read_text(encoding="utf-8"))
    pairs = dataset["comparisons"]
    reference = pairs[0]["cases"]["CH4"]["frequency_cm1"]
    _, _, zoom = physical_state(0, reference)
    zoom_end = zoom + 2
    end = float(np.ceil((zoom_end + 0.4 + 9) * 60) / 60)
    max_clock_error = 0.0
    for row in records["frames"]:
        q, t, _ = physical_state(row["v"], reference)
        max_clock_error = max(max_clock_error, abs(row["q"] - q), abs(row["time_fs"] - t))
    if max_clock_error > 1e-9:
        raise AssertionError(("clock mismatch", max_clock_error))
    if abs(physical_state(zoom, reference)[1] - 650) > 1e-9:
        raise AssertionError("Zoom not at 650 fs")
    if records["source_data_sha256"] != hashlib.sha256(data.read_bytes()).hexdigest():
        raise AssertionError("Renderer used a different dataset")
    count = 0
    pts_error = 0.0
    enlarged = 0
    phase_checks = 0
    maximum_curve_error = 0.0
    previous = None
    motion = []
    timer_frames = 0
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
            if v < zoom_end:
                continue
            enlarged += 1
            roi = np.concatenate((pixels[338:490, 70:1850:2], pixels[590:742, 70:1850:2]), axis=0)
            if previous is not None:
                motion.append(float(np.mean(abs(roi.astype(np.int16) - previous.astype(np.int16)))))
            previous = roi.copy()
            q, _, _ = physical_state(v, reference)
            for k, centre in [(0, 1.05), (1, -1.05)]:
                for name in ["CH4", "CH3D"]:
                    ratio = pairs[k]["cases"][name]["frequency_cm1"] / reference
                    for scene_x in np.linspace(-6.8, 6.8, 12) * 1.05:
                        phase = q - 8 + 8 * (scene_x + 7.245) / 14.49
                        expected = np.array(
                            [
                                (scene_x + 8) * 120,
                                (4.5 - (centre + 0.6 * np.sin(np.pi * ratio * phase))) * 120,
                            ]
                        )
                        x, y = np.rint(expected).astype(int)
                        patch = pixels[y - 4 : y + 5, x - 4 : x + 5].astype(float)
                        yy, xx = np.nonzero(
                            (patch[:, :, 0] > 120)
                            & (patch[:, :, 1] > 100)
                            & (patch.max(axis=2) > 140)
                        )
                        if not len(xx):
                            raise AssertionError(("Missing wave", v, k, name))
                        error = float(
                            np.min(
                                np.linalg.norm(
                                    np.column_stack((x - 4 + xx, y - 4 + yy)) - expected, axis=1
                                )
                            )
                        )
                        if error >= 3:
                            raise AssertionError(("Wave phase mismatch", v, k, name, error))
                        maximum_curve_error = max(maximum_curve_error, error)
                        phase_checks += 1
    if count != round(end * 60) or pts_error > 1e-9:
        raise AssertionError(("Frame-grid/duration mismatch", count, end, pts_error))
    if not motion or min(motion) <= 0.2:
        raise AssertionError("Frozen or discontinuous finale")
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
