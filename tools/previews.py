"""Create four inline README GIF chapters directly from the final MP4.

The approved film stays unchanged. FFmpeg decodes short, explicitly specified
segments, scales them, and builds a per-segment palette. GIF frame rate is only
a lightweight presentation format; it is not a new molecular simulation.
"""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
# The frozen and freshly generated data share the same scene schedule. All
# excerpts below use actual video seconds, including the extra comparison time.
CHAPTERS = [
    ("01-isotope-nuclei", 5.6, 5.0, 30),
    ("02-component-guides", 24.0, 8.0, 12),
    ("03-two-mode-comparison", 34.0, 10.15, 12),
    ("04-enlarged-waves", 46.6, 5.0, 12),
]


def make_previews(video: Path, output: Path, ffmpeg: str, width: int = 960) -> list[dict]:
    """Encode each named chapter as a looping GIF with a dedicated palette.

    Input frames are never edited into new atomic trajectories. The fast opening
    uses 30 fps, above twice its largest displayed vibrational frequency; the
    already slowed chapters use 12 fps. Palette generation/use share a split
    filter, preserving antialiased caption edges against the dark field.
    """
    if not video.is_file():
        raise FileNotFoundError(video)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, start, duration, fps in CHAPTERS:
        target = output / (name + ".gif")
        if target.exists():
            raise FileExistsError(f"Protecting existing preview: {target}")
        filters = (
            f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
            # Include static labels in the palette. A changed-pixels-only
            # histogram can drop the cyan/pink mode colours from these clips.
            "[a]palettegen=max_colors=192:stats_mode=full:reserve_transparent=0[p];"
            "[b][p]paletteuse=dither=none"
        )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            str(video),
            "-filter_complex",
            filters,
            "-an",
            "-loop",
            "0",
            str(target),
        ]
        subprocess.run(command, check=True)
        records.append(
            {
                "file": target.name,
                "start_seconds": start,
                "duration_seconds": duration,
                "width": width,
                "fps": fps,
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    return records


def main() -> None:
    """Parse source movie, destination directory and FFmpeg executable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "media/methane-isotope-effect.mp4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "media")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()
    binary = shutil.which(args.ffmpeg) or args.ffmpeg
    print(json.dumps(make_previews(args.video, args.output_dir, binary, args.width), indent=2))


if __name__ == "__main__":
    main()
