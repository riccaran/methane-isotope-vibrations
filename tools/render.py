"""Run the complete calculation-to-video pipeline or render frozen reference data.

Examples, from the repository root::

    python tools/render.py --from-scratch --previews
    python tools/render.py --data data/methane_modes.json

PySCF can run in a separate Linux/WSL interpreter with ``--qm-wsl`` and
``--qm-python .venv-qm/bin/python`` while the Manim renderer stays on Windows.
No private application or absolute machine-specific installation path is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export_video(ffmpeg: str, source: Path, target: Path) -> None:
    """Copy native H.264 packets onto an exact 60-fps clock, without reencoding."""
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-bsf:v",
            "setts=pts=round(PTS*TB*60)/(TB*60):dts=round(DTS*TB*60)/(TB*60):duration=1/(TB*60)",
            "-video_track_timescale",
            "60000",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )


def check_ffmpeg(ffmpeg: str, work: Path) -> None:
    """Exercise the actual encoder and timestamp options before costly stages."""
    import av

    with tempfile.TemporaryDirectory(dir=work) as directory:
        source, target = [Path(directory) / name for name in ["input.mp4", "output.mp4"]]
        try:
            run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=16x16:r=60",
                    "-frames:v",
                    "3",
                    "-c:v",
                    "libx264",
                    str(source),
                ]
            )
            export_video(ffmpeg, source, target)
            with av.open(str(target)) as movie:
                if [float(frame.time) for frame in movie.decode(video=0)] != [
                    i / 60 for i in range(3)
                ]:
                    raise ValueError("Probe timestamps are not on the 60-fps grid")
        except (subprocess.CalledProcessError, ValueError) as error:
            raise RuntimeError(
                "FFmpeg cannot encode/export this pipeline. Use a build with libx264 and setts duration support via --ffmpeg."
            ) from error


def ffmpeg_executable(value: str | None) -> str:
    """Resolve FFmpeg from an explicit path or PATH, without bundling a binary.

    Argument arrays are passed directly to subprocess throughout this tool;
    spaces in paths therefore need no string-built shell command or escaping.
    """
    found = shutil.which(value or "ffmpeg")
    if found:
        return found
    if value and Path(value).is_file():
        return str(Path(value).resolve())
    raise FileNotFoundError("FFmpeg not found. Install it on PATH or pass --ffmpeg /path/to/ffmpeg")


def wsl_path(path: Path) -> str:
    """Translate an absolute Windows drive path to its standard WSL mount path.

    Relative interpreter paths are handled by WSL's explicit repository cwd.
    UNC/network shares are rejected rather than silently mapping elsewhere.
    """
    windows = PureWindowsPath(str(path.resolve()))
    if len(windows.drive) != 2 or windows.drive[1] != ":":
        raise ValueError("The WSL convenience runner requires a local Windows drive path")
    return "/mnt/" + windows.drive[0].lower() + "/" + ("/".join(windows.parts[1:]))


def run(command: list[str], environment: dict | None = None) -> None:
    """Run one dependent pipeline stage synchronously and stop on failure.

    A later extraction/render is never started after a failed calculation.
    Stdout and stderr remain visible so long scientific jobs can be observed.
    """
    print("Running: " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    """Parse a reproducible run configuration and execute all requested stages.

    ``--from-scratch`` constructs/optimizes methane, computes two Hessians,
    extracts modes, renders, exports and verifies the result. Without it, the
    checked-in dataset supplies the approved reference movie's numerical input.
    Existing final videos are protected; choose a new work directory to rerun.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    calculation = parser.add_mutually_exclusive_group()
    calculation.add_argument("--from-scratch", action="store_true")
    calculation.add_argument(
        "--calculation",
        type=Path,
        help="Extract from an existing electronic calculation instead of repeating DFT",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed calculation/scene stages only when their inputs match",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/methane_modes.json")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build/run")
    parser.add_argument("--ffmpeg", help="FFmpeg executable path, otherwise searched on PATH")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--qm-python", help="Optional separate PySCF Python interpreter")
    parser.add_argument(
        "--qm-wsl", action="store_true", help="Run the PySCF stage under WSL on Windows"
    )
    parser.add_argument("--wsl-distribution", default="Ubuntu")
    parser.add_argument(
        "--previews", action="store_true", help="Also create the four README-style chapter GIFs"
    )
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    video = work / "methane-isotope-effect.mp4"
    if video.exists():
        raise FileExistsError(f"Protecting existing final video: {video}")
    ffmpeg = ffmpeg_executable(args.ffmpeg)
    check_ffmpeg(ffmpeg, work)
    from methane import model, modes

    data = args.data.resolve()
    new_quantum_jobs = 0
    if args.from_scratch:
        raw = work / "calculation.json"
        data = work / "methane_modes.json"
        if args.qm_wsl:
            if os.name != "nt":
                raise ValueError("--qm-wsl applies only to a Windows host")
            prefix = [
                "wsl.exe",
                "--distribution",
                args.wsl_distribution,
                "--cd",
                wsl_path(ROOT),
                "--exec",
                args.qm_python or "python3",
            ]
            raw_argument = wsl_path(raw)
        else:
            prefix = [args.qm_python or sys.executable]
            raw_argument = str(raw)
        # One common electronic potential serves both isotope mass assignments.
        if not (args.resume and raw.exists()):
            run(
                prefix
                + [
                    "-m",
                    "methane.chemistry",
                    "--output",
                    raw_argument,
                    "--threads",
                    str(args.threads),
                    "--memory-mb",
                    str(args.memory_mb),
                ]
            )
            new_quantum_jobs = 1
        else:
            saved = json.loads(raw.read_text(encoding="utf-8"))
            if not (
                saved.get("internal_stable")
                and saved.get("external_stable")
                and saved.get("hessian_half_step_hartree_per_bohr2") is not None
            ):
                raise ValueError("Cannot resume from an incomplete quantum calculation")
    elif args.calculation:
        raw = args.calculation.resolve()
        data = work / "methane_modes.json"
    if args.from_scratch or args.calculation:
        # Re-extraction is cheap. Replace a previous intermediate only on success.
        with tempfile.TemporaryDirectory(dir=work) as directory:
            extracted = Path(directory) / "modes.json"
            modes.extract(raw, extracted)
            extracted.replace(data)
    if not data.is_file():
        raise FileNotFoundError(data)
    model.load_dataset(data)
    metadata = work / "render_metadata.json"
    environment = os.environ.copy()
    environment["METHANE_DATA_FILE"] = str(data)
    environment["METHANE_METADATA"] = str(metadata)
    environment["METHANE_NEW_QUANTUM_JOBS"] = str(new_quantum_jobs)
    media = work / "manim"
    state_file = work / "scene_inputs.json"
    fingerprint = {
        "data": hashlib.sha256(data.read_bytes()).hexdigest(),
        "manim_version": importlib.metadata.version("manim"),
        **{
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ["methane/model.py", "methane/scene.py", "manim.cfg"]
        },
    }
    state = json.loads(state_file.read_text()) if args.resume and state_file.exists() else {}
    candidates = list((media / "videos").rglob("methane_scene.mp4"))
    reuse = (
        state.get("inputs") == fingerprint
        and len(candidates) == 1
        and metadata.exists()
        and state.get("video_sha256") == hashlib.sha256(candidates[0].read_bytes()).hexdigest()
        and state.get("metadata_sha256") == hashlib.sha256(metadata.read_bytes()).hexdigest()
    )
    if not reuse:
        run(
            [
                sys.executable,
                "-m",
                "manim",
                "--config_file",
                str(ROOT / "manim.cfg"),
                "--resolution",
                "1920,1080",
                "--fps",
                "60",
                "--media_dir",
                str(media),
                "--verbosity",
                "WARNING",
                "-o",
                "methane_scene",
                str(ROOT / "methane/scene.py"),
                "MethaneComparison",
            ],
            environment,
        )
    else:
        print(
            "Reusing validated native scene inputs; encoded output will be verified again.",
            flush=True,
        )
    candidates = list((media / "videos").rglob("methane_scene.mp4"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one native movie, found {candidates}")
    state_file.write_text(
        json.dumps(
            {
                "inputs": fingerprint,
                "video_sha256": hashlib.sha256(candidates[0].read_bytes()).hexdigest(),
                "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Normalize packet timestamps to the exact 60-fps grid without reencoding
    # a single image. This retains Manim's native compressed image payloads.
    pending = work / "methane-isotope-effect.pending.mp4"
    export_video(ffmpeg, candidates[0], pending)
    run(
        [
            sys.executable,
            str(ROOT / "tools/verify_render.py"),
            "--video",
            str(pending),
            "--metadata",
            str(metadata),
            "--data",
            str(data),
            "--output",
            str(work / "verification.json"),
        ]
    )
    if args.previews:
        with tempfile.TemporaryDirectory(dir=work) as directory:
            run(
                [
                    sys.executable,
                    str(ROOT / "tools/previews.py"),
                    "--video",
                    str(pending),
                    "--output-dir",
                    directory,
                    "--ffmpeg",
                    ffmpeg,
                ]
            )
            (work / "previews").mkdir(exist_ok=True)
            for preview in Path(directory).glob("*.gif"):
                preview.replace(work / "previews" / preview.name)
    pending.rename(video)
    print(
        json.dumps(
            {
                "video": str(video),
                "data": str(data),
                "metadata": str(metadata),
                "verification": str(work / "verification.json"),
                "from_scratch": args.from_scratch,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
