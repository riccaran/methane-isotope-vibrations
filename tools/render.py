"""Calculate methane isotope modes, render the film and verify the encoded output.

Use --from-scratch for a new calculation, --data for existing modes, or
--calculation for an existing Hessian. PySCF may run in a separate WSL Python.
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


def run(command: list[str], environment: dict | None = None) -> None:
    """Run a dependent stage in the repository, stopping immediately on failure."""
    print("Running: " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def export_video(ffmpeg: str, source: Path, target: Path) -> None:
    """Copy H.264 packets onto the exact 60-fps grid without reencoding images."""
    run([
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(source),
        "-map", "0:v:0", "-an", "-c:v", "copy", "-bsf:v",
        "setts=pts=round(PTS*TB*60)/(TB*60):dts=round(DTS*TB*60)/(TB*60):duration=1/(TB*60)",
        "-video_track_timescale", "60000", "-movflags", "+faststart", str(target),
    ])


def check_ffmpeg(ffmpeg: str, work: Path) -> None:
    """Exercise encoding and timestamp export before starting expensive stages."""
    import av

    with tempfile.TemporaryDirectory(dir=work) as directory:
        source, target = [Path(directory) / name for name in ("input.mp4", "output.mp4")]
        try:
            run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "color=c=black:s=16x16:r=60", "-frames:v", "3",
                "-c:v", "libx264", str(source),
            ])
            export_video(ffmpeg, source, target)
            with av.open(str(target)) as movie:
                if [float(frame.time) for frame in movie.decode(video=0)] != [i / 60 for i in range(3)]:
                    raise ValueError("Probe timestamps are not on the 60-fps grid")
        except (subprocess.CalledProcessError, ValueError) as error:
            raise RuntimeError(
                "FFmpeg cannot encode/export this pipeline. Use a build with libx264 and setts duration support via --ffmpeg."
            ) from error


def ffmpeg_executable(value: str | None) -> str:
    """Resolve an explicit executable or PATH entry; argument arrays preserve spaces."""
    found = shutil.which(value or "ffmpeg")
    if found:
        return found
    if value and Path(value).is_file():
        return str(Path(value).resolve())
    raise FileNotFoundError("FFmpeg not found. Install it on PATH or pass --ffmpeg /path/to/ffmpeg")


def wsl_path(path: Path) -> str:
    """Translate a local Windows drive path to WSL; reject UNC/network paths."""
    windows = PureWindowsPath(str(path.resolve()))
    if len(windows.drive) != 2 or windows.drive[1] != ":":
        raise ValueError("The WSL convenience runner requires a local Windows drive path")
    return "/mnt/" + windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])


def prepare_data(args: argparse.Namespace, work: Path) -> tuple[Path, int]:
    """Calculate or reuse a Hessian, then atomically extract and validate its modes."""
    from methane import model, modes

    data, jobs = args.data.resolve(), 0
    if args.from_scratch:
        raw, data = work / "calculation.json", work / "methane_modes.json"
        if args.qm_wsl:
            if os.name != "nt":
                raise ValueError("--qm-wsl applies only to a Windows host")
            prefix = ["wsl.exe", "--distribution", args.wsl_distribution, "--cd",
                      wsl_path(ROOT), "--exec", args.qm_python or "python3"]
            raw_argument = wsl_path(raw)
        else:
            prefix, raw_argument = [args.qm_python or sys.executable], str(raw)
        if args.resume and raw.exists():
            saved = json.loads(raw.read_text(encoding="utf-8"))
            if not (saved.get("internal_stable") and saved.get("external_stable")
                    and saved.get("hessian_half_step_hartree_per_bohr2") is not None):
                raise ValueError("Cannot resume from an incomplete quantum calculation")
        else:
            run(prefix + ["-m", "methane.chemistry", "--output", raw_argument,
                          "--threads", str(args.threads), "--memory-mb", str(args.memory_mb)])
            jobs = 1
    elif args.calculation:
        raw, data = args.calculation.resolve(), work / "methane_modes.json"
    if args.from_scratch or args.calculation:
        with tempfile.TemporaryDirectory(dir=work) as directory:
            extracted = Path(directory) / "modes.json"
            modes.extract(raw, extracted)
            extracted.replace(data)
    model.load_dataset(data)
    return data, jobs


def render_scene(work: Path, data: Path, environment: dict, resume: bool) -> Path:
    """Reuse only an unchanged completed scene; otherwise render and fingerprint it."""
    media, metadata = work / "manim", work / "render_metadata.json"
    state_file = work / "scene_inputs.json"
    fingerprint = {
        "data": hashlib.sha256(data.read_bytes()).hexdigest(),
        "manim_version": importlib.metadata.version("manim"),
        **{name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
           for name in ("methane/model.py", "methane/scene.py", "manim.cfg")},
    }
    state = json.loads(state_file.read_text()) if resume and state_file.exists() else {}
    candidates = list((media / "videos").rglob("methane_scene.mp4"))
    reuse = (
        state.get("inputs") == fingerprint and len(candidates) == 1 and metadata.exists()
        and state.get("video_sha256") == hashlib.sha256(candidates[0].read_bytes()).hexdigest()
        and state.get("metadata_sha256") == hashlib.sha256(metadata.read_bytes()).hexdigest()
    )
    if reuse:
        print("Reusing validated native scene inputs; encoded output will be verified again.", flush=True)
    else:
        run([
            sys.executable, "-m", "manim", "--config_file", str(ROOT / "manim.cfg"),
            "--resolution", "1920,1080", "--fps", "60", "--media_dir", str(media),
            "--verbosity", "WARNING", "-o", "methane_scene",
            str(ROOT / "methane/scene.py"), "MethaneComparison",
        ], environment)
        candidates = list((media / "videos").rglob("methane_scene.mp4"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one native movie, found {candidates}")
    state_file.write_text(json.dumps({
        "inputs": fingerprint,
        "video_sha256": hashlib.sha256(candidates[0].read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    return candidates[0]


def main() -> None:
    """Orchestrate calculation, recoverable rendering and verified final publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    calculation = parser.add_mutually_exclusive_group()
    calculation.add_argument("--from-scratch", action="store_true")
    calculation.add_argument("--calculation", type=Path, help="Reuse an electronic calculation")
    parser.add_argument("--resume", action="store_true", help="Reuse completed stages with matching inputs")
    parser.add_argument("--data", type=Path, default=ROOT / "data/methane_modes.json")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build/run")
    parser.add_argument("--ffmpeg", help="FFmpeg executable path, otherwise searched on PATH")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--qm-python", help="Optional separate PySCF Python interpreter")
    parser.add_argument("--qm-wsl", action="store_true", help="Run PySCF under WSL on Windows")
    parser.add_argument("--wsl-distribution", default="Ubuntu")
    parser.add_argument("--previews", action="store_true", help="Also generate the four chapter GIFs")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    video = work / "methane-isotope-effect.mp4"
    if video.exists():
        raise FileExistsError(f"Protecting existing final video: {video}")
    ffmpeg = ffmpeg_executable(args.ffmpeg)
    check_ffmpeg(ffmpeg, work)
    data, jobs = prepare_data(args, work)
    metadata = work / "render_metadata.json"
    environment = dict(os.environ, METHANE_DATA_FILE=str(data), METHANE_METADATA=str(metadata),
                       METHANE_NEW_QUANTUM_JOBS=str(jobs))
    native = render_scene(work, data, environment, args.resume)
    pending = work / "methane-isotope-effect.pending.mp4"
    export_video(ffmpeg, native, pending)
    run([
        sys.executable, str(ROOT / "tools/verify_render.py"), "--video", str(pending),
        "--metadata", str(metadata), "--data", str(data), "--output", str(work / "verification.json"),
    ])
    if args.previews:
        with tempfile.TemporaryDirectory(dir=work) as directory:
            run([sys.executable, str(ROOT / "tools/previews.py"), "--video", str(pending),
                 "--output-dir", directory, "--ffmpeg", ffmpeg])
            (work / "previews").mkdir(exist_ok=True)
            for preview in Path(directory).glob("*.gif"):
                preview.replace(work / "previews" / preview.name)
    pending.rename(video)
    print(json.dumps({"video": str(video), "data": str(data), "metadata": str(metadata),
                      "verification": str(work / "verification.json"), "from_scratch": args.from_scratch}, indent=2))


if __name__ == "__main__":
    main()
