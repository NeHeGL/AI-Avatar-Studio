"""Turn a portrait still + speech wav into a talking-head mp4 via ComfyUI InfiniteTalk."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / "Scripts" / "python.exe"
COMFY_RUN = HERE / "comfy_run.py"

PROMPT = (
    "a person speaking naturally to the camera, subtle head movement, natural blinking, "
    "soft daylight, photorealistic, upper body portrait"
)


def render_talk_video(image: Path, audio_wav: Path, out_mp4: Path, *, size: int = 640, steps: int = 6) -> Path:
    """Run InfiniteTalk. Raises on failure. Typically tens of seconds to a few minutes per clip."""
    if not image.exists():
        raise FileNotFoundError(image)
    if not audio_wav.exists():
        raise FileNotFoundError(audio_wav)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if not COMFY_RUN.exists():
        raise RuntimeError("comfy_run.py missing — install ComfyUI models first (ml/download_models.py)")
    cmd = [
        str(PY),
        str(COMFY_RUN),
        "--image",
        str(image),
        "--audio",
        str(audio_wav),
        "--out",
        str(out_mp4),
        "--prompt",
        PROMPT,
        "--size",
        str(size),
        "--steps",
        str(steps),
    ]
    subprocess.run(cmd, check=True, cwd=str(HERE))
    if not out_mp4.exists():
        raise RuntimeError("ComfyUI finished but output mp4 was not written")
    return out_mp4


def portrait_for_profile(scan_dir: Path) -> Path:
    for pattern in ("angle_100_*.jpg", "angle_100_*.jpeg", "angle_13_*.jpg"):
        hits = sorted(scan_dir.glob(pattern))
        if hits:
            return hits[-1]
    hits = sorted(scan_dir.glob("angle_*.jpg"))
    if not hits:
        raise FileNotFoundError(f"no scan photos in {scan_dir}")
    return hits[0]


def enabled() -> bool:
    return os.environ.get("AVATAR_PHOTOREAL_VIDEO", "").strip() == "1"