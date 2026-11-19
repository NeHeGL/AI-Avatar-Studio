"""Server-side face landmarks for the avatar.

Browsers differ (privacy settings, GPU/CPU delegates) in whether MediaPipe can analyse a still
photo — in some, it hangs the whole page. So the server finds the landmarks once with MediaPipe
for Python, caches them next to the photo, and the page just uses the numbers.

Needs a Python with mediapipe + opencv. By default the Camera-Effects project's venv is used;
override with the AVATAR_LANDMARK_PYTHON / AVATAR_LANDMARK_MODEL environment variables.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "detect_landmarks.py"
_CAMERA_EFFECTS = ROOT.parent.parent / "Published" / "Camera-Effects"


def _python() -> Path | None:
    for cand in (os.environ.get("AVATAR_LANDMARK_PYTHON"), str(_CAMERA_EFFECTS / ".venv" / "Scripts" / "python.exe")):
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _model() -> Path | None:
    for cand in (os.environ.get("AVATAR_LANDMARK_MODEL"), str(ROOT / "face_landmarker.task"), str(_CAMERA_EFFECTS / "face_landmarker.task")):
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def landmarks_for(image_path: str | Path) -> dict:
    """Return {"w","h","landmarks":[[x,y,z]…]} for the photo, or {"error": "..."}."""
    image = Path(image_path)
    if not image.exists():
        return {"error": f"image not found: {image.name}"}
    sidecar = image.with_suffix(".lm.json")
    if sidecar.exists() and sidecar.stat().st_mtime >= image.stat().st_mtime:
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    py, model = _python(), _model()
    if py is None or model is None:
        return {"error": "landmark helper unavailable (need a Python with mediapipe and the face_landmarker.task model)"}
    try:
        proc = subprocess.run(
            [str(py), str(SCRIPT), str(image), str(model)],
            capture_output=True,
            text=True,
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
        data = json.loads(lines[-1]) if lines else {"error": (proc.stderr or "no output").strip()[-200:]}
    except subprocess.TimeoutExpired:
        return {"error": "landmark detection timed out"}
    except (OSError, ValueError) as exc:
        return {"error": f"landmark detection failed: {exc}"}
    if "landmarks" in data:
        try:
            sidecar.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass
    return data
